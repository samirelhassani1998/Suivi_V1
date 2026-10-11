# Conserver la connexion WHOOP entre les visites

Le mode personnel conserve les jetons OAuth WHOOP chiffrés dans une base PostgreSQL externe. Après une première autorisation, une nouvelle session authentifiée retrouve la connexion dans `Whoop` ou `Boxe` et tente une synchronisation. Les récupérations, nuits, cycles, séances et autres données physiologiques restent uniquement dans la session Streamlit : elles ne sont écrites ni dans cette base, ni dans Google Sheets.

Cette option est désactivée par défaut. Sans elle, la connexion reste limitée à la session. Le code et la migration sont fournis ; aucun backend de production n'a été provisionné ou configuré dans le cadre de cette livraison. L'activation nécessite les opérations ci-dessous sur votre infrastructure existante.

## Périmètre personnel

L'application utilise un mot de passe commun, sans compte individuel : **toute personne qui connaît ce mot de passe peut retrouver la même liaison WHOOP et consulter ses données**. Ce mode est réservé à votre déploiement personnel ; il ne fournit aucune séparation entre plusieurs utilisateurs.

Il exige `enabled = true`, `mode = "personal"`, une protection `[auth]` active avec un mot de passe configuré et une session authentifiée. Le mode démo et une session non authentifiée ne chargent pas le coffre. Les identifiants WHOOP doivent venir des secrets serveur ou des variables d'environnement ; la saisie manuelle propre à un onglet est ignorée en mode persistant.

Le stockage local, notamment SQLite, ne convient pas ici : Streamlit Community Cloud ne garantit pas la conservation des fichiers locaux. Une base externe permet de retrouver les jetons après le remplacement ou le redémarrage de l'instance. [Documentation Streamlit sur les connexions et le stockage local](https://docs.streamlit.io/develop/concepts/connections/connecting-to-data).

## 1. Préparer une base PostgreSQL existante

Il faut une base accessible depuis le serveur Streamlit, un accès administrateur pour installer le schéma et un rôle dédié pour l'application. Un projet Supabase existant convient ; la connexion utilise PostgreSQL directement, sans clé API Supabase ni Data API.

Avec un rôle administrateur, exécuter une fois le fichier [sql/whoop_token_store.sql](sql/whoop_token_store.sql), dans l'éditeur SQL du fournisseur ou avec `psql`. Il crée :

- le schéma privé `whoop_private` ;
- la table `whoop_private.oauth_connections`, contenant une clé `namespace` et un `payload` chiffré ;
- les restrictions d'accès pour le rôle `PUBLIC`.

L'application ne lance pas cette migration et n'a pas besoin du droit de créer des objets. Dans Supabase, **ne pas ajouter `whoop_private` aux schémas exposés par l'API**, et ne donner aucun accès à cette table aux rôles `anon` ou `authenticated`.

Créer un rôle de connexion dédié, puis lui attribuer les droits suivants. Remplacer `postgres` par le nom réel de votre base si nécessaire ; ne pas réutiliser un rôle administrateur dans les secrets de l'application.

```sql
CREATE ROLE suivi_whoop LOGIN
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;

GRANT CONNECT ON DATABASE postgres TO suivi_whoop;
GRANT USAGE ON SCHEMA whoop_private TO suivi_whoop;
GRANT SELECT, INSERT, UPDATE, DELETE
    ON whoop_private.oauth_connections TO suivi_whoop;
```

Définir son mot de passe dans l'interface sécurisée du fournisseur ou avec la commande interactive `\password suivi_whoop` de `psql`. Cette commande n'est pas du SQL à coller dans un éditeur web. Ne pas enregistrer le mot de passe dans un fichier SQL versionné ou dans l'historique d'une commande shell.

Sur Supabase, ouvrir **Connect** et copier l'URI correspondant au réseau du déploiement. La connexion directe convient si son adresse est joignable ; le **Session pooler** permet notamment un accès depuis un réseau IPv4. Avec ce pooler, le nom de connexion d'un rôle personnalisé est `suivi_whoop.PROJECT_REF` ; reprendre l'hôte et le port fournis, sans les déduire du nom de région. Encoder les caractères réservés du mot de passe dans l'URI. [Instructions de connexion Supabase](https://supabase.com/docs/guides/database/connecting-to-postgres).

## 2. Générer et conserver la clé de chiffrement

Dans un environnement Python local où les dépendances du projet sont installées, générer une clé une seule fois :

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode('ascii'))"
```

Copier le résultat dans les secrets du déploiement et en conserver une sauvegarde privée. Ne placer **ni la clé, ni l'URI PostgreSQL, ni les jetons dans Git, Google Sheets ou une capture partagée**. Les dépendances sont fixées à `cryptography==50.0.2` pour Fernet et `psycopg[binary]==3.3.6` pour PostgreSQL.

La clé doit rester identique après un redéploiement. La remplacer sans migration rend les liaisons existantes illisibles ; le code refuse de les écraser avec une autre clé. Il n'y a pas de rotation de clé automatique. Une restauration exige la clé d'origine ; à défaut, l'administrateur doit retirer la liaison concernée avant une nouvelle autorisation WHOOP.

## 3. Configurer Streamlit

Dans Streamlit Community Cloud, ouvrir les paramètres de l'application, rubrique **Secrets**, et compléter les secrets existants avec les blocs suivants. Modifier les blocs déjà présents sans créer de doublon ; conserver notamment la configuration de la source de poids. En local, utiliser `.streamlit/secrets.toml`, ignoré par Git. Les valeurs en majuscules sont des exemples à remplacer. [Gestion officielle des secrets Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

```toml
[auth]
required = true
password = "VOTRE_MOT_DE_PASSE_APPLICATION"

[whoop]
client_id = "VOTRE_CLIENT_ID_WHOOP"
client_secret = "VOTRE_CLIENT_SECRET_WHOOP"
redirect_uri = "https://votre-app.streamlit.app/"

[whoop_persistence]
enabled = true
mode = "personal"
database_url = "postgresql://suivi_whoop:MOT_DE_PASSE_ENCODE@HOTE:5432/BASE?sslmode=require"
encryption_key = "CLE_FERNET_GENEREE_LOCALEMENT"
namespace = "suivi-personal"
```

Pour Supabase, remplacer l'URI générique entière par celle préparée à l'étape 1, avec le rôle dédié. Les variables `WHOOP_CLIENT_ID`, `WHOOP_CLIENT_SECRET` et `WHOOP_REDIRECT_URI` restent une alternative aux secrets WHOOP. La redirection doit correspondre exactement à l'une des URL enregistrées dans votre application développeur WHOOP.

Conserver un `namespace` stable. Utiliser un nom différent pour un déploiement de test indépendant : deux déploiements partageant la base, le namespace et le client WHOOP partagent la liaison. Le code associe le namespace au `client_id` puis en stocke une empreinte, et non la valeur littérale `suivi-personal`.

Pour un serveur distant, le client exige TLS : `sslmode=require`, `verify-ca` ou `verify-full`. Sans option explicite, il applique `require`. `require` chiffre la connexion ; pour vérifier aussi l'identité du serveur, utiliser `verify-full` avec le certificat racine approprié disponible sur le serveur Streamlit et un paramètre `sslrootcert` adapté. Le mode `verify-full` vérifie également le nom d'hôte. Les modes permettant un repli sans chiffrement sont refusés pour une base distante. [Modes TLS PostgreSQL](https://www.postgresql.org/docs/current/libpq-ssl.html).

## 4. Autoriser WHOOP puis vérifier deux sessions

Après installation de la migration et des secrets :

1. Ouvrir l'application et saisir son mot de passe.
2. Aller dans `Whoop`, puis cliquer sur **Autoriser l'accès WHOOP** et accepter la demande. Une connexion jusque-là limitée à une session n'est pas migrée automatiquement : effectuer cette première autorisation après activation du mode persistant.
3. Vérifier que la connexion est annoncée comme conservée entre les visites et que la synchronisation affiche les données attendues.
4. Ouvrir l'application dans un deuxième onglet ou une fenêtre privée, s'authentifier, puis ouvrir `Whoop` ou `Boxe`. La même liaison doit être retrouvée sans nouvelle demande de consentement WHOOP ; une synchronisation est tentée pour remplir cette nouvelle session.
5. Fermer puis rouvrir une session, ou vérifier après un redémarrage contrôlé du déploiement, en conservant la base et les mêmes secrets. La connexion doit être retrouvée de la même manière.

Le scope `offline` est imposé en mode persistant pour obtenir le jeton de renouvellement. WHOOP renouvelle ensemble le jeton d'accès et celui de renouvellement, en invalidant les anciens. Ce comportement exige d'utiliser la dernière paire disponible, notamment avec plusieurs onglets. [Flux OAuth WHOOP](https://developer.whoop.com/docs/developing/oauth/).

La restauration déclenche une seule tentative de synchronisation par liaison restaurée dans la session. Elle ne lance ni tâche de fond permanente, ni boucle de tentatives. En cas d'échec, utiliser **Synchroniser maintenant** après résolution du problème. Le renouvellement du jeton intervient lors d'une synchronisation quand il est nécessaire.

## Déconnexion et révocation

**Déconnecter WHOOP** ou **Réinitialiser la connexion** supprime la liaison du coffre et les données WHOOP de la session courante. Les autres sessions retirent cette connexion et leurs données au prochain rerun de `Whoop` ou `Boxe` qui relit le coffre ; un onglet immobile n'est pas effacé à distance instantanément. Pour le vérifier, déconnecter dans le premier onglet puis déclencher un rerun dans le deuxième : il doit proposer de relier WHOOP de nouveau.

Cette action **ne révoque pas l'autorisation accordée chez WHOOP**. Pour retirer aussi cette autorisation, révoquer l'accès de l'application depuis WHOOP. Si la suppression en base échoue, l'application signale l'erreur et ne confirme pas une déconnexion durable réussie.

Passer `enabled` à `false` désactive l'utilisation du coffre, sans supprimer sa ligne. Pour retirer la liaison, se déconnecter pendant que la configuration et la base sont encore disponibles, puis désactiver l'option si souhaité. Une suppression dans l'application ne purge pas les éventuelles sauvegardes conservées par le fournisseur PostgreSQL.

## Erreurs et limites de continuité

| Situation | Comportement et action |
| --- | --- |
| Mode personnel absent, protection désactivée ou session non authentifiée | La connexion durable n'est pas chargée. Corriger les secrets ou s'authentifier. |
| Base indisponible, schéma absent, droits insuffisants ou configuration TLS invalide | L'accès au coffre échoue ; aucun ancien jeton de l'onglet n'est utilisé en remplacement. Corriger la configuration puis relancer la synchronisation ou la page. |
| Mauvaise clé Fernet ou contenu illisible | L'accès est refusé et la ligne n'est pas écrasée. Restaurer la clé correcte. |
| Erreur réseau WHOOP ou erreur temporaire du service | La liaison n'est pas supprimée automatiquement. Réessayer manuellement quand le service est disponible. |
| Jeton révoqué ou `invalid_grant` persistant | Refaire l'autorisation WHOOP. Une liaison conservée n'est pas une garantie d'accès perpétuel. |

Chaque renouvellement lit le dernier jeton sous verrou transactionnel PostgreSQL, appelle WHOOP puis enregistre la nouvelle paire avant de libérer le verrou. Les onglets coopérant avec le même coffre évitent ainsi de renouveler simultanément un ancien jeton. Une liaison supprimée n'est pas recréée par le rafraîchissement d'un onglet resté ouvert.

L'appel WHOOP et la validation de la transaction PostgreSQL ne forment toutefois pas une transaction distribuée. Si WHOOP a renouvelé les jetons mais que le processus s'arrête avant l'enregistrement, ou si cette validation échoue, le coffre peut contenir une paire déjà invalidée. Une nouvelle autorisation est alors nécessaire. Les interruptions de service, la révocation et les politiques d'expiration de WHOOP restent également possibles.

La base contient le contenu OAuth chiffré et l'identifiant technique de la liaison. Le serveur Streamlit détient la clé et peut déchiffrer les jetons pour appeler WHOOP : cette protection ne remplace ni la sécurité du déploiement ni celle de ses secrets. Les données physiologiques sont rechargées depuis WHOOP dans chaque session ; leur disponibilité dépend donc aussi de l'API WHOOP.
