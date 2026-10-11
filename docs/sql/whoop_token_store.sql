-- À exécuter une fois avec un rôle d'administration, jamais par l'application.
-- Schéma privé : ne PAS ajouter whoop_private aux schémas exposés par l'API
-- Supabase/PostgREST. Les jetons sont chiffrés par l'application avant insertion.
CREATE SCHEMA IF NOT EXISTS whoop_private;

CREATE TABLE IF NOT EXISTS whoop_private.oauth_connections (
    namespace TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);

REVOKE ALL ON SCHEMA whoop_private FROM PUBLIC;
REVOKE ALL ON TABLE whoop_private.oauth_connections FROM PUBLIC;

-- Utiliser un rôle de connexion dédié, sans accès des rôles anon/authenticated.
-- Le propriétaire de la base peut lui accorder uniquement :
-- GRANT USAGE ON SCHEMA whoop_private TO votre_role_prive;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON whoop_private.oauth_connections TO votre_role_prive;
-- La clé Fernet reste dans les secrets du déploiement, séparément de la base.
