"""Deployment helper functions."""

import subprocess
import streamlit as st

def get_commit_sha() -> str:
    """Get the short commit SHA for version tracking."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def show_deployment_info():
    """Show deployment info (footer)."""
    st.divider()
    st.caption(f"Streamlit v{st.__version__} | Commit: {get_commit_sha()}")
