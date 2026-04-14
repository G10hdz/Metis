#!/bin/bash
# Sync changes from private Metis to Metis-public
# Usage: ./scripts/sync-to-public.sh
# 
# This copies safe files from the working copy to the public repo.
# It NEVER copies .env, databases, logs, or any sensitive data.

set -e

# Edit these paths to match your setup
PRIVATE="/home/yourusername/Projects/Metis"
PUBLIC="/home/yourusername/Vscode-projects/Metis-public"

echo "🔄 Syncing Metis → Metis-public..."

rsync -av --delete \
    --exclude='.git' \
    --exclude='.env' \
    --exclude='.env.*' \
    --exclude='*.db' \
    --exclude='*.sqlite*' \
    --exclude='*.log' \
    --exclude='ergane-logs.md' \
    --exclude='.pytest_cache' \
    --exclude='__pycache__' \
    --exclude='.claude' \
    --exclude='.qwen' \
    --exclude='.venv' \
    --exclude='*.egg-info' \
    --exclude='metis.log' \
    --exclude='*.lock' \
    --exclude='INIT.md' \
    --exclude='SECURITY_AUDIT.md' \
    "$PRIVATE/" "$PUBLIC/"

echo ""
echo "✅ Sync complete."
echo ""
echo "Review changes in $PUBLIC, then:"
echo "  cd $PUBLIC"
echo "  git add ."
echo "  git commit -m \"sync: update from private\""
echo "  git push origin main"
