#!/bin/bash

# === CONFIGURATION ===
BRANCH="main"  # Change this if you're pushing to a different branch
COMMIT_PREFIX="Batch commit: Creating vector dbs for recipes and wikihow"  # Prefix for commit messages

# === SCRIPT ===
echo "Finding uncommitted files..."
FILES=($(git ls-files --others --exclude-standard; git diff --name-only --cached))

if [ ${#FILES[@]} -eq 0 ]; then
  echo "No files to commit."
  exit 0
fi

COUNT=0
TOTAL=${#FILES[@]}

while [ $COUNT -lt $TOTAL ]; do
  echo "Processing files $COUNT to $((COUNT + 9))..."
  BATCH=("${FILES[@]:$COUNT:10}")

  # Add files
  git add "${BATCH[@]}"

  # Commit
  git commit -m "$COMMIT_PREFIX $((COUNT / 10 + 1))"

  # Push
  git push origin "$BRANCH"
  
  # Move to next batch
  COUNT=$((COUNT + 10))
done

echo "All batches committed and pushed."
