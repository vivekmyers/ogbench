#!/bin/bash

# Fix imports in all agent files
for file in impls/agents/*.py; do
  if [ "$file" != "impls/agents/__init__.py" ]; then
    echo "Fixing imports in $file"
    sed -i '' 's/from utils\./from impls.utils./g' "$file"
  fi
done

echo "All imports fixed!" 