#!/bin/bash

# Taille limite en octets
LIMIT=$((100 * 1024 * 1024))
WARNING=$((50 * 1024 * 1024))

echo "📂 Scan du répertoire : $(pwd)"

# Cherche tous les fichiers et calcule leur taille
find . -type f -print0 | while IFS= read -r -d '' file; do
    size=$(stat -c%s "$file")

    if [ "$size" -ge "$LIMIT" ]; then
        echo "❌ Trop lourd (>100 Mo): $file ($(echo "scale=2; $size/1024/1024" | bc) Mo)"
    elif [ "$size" -ge "$WARNING" ]; then
        echo "⚠️ Volumineux (>50 Mo): $file ($(echo "scale=2; $size/1024/1024" | bc) Mo)"
    fi
done

echo "✅ Scan terminé."

