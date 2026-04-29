#!/bin/bash

# Taille limite en octets (30 Mo)
LIMIT=$((30 * 1024 * 1024))

echo "📂 Scan du répertoire : $(pwd)"
echo "🔍 Vérification des fichiers > 30 Mo..."
echo ""

GITIGNORE=".gitignore"
FOUND_LARGE_FILES=0

# Cherche tous les fichiers et calcule leur taille
find . -type f -not -path "./.git/*" -print0 | while IFS= read -r -d '' file; do
    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
    
    if [ "$size" -ge "$LIMIT" ]; then
        FOUND_LARGE_FILES=1
        size_mb=$(echo "scale=2; $size/1024/1024" | bc)
        echo "⚠️  Fichier volumineux (>30 Mo): $file ($size_mb Mo)"
        
        # Nettoyer le chemin (enlever ./)
        clean_path="${file#./}"
        
        # Vérifier si le fichier est déjà dans .gitignore
        if [ -f "$GITIGNORE" ]; then
            if grep -qF "$clean_path" "$GITIGNORE"; then
                echo "   ✓ Déjà dans .gitignore"
            else
                echo "   ➕ Ajout à .gitignore"
                echo "$clean_path" >> "$GITIGNORE"
            fi
        else
            echo "   ➕ Création de .gitignore et ajout du fichier"
            echo "$clean_path" > "$GITIGNORE"
        fi
        echo ""
    fi
done

echo "✅ Scan terminé."
