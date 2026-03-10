#!/bin/bash

INPUT_DIR="/dmidata/users/maxb/PREPROC"
OUTPUT_DIR="/dmidata/users/maxb/PREPROC"
PREFIX="preproc_batch"
OUTPUT_PREFIX="preproc_CROSCIM"

cd "$INPUT_DIR" || exit 1

# Extraire les résolutions disponibles
res_list=$(ls "$INPUT_DIR"/${PREFIX}_*_x*.nc 2>/dev/null | sed -n 's/.*_x\([0-9]*\)\.nc/\1/p' | sort -n | uniq)

for res in $res_list; do
    echo "🔍 Processing resolution: x${res}"

    files=$(ls "$INPUT_DIR"/${PREFIX}_*_x${res}.nc 2>/dev/null)
    if [ -z "$files" ]; then
        echo "⚠️  No files found for resolution x${res}, skipping..."
        continue
    fi

    # Prendre le premier fichier comme référence
    ref_file=$(echo "$files" | head -n 1)
    ref_vars=$(ncdump -h "$ref_file" | grep 'float\|double\|int' | awk '{print $2}' | sed 's/(.*//')

    echo "📋 Reference file: $ref_file"
    echo "✅ Expected variables: $ref_vars"

    valid_files=()
    for f in $files; do
        all_vars=$(ncdump -h "$f" | grep 'float\|double\|int' | awk '{print $2}' | sed 's/(.*//')
        missing=$(comm -23 <(echo "$ref_vars" | sort) <(echo "$all_vars" | sort))
        if [ -z "$missing" ]; then
            valid_files+=("$f")
        else
            echo "⚠️ Skipping $f (missing variables: $missing)"
        fi
    done

    if [ ${#valid_files[@]} -eq 0 ]; then
        echo "❌ No valid files remain for resolution x${res}, skipping..."
        continue
    fi

    tmpfile="${OUTPUT_DIR}/tmp_x${res}.nc"
    outfile="${OUTPUT_DIR}/${OUTPUT_PREFIX}_x${res}.nc"

    # Step 1: concat across new unlimited dim
    ncecat -O "${valid_files[@]}" "$tmpfile" || { echo "❌ ncecat failed for x${res}, skipping"; continue; }

    # Step 2: rename dim if needed
    if ! ncdump -h "$tmpfile" | grep -q "sample ="; then
        echo "🔄 Renaming 'record' to 'sample'"
        ncrename -O -d record,sample "$tmpfile"
    fi

    # Step 3: save output
    cp -f "$tmpfile" "$outfile"
    rm -f "$tmpfile"

    # Step 4: compression
    ncks -O --cnk_dmn time,1 \
            --cnk_dmn sample,1 \
            --deflate 9 \
            "$outfile" "${outfile%.nc}_compressed.nc"

    mv "${outfile%.nc}_compressed.nc" "$outfile"

    echo "✅ Saved: $outfile"
done

