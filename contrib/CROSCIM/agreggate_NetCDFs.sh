#!/bin/bash

# Dossier contenant les fichiers .nc
INPUT_DIR="/dmidata/users/maxb/PREPROC"   # à adapter si nécessaire
OUTPUT_DIR="/dmidata/users/maxb/PREPROC"  # dossier de sortie
PREFIX="preproc_batch"
OUTPUT_PREFIX="preproc_CROSCIM"

cd "$INPUT_DIR" || exit 1

# Get all unique resolutions from filenames like preproc_batch_*_x10.nc
res_list=$(ls "$INPUT_DIR"/preproc_batch_*_x*.nc | sed -n 's/.*_x\([0-9]*\)\.nc/\1/p' | sort -n | uniq)

for res in $res_list; do
    echo "🔍 Processing resolution: x${res}"

    files=$(ls "$INPUT_DIR"/preproc_batch_*_x${res}.nc 2>/dev/null)
    if [ -z "$files" ]; then
        echo "⚠️  No files found for resolution x${res}, skipping..."
        continue
    fi

    tmpfile="${OUTPUT_DIR}/tmp_x${res}.nc"
    outfile="${OUTPUT_DIR}/${OUTPUT_PREFIX}_x${res}.nc"

    # Step 1: concat across new unlimited dim (record)
    ncecat -O $files "$tmpfile"

    # Step 2: check if "sample" already exists
    if ! ncdump -h "$tmpfile" | grep -q "sample ="; then
        echo "🔄 Renaming 'record' to 'sample'"
        ncrename -O -d record,sample "$tmpfile"
    else
        echo "⚠️  Dimension 'sample' already exists, skipping rename"
    fi

    # Step 3: reorder dimensions (optional)
    #ncpdq -O "$tmpfile" "$outfile"
    cp -rf "$tmpfile" "$outfile"
    rm "$tmpfile"

    # Step 4: compression
    ncks -O --cnk_dmn time,1 \
            --cnk_dmn sample,1 \
            --deflate 9 \
            "$outfile" "${outfile%.nc}_compressed.nc"

    # Replace original with compressed version
    mv "${outfile%.nc}_compressed.nc" "$outfile"

    echo "✅ Saved: $outfile"
done

