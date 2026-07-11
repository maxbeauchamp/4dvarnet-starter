#!/bin/bash

INPUT_DIR="/Odyssey/public/CROSCIM_dataset"
OUTPUT_DIR="/Odyssey/public/CROSCIM_dataset"
PREFIX="preproc_batch"
OUTPUT_PREFIX="preproc_CROSCIM"

# Désactive le locking HDF5 (échoue sur NFS -> NC_EHDFERR pendant l'écriture)
export HDF5_USE_FILE_LOCKING=FALSE

cd "$INPUT_DIR" || exit 1

# ── Step 1: collect all resolutions and all batch IDs ─────────────────────
res_list=$(ls "$INPUT_DIR"/${PREFIX}_*_x*.nc 2>/dev/null \
           | sed -n 's/.*_x\([0-9]*\)\.nc/\1/p' | sort -n | uniq)

batch_list=$(ls "$INPUT_DIR"/${PREFIX}_*_x*.nc 2>/dev/null \
             | sed -n "s|.*/${PREFIX}_\([0-9]*\)_x[0-9]*\.nc|\1|p" | sort -n | uniq)

echo "📐 Resolutions found : $(echo $res_list | tr '\n' ' ')"
echo "📦 Batch IDs found   : $(echo $batch_list | wc -w) batches"

# ── Step 2: validate every file individually ───────────────────────────────
# A file is valid if:
#   a) ncdump -h succeeds (header readable)
#   b) ncecat on the single file succeeds (data readable, no dim-bound errors)
# Result: declare associative array  valid[batch_id,res] = 1

declare -A valid   # valid[batchid_res]=1 if file is OK

_tmpnc=$(mktemp /tmp/ncecat_check_XXXXXX.nc)

echo ""
echo "🔎 Validating all files..."
for res in $res_list; do
    for batch in $batch_list; do
        f="${INPUT_DIR}/${PREFIX}_${batch}_x${res}.nc"
        [ -f "$f" ] || continue

        if ! ncdump -h "$f" &>/dev/null; then
            echo "  ❌ $f  →  corrupt header"
            continue
        fi

        if ! ncecat -O "$f" "$_tmpnc" &>/dev/null; then
            echo "  ❌ $f  →  data-read error (ncecat)"
            continue
        fi

        valid["${batch}_${res}"]=1
    done
done
rm -f "$_tmpnc"

# ── Step 3: keep only batch IDs that are valid for ALL resolutions ─────────
echo ""
echo "🔗 Filtering batch IDs valid across all resolutions..."
good_batches=()
for batch in $batch_list; do
    ok=true
    for res in $res_list; do
        f="${INPUT_DIR}/${PREFIX}_${batch}_x${res}.nc"
        if [ ! -f "$f" ] || [ -z "${valid[${batch}_${res}]+_}" ]; then
            ok=false
            break
        fi
    done
    if $ok; then
        good_batches+=("$batch")
    else
        echo "  ⚠️  Batch $batch excluded (invalid or missing file for at least one resolution)"
    fi
done

echo "✅ ${#good_batches[@]} complete valid batches retained out of $(echo $batch_list | wc -w)"

# ── Step 4: concatenate per resolution using only good batches ─────────────
echo ""
for res in $res_list; do
    echo "🔍 Processing resolution: x${res}"

    valid_files=()
    for batch in "${good_batches[@]}"; do
        valid_files+=("${INPUT_DIR}/${PREFIX}_${batch}_x${res}.nc")
    done

    if [ ${#valid_files[@]} -eq 0 ]; then
        echo "  ❌ No valid files for x${res}, skipping..."
        continue
    fi

    outfile="${OUTPUT_DIR}/${OUTPUT_PREFIX}_x${res}.nc"

    # Concaténation + compression à la volée : écrit directement le fichier
    # compressé, sans matérialiser l'intermédiaire non compressé géant (~180 Go)
    ncecat -O -L 1 --cnk_dmn time,1 "${valid_files[@]}" "$outfile" \
        || { echo "  ❌ ncecat failed for x${res}, skipping"; continue; }

    # Rename record→sample if needed
    if ! ncdump -h "$outfile" | grep -q "sample ="; then
        echo "  🔄 Renaming 'record' to 'sample'"
        ncrename -O -d record,sample "$outfile"
    fi

    echo "  ✅ Saved: $outfile  (${#valid_files[@]} batches)"
done

