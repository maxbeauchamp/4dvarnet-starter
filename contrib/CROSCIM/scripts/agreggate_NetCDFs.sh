#!/bin/bash

INPUT_DIR="/dmidata/users/maxb/PREPROC"
OUTPUT_DIR="/dmidata/users/maxb/PREPROC"
PREFIX="preproc_batch"
OUTPUT_PREFIX="preproc_CROSCIM"

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

    tmpfile="${OUTPUT_DIR}/tmp_x${res}.nc"
    outfile="${OUTPUT_DIR}/${OUTPUT_PREFIX}_x${res}.nc"

    # Concatenate
    ncecat -O "${valid_files[@]}" "$tmpfile" \
        || { echo "  ❌ ncecat failed for x${res}, skipping"; continue; }

    # Rename record→sample if needed
    if ! ncdump -h "$tmpfile" | grep -q "sample ="; then
        echo "  🔄 Renaming 'record' to 'sample'"
        ncrename -O -d record,sample "$tmpfile"
    fi

    cp -f "$tmpfile" "$outfile"
    rm -f "$tmpfile"

    # Compress
    ncks -O --cnk_dmn time,1 \
            --cnk_dmn sample,1 \
            --deflate 9 \
            "$outfile" "${outfile%.nc}_compressed.nc"
    mv "${outfile%.nc}_compressed.nc" "$outfile"

    echo "  ✅ Saved: $outfile  (${#valid_files[@]} batches)"
done

