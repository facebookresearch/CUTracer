set -eu

# Define project root path (absolute path)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project root: $PROJECT_ROOT"

test_trace_formats_clp() {
  # ===== Test Mode 3 (CLP Archive Compression) =====
  echo ""
  echo "  📦 Testing Mode 3 (CLP Archive)..."

  test_dir=$PROJECT_ROOT/tests/py_add
  cd $test_dir

# Clean old clp files
  rm -rf kernel_*triton_poi_fused*.clp

  if ! TRACE_FORMAT_NDJSON=3 \
       CUDA_INJECTION64_PATH="$PROJECT_ROOT/lib/cutracer.so" \
       CUTRACER_INSTRUMENT=reg_trace,mem_trace \
       KERNEL_FILTERS=triton_poi_fused \
       python ./test_add.py; then
    echo "    ❌ Mode 3 execution failed"
    mode3_status="failed"
  else
    # Find generated .clp file (PT2 compiled Triton kernel)
    mode3_archive=$(ls -1dt kernel_*triton_poi_fused*.clp 2>/dev/null | head -n 1)
    if [ -z "$mode3_archive" ]; then
      echo "    ❌ No .clp archive generated"
      mode1_status="failed"
    else
      echo "    ✅ Found: $mode3_archive"
      # Get compressed archive size
      local compressed_archive_size=$(du -sb "$mode3_archive" 2>/dev/null | cut -f1)
      echo "       Compressed archive size: $compressed_archive_size bytes"

      # Decompress for validation
      echo "    🔓 Decompressing..."
      # Decompress CLP archive with clp-s
      clp-s x "$mode3_archive" decompressed
      # Get decompressed file size
      local decompressed_size=$(du -sb decompressed 2>/dev/null | cut -f1)
      local ratio=$(awk "BEGIN {printf \"%.1f\", ($decompressed_size / $compressed_archive_size)}")

      echo "    ✅ Decompression successful"
      echo "       CLP Archive size:  $compressed_archive_size bytes"
      echo "       Decompressed size: $decompressed_size bytes"
      echo "       Compression ratio: ${ratio}x"

      # Validate CLP Archive with decompressed JSON
      echo "    🔍 Validating CLP Archive format..."
      decompressed_file=$(ls -1t decompressed/* 2>/dev/null | head -n 1)
      mv "$decompressed_file" decompressed/mode3_decompressed.ndjson
      decompressed_file="decompressed/mode3_decompressed.ndjson"
      if python3 "$PROJECT_ROOT/scripts/validate_trace.py" --no-color json $decompressed_file >mode3_validation.log 2>&1; then
        mode3_status="passed"

        # Count each trace type separately
        mode3_reg_count=$(grep -ac '"type":"reg_trace"' $decompressed_file 2>/dev/null | tr -d '[:space:]')
        mode3_mem_count=$(grep -ac '"type":"mem_trace"' $decompressed_file 2>/dev/null | tr -d '[:space:]')
        mode3_total_count=$(wc -l < $decompressed_file 2>/dev/null | tr -d '[:space:]')

        echo "    ✅ Mode 3 validation passed"
        echo "       📊 Record breakdown:"
        echo "          reg_trace:  $mode3_reg_count records"
        echo "          mem_trace:  $mode3_mem_count records"
        echo "          Total:      $mode3_total_count records"

        # Show first record of each type (formatted)
        echo "       First reg_trace record (formatted):"
        grep -a '"type":"reg_trace"' $decompressed_file | head -1 | python3 -m json.tool | head -20 | sed 's/^/         /'

        if [ "$mode3_mem_count" -gt 0 ]; then
          echo "       First mem_trace record (formatted):"
          grep -a '"type":"mem_trace"' $decompressed_file | head -1 | python3 -m json.tool | head -20 | sed 's/^/         /'
        fi
      else
        echo "    ❌ Mode 3 validation failed"
        echo "    === Validation errors ==="
        cat mode3_validation.log
        mode3_status="failed"
      fi
    fi
  fi

  # ===== Final Report =====
  echo ""
  echo "  ═══════════════════════════════════════════"
  echo "  📋 Test Summary"
  echo "  ═══════════════════════════════════════════"
  echo "  Mode 3 (CLP):           $mode3_status"
  echo "  ═══════════════════════════════════════════"

  # Determine overall result - all four must pass
  if [ "$mode3_status" = "passed" ]; then
    echo "  ✅ CLP trace format tests passed!"
    return 0
  else
    echo "  ❌ CLP trace format tests failed"
    return 1
  fi
}

test_trace_formats_clp
