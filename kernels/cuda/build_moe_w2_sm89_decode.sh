#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_file="${script_dir}/moe_w2_sm89_decode.cu"
output_file="${1:-/tmp/moe_w2_sm89_decode.cubin}"
nvcc_bin="${NVCC:-nvcc}"
cuobjdump_bin="${CUOBJDUMP:-cuobjdump}"

mkdir -p -- "$(dirname -- "${output_file}")"

"${nvcc_bin}" \
  -std=c++17 \
  -O3 \
  -arch=sm_89 \
  --cubin \
  -lineinfo \
  -Xptxas=-v,-warn-spills \
  -o "${output_file}" \
  "${source_file}"

echo
echo "Resource usage"
"${cuobjdump_bin}" --dump-resource-usage "${output_file}"

sass_file="$(mktemp)"
trap 'rm -f -- "${sass_file}"' EXIT
"${cuobjdump_bin}" --dump-sass "${output_file}" >"${sass_file}"

echo
echo "Kernel symbols"
grep -E 'Function : moe_w2_sm89_decode_' "${sass_file}"

echo
echo "FP8 tensor-core instructions"
if ! grep -E '(Q|H)MMA\..*(E4M3|FP8)' "${sass_file}" | head -n 24; then
  echo "error: no E4M3/FP8 tensor-core instruction found in cubin" >&2
  exit 1
fi

echo
echo "Built ${output_file}"
