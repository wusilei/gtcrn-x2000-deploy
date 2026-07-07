#!/bin/bash
# build_msa.sh — Build DENOISE with MSA acceleration for X2000
# ==============================================================
# Architecture:
#   - MSA files compiled with -mhard-float -mmsa (separate .o)
#   - Main files compiled with -msoft-float
#   - Linked together (MSA functions pass only integer types → ABI safe)
#
# Usage:
#   ./build_msa.sh           # Full build → denoise_linux_q15_msa
#   ./build_msa.sh deploy    # Build + scp to X2000
#   ./build_msa.sh test      # Build unit tests

set -e
cd "$(dirname "$0")"

CC=/home/a/work/mips-gcc720-glibc229/bin/mips-linux-gnu-gcc
CFLAGS_SOFT="-std=c99 -Wall -O3 -march=mips32r2 -msoft-float -DHAS_MSA -I."
CFLAGS_MSA="-std=c99 -Wall -O3 -march=mips32r5 -mhard-float -mfp64 -mmsa -DHAS_MSA -I."
LDFLAGS="-lm -lrt -static"
TARGET="denoise_linux_q15_msa"

# MSA source files (compiled with hard-float + MSA)
MSA_SRCS="msa/bm_fixed_msa.c"
MSA_OBJS=""

# Main source files (compiled with soft-float)
MAIN_SRCS="agc.c denoise_linux.c noise_reduction_q15.c denoise_fp.c denoise_infer.c"
MAIN_OBJS=""

X2000_HOST="root@192.168.42.159"
X2000_PATH="/data/denoise_linux_q15_msa"

echo "=== DENOISE MSA Build ==="
echo ""

# Compile MSA files
echo "--- MSA objects (-mhard-float -mmsa) ---"
for src in $MSA_SRCS; do
    obj="${src%.c}.o"
    echo "  $src → $obj"
    $CC $CFLAGS_MSA -c "$src" -o "$obj"
    MSA_OBJS="$MSA_OBJS $obj"
done
echo ""

# Compile main files
echo "--- Main objects (-msoft-float) ---"
for src in $MAIN_SRCS; do
    obj="${src%.c}.o"
    echo "  $src → $obj"
    $CC $CFLAGS_SOFT -c "$src" -o "$obj"
    MAIN_OBJS="$MAIN_OBJS $obj"
done
echo ""

# Link
echo "--- Link $TARGET ---"
$CC $CFLAGS_SOFT -o "$TARGET" $MAIN_OBJS $MSA_OBJS $LDFLAGS
echo "  Binary: $TARGET ($(ls -lh $TARGET | awk '{print $5}'))"
echo ""

# Verify MSA instructions present
echo "--- MSA instructions in binary ---"
/home/a/work/mips-gcc720-glibc229/bin/mips-linux-gnu-objdump -d "$TARGET" | grep -c 'madd_q.w\|ld.w\|fill.w\|ilvr.w\|slli.w' | while read cnt; do
    echo "  MSA ops: $cnt"
done
echo ""

# Deploy
if [ "${1:-}" = "deploy" ]; then
    echo "--- Deploy to X2000 ---"
    scp "$TARGET" "${X2000_HOST}:${X2000_PATH}"
    ssh "$X2000_HOST" "chmod +x ${X2000_PATH}"
    echo "  Deployed to ${X2000_PATH}"

    # Kill old denoise, start new
    ssh "$X2000_HOST" "killall denoise_linux_q15 2>/dev/null; killall denoise_linux_q15_msa 2>/dev/null; sleep 1"
    echo "  Old processes killed"

    echo ""
    echo "  To start on X2000:"
    echo "    ssh $X2000_HOST ${X2000_PATH} &"
fi

echo "=== Done ==="
