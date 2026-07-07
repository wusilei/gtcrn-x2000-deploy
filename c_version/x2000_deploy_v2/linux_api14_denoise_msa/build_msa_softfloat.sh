#!/bin/bash
# build_msa_softfloat.sh — Build DENOISE with MSA on soft-float base
# Uses assembly bridge: MSA code compiled to asm, nan2008 flag cleared,
# linked with soft-float main code for bit-exact output match.
set -e
cd "$(dirname "$0")"

CC=/home/a/work/mips-gcc720-glibc229/bin/mips-linux-gnu-gcc
OBJCOPY=/home/a/work/mips-gcc720-glibc229/bin/mips-linux-gnu-objcopy

echo "=== 1. MSA .c → .s (hard-float + MSA) ==="
$CC -S -O3 -mmsa -mhard-float -mfp64 -march=mips32r5 -DHAS_MSA -I. msa/bm_fixed_msa.c -o msa/bm_fixed_msa.s

echo "=== 2. Strip .module directives ==="
sed -i '/^\s*\.module\s\+fp=64/d;/^\s*\.module\s\+oddspreg/d' msa/bm_fixed_msa.s

echo "=== 3. Assemble with soft-float + MSA opcodes ==="
$CC -c -Wa,-mmsa -march=mips32r5 -msoft-float -I. msa/bm_fixed_msa.s -o msa/bm_fixed_msa.o

echo "=== 4. Clear nan2008 flag ==="
python3 -c "
import struct
with open('msa/bm_fixed_msa.o','rb') as f: data=bytearray(f.read())
assert data[0:4]==b'\x7fELF'
e='>' if data[5]==2 else '<'
f=struct.unpack(e+'I',data[36:40])[0]
data[36:40]=struct.pack(e+'I',f&~0x400)
with open('msa/bm_fixed_msa_softfloat.o','wb') as f: f.write(data)
"

echo "=== 5. Compile main (soft-float + P0) ==="
for src in agc.c denoise_linux.c noise_reduction_q15.c denoise_fp.c denoise_infer.c; do
    echo "  $src"
    $CC -std=c99 -Wall -O3 -march=mips32r2 -msoft-float -DHAS_MSA -I. -c "$src" -o "${src%.c}.o"
done

echo "=== 6. Link ==="
$CC -march=mips32r2 -msoft-float -o denoise_linux_q15_msa \
    agc.o denoise_linux.o noise_reduction_q15.o denoise_fp.o denoise_infer.o \
    msa/bm_fixed_msa_softfloat.o -lm -lrt -static

echo "  Binary: $(ls -lh denoise_linux_q15_msa | awk '{print $5}')"
echo "  MSA ops: $($OBJCOPY -O binary --only-section=.text denoise_linux_q15_msa /dev/stdout 2>/dev/null | xxd | grep -c 'madd_q\|fill.w\|ld.w') (approx)"
echo ""
echo "=== Done ==="
