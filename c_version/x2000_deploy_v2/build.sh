#!/bin/bash
# ================================================================
# build.sh — GTCRN C Build & Test Script for Linux VM
# ================================================================
# Usage:
#   chmod +x build.sh
#   ./build.sh              # PC GCC build + test
#   ./build.sh x2000        # X2000 MIPS cross-compile (set CROSS first)
#   ./build.sh clean        # Remove build artifacts
#
# Requires: gcc, make (or mips-cross toolchain for X2000)
# ================================================================

set -e
cd "$(dirname "$0")"

# ---- Config ----
CC=${CC:-gcc}
CFLAGS="-Wall -Wextra -O2 -std=c99"
LDFLAGS="-lm"
TARGET_PC="test_cmp_py"
TARGET_X2000="gtcrn_x2000"

# ---- Colors ----
RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GRN}PASS${NC} $*"; }
fail() { echo -e "${RED}FAIL${NC} $*"; }
info() { echo -e "${YLW}INFO${NC} $*"; }

# ---- Build PC test ----
build_pc() {
    info "Building PC test (${CC})..."

    # Build the comparison test (includes full wired inference)
    ${CC} ${CFLAGS} -o ${TARGET_PC} test_cmp_py.c gtcrn_fp.c ${LDFLAGS}
    pass "Compiled ${TARGET_PC}"

    # Build unit test
    ${CC} ${CFLAGS} -o test_gtcrn_pc test_gtcrn_pc.c gtcrn_fp.c ${LDFLAGS}
    pass "Compiled test_gtcrn_pc"

    # Build full test
    ${CC} ${CFLAGS} -o test_gtcrn_full test_gtcrn_full.c gtcrn_fp.c ${LDFLAGS}
    pass "Compiled test_gtcrn_full"
}

# ---- Run tests ----
run_tests() {
    info "Running unit tests..."
    ./test_gtcrn_pc || fail "Unit tests failed"

    info "Running full pipeline test..."
    ./test_gtcrn_full || fail "Full pipeline test failed"

    info "Running Python comparison..."
    if [ -f ref_frame_input.bin ] && [ -f ref_frame_crm.bin ]; then
        ./${TARGET_PC} ref_frame_input.bin ref_frame_crm.bin
        RET=$?
        if [ $RET -eq 0 ]; then
            pass "Python comparison matched!"
        else
            fail "Python comparison — see details above"
        fi
    else
        info "ref_frame_*.bin not found — skipping Python comparison"
        info "Run 'python ../generate_ref_data.py' to generate reference data"
    fi

    echo ""
    pass "All tests complete."
}

# ---- X2000 cross-compile ----
build_x2000() {
    MIPS_CC=${CROSS:-mips-linux-gnu-gcc}
    MIPS_CFLAGS="-Wall -O2 -std=c99 -march=mips32r2 -msoft-float -DNNO_MALLOC"
    MIPS_LDFLAGS="-lm -static"

    info "Cross-compiling for X2000 MIPS32R2 (${MIPS_CC})..."

    if ! command -v ${MIPS_CC} &>/dev/null; then
        fail "Cross-compiler '${MIPS_CC}' not found."
        info "Set CROSS=/path/to/mips-gcc or install mips toolchain."
        return 1
    fi

    # Build standalone inference binary
    ${MIPS_CC} ${MIPS_CFLAGS} -o ${TARGET_X2000} \
        test_cmp_py.c gtcrn_fp.c ${MIPS_LDFLAGS}
    pass "Compiled ${TARGET_X2000} for X2000"

    # Show binary info
    file ${TARGET_X2000}
    size ${TARGET_X2000}

    info "X2000 binary ready: ${TARGET_X2000}"
    info "Copy to X2000 and run: ./${TARGET_X2000}"
}

# ---- Clean ----
clean() {
    rm -f ${TARGET_PC} ${TARGET_X2000} test_gtcrn_pc test_gtcrn_full *.o
    info "Cleaned build artifacts."
}

# ---- Main ----
case "${1:-all}" in
    pc)     build_pc && run_tests ;;
    x2000)  build_x2000 ;;
    all)    build_pc && run_tests ;;
    clean)  clean ;;
    *)      echo "Usage: $0 {pc|all|x2000|clean}" ;;
esac
