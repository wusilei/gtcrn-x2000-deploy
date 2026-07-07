/**
 * msa_init.c — Enable MSA on MIPS32R5 XBurst2 without switching float ABI
 *
 * MSA requires FPU in FR=1 (64-bit FPU register mode) and CU1=1.
 * This function enables the FPU just enough for MSA instructions to work,
 * without changing the soft-float ABI (no float params in FPU regs).
 *
 * Call once before any MSA-accelerated function.
 */

void msa_init(void) {
    unsigned int status, config5;

    /* Enable CP1 (FPU) in Status register */
    __asm__ volatile (
        "mfc0   %0, $12\n\t"       /* read Status */
        "or     %0, %0, 0x20000000\n\t" /* set CU1 (bit 29) */
        "mtc0   %0, $12\n\t"       /* write back */
        : "=r"(status)
    );

    /* Set FR=1 in Config5 register (64-bit FPU mode, required for MSA) */
    __asm__ volatile (
        ".set    push\n\t"
        ".set    mips32r2\n\t"
        "mfc0   %0, $16, 5\n\t"    /* read Config5 */
        "or     %0, %0, 0x400\n\t" /* set FR (bit 10) */
        "mtc0   %0, $16, 5\n\t"    /* write back */
        ".set    pop\n\t"
        : "=r"(config5)
    );

    /* Enable MSA in Config5 register (MSAEn, bit 27) */
    __asm__ volatile (
        ".set    push\n\t"
        ".set    mips32r2\n\t"
        "mfc0   %0, $16, 5\n\t"
        "or     %0, %0, 0x08000000\n\t" /* set MSAEn (bit 27) */
        "mtc0   %0, $16, 5\n\t"
        ".set    pop\n\t"
        : "=r"(config5)
    );
}
