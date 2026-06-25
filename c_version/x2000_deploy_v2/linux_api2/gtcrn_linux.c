/**
 * gtcrn_linux.c — GTCRN Linux 侧降噪主程序
 * =========================================
 * iccom rf11 读入 → GTCRN 降噪 → iccom wf12 写出
 *
 * 数据流: RTOS 每 25ms 发 200×int16 @ 8kHz → rf11
 *         Linux 处理后每 25ms 回 200×int16 @ 8kHz → wf12
 *
 * 编译: mips-linux-gnu-gcc -O3 -std=c99 -march=mips32r2 -msoft-float \
 *         -o gtcrn_linux gtcrn_linux.c noise_reduction.c \
 *         gtcrn_fp.c gtcrn_infer.c kiss_fft.c kiss_fftr.c -lm -static
 *
 * 运行: ./gtcrn_linux           (iccom 模式)
 *       ./gtcrn_linux test      (stdin→GTCRN→stdout, PC 调试)
 */
#include "noise_reduction.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#define FRAME_IN  200   /* 25ms @ 8kHz */

int main(int argc, char **argv) {
    int test_mode = (argc > 1 && strcmp(argv[1], "test") == 0);

    if (test_mode) {
        /* ═══════════════════════════════════════════════════
         *  自测模式: stdin → GTCRN → stdout (PC 调试用)
         * ═══════════════════════════════════════════════════ */
        fprintf(stderr, "GTCRN Linux — self-test mode (stdin→GTCRN→stdout)\n");
        noise_init();

        short in[FRAME_IN], out[FRAME_IN];
        while (fread(in, sizeof(short), FRAME_IN, stdin) == FRAME_IN) {
            noise_reduction(in, out);
            fwrite(out, sizeof(short), FRAME_IN, stdout);
            fflush(stdout);
        }
        noise_deinit();
    } else {
        /* ═══════════════════════════════════════════════════
         *  iccom 模式: rf11→GTCRN→wf12 (X2000 板上运行)
         * ═══════════════════════════════════════════════════ */
        fprintf(stderr, "GTCRN Linux — iccom mode (rf11→GTCRN→wf12)\n");

        int fd_rf = open("/dev/iccom_rf11", O_RDONLY);
        if (fd_rf < 0) { perror("open rf11"); return 1; }

        int fd_wf = open("/dev/iccom_wf12", O_WRONLY);
        if (fd_wf < 0) { perror("open wf12"); close(fd_rf); return 1; }

        noise_init();
        fprintf(stderr, "GTCRN ready. Waiting for audio from rtos...\n");

        while (1) {
            short in[FRAME_IN], out[FRAME_IN];
            ssize_t nr = read(fd_rf, in, FRAME_IN * sizeof(short));
            if (nr != FRAME_IN * sizeof(short)) {
                fprintf(stderr, "rf11 short read: %zd/%d\n", nr, FRAME_IN * (int)sizeof(short));
                continue;
            }
            noise_reduction(in, out);
            write(fd_wf, out, FRAME_IN * sizeof(short));
        }

        noise_deinit();
        close(fd_rf);
        close(fd_wf);
    }
    return 0;
}
