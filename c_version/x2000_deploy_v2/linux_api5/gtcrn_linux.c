/**
 * gtcrn_linux.c — GTCRN Linux 侧降噪主程序
 * =========================================
 * iccom rf11 读入 → GTCRN 降噪 → iccom wf12 写出
 *
 * 数据流: RTOS 每 50ms 发 400×int16 @ 8kHz → rf11
 *         Linux 处理后每 50ms 回 400×int16 @ 8kHz → wf12
 *         内部: 400 拆成 2×200, noise_reduction 每 25ms 调用一次
 *
 * 编译: mips-linux-gnu-gcc -O3 -std=c99 -march=mips32r2 -msoft-float \
 *         -o gtcrn_linux gtcrn_linux.c noise_reduction.c \
 *         gtcrn_fp.c gtcrn_infer.c kiss_fft.c kiss_fftr.c -lm -static
 *
 * 运行: ./gtcrn_linux           (iccom 模式)
 *       ./gtcrn_linux test      (16kHz stdin→降采样8kHz→GTCRN→8kHz stdout)
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
    int test16k_mode = (argc > 1 && strcmp(argv[1], "test16k") == 0);

    if (test_mode) {
        /* ============================================================
         *  test 模式: stdin(8kHz PCM) → GTCRN → stdout(8kHz)
         *  用于对讲机实录等原生 8kHz 音频
         * ============================================================ */
        fprintf(stderr, "GTCRN Linux — test mode (8kHz stdin→GTCRN→8kHz stdout)\n");
        noise_init();
        short in[FRAME_IN], out[FRAME_IN];
        while (fread(in, sizeof(short), FRAME_IN, stdin) == FRAME_IN) {
            noise_reduction(in, out);
            fwrite(out, sizeof(short), FRAME_IN, stdout);
            fflush(stdout);
        }
        noise_deinit();
    } else if (test16k_mode) {
        /* ============================================================
         *  test16k 模式: stdin(16kHz PCM) → 降采样8kHz → GTCRN → stdout(8kHz)
         *  用于 MATLAB 导出的 16kHz 测试文件
         * ============================================================ */
        fprintf(stderr, "GTCRN Linux — test16k mode (16kHz stdin→downsample→GTCRN→8kHz stdout)\n");
        noise_init();
        #define FRAME_16K_IN  (FRAME_IN * 2)
        short raw16[FRAME_16K_IN], in8[FRAME_IN], out8[FRAME_IN];
        while (fread(raw16, sizeof(short), FRAME_16K_IN, stdin) == FRAME_16K_IN) {
            for (int i = 0; i < FRAME_IN; i++)
                in8[i] = (short)(((int)raw16[i*2] + (int)raw16[i*2+1]) >> 1);
            noise_reduction(in8, out8);
            fwrite(out8, sizeof(short), FRAME_IN, stdout);
            fflush(stdout);
        }
        noise_deinit();
    } else {
        /* ============================================================
         *  iccom 模式: rf11(400×int16@50ms)→GTCRN→wf12(400×int16@50ms)
         *
         *  DSP 每 50ms 发 400 样本, Linux 内部分 2×200 处理,
         *  noise_reduction API 不变 (仍 200/次, 25ms/次).
         * ============================================================ */
        fprintf(stderr, "GTCRN Linux — iccom mode (rf11→GTCRN→wf12, 400smp/50ms)\n");

        int fd_rf = open("/dev/iccom_rf11", O_RDONLY);
        if (fd_rf < 0) { perror("open rf11"); return 1; }

        int fd_wf = open("/dev/iccom_wf12", O_WRONLY);
        if (fd_wf < 0) { perror("open wf12"); close(fd_rf); return 1; }

        noise_init();
        fprintf(stderr, "GTCRN ready. Waiting for audio from rtos...\n");

        #define FRAME_50MS (FRAME_IN * 2)  /* 400 samples @ 8kHz = 50ms */
        while (1) {
            short in[FRAME_50MS], out[FRAME_50MS];
            ssize_t nr = read(fd_rf, in, FRAME_50MS * sizeof(short));
            if (nr != FRAME_50MS * sizeof(short)) {
                fprintf(stderr, "rf11 short read: %zd/%d\n", nr, FRAME_50MS * (int)sizeof(short));
                continue;
            }
            /* 400 拆 2×200: noise_reduction 保持 25ms 帧不变 */
            noise_reduction(in,           out);       /* 0~199 → 0~199   */
            noise_reduction(in + FRAME_IN, out + FRAME_IN); /* 200~399 → 200~399 */
            write(fd_wf, out, FRAME_50MS * sizeof(short));
        }

        noise_deinit();
        close(fd_rf);
        close(fd_wf);
    }
    return 0;
}
