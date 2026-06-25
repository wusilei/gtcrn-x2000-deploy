/**
 * gtcrn_ll.c — GTCRN Linux 侧低延迟降噪主程序 (25ms iccom)
 * ============================================================
 * iccom rf11 读入 → GTCRN 降噪 → iccom wf12 写出
 *
 * 数据流: DSP 每 25ms 发 200×int16 (400 bytes) 纯 PCM @ 8kHz → rf11
 *         Linux 每 25ms 回 200×int16 (400 bytes) 纯 PCM @ 8kHz → wf12
 *
 * 编译: mips-linux-gnu-gcc -O3 -std=c99 -march=mips32r2 -msoft-float \
 *         -o gtcrn_ll_mips gtcrn_ll.c noise_reduction.c \
 *         gtcrn_fp.c gtcrn_infer.c kiss_fft.c kiss_fftr.c -lm -static
 *
 * 运行: ./gtcrn_ll_mips           (iccom 模式)
 *       ./gtcrn_ll_mips test      (8kHz stdin→GTCRN→8kHz stdout)
 */
#define _POSIX_C_SOURCE 199309L
#include "noise_reduction.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#define FRAME_IN    200            /* 25ms @ 8kHz PCM 样本数 */

int main(int argc, char **argv) {
    int test_mode = (argc > 1 && strcmp(argv[1], "test") == 0);

    if (test_mode) {
        /* ============================================================
         *  test 模式: stdin(8kHz PCM) → GTCRN → stdout(8kHz)
         *  用于对讲机实录等原生 8kHz 音频
         * ============================================================ */
        fprintf(stderr, "GTCRN LL — test mode (8kHz stdin→GTCRN→8kHz stdout)\n");
        noise_init();
        short in[FRAME_IN], out[FRAME_IN];
        while (fread(in, sizeof(short), FRAME_IN, stdin) == FRAME_IN) {
            noise_reduction(in, out);
            fwrite(out, sizeof(short), FRAME_IN, stdout);
            fflush(stdout);
        }
        noise_deinit();

    } else {
        /* ============================================================
         *  iccom 模式: rf11(200×int16@25ms)→GTCRN→wf12(200×int16@25ms)
         *  DSP 每 25ms 发 200 shorts (400 bytes) 纯 PCM @ 8kHz
         * ============================================================ */
        fprintf(stderr, "GTCRN LL — iccom mode (rf11→GTCRN→wf12, 200 smp/25ms)\n");

        int fd_rf = open("/dev/iccom_rf11", O_RDONLY);
        if (fd_rf < 0) { perror("open rf11"); return 1; }

        int fd_wf = open("/dev/iccom_wf12", O_WRONLY);
        if (fd_wf < 0) { perror("open wf12"); close(fd_rf); return 1; }

        noise_init();
        fprintf(stderr, "GTCRN LL ready. Waiting for audio from rtos...\n");

        while (1) {
            short in_pcm[FRAME_IN], out_pcm[FRAME_IN];
            struct timespec t0, t1;

            ssize_t nr = read(fd_rf, in_pcm, FRAME_IN * sizeof(short));
            if (nr != FRAME_IN * sizeof(short)) {
                fprintf(stderr, "rf11 short read: %zd/%d\n",
                        nr, FRAME_IN * (int)sizeof(short));
                continue;
            }

            clock_gettime(CLOCK_MONOTONIC, &t0);
            noise_reduction(in_pcm, out_pcm);
            clock_gettime(CLOCK_MONOTONIC, &t1);

            long us = (t1.tv_sec - t0.tv_sec) * 1000000L
                    + (t1.tv_nsec - t0.tv_nsec) / 1000L;

            write(fd_wf, out_pcm, FRAME_IN * sizeof(short));

            static int frame_cnt = 0;
            static long us_sum = 0, us_min = 999999, us_max = 0;
            frame_cnt++;
            us_sum += us;
            if (us < us_min) us_min = us;
            if (us > us_max) us_max = us;

            if (frame_cnt % 100 == 0) {
                fprintf(stderr, "frame %d: avg=%ldus min=%ldus max=%ldus (%.1fms)\n",
                        frame_cnt, us_sum/100, us_min, us_max, (us_sum/100)/1000.0);
                us_sum = 0; us_min = 999999; us_max = 0;
            }
        }

        noise_deinit();
        close(fd_rf);
        close(fd_wf);
    }
    return 0;
}
