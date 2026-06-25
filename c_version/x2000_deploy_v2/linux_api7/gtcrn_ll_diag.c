/**
 * gtcrn_ll_diag.c — GTCRN iccom 诊断版 (带 write 校验 + 音频统计)
 * ================================================================
 * 每 50 帧打印一次概要: 读写字节数, PCM RMS, write 状态
 * 编译: mips-linux-gnu-gcc -O3 -std=c99 -march=mips32r2 -msoft-float \
 *         -o gtcrn_ll_diag gtcrn_ll_diag.c noise_reduction.c \
 *         gtcrn_fp.c gtcrn_infer.c kiss_fft.c kiss_fftr.c -lm -static
 */
#include "noise_reduction.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>

#define FRAME_IN 200

int main(void) {
    int fd_rf = open("/dev/iccom_rf11", O_RDONLY);
    if (fd_rf < 0) { fprintf(stderr, "open rf11: %s\n", strerror(errno)); return 1; }
    int fd_wf = open("/dev/iccom_wf12", O_WRONLY);
    if (fd_wf < 0) { fprintf(stderr, "open wf12: %s\n", strerror(errno)); close(fd_rf); return 1; }

    noise_init();
    fprintf(stderr, "GTCRN DIAG: rf11→GTCRN→wf12, 200 smp/25ms\n");
    fprintf(stderr, "frame | rd_bytes | wr_bytes | in_rms | out_rms | note\n");
    fprintf(stderr, "------+----------+----------+--------+---------+-----\n");

    short in_pcm[FRAME_IN], out_pcm[FRAME_IN];
    int frame = 0;
    int write_errors = 0;
    int read_errors = 0;

    while (1) {
        /* ── Read ── */
        ssize_t nr = read(fd_rf, in_pcm, sizeof(in_pcm));
        if (nr != sizeof(in_pcm)) {
            read_errors++;
            if (frame < 10 || read_errors % 50 == 1)
                fprintf(stderr, " #%d | %8zd |        - |      - |       - | short read\n",
                        frame, nr);
            if (read_errors > 100) {
                fprintf(stderr, "Too many read errors (%d), exiting\n", read_errors);
                break;
            }
            continue;
        }

        /* ── Compute input RMS ── */
        int64_t in_sq = 0;
        for (int i = 0; i < FRAME_IN; i++)
            in_sq += (int64_t)in_pcm[i] * in_pcm[i];
        int in_rms = (int)sqrt((double)in_sq / FRAME_IN);

        /* ── GTCRN ── */
        noise_reduction(in_pcm, out_pcm);

        /* ── Compute output RMS ── */
        int64_t out_sq = 0;
        for (int i = 0; i < FRAME_IN; i++)
            out_sq += (int64_t)out_pcm[i] * out_pcm[i];
        int out_rms = (int)sqrt((double)out_sq / FRAME_IN);

        /* ── Write ── */
        ssize_t nw = write(fd_wf, out_pcm, sizeof(out_pcm));
        const char *note = "OK";
        if (nw != sizeof(out_pcm)) {
            write_errors++;
            if (nw < 0)
                note = strerror(errno);
            else
                note = "partial";
        }

        /* ── Print summary every 50 frames ── */
        if (frame < 10 || frame % 50 == 0) {
            fprintf(stderr, " %4d | %8zd | %8zd | %6d | %7d | %s\n",
                    frame, nr, nw, in_rms, out_rms, note);
            fflush(stderr);
        }

        frame++;
    }

    noise_deinit();
    close(fd_rf);
    close(fd_wf);
    return 0;
}
