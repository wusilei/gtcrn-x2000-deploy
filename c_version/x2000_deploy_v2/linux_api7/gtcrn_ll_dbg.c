/**
 * gtcrn_ll_dbg.c — GTCRN 25ms iccom 诊断版本
 * ==========================================
 * 打印前 N 个包的内容 (seq_num, padding, PCM RMS)
 * 用于 DSP 联调时确认数据格式
 *
 * 编译: mips-linux-gnu-gcc -O3 -std=c99 -march=mips32r2 -msoft-float \
 *         -o gtcrn_ll_dbg gtcrn_ll_dbg.c noise_reduction.c \
 *         gtcrn_fp.c gtcrn_infer.c kiss_fft.c kiss_fftr.c -lm -static
 */
#include "noise_reduction.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>

#define FRAME_IN    200
#define HDR_LEN     2
#define PKT_LEN     (HDR_LEN + FRAME_IN)  /* 202 shorts */

int main(int argc, char **argv) {
    int fd_rf = open("/dev/iccom_rf11", O_RDONLY);
    if (fd_rf < 0) { perror("open rf11"); return 1; }

    int fd_wf = open("/dev/iccom_wf12", O_WRONLY);
    if (fd_wf < 0) { perror("open wf12"); close(fd_rf); return 1; }

    noise_init();
    fprintf(stderr, "GTCRN LL DEBUG — iccom mode (rf11→GTCRN→wf12, %d smp/25ms)\n", FRAME_IN);
    fprintf(stderr, "PKT_LEN=%d shorts = %d bytes\n", PKT_LEN, PKT_LEN * (int)sizeof(short));
    fprintf(stderr, "Waiting for data...\n");

    int pkt_count = 0;
    int diag_count = 20;  /* 打印前 20 个包 */

    while (1) {
        short in_pkt[PKT_LEN], out_pkt[PKT_LEN];
        ssize_t nr = read(fd_rf, in_pkt, PKT_LEN * sizeof(short));

        if (nr < 0) {
            perror("read rf11");
            continue;
        }

        if (pkt_count < diag_count) {
            fprintf(stderr, "\n=== pkt #%d: read %zd bytes (%zd shorts) ===\n",
                    pkt_count, nr, nr / sizeof(short));
        }

        if (nr != PKT_LEN * sizeof(short)) {
            if (pkt_count < 5) {
                fprintf(stderr, "  ⚠️ short read: %zd/%d bytes — DSP may be sending different size\n",
                        nr, PKT_LEN * (int)sizeof(short));
                /* 尝试打印实际读到的数据 */
                int nshorts = nr / (int)sizeof(short);
                if (nshorts > 0) {
                    fprintf(stderr, "  First %d shorts: ", nshorts < 8 ? nshorts : 8);
                    for (int i = 0; i < nshorts && i < 8; i++)
                        fprintf(stderr, "%d ", in_pkt[i]);
                    fprintf(stderr, "\n");
                }
            }
            /* 数据不足, 继续读 */
            continue;
        }

        /* 诊断打印 */
        if (pkt_count < diag_count) {
            int16_t seq   = in_pkt[0];
            int16_t padding = in_pkt[1];

            /* 计算 PCM RMS */
            int64_t sum_sq = 0;
            int16_t pcm_min = 32767, pcm_max = -32768;
            for (int i = HDR_LEN; i < PKT_LEN; i++) {
                int v = in_pkt[i];
                sum_sq += (int64_t)v * v;
                if (v < pcm_min) pcm_min = v;
                if (v > pcm_max) pcm_max = v;
            }
            int rms = (int)sqrt((double)sum_sq / FRAME_IN);

            fprintf(stderr, "  seq=%d pad=%d | PCM: rms=%d min=%d max=%d | samples[2..7]=",
                    seq, padding, rms, pcm_min, pcm_max);
            for (int i = HDR_LEN; i < HDR_LEN + 6 && i < PKT_LEN; i++)
                fprintf(stderr, "%d ", in_pkt[i]);
            fprintf(stderr, "\n");
        }

        /* Header 透传 */
        out_pkt[0] = in_pkt[0];
        out_pkt[1] = in_pkt[1];

        /* PCM 降噪 */
        noise_reduction(in_pkt + HDR_LEN, out_pkt + HDR_LEN);

        ssize_t nw = write(fd_wf, out_pkt, PKT_LEN * sizeof(short));
        if (pkt_count < diag_count) {
            fprintf(stderr, "  → wrote %zd/%d bytes to wf12\n",
                    nw, PKT_LEN * (int)sizeof(short));
        }

        pkt_count++;
    }

    noise_deinit();
    close(fd_rf);
    close(fd_wf);
    return 0;
}
