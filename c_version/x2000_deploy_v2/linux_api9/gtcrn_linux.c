/**
 * gtcrn_linux.c — GTCRN Linux 侧降噪主程序 (v16: AGC前置)
 * ============================================================
 * iccom rf11 读入 → AGC 增益 → GTCRN 降噪 → iccom wf12 写出
 *
 * 数据流: DSP 每 50ms 发 400×int16 @ 8kHz → rf11
 *         Linux: 能量采集 → AGC(先增益) → GTCRN×2(后降噪) → wf12
 *
 * 编译: mips-linux-gnu-gcc -O3 -std=c99 -march=mips32r2 -msoft-float \
 *         -o gtcrn_linux_q15 agc.c gtcrn_linux.c noise_reduction_q15.c \
 *         gtcrn_fp.c gtcrn_infer.c -lm -lrt -static
 *
 * 运行: ./gtcrn_linux_q15          (iccom 模式, 降噪+AGC)
 *       ./gtcrn_linux_q15 bypass    (iccom 模式, 直通, 无降噪无AGC)
 *       ./gtcrn_linux_q15 test      (stdin→GTCRN+AGC→stdout)
 */
#include "noise_reduction.h"
#include "agc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#define FRAME_IN   200   /* 25ms @ 8kHz */

int main(int argc, char **argv) {
    int test_mode   = (argc > 1 && strcmp(argv[1], "test") == 0);
    int bypass_mode = (argc > 1 && strcmp(argv[1], "bypass") == 0);

    if (test_mode) {
        /* ============================================================
         *  test 模式: stdin(8kHz PCM) → GTCRN+AGC → stdout(8kHz)
         * ============================================================ */
        fprintf(stderr, "GTCRN v16 — test mode (8kHz stdin→AGC→GTCRN→8kHz stdout)\n");
        noise_init();
        agc_init();
        short in[FRAME_IN], out[FRAME_IN];
        while (fread(in, sizeof(short), FRAME_IN, stdin) == FRAME_IN) {
            int energy = energy_calc_frame(in, FRAME_IN);
            agc_process(in, FRAME_IN, energy);
            noise_reduction(in, out);
            fwrite(out, sizeof(short), FRAME_IN, stdout);
            fflush(stdout);
        }
        noise_deinit();
    } else if (bypass_mode) {
        /* ============================================================
         *  bypass 模式: iccom 直通, 无降噪无AGC
         * ============================================================ */
        fprintf(stderr, "GTCRN v15 — BYPASS mode (rf11→wf12 passthrough)\n");
        int fd_rf = open("/dev/iccom_rf11", O_RDONLY);
        int fd_wf = open("/dev/iccom_wf12", O_WRONLY);
        if (fd_rf < 0 || fd_wf < 0) { perror("open"); return 1; }
        #define FRAME_50MS_BP (FRAME_IN * 2)
        short buf[FRAME_50MS_BP];
        while (1) {
            if (read(fd_rf, buf, sizeof(buf)) != sizeof(buf)) continue;
            write(fd_wf, buf, sizeof(buf));
        }
    } else {
        /* ============================================================
         *  iccom 模式 (默认): rf11→GTCRN+AGC→wf12
         *  降噪开关: wp10 控制通道 (write '1'=启用, '0'=旁路)
         * ============================================================ */
        fprintf(stderr, "GTCRN v16 — iccom mode (rf11→AGC→GTCRN→wf12, 400smp/50ms)\n");

        int fd_rf = open("/dev/iccom_rf11", O_RDONLY);
        int fd_wf = open("/dev/iccom_wf12", O_WRONLY);
        if (fd_rf < 0 || fd_wf < 0) { perror("open"); return 1; }

        noise_init();
        agc_init();
        fprintf(stderr, "GTCRN+AGC ready. noise=ON, agc=ON\n");

        #define FRAME_50MS (FRAME_IN * 2)  /* 400 shorts = 50ms */
        short in[FRAME_50MS], out[FRAME_50MS];
        while (1) {
            ssize_t nr = read(fd_rf, in, FRAME_50MS * sizeof(short));
            if (nr != FRAME_50MS * sizeof(short)) {
                fprintf(stderr, "rf11 short read: %zd/%d\n",
                        nr, FRAME_50MS * (int)sizeof(short));
                continue;
            }

            /* 帧0: 能量采集 → AGC(先增益) → GTCRN(后降噪) */
            int energy0 = energy_calc_frame(in, FRAME_IN);
            agc_process(in, FRAME_IN, energy0);
            noise_reduction(in, out);

            /* 帧1 */
            int energy1 = energy_calc_frame(in + FRAME_IN, FRAME_IN);
            agc_process(in + FRAME_IN, FRAME_IN, energy1);
            noise_reduction(in + FRAME_IN, out + FRAME_IN);

            write(fd_wf, out, FRAME_50MS * sizeof(short));
        }

        noise_deinit();
        close(fd_rf);
        close(fd_wf);
    }
    return 0;
}
