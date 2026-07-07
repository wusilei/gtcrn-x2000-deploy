/**
 * gtcrn_linux.c — GTCRN Linux 侧降噪主程序 (v20 — linux_api14, 前置AGC)
 * =====================================================================
 * 新管线 (3 阶段):
 *   1. MIC → energy_calculate_and_smooth_s16 → current_energy + energy_out
 *   2. 二分阈值: current_energy/energy_out → goal_val
 *   3. MIC → voice_NoiseReductionAndAGC(in, out, goal, state_flag=3)
 *      (内部: AGC先 → 降噪后, 前置AGC)
 *
 * iccom: DSP 每 50ms 发 400×int16 @ 8kHz → rf11
 *        Linux: 能量采集 → 阈值决策 → 联合AGC+降噪 → wf12
 *
 * 编译: mips-linux-gnu-gcc -O3 -std=c99 -march=mips32r2 -msoft-float \
 *         -o gtcrn_linux_q15 agc.c gtcrn_linux.c noise_reduction_q15.c \
 *         gtcrn_fp.c gtcrn_infer.c -lm -lrt -static
 */
#include "agc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#define FRAME_IN   200

int main(int argc, char **argv) {
    int test_mode    = (argc > 1 && strcmp(argv[1], "test") == 0);
    int bypass_mode  = (argc > 1 && strcmp(argv[1], "bypass") == 0);
    int measure_mode = (argc > 1 && strcmp(argv[1], "measure") == 0);

    if (measure_mode) {
        fprintf(stderr, "GTCRN v20 — measure mode (8kHz stdin→NR→8kHz stdout, no AGC)\n");
        agc_init();
        short in[FRAME_IN], out[FRAME_IN]; int fid = 0;
        while (fread(in, sizeof(short), FRAME_IN, stdin) == FRAME_IN) {
            voice_NoiseReductionAndAGC(in, out, AGC_GOAL_HIGH, STATE_FLAG_DENOISE);
            fwrite(out, sizeof(short), FRAME_IN, stdout); fflush(stdout); fid++;
        }
        fprintf(stderr, "measure done: %d frames\n", fid);
    } else if (test_mode) {
        fprintf(stderr, "GTCRN v20 — test mode (8kHz stdin→AGC→NR→8kHz stdout)\n");
        agc_init();
        short in[FRAME_IN], out[FRAME_IN];
        unsigned short energy_out = 0;
        while (fread(in, sizeof(short), FRAME_IN, stdin) == FRAME_IN) {
            unsigned short cur = energy_calculate_and_smooth_s16(
                in, FRAME_IN, ENERGY_HISTORY_FAC, &energy_out);
            int hi = (cur > ENERGY_ABS_THR_HIGH || energy_out > ENERGY_SMOOTH_THR);
            long long goal = hi ? AGC_GOAL_HIGH : AGC_GOAL_LOW;
            voice_NoiseReductionAndAGC(in, out, goal, STATE_FLAG_BOTH);
            fwrite(out, sizeof(short), FRAME_IN, stdout); fflush(stdout);
        }
    } else if (bypass_mode) {
        fprintf(stderr, "GTCRN v20 — BYPASS mode (rf11→wf12 passthrough)\n");
        int fd_rf = open("/dev/iccom_rf11", O_RDONLY);
        int fd_wf = open("/dev/iccom_wf12", O_WRONLY);
        if (fd_rf < 0 || fd_wf < 0) { perror("open"); return 1; }
        #define FRAME_50MS_BP (FRAME_IN * 2)
        short buf[FRAME_50MS_BP];
        while (1) { if (read(fd_rf, buf, sizeof(buf)) != sizeof(buf)) continue; write(fd_wf, buf, sizeof(buf)); }
    } else {
        fprintf(stderr, "GTCRN v20 — iccom mode (rf11→AGC→NR→wf12, 400smp/50ms, state_flag=3)\n");
        int fd_rf = open("/dev/iccom_rf11", O_RDONLY);
        int fd_wf = open("/dev/iccom_wf12", O_WRONLY);
        if (fd_rf < 0 || fd_wf < 0) { perror("open"); return 1; }

        agc_init();
        fprintf(stderr, "GTCRN+AGC ready. noise=ON, agc=ON (state_flag=3, pre-AGC)\n");

        #define FRAME_50MS (FRAME_IN * 2)
        short in[FRAME_50MS], out[FRAME_50MS];
        unsigned short energy_out = 0;

        while (1) {
            ssize_t nr = read(fd_rf, in, FRAME_50MS * sizeof(short));
            if (nr != FRAME_50MS * sizeof(short)) {
                fprintf(stderr, "rf11 short read: %zd/%d\n", nr, FRAME_50MS * (int)sizeof(short));
                continue;
            }

            /* 帧0 */
            unsigned short cur0 = energy_calculate_and_smooth_s16(in, FRAME_IN, ENERGY_HISTORY_FAC, &energy_out);
            int hi0 = (cur0 > ENERGY_ABS_THR_HIGH || energy_out > ENERGY_SMOOTH_THR);
            voice_NoiseReductionAndAGC(in, out, hi0 ? AGC_GOAL_HIGH : AGC_GOAL_LOW, STATE_FLAG_BOTH);

            /* 帧1 */
            unsigned short cur1 = energy_calculate_and_smooth_s16(in + FRAME_IN, FRAME_IN, ENERGY_HISTORY_FAC, &energy_out);
            int hi1 = (cur1 > ENERGY_ABS_THR_HIGH || energy_out > ENERGY_SMOOTH_THR);
            voice_NoiseReductionAndAGC(in + FRAME_IN, out + FRAME_IN, hi1 ? AGC_GOAL_HIGH : AGC_GOAL_LOW, STATE_FLAG_BOTH);

            write(fd_wf, out, FRAME_50MS * sizeof(short));
        }
        close(fd_rf); close(fd_wf);
    }
    return 0;
}
