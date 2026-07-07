/**
 * gtcrn_linux.c — GTCRN Linux 侧降噪主程序 (v23 — linux_api17, 混合AGC + dump)
 * ============================================================================
 * 混合AGC管线 (pre×1.5→NR→post×2.5) + SIGUSR1/SIGUSR2 波形抓取
 *
 * dump: kill -USR1 <pid> = toggle DUMP_PRE, kill -USR2 <pid> = toggle DUMP_POST
 */
#include "agc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>

#define FRAME_IN   200

static volatile int g_dump_pre = 0, g_dump_post = 0, g_dump_req = 0;
static int g_frame_counter = 0;

static void sigusr1_handler(int sig) { (void)sig; g_dump_pre = !g_dump_pre; g_dump_req = 1; }
static void sigusr2_handler(int sig) { (void)sig; g_dump_post = !g_dump_post; g_dump_req = 1; }

static void dump_frame(const char *tag, int fid, const short *pcm, int n) {
    fprintf(stderr, "%s,%d,%d", tag, fid, n);
    for (int i = 0; i < n; i++) fprintf(stderr, ",%d", (int)pcm[i]);
    fprintf(stderr, "\n");
}

int main(int argc, char **argv) {
    int test_mode    = (argc > 1 && strcmp(argv[1], "test") == 0);
    int bypass_mode  = (argc > 1 && strcmp(argv[1], "bypass") == 0);
    int measure_mode = (argc > 1 && strcmp(argv[1], "measure") == 0);

    signal(SIGUSR1, sigusr1_handler);
    signal(SIGUSR2, sigusr2_handler);

    if (measure_mode) {
        fprintf(stderr, "GTCRN v23 — measure mode (no AGC)\n");
        agc_init(); short in[FRAME_IN], out[FRAME_IN]; int fid = 0;
        while (fread(in, sizeof(short), FRAME_IN, stdin) == FRAME_IN) {
            voice_NoiseReductionAndAGC(in, out, AGC_GOAL_HIGH, STATE_FLAG_DENOISE);
            fwrite(out, sizeof(short), FRAME_IN, stdout); fflush(stdout); fid++;
        }
        fprintf(stderr, "measure done: %d frames\n", fid);
    } else if (test_mode) {
        fprintf(stderr, "GTCRN v23 — test mode (hybrid AGC)\n");
        agc_init(); short in[FRAME_IN], out[FRAME_IN];
        unsigned short energy_out = 0;
        while (fread(in, sizeof(short), FRAME_IN, stdin) == FRAME_IN) {
            unsigned short cur = energy_calculate_and_smooth_s16(in, FRAME_IN, ENERGY_HISTORY_FAC, &energy_out);
            int hi = (cur > ENERGY_ABS_THR_HIGH || energy_out > ENERGY_SMOOTH_THR);
            voice_NoiseReductionAndAGC(in, out, hi ? AGC_GOAL_HIGH : AGC_GOAL_LOW, STATE_FLAG_BOTH);
            fwrite(out, sizeof(short), FRAME_IN, stdout); fflush(stdout);
        }
    } else if (bypass_mode) {
        fprintf(stderr, "GTCRN v23 — BYPASS\n");
        int fd_rf = open("/dev/iccom_rf11", O_RDONLY), fd_wf = open("/dev/iccom_wf12", O_WRONLY);
        if (fd_rf < 0 || fd_wf < 0) { perror("open"); return 1; }
        short buf[FRAME_IN*2];
        while (1) { if (read(fd_rf, buf, sizeof(buf)) != sizeof(buf)) continue; write(fd_wf, buf, sizeof(buf)); }
    } else {
        fprintf(stderr, "GTCRN v23 — iccom (hybrid-AGC, state_flag=3)\n");
        fprintf(stderr, "  dump: kill -USR1 = DUMP_PRE, kill -USR2 = DUMP_POST\n");
        int fd_rf = open("/dev/iccom_rf11", O_RDONLY), fd_wf = open("/dev/iccom_wf12", O_WRONLY);
        if (fd_rf < 0 || fd_wf < 0) { perror("open"); return 1; }
        agc_init();
        fprintf(stderr, "ready. dump_pre=OFF dump_post=OFF\n");

        #define F50 (FRAME_IN*2)
        short in[F50], out[F50]; unsigned short energy_out = 0;
        while (1) {
            if (read(fd_rf, in, sizeof(in)) != sizeof(in)) { fprintf(stderr, "short read\n"); continue; }
            if (g_dump_req) { g_dump_req = 0; fprintf(stderr, "# DUMP: pre=%d post=%d\n", g_dump_pre, g_dump_post); }

            if (g_dump_pre) dump_frame("DUMP_PRE", g_frame_counter, in, FRAME_IN);
            unsigned short c0 = energy_calculate_and_smooth_s16(in, FRAME_IN, ENERGY_HISTORY_FAC, &energy_out);
            voice_NoiseReductionAndAGC(in, out, (c0 > ENERGY_ABS_THR_HIGH || energy_out > ENERGY_SMOOTH_THR) ? AGC_GOAL_HIGH : AGC_GOAL_LOW, STATE_FLAG_BOTH);
            if (g_dump_post) dump_frame("DUMP_POST", g_frame_counter, out, FRAME_IN);
            g_frame_counter++;

            if (g_dump_pre) dump_frame("DUMP_PRE", g_frame_counter, in + FRAME_IN, FRAME_IN);
            unsigned short c1 = energy_calculate_and_smooth_s16(in + FRAME_IN, FRAME_IN, ENERGY_HISTORY_FAC, &energy_out);
            voice_NoiseReductionAndAGC(in + FRAME_IN, out + FRAME_IN, (c1 > ENERGY_ABS_THR_HIGH || energy_out > ENERGY_SMOOTH_THR) ? AGC_GOAL_HIGH : AGC_GOAL_LOW, STATE_FLAG_BOTH);
            if (g_dump_post) dump_frame("DUMP_POST", g_frame_counter, out + FRAME_IN, FRAME_IN);
            g_frame_counter++;

            write(fd_wf, out, sizeof(out));
        }
    }
    return 0;
}
