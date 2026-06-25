/**
 * gtcrn_passthru.c — iccom 直通测试 (rf11 → wf12, 无 GTCRN)
 * ==========================================================
 * 隔离 iccom 链路问题 vs GTCRN 处理问题
 * 编译: mips-linux-gnu-gcc -O3 -std=c99 -static -o gtcrn_passthru gtcrn_passthru.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

#define PKT_LEN 200  /* 200 shorts = 400 bytes, raw PCM */

int main(void) {
    int fd_rf = open("/dev/iccom_rf11", O_RDONLY);
    if (fd_rf < 0) { perror("open rf11"); return 1; }
    int fd_wf = open("/dev/iccom_wf12", O_WRONLY);
    if (fd_wf < 0) { perror("open wf12"); close(fd_rf); return 1; }

    fprintf(stderr, "PASSTHRU: rf11→wf12 (%d shorts/pkt)\n", PKT_LEN);
    fprintf(stderr, "Waiting for data...\n");

    int16_t buf[PKT_LEN];
    int count = 0;
    while (1) {
        ssize_t nr = read(fd_rf, buf, sizeof(buf));
        if (nr != sizeof(buf)) {
            fprintf(stderr, "  #%d short read: %zd/%zu\n", count, nr, sizeof(buf));
            continue;
        }
        if (count < 10) {
            fprintf(stderr, "  #%d: hdr=[%d,%d] → passthru\n", count, buf[0], buf[1]);
        }
        write(fd_wf, buf, sizeof(buf));
        count++;
    }
    return 0;
}
