#include "noise_reduction.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#define FRAME_IN 200
#define HDR_LEN  2
#define PKT_LEN  (HDR_LEN + FRAME_IN)

int main(void) {
    int fd_rf = open("/dev/iccom_rf11", O_RDONLY);
    if (fd_rf < 0) { perror("open rf11"); return 1; }
    int fd_wf = open("/dev/iccom_wf12", O_WRONLY);
    if (fd_wf < 0) { perror("open wf12"); close(fd_rf); return 1; }

    noise_init();
    fprintf(stderr, "GTCRN SEQ — verifying seq passthrough\n");
    fprintf(stderr, "frame | rd | in_seq0 | in_seq1 | out_seq0 | out_seq1 | wr | match\n");
    fprintf(stderr, "------+----+---------+---------+----------+----------+----+------\n");

    short in_pkt[PKT_LEN], out_pkt[PKT_LEN];
    int frame = 0;

    while (frame < 100) {
        ssize_t nr = read(fd_rf, in_pkt, sizeof(in_pkt));
        if (nr != sizeof(in_pkt)) {
            fprintf(stderr, "%5d |%4zd|       - |       - |        - |        - |  - | short rd\n",
                    frame, nr);
            continue;
        }

        /* Header passthrough */
        out_pkt[0] = in_pkt[0];
        out_pkt[1] = in_pkt[1];

        /* GTCRN */
        noise_reduction(in_pkt + HDR_LEN, out_pkt + HDR_LEN);

        ssize_t nw = write(fd_wf, out_pkt, sizeof(out_pkt));

        /* Print every frame for first 5, then every 20 */
        if (frame < 5 || frame % 20 == 0) {
            int match = (out_pkt[0]==in_pkt[0] && out_pkt[1]==in_pkt[1]);
            fprintf(stderr, "%5d |%4zd| %7d | %7d | %8d | %8d |%4zd| %s\n",
                    frame, nr, in_pkt[0], in_pkt[1],
                    out_pkt[0], out_pkt[1], nw,
                    match ? "OK" : "MISMATCH");
            fflush(stderr);
        }

        /* Check for write errors */
        if (nw != sizeof(out_pkt)) {
            fprintf(stderr, "%5d | WRITE ERROR: %zd/%zu (%s)\n",
                    frame, nw, sizeof(out_pkt), nw<0?strerror(errno):"partial");
        }

        /* Check for seq mismatch */
        if (out_pkt[0] != in_pkt[0] || out_pkt[1] != in_pkt[1]) {
            fprintf(stderr, "%5d | SEQ MISMATCH: in=[%d,%d] out=[%d,%d]\n",
                    frame, in_pkt[0], in_pkt[1], out_pkt[0], out_pkt[1]);
        }

        frame++;
    }

    noise_deinit();
    close(fd_rf);
    close(fd_wf);
    return 0;
}
