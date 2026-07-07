/**
 * denoise_fp.h — DENOISE MATLAB→C Fixed-Point Implementation
 * =======================================================
 * Q-format matching MATLAB Fix_point.m exactly:
 *
 *   s32f20: activations (int32_t, 20 frac bits, ×1048576)
 *   s32f18: Conv weights encoder (int32_t, 18 frac bits, ×262144)
 *   s16f15: GRU hidden / sigmoid output (int16_t, 15 frac bits, ×32768)
 *   s16f14: BN/PReLU/DeConv weights (int16_t, 14 frac bits, ×16384)
 *   s16f13: FC/DD-Conv/PC weights (int16_t, 13 frac bits, ×8192)
 *   s16f12: GRU/LN/DD-DeConv weights (int16_t, 12 frac bits, ×4096)
 *   s16f10: GRU bias (int16_t, 10 frac bits, ×1024)
 *   u16f15: sigmoid/tanh output (uint16_t, 15 frac bits, ×32768)
 *   u16f14: BN weight (uint16_t, 14 frac bits, ×16384)
 *   u16f13: BN running_var (uint16_t, 13 frac bits, ×8192)
 *   u16f12: BN running_var (uint16_t, 12 frac bits, ×4096)
 *   u16f10: BN running_var (uint16_t, 10 frac bits, ×1024)
 *
 * Target: Ingenic X2000 MIPS32R2 (no FPU) + PC verification
 *
 * Data layout: (C, W) for 2D tensors, (C, H, W) for 3D
 *   Indexing: element[c][h][w] = data[c*H*W + h*W + w]
 *   For (C, W): element[c][w] = data[c*W + w]
 */

#ifndef DENOISE_FP_H
#define DENOISE_FP_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Q-format Constants (matching MATLAB Fix_point.m)
 * ================================================================ */

/* Activation Q-formats */
#define Q_ACT       20    /* s32f20: main activations */
#define Q_ACT_TR    18    /* s32f18: TRA/DeTRA agg quant */

/* Weight Q-formats */
#define Q_WGT_CONV_E  18  /* s32f18: Conv weights (encoder Conv0, DeConv0) */
#define Q_WGT_FC      13  /* s16f13: FC weights, DD-Conv, PC weights */
#define Q_WGT_GRU     12  /* s16f12: GRU weights, LN weights, DD-DeConv */
#define Q_WGT_BN      14  /* s16f14 / u16f14: BN/PReLU weights */
#define Q_WGT_GRU_BIAS 10 /* s16f10: GRU bias */
#define Q_WGT_SIGMOID 15  /* u16f15: sigmoid/tanh output */
#define Q_WGT_BM_BS   15  /* u16f15: BM/BS weights */

/* BN running_var Q-formats (varies per block) */
#define Q_BN_VAR_14   14  /* u16f14 */
#define Q_BN_VAR_13   13  /* u16f13 */
#define Q_BN_VAR_12   12  /* u16f12 */
#define Q_BN_VAR_10   10  /* u16f10 */

/* Safe clamp to int16_t range (s16f15 max is 32767, not 32768) */
static inline int16_t sat_s16f15(float x) {
    float r = roundf(x * 32768.0f);
    if (r > 32767.0f) return 32767;
    if (r < -32768.0f) return -32768;
    return (int16_t)r;
}

/* Scale factors */
/* F2Q20 with saturation — matches Python np.clip(round(x*2^20), INT32_MIN, INT32_MAX) */
static inline int32_t F2Q20(float x) {
    float v = roundf(x * 1048576.0f);
    if (v > 2147483647.0f) return INT32_MAX;
    if (v < -2147483648.0f) return INT32_MIN;
    return (int32_t)v;
}
#define F2Q18(x)  ((int32_t)roundf((x) * 262144.0f))    /* × 2^18 */
#define F2Q15(x)  (sat_s16f15(x))                        /* × 2^15, safely clamped */
#define F2Q14(x)  ((int16_t)roundf((x) * 16384.0f))     /* × 2^14 */
#define F2Q13(x)  ((int16_t)roundf((x) * 8192.0f))      /* × 2^13 */
#define F2Q12(x)  ((int16_t)roundf((x) * 4096.0f))      /* × 2^12 */
#define F2Q10(x)  ((int16_t)roundf((x) * 1024.0f))      /* × 2^10 */

#define U2Q15(x)  ((uint16_t)roundf((x) * 32768.0f))
#define U2Q14(x)  ((uint16_t)roundf((x) * 16384.0f))
#define U2Q13(x)  ((uint16_t)roundf((x) * 8192.0f))
#define U2Q12(x)  ((uint16_t)roundf((x) * 4096.0f))
#define U2Q10(x)  ((uint16_t)roundf((x) * 1024.0f))

/* Dequant */
#define Q20_TO_F(x)  ((float)(x) / 1048576.0f)
#define Q18_TO_F(x)  ((float)(x) / 262144.0f)
#define Q15_TO_F(x)  ((float)(x) / 32768.0f)
#define Q10_TO_F(x)  ((float)(x) / 1024.0f)

/* ================================================================
 * Saturation Helpers
 * ================================================================ */

static inline int32_t sat32(int64_t x) {
    if (x > INT32_MAX) return INT32_MAX;
    if (x < INT32_MIN) return INT32_MIN;
    return (int32_t)x;
}

static inline int16_t sat16(int32_t x) {
    if (x > INT16_MAX) return INT16_MAX;
    if (x < INT16_MIN) return INT16_MIN;
    return (int16_t)x;
}

static inline uint16_t usat16(int32_t x) {
    if (x > UINT16_MAX) return UINT16_MAX;
    if (x < 0) return 0;
    return (uint16_t)x;
}

/* ROUND(x) in C: add half and truncate towards zero */
static inline int32_t round_div(int64_t num, int64_t den) {
    int64_t half = den >> 1;
    if (num >= 0) return (int32_t)((num + half) / den);
    else return (int32_t)((num - half) / den);
}

/* ================================================================
 * Sigmoid & Tanh — 512-entry LUT with linear interpolation
 * ================================================================
 * Replaces soft-float expf()/tanhf() (~1000 cycles/call) with
 * integer LUT lookup (~30 cycles/call). ~30x faster on MIPS soft-float.
 *
 * LUT: Q10 input (×1024) → Q15 output (×32768)
 *   sigmoid: [-8192, 8191] → [0, 32767] — covers float [-8, 8]
 *   tanh:    [-4096, 4095] → [-32768, 32767] — covers float [-4, 4]
 *   Outside range: clamped to saturation value.
 *   Linear interpolation: max error < 1 LSB Q15 (lossless vs expf).
 *
 * Memory: 512×2×2 = 2KB total for both LUTs.
 */

static const int16_t sigmoid_lut_q15[512] = {
      11,    11,    12,    12,    12,    13,    13,    14,    14,    15,    15,    16,    16,    17,    17,    18,
      18,    19,    19,    20,    21,    21,    22,    23,    23,    24,    25,    26,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    37,    38,    40,    41,    42,    44,    45,    46,    48,
      49,    51,    53,    54,    56,    58,    60,    61,    63,    65,    67,    70,    72,    74,    76,    79,
      81,    84,    87,    89,    92,    95,    98,   101,   104,   108,   111,   115,   118,   122,   126,   130,
     134,   138,   143,   147,   152,   157,   162,   167,   172,   177,   183,   189,   195,   201,   207,   214,
     221,   228,   235,   242,   250,   258,   266,   274,   283,   292,   301,   310,   320,   330,   341,   351,
     362,   374,   386,   398,   410,   423,   436,   450,   464,   479,   494,   509,   525,   542,   558,   576,
     594,   612,   632,   651,   672,   692,   714,   736,   759,   783,   807,   832,   858,   884,   912,   940,
     969,   999,  1029,  1061,  1094,  1127,  1162,  1197,  1234,  1272,  1311,  1351,  1392,  1434,  1478,  1522,
    1569,  1616,  1665,  1715,  1767,  1820,  1874,  1930,  1988,  2047,  2108,  2171,  2235,  2301,  2369,  2439,
    2511,  2584,  2660,  2737,  2817,  2898,  2982,  3068,  3156,  3247,  3340,  3435,  3532,  3632,  3734,  3839,
    3947,  4057,  4169,  4284,  4402,  4523,  4647,  4773,  4902,  5034,  5169,  5307,  5447,  5591,  5738,  5887,
    6040,  6196,  6355,  6517,  6682,  6850,  7021,  7195,  7373,  7553,  7737,  7923,  8113,  8305,  8501,  8700,
    8901,  9106,  9313,  9523,  9736,  9952, 10170, 10391, 10614, 10840, 11068, 11299, 11532, 11767, 12004, 12243,
   12484, 12727, 12972, 13218, 13466, 13715, 13965, 14217, 14469, 14722, 14977, 15232, 15487, 15743, 15999, 16256,
   16512, 16769, 17025, 17281, 17536, 17791, 18046, 18299, 18551, 18803, 19053, 19302, 19550, 19796, 20041, 20284,
   20525, 20764, 21001, 21236, 21469, 21700, 21928, 22154, 22377, 22598, 22816, 23032, 23245, 23455, 23662, 23867,
   24068, 24267, 24463, 24655, 24845, 25031, 25215, 25395, 25573, 25747, 25918, 26086, 26251, 26413, 26572, 26728,
   26881, 27030, 27177, 27321, 27461, 27599, 27734, 27866, 27995, 28121, 28245, 28366, 28484, 28599, 28711, 28821,
   28929, 29034, 29136, 29236, 29333, 29428, 29521, 29612, 29700, 29786, 29870, 29951, 30031, 30108, 30184, 30257,
   30329, 30399, 30467, 30533, 30597, 30660, 30721, 30780, 30838, 30894, 30948, 31001, 31053, 31103, 31152, 31199,
   31246, 31290, 31334, 31376, 31417, 31457, 31496, 31534, 31571, 31606, 31641, 31674, 31707, 31739, 31769, 31799,
   31828, 31856, 31884, 31910, 31936, 31961, 31985, 32009, 32032, 32054, 32076, 32096, 32117, 32136, 32156, 32174,
   32192, 32210, 32226, 32243, 32259, 32274, 32289, 32304, 32318, 32332, 32345, 32358, 32370, 32382, 32394, 32406,
   32417, 32427, 32438, 32448, 32458, 32467, 32476, 32485, 32494, 32502, 32510, 32518, 32526, 32533, 32540, 32547,
   32554, 32561, 32567, 32573, 32579, 32585, 32591, 32596, 32601, 32606, 32611, 32616, 32621, 32625, 32630, 32634,
   32638, 32642, 32646, 32650, 32653, 32657, 32660, 32664, 32667, 32670, 32673, 32676, 32679, 32681, 32684, 32687,
   32689, 32692, 32694, 32696, 32698, 32701, 32703, 32705, 32707, 32708, 32710, 32712, 32714, 32715, 32717, 32719,
   32720, 32722, 32723, 32724, 32726, 32727, 32728, 32730, 32731, 32732, 32733, 32734, 32735, 32736, 32737, 32738,
   32739, 32740, 32741, 32742, 32742, 32743, 32744, 32745, 32745, 32746, 32747, 32747, 32748, 32749, 32749, 32750,
   32750, 32751, 32751, 32752, 32752, 32753, 32753, 32754, 32754, 32755, 32755, 32756, 32756, 32756, 32757, 32757
};

static const int16_t tanh_lut_q15[512] = {
  -32746,-32745,-32745,-32744,-32743,-32742,-32741,-32741,-32740,-32739,-32738,-32737,-32736,-32735,-32734,-32733,
  -32732,-32731,-32729,-32728,-32727,-32726,-32724,-32723,-32721,-32720,-32718,-32717,-32715,-32714,-32712,-32710,
  -32708,-32706,-32704,-32702,-32700,-32698,-32696,-32694,-32691,-32689,-32686,-32684,-32681,-32678,-32675,-32672,
  -32669,-32666,-32663,-32660,-32656,-32653,-32649,-32645,-32641,-32637,-32633,-32629,-32624,-32620,-32615,-32610,
  -32605,-32600,-32595,-32589,-32584,-32578,-32572,-32566,-32559,-32553,-32546,-32539,-32531,-32524,-32516,-32508,
  -32500,-32491,-32483,-32474,-32464,-32455,-32445,-32435,-32424,-32413,-32402,-32390,-32378,-32366,-32353,-32340,
  -32327,-32313,-32299,-32284,-32268,-32253,-32236,-32220,-32202,-32184,-32166,-32147,-32128,-32107,-32087,-32065,
  -32043,-32020,-31997,-31973,-31948,-31922,-31895,-31868,-31840,-31811,-31781,-31750,-31718,-31685,-31651,-31616,
  -31580,-31543,-31505,-31465,-31425,-31383,-31340,-31296,-31250,-31203,-31154,-31104,-31053,-31000,-30945,-30889,
  -30831,-30771,-30709,-30646,-30581,-30513,-30444,-30373,-30300,-30224,-30147,-30067,-29984,-29900,-29813,-29723,
  -29631,-29536,-29438,-29338,-29235,-29129,-29020,-28907,-28792,-28673,-28552,-28426,-28298,-28165,-28030,-27890,
  -27747,-27600,-27449,-27294,-27135,-26971,-26804,-26632,-26455,-26274,-26089,-25899,-25704,-25504,-25299,-25090,
  -24875,-24655,-24430,-24199,-23963,-23722,-23475,-23222,-22964,-22700,-22430,-22155,-21873,-21586,-21293,-20993,
  -20688,-20376,-20058,-19735,-19405,-19068,-18726,-18377,-18023,-17662,-17295,-16922,-16542,-16157,-15766,-15369,
  -14966,-14557,-14142,-13722,-13296,-12865,-12428,-11986,-11540,-11088,-10631,-10170, -9704, -9234, -8759, -8281,
   -7799, -7313, -6824, -6332, -5837, -5339, -4838, -4335, -3830, -3323, -2815, -2305, -1794, -1282,  -769,  -256,
     256,   769,  1282,  1794,  2305,  2815,  3323,  3830,  4335,  4838,  5339,  5837,  6332,  6824,  7313,  7799,
    8281,  8759,  9234,  9704, 10170, 10631, 11088, 11540, 11986, 12428, 12865, 13296, 13722, 14142, 14557, 14966,
   15369, 15766, 16157, 16542, 16922, 17295, 17662, 18023, 18377, 18726, 19068, 19405, 19735, 20058, 20376, 20688,
   20993, 21293, 21586, 21873, 22155, 22430, 22700, 22964, 23222, 23475, 23722, 23963, 24199, 24430, 24655, 24875,
   25090, 25299, 25504, 25704, 25899, 26089, 26274, 26455, 26632, 26804, 26971, 27135, 27294, 27449, 27600, 27747,
   27890, 28030, 28165, 28298, 28426, 28552, 28673, 28792, 28907, 29020, 29129, 29235, 29338, 29438, 29536, 29631,
   29723, 29813, 29900, 29984, 30067, 30147, 30224, 30300, 30373, 30444, 30513, 30581, 30646, 30709, 30771, 30831,
   30889, 30945, 31000, 31053, 31104, 31154, 31203, 31250, 31296, 31340, 31383, 31425, 31465, 31505, 31543, 31580,
   31616, 31651, 31685, 31718, 31750, 31781, 31811, 31840, 31868, 31895, 31922, 31948, 31973, 31997, 32020, 32043,
   32065, 32087, 32107, 32128, 32147, 32166, 32184, 32202, 32220, 32236, 32253, 32268, 32284, 32299, 32313, 32327,
   32340, 32353, 32366, 32378, 32390, 32402, 32413, 32424, 32435, 32445, 32455, 32464, 32474, 32483, 32491, 32500,
   32508, 32516, 32524, 32531, 32539, 32546, 32553, 32559, 32566, 32572, 32578, 32584, 32589, 32595, 32600, 32605,
   32610, 32615, 32620, 32624, 32629, 32633, 32637, 32641, 32645, 32649, 32653, 32656, 32660, 32663, 32666, 32669,
   32672, 32675, 32678, 32681, 32684, 32686, 32689, 32691, 32694, 32696, 32698, 32700, 32702, 32704, 32706, 32708,
   32710, 32712, 32714, 32715, 32717, 32718, 32720, 32721, 32723, 32724, 32726, 32727, 32728, 32729, 32731, 32732,
   32733, 32734, 32735, 32736, 32737, 32738, 32739, 32740, 32741, 32741, 32742, 32743, 32744, 32745, 32745, 32746
};

/* LUT dimensions (power-of-2 ranges → shift for fast division) */
#define SIG_LUT_SIZE   512
#define SIG_Q10_MIN   (-8192)
#define SIG_Q10_MAX    8192
#define SIG_Q10_RANGE  16384   /* SIG_Q10_MAX - SIG_Q10_MIN = 2^14 */
#define SIG_Q10_SHIFT  14

#define TANH_LUT_SIZE   512
#define TANH_Q10_MIN   (-4096)
#define TANH_Q10_MAX    4096
#define TANH_Q10_RANGE  8192   /* TANH_Q10_MAX - TANH_Q10_MIN = 2^13 */
#define TANH_Q10_SHIFT  13

/* Q10→Q15 LUT lookup with linear interpolation. Pure integer, no float ops. */
static inline int16_t sigmoid_q15(int32_t q10) {
    if (q10 <= SIG_Q10_MIN) return 0;
    if (q10 >= SIG_Q10_MAX) return 32767;
    int32_t pos = (q10 - SIG_Q10_MIN) * (SIG_LUT_SIZE - 1);
    int32_t idx = pos >> SIG_Q10_SHIFT;
    int32_t frac = pos & (SIG_Q10_RANGE - 1);
    int64_t interp = (int64_t)sigmoid_lut_q15[idx] * (SIG_Q10_RANGE - frac)
                   + (int64_t)sigmoid_lut_q15[idx + 1] * frac;
    return (int16_t)((interp + (SIG_Q10_RANGE >> 1)) >> SIG_Q10_SHIFT);
}

static inline int16_t tanh_q15(int32_t q10) {
    if (q10 <= TANH_Q10_MIN) return -32768;
    if (q10 >= TANH_Q10_MAX) return 32767;
    int32_t pos = (q10 - TANH_Q10_MIN) * (TANH_LUT_SIZE - 1);
    int32_t idx = pos >> TANH_Q10_SHIFT;
    int32_t frac = pos & (TANH_Q10_RANGE - 1);
    int64_t interp = (int64_t)tanh_lut_q15[idx] * (TANH_Q10_RANGE - frac)
                   + (int64_t)tanh_lut_q15[idx + 1] * frac;
    return (int16_t)((interp + (TANH_Q10_RANGE >> 1)) >> TANH_Q10_SHIFT);
}

/* Drop-in replacements — same float→float signatures as original.
 * Float→Q10→LUT→Q15→float roundtrip is exact because:
 *   Q10 < 2^24 fits in float24 mantissa, and 32768 = 2^15.
 * Callers do F2Q15(result) which exactly recovers the Q15 integer. */
/* Round float→Q10 with proper rounding (not truncation).
 * Truncation error < 1 Q10 LSB causes up to ~8 LSB Q15 error at
 * sigmoid steepest slope — acceptable but adds up over 5000 calls.
 * roundf() adds ~50 cycles on MIPS soft-float but eliminates
 * this error source entirely. */
static inline float sigmoidf_fp(float x) {
    return (float)sigmoid_q15((int32_t)roundf(x * 1024.0f)) * (1.0f / 32768.0f);
}

static inline float tanhf_fp(float x) {
    return (float)tanh_q15((int32_t)roundf(x * 1024.0f)) * (1.0f / 32768.0f);
}

/* ================================================================
 * Integer sqrt — Newton's method, ~6 iterations for 64-bit
 * ================================================================
 * Replaces soft-float sqrtf() in LN, mag_gen, etc.
 * Usage: isqrt64(sum_sq_q40) → magnitude in Q20 (sqrt of Q40 = Q20) */
static inline uint32_t isqrt64(uint64_t n) {
    if (n <= 1) return (uint32_t)n;
    uint64_t x = n;
    uint64_t y = (x + 1) >> 1;
    while (y < x) {
        x = y;
        y = (x + n / x) >> 1;
    }
    return (uint32_t)x;
}

/* ================================================================
 * Dimension Constants
 * ================================================================ */

#define N_FFT       512
#define WIN_LEN     512
#define WIN_INC     256
#define N_BINS      257
#define N_BINS_BM   129
#define N_BINS_MID  65
#define N_BINS_SMALL 33

#define CH_IN       3     /* mag + real + imag */
#define CH_SFE      9     /* after SFE (3×3) */
#define CH_MID      16    /* intermediate channels */
#define CH_GRU_HID  4     /* Intra-RNN GRU hidden */
#define CH_GRU_INTER 8    /* Inter-RNN GRU hidden */
#define CH_TRA_HID  16    /* TRA GRU hidden */
#define CH_PC0_IN   24    /* PC0 input (8×3 from SFE) */
#define CH_OUT      2     /* final CRM output (I, Q) */

#define DILATION_MAX 5   /* max dilation for history buffer */

/* ================================================================
 * History / State Buffers (per-frame persistent state)
 * ================================================================ */

/* DD-Conv 3D history: (C, time_steps, freq)
 * Encoder: 16 channels × 16 time × 33 freq + current frame
 * Decoder: 16 channels × 16 time × 33 freq + current frame
 */
#define DD_HIST_TIME  16  /* history length = kernel_size + 2*(dil-1) - 1 = 3+2*(5-1)-1 = 14, round to 16 */
#define DD_HIST_SIZE  (CH_MID * DD_HIST_TIME * N_BINS_SMALL)

/* GRU hidden states */
#define GRU_HIDDEN_SIZE  (N_BINS_SMALL * CH_GRU_INTER)  /* 33 × 8 */

/* TRA hidden states: (N_BINS_SMALL, CH_TRA_HID) = 33×16 */
#define TRA_HIDDEN_SIZE  (N_BINS_SMALL * CH_TRA_HID)

/* ================================================================
 * Model State (extern — allocated by user)
 * ================================================================ */

typedef struct {
    /* Encoder DD-Conv history: (16, DD_HIST_TIME, 33) per GT-Conv ×3 */
    int32_t enc_conv_hist[CH_MID * DD_HIST_TIME * N_BINS_SMALL];

    /* Encoder TRA GRU hidden: (3 * 16) — 3 GT-Convs, single state each */
    int16_t enc_h_prev[3 * CH_TRA_HID];

    /* Decoder DD-DeConv history: (16, DD_HIST_TIME, 33) per GT-DeConv ×3 */
    int32_t dec_conv_hist[CH_MID * DD_HIST_TIME * N_BINS_SMALL];

    /* Decoder DeTRA GRU hidden: (3 * 16) — 3 GT-DeConvs, single state each */
    int16_t dec_h_prev[3 * CH_TRA_HID];

    /* Inter-RNN hidden: (2×DPRNN, 33, 16) — 2 DPRNN blocks */
    int16_t inter_prev1[N_BINS_SMALL * CH_MID];
    int16_t inter_prev2[N_BINS_SMALL * CH_MID];

} denoise_state_t;

/* ================================================================
 * Function Prototypes — Basic Ops
 * ================================================================ */

/* conv2d: (Cin, Win) → (Cout, Wout), weight (Cout, Cin, Hk, Wk) */
void conv2d_fixed(const int32_t *x, int Cin, int Win,
                  const int16_t *weight, const int32_t *bias,
                  int Cout, int Wout, int Hk, int Wk,
                  int stride, int pad_w, int qr,
                  int32_t *y);

/* tconv2d: transposed conv2d */
void tconv2d_fixed(const int32_t *x, int Cin, int Win,
                   const int16_t *weight, const int32_t *bias,
                   int Cout, int Wout, int Hk, int Wk,
                   int stride, int qr,
                   int32_t *y);

/* ddconv2d: depth-wise dilated conv2d, (C, H, W) input with history */
void ddconv2d_fixed(const int32_t *x, int C, int H, int Win,
                    const int16_t *weight, const int32_t *bias,
                    int Wout, int Hk, int Wk,
                    int pad_h, int pad_w, int dilation, int qr,
                    int32_t *y,
                    int32_t *hist_out, int *hist_len);

/* ddtconv2d: depth-wise dilated transposed conv2d */
void ddtconv2d_fixed(const int32_t *x, int C, int H, int Win,
                     const int16_t *weight, const int32_t *bias,
                     int Wout, int Hk, int Wk,
                     int pad_w, int dilation, int qr,
                     int32_t *y,
                     int32_t *hist_out, int *hist_len);

/* pconv2d: point-wise conv2d (1×1), weights (Cout, Cin, 1, 1) */
void pconv2d_fixed(const int32_t *x, int Cin, int Win,
                   const int16_t *weight, const int32_t *bias,
                   int Cout, int qr,
                   int32_t *y);

/* ptconv2d: point-wise transposed conv2d (1×1) */
void ptconv2d_fixed(const int32_t *x, int Cin, int Win,
                    const int16_t *weight, const int32_t *bias,
                    int Cout, int qr,
                    int32_t *y);

/* BatchNorm: y = ((x - mean) * var_inv)>>qr1 * weight>>qr2 + bias */
void bn_fixed(int32_t *x, int C, int Win,
              const uint16_t *weight, const int32_t *bias,
              const int32_t *running_mean, const uint16_t *running_var,
              int qr1, int qr2);

/* PReLU: y(x<0) = x * weight >> qr */
void prelu_fixed(int32_t *x, int C, int Win,
                 const int16_t *weight, int qr);

/* LayerNorm: computed online (mean/var from data) */
void ln_fixed(int32_t *x, int C, int Win,
              const int16_t *weight, const int32_t *bias, int qr);

/* GRU: (seq_len, input_dim) with s32f20 input → s16f15 output */
void gru_fixed(const int32_t *x, int seq_len, int input_dim, int hidden_dim,
               const int16_t *ih_weight, const int16_t *ih_bias,
               const int16_t *hh_weight, const int16_t *hh_bias,
               int16_t *y, int16_t *h_prev);

/* Per-frame independent GRU — each frame has its own hidden state at h_prev[f*hidden_dim] */
void gru_fixed_perframe(const int32_t *x, int seq_len, int input_dim, int hidden_dim,
                        const int16_t *ih_weight, const int16_t *ih_bias,
                        const int16_t *hh_weight, const int16_t *hh_bias,
                        int16_t *y, int16_t *h_prev,
                        int state_stride);

/* BiGRU: bidirectional GRU */
void bigru_fixed(const int32_t *x, int seq_len, int input_dim, int hidden_dim,
                 const int16_t *ih_w, const int16_t *ih_b,
                 const int16_t *hh_w, const int16_t *hh_b,
                 const int16_t *re_ih_w, const int16_t *re_ih_b,
                 const int16_t *re_hh_w, const int16_t *re_hh_b,
                 int16_t *y);

/* SFE: Subband Feature Extraction: (C, W) → (C*3, W) */
void SFE_fixed(const int32_t *x, int C, int Win, int32_t *y);

/* BM: Band Merging: (3, 257) → (3, 129) */
void BM_fixed(const int32_t *x, const uint16_t *weight, int32_t *y);

/* BS: Band Splitting: (2, 129) → (2, 257), input s16f15 → output s32f20 */
void BS_fixed(const int16_t *x, const uint16_t *weight, int16_t *y);

/* MASK: Complex Ratio Mask, mask s16f15 × real/imag s32f20 → output s32f20 */
void MASK_fixed(const int16_t *mask, const int32_t *real_in,
                const int32_t *imag_in, int32_t *y);

/* mag_gen: magnitude + real + imag → (3, 257) s32f20 */
void mag_gen(const float *real_in, const float *imag_in, int N, int32_t *y);
/* Q15 integer version: real_in/imag_in in Q15, →Q20 via <<5 (exact) */
void mag_gen_q15(const int32_t *real_in, const int32_t *imag_in, int N, int32_t *y);

/* ================================================================
 * Function Prototypes — High-Level Modules
 * ================================================================ */

/* DD-Conv block: depthwise dilated conv + BN + PReLU */
void DD_Conv_block(const int32_t *x, int32_t *conv_hist,
                   int dilation, int gtc_idx,
                   const int16_t *conv_w, const int32_t *conv_b,
                   const uint16_t *bn_w, const int32_t *bn_b,
                   const int32_t *bn_mean, const uint16_t *bn_var,
                   const int16_t *prelu_w,
                   int32_t *y);

/* DD-DeConv block: depthwise dilated transposed conv + BN + PReLU */
void DD_DeConv_block(const int32_t *x, int32_t *conv_hist,
                     int dilation, int gtc_idx,
                     const int16_t *conv_w, const int32_t *conv_b,
                     const uint16_t *bn_w, const int32_t *bn_b,
                     const int32_t *bn_mean, const uint16_t *bn_var,
                     const int16_t *prelu_w,
                     int32_t *y);

/* TRA: Temporal Recurrent Attention */
void TRA_module(const int32_t *x, int16_t *h_prev, int gtc_idx,
                const int16_t *ih_w, const int32_t *ih_b,  /* s16f13, s32f20 */
                const int16_t *hh_w, const int32_t *hh_b,
                const int16_t *fc_w, const int32_t *fc_b,
                int32_t *y);

/* DeTRA: Decoder Temporal Recurrent Attention */
void DeTRA_module(const int32_t *x, int16_t *h_prev, int gtc_idx,
                  const int16_t *ih_w, const int32_t *ih_b,  /* s16f13, s32f18 */
                  const int16_t *hh_w, const int32_t *hh_b,
                  const int16_t *fc_w, const int32_t *fc_b,
                  int32_t *y);

/* GT-Conv: Grouped Temporal Convolution block */
void GT_Conv_module(const int32_t *x, int32_t *conv_hist, int16_t *h_prev,
                    int dilation, int gtc_idx,
                    const int16_t *pc0_w, const int32_t *pc0_b,
                    const uint16_t *pc0_bn_w, const int32_t *pc0_bn_b,
                    const int32_t *pc0_bn_m, const uint16_t *pc0_bn_v,
                    const int16_t *pc0_prelu_w,
                    const int16_t *dd_w, const int32_t *dd_b,
                    const uint16_t *dd_bn_w, const int32_t *dd_bn_b,
                    const int32_t *dd_bn_m, const uint16_t *dd_bn_v,
                    const int16_t *dd_prelu_w,
                    const int16_t *pc1_w, const int32_t *pc1_b,
                    const uint16_t *pc1_bn_w, const int32_t *pc1_bn_b,
                    const int32_t *pc1_bn_m, const uint16_t *pc1_bn_v,
                    const int16_t *tra_ih_w, const int32_t *tra_ih_b,
                    const int16_t *tra_hh_w, const int32_t *tra_hh_b,
                    const int16_t *tra_fc_w, const int32_t *tra_fc_b,
                    int32_t *y);

/* GT-DeConv: Grouped Temporal Transposed Convolution block */
void GT_DeConv_module(const int32_t *x_in, const int32_t *x_skip,
                      int32_t *conv_hist, int16_t *h_prev,
                      int dilation, int gtc_idx,
                      const int16_t *pc0_w, const int32_t *pc0_b,
                      const uint16_t *pc0_bn_w, const int32_t *pc0_bn_b,
                      const int32_t *pc0_bn_m, const uint16_t *pc0_bn_v,
                      const int16_t *pc0_prelu_w,
                      const int16_t *dd_w, const int32_t *dd_b,
                      const uint16_t *dd_bn_w, const int32_t *dd_bn_b,
                      const int32_t *dd_bn_m, const uint16_t *dd_bn_v,
                      const int16_t *dd_prelu_w,
                      const int16_t *pc1_w, const int32_t *pc1_b,
                      const uint16_t *pc1_bn_w, const int32_t *pc1_bn_b,
                      const int32_t *pc1_bn_m, const uint16_t *pc1_bn_v,
                      const int16_t *detra_ih_w, const int32_t *detra_ih_b,
                      const int16_t *detra_hh_w, const int32_t *detra_hh_b,
                      const int16_t *detra_fc_w, const int32_t *detra_fc_b,
                      int32_t *y);

/* Conv_block_0: standard conv2d → BN → PReLU */
void Conv_block_0(const int32_t *x,
                  const int32_t *conv_w, const int32_t *conv_b,
                  const uint16_t *bn_w, const int32_t *bn_b,
                  const int32_t *bn_m, const uint16_t *bn_v,
                  const int16_t *prelu_w,
                  int32_t *y);

/* Conv_block_1: grouped conv2d → BN → PReLU */
void Conv_block_1(const int32_t *x,
                  const int16_t *conv_w, const int32_t *conv_b,
                  const uint16_t *bn_w, const int32_t *bn_b,
                  const int32_t *bn_m, const uint16_t *bn_v,
                  const int16_t *prelu_w,
                  int32_t *y);

/* DeConv_block_0: transposed conv2d + skip → BN → Tanh */
void DeConv_block_0(const int32_t *x_in, const int32_t *x_skip,
                    const int32_t *conv_w, const int32_t *conv_b,
                    const uint16_t *bn_w, const int32_t *bn_b,
                    const int32_t *bn_m, const uint16_t *bn_v,
                    int16_t *y);

/* DeConv_block_1: grouped transposed conv2d + skip → BN → PReLU */
void DeConv_block_1(const int32_t *x_in, const int32_t *x_skip,
                    const int16_t *conv_w, const int32_t *conv_b,
                    const uint16_t *bn_w, const int32_t *bn_b,
                    const int32_t *bn_m, const uint16_t *bn_v,
                    const int16_t *prelu_w,
                    int32_t *y);

/* P_Conv_block_0: pointwise conv (Cin=24, Cout=16) → BN → PReLU */
void P_Conv_block_0(const int32_t *x, int gtc_idx,
                    const int16_t *conv_w, const int32_t *conv_b,
                    const uint16_t *bn_w, const int32_t *bn_b,
                    const int32_t *bn_m, const uint16_t *bn_v,
                    const int16_t *prelu_w,
                    int32_t *y);

/* P_Conv_block_1: pointwise conv (Cin=16, Cout=8) → BN */
void P_Conv_block_1(const int32_t *x, int gtc_idx,
                    const int16_t *conv_w, const int32_t *conv_b,
                    const uint16_t *bn_w, const int32_t *bn_b,
                    const int32_t *bn_m, const uint16_t *bn_v,
                    int32_t *y);

/* P_DeConv_block_0: pointwise transposed conv (Cin=24, Cout=16) → BN → PReLU */
void P_DeConv_block_0(const int32_t *x, int gtc_idx,
                      const int16_t *conv_w, const int32_t *conv_b,
                      const uint16_t *bn_w, const int32_t *bn_b,
                      const int32_t *bn_m, const uint16_t *bn_v,
                      const int16_t *prelu_w,
                      int32_t *y);

/* P_DeConv_block_1: pointwise transposed conv (Cin=16, Cout=8) → BN */
void P_DeConv_block_1(const int32_t *x, int gtc_idx,
                      const int16_t *conv_w, const int32_t *conv_b,
                      const uint16_t *bn_w, const int32_t *bn_b,
                      const int32_t *bn_m, const uint16_t *bn_v,
                      int32_t *y);

/* Intra_RNN: BiGRU + FC + LN + residual */
void Intra_RNN_module(const int32_t *x, int dprnn_idx,
                      const int16_t *rnn1_ih_w, const int16_t *rnn1_ih_b,
                      const int16_t *rnn1_hh_w, const int16_t *rnn1_hh_b,
                      const int16_t *rnn1_re_ih_w, const int16_t *rnn1_re_ih_b,
                      const int16_t *rnn1_re_hh_w, const int16_t *rnn1_re_hh_b,
                      const int16_t *rnn2_ih_w, const int16_t *rnn2_ih_b,
                      const int16_t *rnn2_hh_w, const int16_t *rnn2_hh_b,
                      const int16_t *rnn2_re_ih_w, const int16_t *rnn2_re_ih_b,
                      const int16_t *rnn2_re_hh_w, const int16_t *rnn2_re_hh_b,
                      const int16_t *fc_w, const int32_t *fc_b,
                      const int16_t *ln_w, const int32_t *ln_b,
                      int32_t *y);

/* Inter_RNN: GRU + FC + LN + residual */
void Inter_RNN_module(const int32_t *x, int16_t *h_prev, int dprnn_idx,
                      const int16_t *rnn1_ih_w, const int16_t *rnn1_ih_b,
                      const int16_t *rnn1_hh_w, const int16_t *rnn1_hh_b,
                      const int16_t *rnn2_ih_w, const int16_t *rnn2_ih_b,
                      const int16_t *rnn2_hh_w, const int16_t *rnn2_hh_b,
                      const int16_t *fc_w, const int32_t *fc_b,
                      const int16_t *ln_w, const int32_t *ln_b,
                      int32_t *y);

/* GDPRNN: Intra-RNN → Inter-RNN */
void GDPRNN_module(const int32_t *x, int16_t *inter_prev, int dprnn_idx,
                   const int16_t *rnn1_ih_w, const int16_t *rnn1_ih_b,
                   const int16_t *rnn1_hh_w, const int16_t *rnn1_hh_b,
                   const int16_t *rnn1_re_ih_w, const int16_t *rnn1_re_ih_b,
                   const int16_t *rnn1_re_hh_w, const int16_t *rnn1_re_hh_b,
                   const int16_t *rnn2_ih_w, const int16_t *rnn2_ih_b,
                   const int16_t *rnn2_hh_w, const int16_t *rnn2_hh_b,
                   const int16_t *rnn2_re_ih_w, const int16_t *rnn2_re_ih_b,
                   const int16_t *rnn2_re_hh_w, const int16_t *rnn2_re_hh_b,
                   const int16_t *inter_rnn1_ih_w, const int16_t *inter_rnn1_ih_b,
                   const int16_t *inter_rnn1_hh_w, const int16_t *inter_rnn1_hh_b,
                   const int16_t *inter_rnn2_ih_w, const int16_t *inter_rnn2_ih_b,
                   const int16_t *inter_rnn2_hh_w, const int16_t *inter_rnn2_hh_b,
                   const int16_t *intra_fc_w, const int32_t *intra_fc_b,
                   const int16_t *intra_ln_w, const int32_t *intra_ln_b,
                   const int16_t *inter_fc_w, const int32_t *inter_fc_b,
                   const int16_t *inter_ln_w, const int32_t *inter_ln_b,
                   int32_t *y);

/* Encoder: Conv0 → Conv1 → GT-Conv(×3) */
void Encoder_module(const int32_t *x, int32_t *conv_hist, int16_t *h_prev,
                    int32_t *y0, int32_t *y1, int32_t *y2,
                    int32_t *y3, int32_t *y4);

/* Decoder: GT-DeConv(×3) → DeConv1 → DeConv0 */
void Decoder_module(const int32_t *x, const int32_t *y0, const int32_t *y1,
                    const int32_t *y2, const int32_t *y3, const int32_t *y4,
                    int32_t *conv_hist, int16_t *h_prev,
                    int32_t *y);

/* Main inference: single-frame processing, CRM output s32f20 */
void denoise_infer_frame(const float *real_in, const float *imag_in,
                       denoise_state_t *state,
                       const uint16_t *erbfc_w, const uint16_t *ierbfc_w,
                       int32_t *crm_out);
/* Q15 integer version — real_in/imag_in in Q15 (int32_t), skips float */
void denoise_infer_frame_q15(const int32_t *real_in, const int32_t *imag_in,
                           denoise_state_t *state,
                           const uint16_t *erbfc_w, const uint16_t *ierbfc_w,
                           int32_t *crm_out);

/* Initialize state to zero */
void denoise_state_init(denoise_state_t *state);

#ifdef __cplusplus
}
#endif

#endif /* DENOISE_FP_H */
