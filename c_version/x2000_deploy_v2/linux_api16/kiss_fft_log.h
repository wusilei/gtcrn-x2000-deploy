/* kiss_fft_log.h — stderr references PHYSICALLY REMOVED.
 * Original: #define KISS_FFT_ERROR(...) fprintf(stderr, __VA_ARGS__)
 * This empty header eliminates all stderr/fprintf/fputc/fwrite references.
 */
#define KISS_FFT_ERROR(...)  ((void)0)
#define KISS_FFT_WARNING(...) ((void)0)
#define KISS_FFT_INFO(...)  ((void)0)
