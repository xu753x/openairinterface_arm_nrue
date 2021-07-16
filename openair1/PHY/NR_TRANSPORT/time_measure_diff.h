#include <time.h>

//测时间函数
extern struct timespec nr_t[];

#define NR_TIMESPEC_TO_DOUBLE_US( nr_t )    ( ( (double)nr_t.tv_sec * 1000000 ) + ( (double)nr_t.tv_nsec / 1000 ) )

extern  struct timespec  nr_get_timespec_diff(
              struct timespec *start,
              struct timespec *stop );
//--------------------------------------------------------------------------------------------------------
#if 0
//nr_ulsch_unscrambling_optim_fpga_ldpc测时间
extern double   unscramble_llr128_gettime_cur, unscramble_llr8_gettime_cur;
extern struct timespec unscramble_llr128_start, unscramble_llr8_start;
extern struct timespec unscramble_llr128_stop, unscramble_llr8_stop;

//nr_ulsch_decoding_fpga_ldpc测时间
extern double   decode_clock_gettime_cur;
extern struct timespec decode_start;
extern struct timespec decode_stop;

//nr_ulsch_procedures_fpga_ldpc测时间
extern double   unscrambing_gettime_cur,decoding_gettime_cur;
extern struct timespec unscrambing_start,decoding_start;
extern struct timespec unscrambing_stop,decoding_stop;

//phy_procedures_gNB_uespec_RX测总时间
extern double   ulsch_clock_gettime_cur;
extern struct timespec ulsch_start;
extern struct timespec ulsch_stop;
#endif
extern int8_t ul_de_llr8[];
