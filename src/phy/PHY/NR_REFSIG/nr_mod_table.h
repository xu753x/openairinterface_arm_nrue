

#ifndef __PHY_NR_REFSIG_NR_MOD_TABLE__H__
#define __PHY_NR_REFSIG_NR_MOD_TABLE__H__

#define NR_MOD_TABLE_SIZE_SHORT 686
#define NR_MOD_TABLE_BPSK_OFFSET 1
#define NR_MOD_TABLE_QPSK_OFFSET 3
#define NR_MOD_TABLE_QAM16_OFFSET 7
#define NR_MOD_TABLE_QAM64_OFFSET 23
#define NR_MOD_TABLE_QAM256_OFFSET 87

extern short nr_qpsk_mod_table[8];

extern int32_t nr_16qam_mod_table[16];
#if defined(__SSE2__)
extern __m128i nr_qpsk_byte_mod_table[2048];
#endif

extern int64_t nr_16qam_byte_mod_table[1024];

extern int64_t nr_64qam_mod_table[4096];

extern int32_t nr_256qam_mod_table[512];
#endif
