

/* \file config_ue.c
 * \brief common utility functions for NR (gNB and UE)
 * \author R. Knopp,
 * \date 2019
 * \version 0.1
 * \company Eurecom
 * \email: knopp@eurecom.fr
 * \note
 * \warning
 */

#ifndef __COMMON_UTILS_NR_NR_COMMON__H__
#define __COMMON_UTILS_NR_NR_COMMON__H__

#include <stdint.h>
#include "assertions.h"

typedef struct nr_bandentry_s {
  int16_t band;
  uint64_t ul_min;
  uint64_t ul_max;
  uint64_t dl_min;
  uint64_t dl_max;
  uint64_t step_size;
  uint64_t N_OFFs_DL;
  uint8_t deltaf_raster;
} nr_bandentry_t;

extern const size_t nr_bandtable_size;
extern nr_bandentry_t nr_bandtable[];

int NRRIV2BW(int locationAndBandwidth,int N_RB);
int NRRIV2PRBOFFSET(int locationAndBandwidth,int N_RB);
int PRBalloc_to_locationandbandwidth0(int NPRB,int RBstart,int BWPsize);
int PRBalloc_to_locationandbandwidth(int NPRB,int RBstart);
extern uint16_t nr_target_code_rate_table1[29];
extern uint16_t nr_target_code_rate_table2[28];
extern uint16_t nr_target_code_rate_table3[29];
extern uint16_t nr_tbs_table[93];
uint8_t nr_get_Qm(uint8_t Imcs, uint8_t table_idx);
uint32_t nr_get_code_rate(uint8_t Imcs, uint8_t table_idx);
int get_subband_size(int NPRB,int size);
void SLIV2SL(int SLIV,int *S,int *L);


#define CEILIDIV(a,b) ((a+b-1)/b)
#define ROUNDIDIV(a,b) (((a<<1)+b)/(b<<1))

#define cmax(a,b)  ((a>b) ? (a) : (b))
#define cmax3(a,b,c) ((cmax(a,b)>c) ? (cmax(a,b)) : (c))
#define cmin(a,b)  ((a<b) ? (a) : (b))

#ifdef __cplusplus
#ifdef min
#undef min
#undef max
#endif
#else
#define max(a,b) cmax(a,b)
#define min(a,b) cmin(a,b)
#endif

#endif
