



#ifndef __PHY_LTE_TRANSPORT_PUCCH_EXTERN__H__
#define __PHY_LTE_TRANSPORT_PUCCH_EXTERN__H__

#include <stdint.h>

/* PUCCH format3 >> */
#define D_I             0
#define D_Q             1
#define D_IQDATA        2
#define D_NSLT1SF       2
#define D_NSYM1SLT      7
#define D_NSYM1SF       2*7
#define D_NSC1RB        12
#define D_NRB1PUCCH     2
#define D_NPUCCH_SF5    5
#define D_NPUCCH_SF4    4

extern int16_t W4[3][4];

extern int16_t W3_re[3][6];


extern int16_t W3_im[3][6];

extern int16_t alpha_re[12];
extern int16_t alpha_im[12];

extern char *pucch_format_string[];

extern uint8_t chcod_tbl[128][48];

extern int16_t W5_fmt3_re[5][5];

extern int16_t W5_fmt3_im[5][5];

extern int16_t W4_fmt3[4][4];

extern int16_t W2[2];

extern int16_t RotTBL_re[4];
extern int16_t RotTBL_im[4];

//np4_tbl, np5_tbl
extern uint8_t Np5_TBL[5];
extern uint8_t Np4_TBL[4];

// alpha_TBL
extern int16_t alphaTBL_re[12];
extern int16_t alphaTBL_im[12];

#endif
