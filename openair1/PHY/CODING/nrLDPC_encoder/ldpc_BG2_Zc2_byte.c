#if defined(__x86_64__) || defined(__i386__)
#include "PHY/sse_intrin.h"
#elif defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
#endif
#define scalar_xor(a,b) ((a)^(b))
// generated code for Zc=2, byte encoding
static inline void ldpc_BG2_Zc2_byte(uint8_t *c,uint8_t *d) {
  uint8_t *csimd=(uint8_t *)c,*dsimd=(uint8_t *)d;

  uint8_t *c2,*d2;

  int i2;
  for (i2=0; i2<2; i2++) {
     c2=&csimd[i2];
     d2=&dsimd[i2];

//row: 0
     d2[0]=scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[36],scalar_xor(c2[37],c2[37]))))))))))))))))))))))))));

//row: 1
     d2[2]=scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[37],c2[37]))))))))))))))))))))))))))))))));

//row: 2
     d2[4]=scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[36],scalar_xor(c2[37],c2[37]))))))))))))))))))))))))))))))))))))))));

//row: 3
     d2[6]=scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[29],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[36],c2[37]))))))))))))))))))))))))))))))))));

//row: 4
     d2[8]=scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[36],c2[36]))))))))))))))))))))))))))))))))));

//row: 5
     d2[10]=scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[36],c2[36]))))))))))))))))))))))))))))))))))));

//row: 6
     d2[12]=scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[37],c2[36]))))))))))))))))))))))))))))))))))));

//row: 7
     d2[14]=scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[21],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[28],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[36],c2[37]))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));

//row: 8
     d2[16]=scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[36],scalar_xor(c2[37],c2[37]))))))))))))))))))))))))))))))))))))))))));

//row: 9
     d2[18]=scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[21],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[36],scalar_xor(c2[36],c2[36])))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));

//row: 10
     d2[20]=scalar_xor(c2[1],scalar_xor(c2[5],scalar_xor(c2[24],c2[29])));

//row: 11
     d2[22]=scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[29],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[28],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[37],c2[36])))))))))))))))))))))))))))))))))))));

//row: 12
     d2[24]=scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[37],c2[37]))))))))))))))))))))))))))))))))));

//row: 13
     d2[26]=scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[29],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[36],c2[37])))))))))))))))))))))))))))))))))))));

//row: 14
     d2[28]=scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[36],c2[37])))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));

//row: 15
     d2[30]=scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[21],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[36],scalar_xor(c2[36],c2[36]))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));

//row: 16
     d2[32]=scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[21],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[37],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[36],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[37],scalar_xor(c2[37],scalar_xor(c2[37],c2[36])))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));

//row: 17
     d2[34]=scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[37],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[36],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[37],scalar_xor(c2[37],c2[37])))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));

//row: 18
     d2[36]=scalar_xor(c2[0],scalar_xor(c2[24],c2[28]));

//row: 19
     d2[38]=scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[37],scalar_xor(c2[36],c2[36]))))))))))))))))))))))))))));

//row: 20
     d2[40]=scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[36],c2[36]))))))))))))))))))))))))))))))))));

//row: 21
     d2[42]=scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[29],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[36],c2[37]))))))))))))))))))))))))))))))))))));

//row: 22
     d2[44]=scalar_xor(c2[4],c2[9]);

//row: 23
     d2[46]=scalar_xor(c2[1],scalar_xor(c2[13],c2[20]));

//row: 24
     d2[48]=scalar_xor(c2[4],scalar_xor(c2[9],c2[36]));

//row: 25
     d2[50]=scalar_xor(c2[0],c2[20]);

//row: 26
     d2[52]=scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[21],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[36],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[37],c2[37])))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));

//row: 27
     d2[54]=scalar_xor(c2[0],c2[25]);

//row: 28
     d2[56]=scalar_xor(c2[4],scalar_xor(c2[9],c2[21]));

//row: 29
     d2[58]=scalar_xor(c2[0],c2[16]);

//row: 30
     d2[60]=scalar_xor(c2[9],scalar_xor(c2[20],scalar_xor(c2[29],c2[36])));

//row: 31
     d2[62]=scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[21],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[28],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[37],c2[36])))))))))))))))))))))))))))))))))));

//row: 32
     d2[64]=scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[36],scalar_xor(c2[37],c2[37]))))))))))))))))))))))))))))))))))))))))));

//row: 33
     d2[66]=scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[37],scalar_xor(c2[36],c2[36]))))))))))))))))))))))))))));

//row: 34
     d2[68]=scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[29],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[36],c2[37]))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));

//row: 35
     d2[70]=scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[36],c2[36]))))))))))))))))))))))))))))))))));

//row: 36
     d2[72]=scalar_xor(c2[0],scalar_xor(c2[8],c2[28]));

//row: 37
     d2[74]=scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[1],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[12],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[21],scalar_xor(c2[21],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[37],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[36],c2[36])))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));

//row: 38
     d2[76]=scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[37],c2[37]))))))))))))))))))))))))))))))))));

//row: 39
     d2[78]=scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[28],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[36],scalar_xor(c2[37],c2[37]))))))))))))))))))))))))))))))))))))))))));

//row: 40
     d2[80]=scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[8],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[16],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[21],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[28],scalar_xor(c2[28],scalar_xor(c2[29],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[32],scalar_xor(c2[32],scalar_xor(c2[33],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[36],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[36],c2[37]))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));

//row: 41
     d2[82]=scalar_xor(c2[1],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[0],scalar_xor(c2[5],scalar_xor(c2[4],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[5],scalar_xor(c2[8],scalar_xor(c2[9],scalar_xor(c2[9],scalar_xor(c2[12],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[13],scalar_xor(c2[16],scalar_xor(c2[17],scalar_xor(c2[17],scalar_xor(c2[20],scalar_xor(c2[21],scalar_xor(c2[21],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[25],scalar_xor(c2[24],scalar_xor(c2[29],scalar_xor(c2[29],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[33],scalar_xor(c2[37],scalar_xor(c2[36],scalar_xor(c2[37],c2[37]))))))))))))))))))))))))))))))))));
  }
}
