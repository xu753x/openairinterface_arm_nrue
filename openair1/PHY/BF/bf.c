
/*! \file openairinterface5g/openair1/PHY/BF/bf.c
 * \brief merge ISIP beamforming and QR decomposer
 * \author NCTU OpinConnect Terng-Yin Hsu, Sendren Xu, WEI-YING LIN, Min-Hsun Wu
 * \email  a22490010@gmail.com
 * \date   25-9-2021
 * \version 1.0
 * \note
 * \warning
 */

//#include "PHY/defs_common.h"
// #include "/home/isip/OAI/minhsun/openairinterface5g/openair1/PHY/defs_common.h"
// #include "PHY/defs_eNB.h"
// #include "PHY/phy_extern.h"
// #include "PHY/CODING/coding_defs.h"
// #include "PHY/CODING/coding_extern.h"
// #include "PHY/CODING/lte_interleaver_inline.h"
// #include "PHY/LTE_TRANSPORT/transport_eNB.h"
// #include "modulation_eNB.h"
// #include "nr_modulation.h"
// #include "common/utils/LOG/vcd_signal_dumper.h"

// #include "tools_defs.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h>
#include <xmmintrin.h>

#include "bf_struct.h"
#include "bf.h"

#define _mm_sign_epi16(xmmx,xmmy) _mm_xor_si128((xmmx),_mm_cmpgt_epi16(_mm_setzero_si128(),(xmmy)))



  /** \brief This function performs beamforming precoding for common
   * data
      @param txdataF Table of pointers for frequency-domain TX signals
      @param txdataF_BF Table of pointers for frequency-domain TX signals
      @param frame_parms Frame descriptor structure
  after beamforming
      @param beam_weights Beamforming weights applied on each
  antenna element and each carrier
      @param slot Slot number
      @param symbol Symbol index on which to act
      @param aa physical antenna index
      @param p logical antenna index
  */
 
//int counter = 0;
//struct timespec start_symbol_ts, end_symbol_ts ;

int nr_beam_precoding(int32_t **txdataF,
	                  int32_t **txdataF_BF,
                      NR_DL_FRAME_PARMS *frame_parms,
	                  int32_t ***beam_weights,
                      int slot,
                      int symbol,
                      int aa,
                      int nb_antenna_ports)
{
  uint8_t p;
  //struct timespec start_ts_1, end_ts_1;
  //int counter = 0; 
  // clear txdata_BF[aa][re] for each call of ue_spec_beamforming
  // printf("frame_parms->ofdm_symbol_size = %d\n",frame_parms->ofdm_symbol_size);
  // frame_parms->ofdm_symbol_size=4096 //RB=273
  // frame_parms->ofdm_symbol_size=2048 //RB=106
  // printf("txdataF_BF[%d][%d] : %04d\n",aa,symbol*frame_parms->ofdm_symbol_size,txdataF_BF[aa][symbol*frame_parms->ofdm_symbol_size]);
  // printf("&txdataF_BF[%d][%d] : %04p\n",aa,symbol*frame_parms->ofdm_symbol_size,&txdataF_BF[aa][symbol*frame_parms->ofdm_symbol_size]);
    
  memset(&txdataF_BF[aa][symbol*frame_parms->ofdm_symbol_size],0,sizeof(int32_t)*(frame_parms->ofdm_symbol_size));
  
  // printf("txdataF_BF[%d][%d] : %04d\n",aa,symbol*frame_parms->ofdm_symbol_size,txdataF_BF[aa][symbol*frame_parms->ofdm_symbol_size]);
  // printf("&txdataF_BF[%d][%d] : %04p\n",aa,symbol*frame_parms->ofdm_symbol_size,&txdataF_BF[aa][symbol*frame_parms->ofdm_symbol_size]);
  
  for (p=0; p<nb_antenna_ports; p++) {
    if ((frame_parms->L_ssb >> p) & 0x01) {
      
      
     //clock_gettime(CLOCK_MONOTONIC, &start_symbol_ts); 
    
      // printf("&txdataF[p][symbol*frame_parms->ofdm_symbol_size] :%x\n",&txdataF[p][symbol*frame_parms->ofdm_symbol_size]);
      // printf("&txdataF_BF[aa][symbol*frame_parms->ofdm_symbol_size] : %p\n",&txdataF_BF[aa][symbol*frame_parms->ofdm_symbol_size]);
      // printf("&beam_weights[p][aa] :%x\n",&beam_weights[p][aa]);
      // int f = symbol*frame_parms->ofdm_symbol_size;

      multadd_cpx_vector((int16_t*)&txdataF[p][symbol*frame_parms->ofdm_symbol_size],
                          (int16_t*)&beam_weights[p][aa], 
                          (int16_t*)&txdataF_BF[aa][symbol*frame_parms->ofdm_symbol_size], 
                          0, 
                          frame_parms->ofdm_symbol_size, 
                          15
                          );

      // if(check_time){
      //  clock_gettime(CLOCK_MONOTONIC, &end_symbol_ts); 
      // }
      
      
      /*
      if (symbol<(frame_parms->symbols_per_slot)*0.5)
      {
        //printf("[1] symbol*frame_parms->ofdm_symbol_size : %d\n",symbol*frame_parms->ofdm_symbol_size);
        multadd_cpx_vector((int16_t*)&txdataF[p][symbol*frame_parms->ofdm_symbol_size],
                          (int16_t*)&beam_weights[p][aa], 
                          (int16_t*)&txdataF_BF[aa][symbol*frame_parms->ofdm_symbol_size], 
                          0, 
                          frame_parms->ofdm_symbol_size, 
                          15
                          ); 
        
      }
      
      // if(check_time){
      //  clock_gettime(CLOCK_MONOTONIC, &end_symbol_ts); 
      // }  
      //  counter++;

      if (symbol<(frame_parms->symbols_per_slot)*0.5)
      {
        //printf("[2] symbol*frame_parms->ofdm_symbol_size : %d\n",(symbol*frame_parms->ofdm_symbol_size)+(7*frame_parms->ofdm_symbol_size));
        multadd_cpx_vector((int16_t*)&txdataF[p][(symbol*frame_parms->ofdm_symbol_size)+(7*frame_parms->ofdm_symbol_size)],
                          (int16_t*)&beam_weights[p][aa], 
                          (int16_t*)&txdataF_BF[aa][(symbol*frame_parms->ofdm_symbol_size)+(7*frame_parms->ofdm_symbol_size)], 
                          0, 
                          frame_parms->ofdm_symbol_size, 
                          15
                          ); 
      }
      */  

                       
    }
  }
  //printf("1 symbol [%d] : %.2f usec\n" , counter,(end_symbol_ts.tv_nsec - start_symbol_ts.tv_nsec) *1.0 / 1000);
  //counter++;
  // printf("txdataF_BF[%d][%d] : %04d\n",aa,symbol*frame_parms->ofdm_symbol_size,txdataF_BF[aa][symbol*frame_parms->ofdm_symbol_size]);
  // printf("&txdataF_BF[%d][%d] : %04p\n",aa,symbol*frame_parms->ofdm_symbol_size,&txdataF_BF[aa][symbol*frame_parms->ofdm_symbol_size]);

  return 0;
}


#if defined(__x86_64__) || defined(__i386__)
int16_t conjug[8]__attribute__((aligned(16))) = {-1,1,-1,1,-1,1,-1,1} ;
int16_t conjug2[8]__attribute__((aligned(16))) = {1,-1,1,-1,1,-1,1,-1} ;

#define simd_q15_t __m128i
#define simdshort_q15_t __m64
#define set1_int16(a) _mm_set1_epi16(a)
#define setr_int16(a0, a1, a2, a3, a4, a5, a6, a7) _mm_setr_epi16(a0, a1, a2, a3, a4, a5, a6, a7 )
#elif defined(__arm__)
int16_t conjug[4]__attribute__((aligned(16))) = {-1,1,-1,1} ;
#define simd_q15_t int16x8_t
#define simdshort_q15_t int16x4_t
#define _mm_empty()
#define _m_empty()
#endif


int multadd_cpx_vector(int16_t *x1,
                    int16_t *x2,
                    int16_t *y,
                    uint8_t zero_flag,
                    uint32_t N,
                    int output_shift)
{
  // Multiply elementwise the complex conjugate of x1 with x2.
  // x1       - input 1    in the format  |Re0 Im0 Re1 Im1|,......,|Re(N-2)  Im(N-2) Re(N-1) Im(N-1)|
  //            We assume x1 with a dinamic of 15 bit maximum
  //
  // x2       - input 2    in the format  |Re0 Im0 Re1 Im1|,......,|Re(N-2)  Im(N-2) Re(N-1) Im(N-1)|
  //            We assume x2 with a dinamic of 14 bit maximum
  ///
  // y        - output     in the format  |Re0 Im0 Re1 Im1|,......,|Re(N-2)  Im(N-2) Re(N-1) Im(N-1)|
  //
  // zero_flag - Set output (y) to zero prior to disable accumulation
  //
  // N        - the size f the vectors (this function does N cpx mpy. WARNING: N>=4;
  //
  // output_shift  - shift to be applied to generate output
  

  uint32_t i;                 // loop counter
  simd_q15_t *x1_128;
  simd_q15_t *x2_128;
  simd_q15_t *y_128;
#if defined(__x86_64__) || defined(__i386__)
  simd_q15_t tmp_re,tmp_im;
  simd_q15_t tmpy0,tmpy1;
#elif defined(__arm__)
  int32x4_t tmp_re,tmp_im;
  int32x4_t tmp_re1,tmp_im1;
  int16x4x2_t tmpy;
  int32x4_t shift = vdupq_n_s32(-output_shift);
#endif
  x1_128 = (simd_q15_t *)&x1[0];
  x2_128 = (simd_q15_t *)&x2[0];
  y_128  = (simd_q15_t *)&y[0];
  



  // we compute 4 cpx multiply for each loop
  for(i=0; i<(N>>2); i++) {
#if defined(__x86_64__) || defined(__i386__)
    tmp_re = _mm_sign_epi16(*x1_128,*(__m128i*)&conjug2[0]);
    tmp_re = _mm_madd_epi16(tmp_re,*x2_128);
    tmp_im = _mm_shufflelo_epi16(*x1_128,_MM_SHUFFLE(2,3,0,1));
    tmp_im = _mm_shufflehi_epi16(tmp_im,_MM_SHUFFLE(2,3,0,1));
    tmp_im = _mm_madd_epi16(tmp_im,*x2_128);
    tmp_re = _mm_srai_epi32(tmp_re,output_shift);
    tmp_im = _mm_srai_epi32(tmp_im,output_shift);
    tmpy0  = _mm_unpacklo_epi32(tmp_re,tmp_im);
    //print_ints("unpack lo:",&tmpy0[i]);
    tmpy1  = _mm_unpackhi_epi32(tmp_re,tmp_im);
    //print_ints("unpack hi:",&tmpy1[i]);
    if (zero_flag == 1)
      *y_128 = _mm_packs_epi32(tmpy0,tmpy1);
    else
      *y_128 = _mm_adds_epi16(*y_128,_mm_packs_epi32(tmpy0,tmpy1));
    //print_shorts("*y_128:",&y_128[i]);
#elif defined(__arm__)
    msg("mult_cpx_vector not implemented for __arm__");
#endif
    x1_128++;
    x2_128++;
    y_128++;
  }

  _mm_empty();
  _m_empty();

  return(0);
  
}

// void BF_kill_memmory(){
//   free(txdataF_BF);
// }

// void BF_init(){

//   int l,j,p,re;
  
//    NR_DL_FRAME_PARMS *fp = frame_parms;

//   //Table of pointers for frequency-domain TX signals
//   int32_t **txdataF_BF; 
  
//   //Beamforming weights applied on each antenna element and each carrier
//   int32_t **beam_weights[NUMBER_OF_gNB_MAX+1][15];
//   int32_t ***bws = beam_weights[0];
//   //printf("beam_weights[0] : %p\n",beam_weights[0]);
  
//   int tti_tx = 0;
  
//   // number of TX paths on device
//   int nb_tx = 1; 

//   // Number of NR L1 instances in this node
//   RC.nb_nr_L1_inst = 1;

  
//   //nr_init_ru.c
//   // allocate IFFT input buffers (TX)
//     txdataF_BF = (int32_t **)malloc16(nb_tx*sizeof(int32_t*));
//       //LOG_I(PHY,"[INIT] common.txdata_BF= %p (%lu bytes)\n",ru->common.txdataF_BF,ru->nb_tx*sizeof(int32_t*));
//         for (i=0; i<nb_tx; i++) {
//           txdataF_BF[i] = (int32_t*)malloc16_clear(fp->samples_per_subframe_wCP*sizeof(int32_t) );
//             //LOG_I(PHY,"txdataF_BF[%d] %p for RU %d\n",i,ru->common.txdataF_BF[i],ru->idx);
//         }

//     //int l_ind = 0;
//       for (i=0; i<RC.nb_nr_L1_inst; i++) {
//         for (p=0;p<fp->Lmax;p++) {
//           if ((fp->L_ssb >> p) & 0x01) {
// 	            beam_weights[i][p] = (int32_t **)malloc16_clear(nb_tx*sizeof(int32_t*));
// 	              for (j=0; j<nb_tx; j++) {
// 	                beam_weights[i][p][j] = (int32_t *)malloc16_clear(fp->ofdm_symbol_size*sizeof(int32_t));
//                     for (re=0; re<fp->ofdm_symbol_size; re++){
//                       beam_weights[i][p][j][re] = 0x00007fff/nb_tx; //lte
//                       //beam_weights[i][p][j][re] = ru->bw_list[i][l_ind];//nr
//                       } 
//                       //printf("Beam Weight %08x for beam %d and tx %d\n",ru->bw_list[i][l_ind],p,j);
//                       //l_ind++; 
//   	                } // for j
// 	                }  // for p
//                 }
//                }   

// }