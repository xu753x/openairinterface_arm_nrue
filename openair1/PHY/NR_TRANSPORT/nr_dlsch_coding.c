/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

/*! \file PHY/LTE_TRANSPORT/dlsch_coding.c
* \brief Top-level routines for implementing LDPC-coded (DLSCH) transport channels from 38-212, 15.2
* \author H.Wang
* \date 2018
* \version 0.1
* \company Eurecom
* \email:
* \note
* \warning
*/

#include "PHY/defs_gNB.h"
#include "PHY/phy_extern.h"
#include "PHY/CODING/coding_extern.h"
#include "PHY/CODING/coding_defs.h"
#include "PHY/CODING/lte_interleaver_inline.h"
#include "PHY/CODING/nrLDPC_extern.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/NR_TRANSPORT/nr_transport_common_proto.h"
#include "PHY/NR_TRANSPORT/nr_dlsch.h"
#include "openair2/LAYER2/NR_MAC_gNB/mac_proto.h"
#include "SCHED_NR/sched_nr.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "common/utils/LOG/log.h"
#include <syscall.h>

//#define DEBUG_DLSCH_CODING
//#define DEBUG_DLSCH_FREE 1


void free_gNB_dlsch(NR_gNB_DLSCH_t **dlschptr, uint16_t N_RB)
{
  int r;

  NR_gNB_DLSCH_t *dlsch = *dlschptr;

  uint16_t a_segments = MAX_NUM_NR_DLSCH_SEGMENTS;  //number of segments to be allocated
  if (dlsch) {

    if (N_RB != 273) {
      a_segments = a_segments*N_RB;
      a_segments = a_segments/273 +1;
    }

#ifdef DEBUG_DLSCH_FREE
    LOG_D(PHY,"Freeing dlsch %p\n",dlsch);
#endif
    NR_DL_gNB_HARQ_t *harq = &dlsch->harq_process;
    if (harq->b) {
      free16(harq->b, a_segments * 1056);
      harq->b = NULL;
#ifdef DEBUG_DLSCH_FREE
      LOG_D(PHY, "Freeing harq->b (%p)\n", harq->b);
#endif

      if (harq->e) {
        free16(harq->e, 14 * N_RB * 12 * 8);
        harq->e = NULL;
#ifdef DEBUG_DLSCH_FREE
        printf("Freeing dlsch process %d e (%p)\n", i, harq->e);
#endif
      }

      if (harq->f) {
        free16(harq->f, 14 * N_RB * 12 * 8);
        harq->f = NULL;
#ifdef DEBUG_DLSCH_FREE
        printf("Freeing dlsch process %d f (%p)\n", i, harq->f);
#endif
      }

#ifdef DEBUG_DLSCH_FREE
      LOG_D(PHY, "Freeing dlsch process %d c (%p)\n", i, harq->c);
#endif

      for (r = 0; r < a_segments; r++) {
#ifdef DEBUG_DLSCH_FREE
        LOG_D(PHY, "Freeing dlsch process %d c[%d] (%p)\n", i, r, harq->c[r]);
#endif

        if (harq->c[r]) {
          free16(harq->c[r], 1056);
          harq->c[r] = NULL;
        }
        if (harq->d[r]) {
          free16(harq->d[r], 3 * 8448);
          harq->d[r] = NULL;
        }
      }
    }
  }

  free16(dlsch, sizeof(NR_gNB_DLSCH_t));
  *dlschptr = NULL;
}

NR_gNB_DLSCH_t *new_gNB_dlsch(NR_DL_FRAME_PARMS *frame_parms,
                              unsigned char Kmimo,
                              unsigned char Mdlharq,
                              uint32_t Nsoft,
                              uint8_t  abstraction_flag,
                              uint16_t N_RB)
{
  unsigned char i,r,aa,layer;
  int re;
  uint16_t a_segments = MAX_NUM_NR_DLSCH_SEGMENTS;  //number of segments to be allocated

  if (N_RB != 273) {
    a_segments = a_segments*N_RB;
    a_segments = a_segments/273 +1;
  }

  uint16_t dlsch_bytes = a_segments*1056;  // allocated bytes per segment

  NR_gNB_DLSCH_t *dlsch = malloc16(sizeof(NR_gNB_DLSCH_t));
  AssertFatal(dlsch, "cannot allocate dlsch\n");

  bzero(dlsch,sizeof(NR_gNB_DLSCH_t));
  dlsch->Kmimo = Kmimo;
  dlsch->Mdlharq = Mdlharq;
  dlsch->Mlimit = 4;
  dlsch->Nsoft = Nsoft;

  for (layer=0; layer<NR_MAX_NB_LAYERS; layer++) {
    dlsch->ue_spec_bf_weights[layer] = (int32_t**)malloc16(64*sizeof(int32_t*));

    for (aa=0; aa<64; aa++) {
      dlsch->ue_spec_bf_weights[layer][aa] = (int32_t *)malloc16(OFDM_SYMBOL_SIZE_COMPLEX_SAMPLES*sizeof(int32_t));
      for (re=0;re<OFDM_SYMBOL_SIZE_COMPLEX_SAMPLES; re++) {
        dlsch->ue_spec_bf_weights[layer][aa][re] = 0x00007fff;
      }
    }

    dlsch->txdataF[layer] = (int32_t *)malloc16((NR_MAX_PDSCH_ENCODED_LENGTH/NR_MAX_NB_LAYERS)*sizeof(int32_t)); // NR_MAX_NB_LAYERS is already included in NR_MAX_PDSCH_ENCODED_LENGTH
    dlsch->txdataF_precoding[layer] = (int32_t *)malloc16(2*14*frame_parms->ofdm_symbol_size*sizeof(int32_t));
  }

  for (int q=0; q<NR_MAX_NB_CODEWORDS; q++)
    dlsch->mod_symbs[q] = (int32_t *)malloc16(NR_MAX_PDSCH_ENCODED_LENGTH*sizeof(int32_t));

  dlsch->calib_dl_ch_estimates = (int32_t**)malloc16(64*sizeof(int32_t*));
  for (aa=0; aa<64; aa++) {
    dlsch->calib_dl_ch_estimates[aa] = (int32_t *)malloc16(OFDM_SYMBOL_SIZE_COMPLEX_SAMPLES*sizeof(int32_t));
  }

  for (i=0; i<20; i++) {
    dlsch->harq_ids[0][i] = 0;
    dlsch->harq_ids[1][i] = 0;
  }

  NR_DL_gNB_HARQ_t *harq = &dlsch->harq_process;
  bzero(harq, sizeof(NR_DL_gNB_HARQ_t));

  harq->b = malloc16(dlsch_bytes);
  AssertFatal(harq->b, "cannot allocate memory for harq->b\n");
  harq->pdu = malloc16(dlsch_bytes);
  AssertFatal(harq->pdu, "cannot allocate memory for harq->pdu\n");
  bzero(harq->pdu, dlsch_bytes);
  nr_emulate_dlsch_payload(harq->pdu, (dlsch_bytes) >> 3);
  bzero(harq->b, dlsch_bytes);

  for (r = 0; r < a_segments; r++) {
    // account for filler in first segment and CRCs for multiple segment case
    // [hna] 8448 is the maximum CB size in NR
    //       68*348 = 68*(maximum size of Zc)
    //       In section 5.3.2 in 38.212, the for loop is up to N + 2*Zc (maximum size of N is 66*Zc, therefore 68*Zc)
    harq->c[r] = malloc16(8448);
    AssertFatal(harq->c[r], "cannot allocate harq->c[%d]\n", r);
    harq->d[r] = malloc16(68 * 384);
    AssertFatal(harq->d[r], "cannot allocate harq->d[%d]\n", r); // max size for coded output
    bzero(harq->c[r], 8448);
    bzero(harq->d[r], (3 * 8448));
    harq->e = malloc16(14 * N_RB * 12 * 8);
    AssertFatal(harq->e, "cannot allocate harq->e\n");
    bzero(harq->e, 14 * N_RB * 12 * 8);
    harq->f = malloc16(14 * N_RB * 12 * 8);
    AssertFatal(harq->f, "cannot allocate harq->f\n");
    bzero(harq->f, 14 * N_RB * 12 * 8);
  }

  return(dlsch);
}

void clean_gNB_dlsch(NR_gNB_DLSCH_t *dlsch)
{

  unsigned char Mdlharq;
  unsigned char i,j,r;

  AssertFatal(dlsch!=NULL,"dlsch is null\n");
  Mdlharq = dlsch->Mdlharq;
  dlsch->rnti = 0;
  dlsch->active = 0;
  NR_DL_gNB_HARQ_t *harq=&dlsch->harq_process;

  for (i=0; i<10; i++) {
    dlsch->harq_ids[0][i] = Mdlharq;
    dlsch->harq_ids[1][i] = Mdlharq;
  }
  for (i=0; i<Mdlharq; i++) {
    for (j=0; j<96; j++)
      for (r=0; r<MAX_NUM_NR_DLSCH_SEGMENTS; r++)
        if (harq->d[r])
          harq->d[r][j] = NR_NULL;
  }
}

int nr_dlsch_encoding(PHY_VARS_gNB *gNB,
		      unsigned char *a,
                      int frame,
                      uint8_t slot,
                      NR_gNB_DLSCH_t *dlsch,
                      NR_DL_FRAME_PARMS* frame_parms,
		      time_stats_t *tinput,time_stats_t *tprep,time_stats_t *tparity,time_stats_t *toutput,
		      time_stats_t *dlsch_rate_matching_stats,time_stats_t *dlsch_interleaving_stats,
		      time_stats_t *dlsch_segmentation_stats)
{

  unsigned int G;
  unsigned int crc=1;
  NR_DL_gNB_HARQ_t *harq = &dlsch->harq_process;
  nfapi_nr_dl_tti_pdsch_pdu_rel15_t *rel15 = &harq->pdsch_pdu.pdsch_pdu_rel15;
  uint16_t nb_rb = rel15->rbSize;
  uint8_t nb_symb_sch = rel15->NrOfSymbols;
  uint32_t A, Kb, F=0;
  uint32_t *Zc = &dlsch->harq_process.Z;
  uint8_t mod_order = rel15->qamModOrder[0];
  uint16_t Kr=0,r;
  uint32_t r_offset=0;
  uint32_t E;
  uint8_t Ilbrm = 1;
  uint32_t Tbslbrm = 950984; //max tbs
  uint8_t nb_re_dmrs;

  if (rel15->dmrsConfigType==NFAPI_NR_DMRS_TYPE1)
    nb_re_dmrs = 6*rel15->numDmrsCdmGrpsNoData;
  else
    nb_re_dmrs = 4*rel15->numDmrsCdmGrpsNoData;

  uint16_t length_dmrs = get_num_dmrs(rel15->dlDmrsSymbPos);
  uint16_t R=rel15->targetCodeRate[0];
  float Coderate = 0.0;
  uint8_t Nl = 4;

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_gNB_DLSCH_ENCODING, VCD_FUNCTION_IN);

  A = rel15->TBSize[0]<<3;

  NR_gNB_SCH_STATS_t *stats=NULL;
  int first_free=-1;
  for (int i=0;i<NUMBER_OF_NR_SCH_STATS_MAX;i++) {
    if (gNB->dlsch_stats[i].rnti == 0 && first_free == -1) {
      first_free = i;
      stats=&gNB->dlsch_stats[i];
    }
    if (gNB->dlsch_stats[i].rnti == dlsch->rnti) {
      stats=&gNB->dlsch_stats[i];
      break;
    }
  }

  if (stats) {
    stats->rnti = dlsch->rnti;
    stats->total_bytes_tx += rel15->TBSize[0];
    stats->current_RI   = rel15->nrOfLayers;
    stats->current_Qm   = rel15->qamModOrder[0];
  }
  G = nr_get_G(nb_rb, nb_symb_sch, nb_re_dmrs, length_dmrs,mod_order,rel15->nrOfLayers);

  LOG_D(PHY,"dlsch coding A %d G %d mod_order %d\n", A,G, mod_order);

  if (A > 3824) {
    // Add 24-bit crc (polynomial A) to payload
    crc = crc24a(a,A)>>8;
    a[A>>3] = ((uint8_t*)&crc)[2];
    a[1+(A>>3)] = ((uint8_t*)&crc)[1];
    a[2+(A>>3)] = ((uint8_t*)&crc)[0];
    //printf("CRC %x (A %d)\n",crc,A);
    //printf("a0 %d a1 %d a2 %d\n", a[A>>3], a[1+(A>>3)], a[2+(A>>3)]);

    harq->B = A+24;
    //    harq->b = a;

    AssertFatal((A / 8) + 4 <= MAX_NR_DLSCH_PAYLOAD_BYTES,
                "A %d is too big (A/8+4 = %d > %d)\n",
                A,
                (A / 8) + 4,
                MAX_NR_DLSCH_PAYLOAD_BYTES);

    memcpy(harq->b, a, (A / 8) + 4); // why is this +4 if the CRC is only 3 bytes?
  }
  else {
    // Add 16-bit crc (polynomial A) to payload
    crc = crc16(a,A)>>16;
    a[A>>3] = ((uint8_t*)&crc)[1];
    a[1+(A>>3)] = ((uint8_t*)&crc)[0];
    //printf("CRC %x (A %d)\n",crc,A);
    //printf("a0 %d a1 %d \n", a[A>>3], a[1+(A>>3)]);

    harq->B = A+16;
    //    harq->b = a;

    AssertFatal((A / 8) + 3 <= MAX_NR_DLSCH_PAYLOAD_BYTES,
                "A %d is too big (A/8+3 = %d > %d)\n",
                A,
                (A / 8) + 3,
                MAX_NR_DLSCH_PAYLOAD_BYTES);

    memcpy(harq->b, a, (A / 8) + 3); // using 3 bytes to mimic the case of 24 bit crc
  }
  if (R<1000)
    Coderate = (float) R /(float) 1024;
  else  // to scale for mcs 20 and 26 in table 5.1.3.1-2 which are decimal and input 2* in nr_tbs_tools
    Coderate = (float) R /(float) 2048;

  if ((A <=292) || ((A<=3824) && (Coderate <= 0.6667)) || Coderate <= 0.25)
    harq->BG = 2;
  else
    harq->BG = 1;

  start_meas(dlsch_segmentation_stats);
  Kb = nr_segmentation(harq->b, harq->c, harq->B, &harq->C, &harq->K, Zc, &harq->F, harq->BG);
  stop_meas(dlsch_segmentation_stats);
  F = harq->F;

  Kr = harq->K;
#ifdef DEBUG_DLSCH_CODING
  uint16_t Kr_bytes;
  Kr_bytes = Kr>>3;
#endif

  //printf("segment Z %d k %d Kr %d BG %d C %d\n", *Zc,harq->K,Kr,BG,harq->C);

  for (r=0; r<harq->C; r++) {
    //d_tmp[r] = &harq->d[r][0];
    //channel_input[r] = &harq->d[r][0];
#ifdef DEBUG_DLSCH_CODING
    LOG_D(PHY,"Encoder: B %d F %d \n",harq->B, harq->F);
    LOG_D(PHY,"start ldpc encoder segment %d/%d\n",r,harq->C);
    LOG_D(PHY,"input %d %d %d %d %d \n", harq->c[r][0], harq->c[r][1], harq->c[r][2],harq->c[r][3], harq->c[r][4]);
    for (int cnt =0 ; cnt < 22*(*Zc)/8; cnt ++){
      LOG_D(PHY,"%d ", harq->c[r][cnt]);
    }
    LOG_D(PHY,"\n");

#endif
    //ldpc_encoder_orig((unsigned char*)harq->c[r],harq->d[r],*Zc,Kb,Kr,BG,0);
    //ldpc_encoder_optim((unsigned char*)harq->c[r],(unsigned char*)&harq->d[r][0],*Zc,Kb,Kr,BG,NULL,NULL,NULL,NULL);
  }
  encoder_implemparams_t impp;
  impp.n_segments=harq->C;
  impp.tprep = tprep;
  impp.tinput = tinput;
  impp.tparity = tparity;
  impp.toutput = toutput;

  for(int j=0;j<(harq->C/8+1);j++) {
    impp.macro_num=j;
    nrLDPC_encoder(harq->c,harq->d,*Zc,Kb,Kr,harq->BG,&impp);
  }

#ifdef DEBUG_DLSCH_CODING
  write_output("enc_input0.m","enc_in0",&harq->c[0][0],Kr_bytes,1,4);
  write_output("enc_output0.m","enc0",&harq->d[0][0],(3*8*Kr_bytes)+12,1,4);
#endif

  F = harq->F;

  Kr = harq->K;
  for (r=0; r<harq->C; r++) {

    if (F>0) {
      for (int k=(Kr-F-2*(*Zc)); k<Kr-2*(*Zc); k++) {
	// writing into positions d[r][k-2Zc] as in clause 5.3.2 step 2) in 38.212
        harq->d[r][k] = NR_NULL;
	//if (k<(Kr-F+8))
	//printf("r %d filler bits [%d] = %d \n", r,k, harq->d[r][k]);
      }
    }

#ifdef DEBUG_DLSCH_CODING
    LOG_D(PHY,"rvidx in encoding = %d\n", rel15->rvIndex[0]);
#endif

    E = nr_get_E(G, harq->C, mod_order, rel15->nrOfLayers, r);

    //#ifdef DEBUG_DLSCH_CODING
    LOG_D(PHY,"Rate Matching, Code segment %d/%d (coded bits (G) %u, E %d, Filler bits %d, Filler offset %d mod_order %d, nb_rb %d)...\n",
	  r,
	  harq->C,
	  G,
	  E,
	  F,
	  Kr-F-2*(*Zc),
	  mod_order,nb_rb);

    // for tbslbrm calculation according to 5.4.2.1 of 38.212
    if (rel15->nrOfLayers < Nl)
      Nl = rel15->nrOfLayers;

    Tbslbrm = nr_compute_tbslbrm(rel15->mcsTable[0],nb_rb,Nl);

    start_meas(dlsch_rate_matching_stats);
    nr_rate_matching_ldpc(Ilbrm,
                          Tbslbrm,
                          harq->BG,
                          *Zc,
                          harq->d[r],
                          harq->e+r_offset,
                          harq->C,
                          F,
                          Kr-F-2*(*Zc),
                          rel15->rvIndex[0],
                          E);
    stop_meas(dlsch_rate_matching_stats);
#ifdef DEBUG_DLSCH_CODING
    for (int i =0; i<16; i++)
      printf("output ratematching e[%d]= %d r_offset %u\n", i,harq->e[i+r_offset], r_offset);
#endif

    start_meas(dlsch_interleaving_stats);
    nr_interleaving_ldpc(E,
			 mod_order,
			 harq->e+r_offset,
			 harq->f+r_offset);
    stop_meas(dlsch_interleaving_stats);

#ifdef DEBUG_DLSCH_CODING
    for (int i =0; i<16; i++)
      printf("output interleaving f[%d]= %d r_offset %u\n", i,harq->f[i+r_offset], r_offset);

    if (r==harq->C-1)
      write_output("enc_output.m","enc",harq->f,G,1,4);
#endif

    r_offset += E;
  }

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_gNB_DLSCH_ENCODING, VCD_FUNCTION_OUT);

  return 0;
}
