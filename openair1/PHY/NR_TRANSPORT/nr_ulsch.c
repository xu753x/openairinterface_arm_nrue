/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
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

/*! \file PHY/NR_TRANSPORT/nr_ulsch.c
* \brief Top-level routines for the reception of the PUSCH TS 38.211 v 15.4.0
* \author Ahmed Hussein
* \date 2019
* \version 0.1
* \company Fraunhofer IIS
* \email: ahmed.hussein@iis.fraunhofer.de
* \note
* \warning
*/

#include <stdint.h>
#include "PHY/NR_TRANSPORT/nr_transport_common_proto.h"
#include "PHY/NR_TRANSPORT/nr_ulsch.h"
#include "PHY/NR_REFSIG/nr_refsig.h"

void init_nr_ulsch(PHY_VARS_gNB *gNB, int size, int N_RB_UL) {
  gNB->ulDataSize=size*2;
  gNB->ulData=calloc(gNB->ulDataSize, sizeof(NR_gNB_ULSCH_t *));
  for (int i=0; i<gNB->ulDataSize; i++) {

    LOG_I(PHY,"Allocating Transport Channel Buffer for ULSCH, UE %d\n",i);
    AssertFatal( NULL != (gNB->ulData[i]=new_gNB_ulsch(MAX_LDPC_ITERATIONS, N_RB_UL, 0)), "");
  }
      /*
      LOG_I(PHY,"Initializing nFAPI for ULSCH, UE %d\n",i);
      // [hna] added here for RT implementation
      uint8_t harq_pid = 0;
      nfapi_nr_ul_config_ulsch_pdu *rel15_ul = &gNB->ulsch[i+1][j]->harq_processes[harq_pid]->ulsch_pdu;
  
      // --------- setting rel15_ul parameters ----------
      rel15_ul->rnti                           = 0x1234;
      rel15_ul->ulsch_pdu_rel15.start_rb       = 0;
      rel15_ul->ulsch_pdu_rel15.number_rbs     = 50;
      rel15_ul->ulsch_pdu_rel15.start_symbol   = 2;
      rel15_ul->ulsch_pdu_rel15.number_symbols = 12;
      rel15_ul->ulsch_pdu_rel15.length_dmrs    = gNB->dmrs_UplinkConfig.pusch_maxLength;
      rel15_ul->ulsch_pdu_rel15.Qm             = 2;
      rel15_ul->ulsch_pdu_rel15.R              = 679;
      rel15_ul->ulsch_pdu_rel15.mcs            = 9;
      rel15_ul->ulsch_pdu_rel15.rv             = 0;
      rel15_ul->ulsch_pdu_rel15.n_layers       = 1;
      ///////////////////////////////////////////////////
      */
}

NR_gNB_ULSCH_t  * find_nr_ulsch(uint16_t rnti, PHY_VARS_gNB *gNB,find_type_t type) {

  uint16_t i;
  NR_gNB_ULSCH_t * free=NULL;
  for (i=0; i<gNB->ulDataSize; i++) {
     if (gNB->ulData[i]->harq_mask>0 && gNB->ulData[i]->rnti==rnti)
        return gNB->ulData[i];
     else
        if (gNB->ulData[i]->harq_mask==0)
          free=gNB->ulData[i];
  }
  if (type == SEARCH_EXIST)
    return NULL;
  if (free==NULL) {
     LOG_E(PHY,"Table full!!!\n");
     return NULL;
  } else {
     free->rnti=0;
     return free;
  }
}

int free_nr_ulsch(uint16_t rnti, PHY_VARS_gNB *gNB) {
    int rm=0;
    for (int j = 0; j < gNB->ulDataSize; j++)
      NR_gNB_ULSCH_t  *ulsch=gNB->ulData[i];
      if (ulsch->rnti == rnti) {
        
        ulsch->rnti = 0;
        ulsch->harq_mask = 0;
        //clean_gNB_ulsch(ulsch);
        for (int h = 0; h < NR_MAX_ULSCH_HARQ_PROCESSES; h++) {
          ulsch->harq_processes[h]->status = SCH_IDLE;
          ulsch->harq_processes[h]->round  = 0;
          ulsch->harq_processes[h]->handled = 0;
        }
        rm++;
      }
    return rm;
}

void nr_fill_ulsch(PHY_VARS_gNB *gNB,
                   int frame,
                   int slot,
                   nfapi_nr_pusch_pdu_t *ulsch_pdu) {

 
  NR_gNB_ULSCH_t  *ulsch = find_nr_ulsch(ulsch_pdu->rnti,gNB,SEARCH_EXIST_OR_FREE);

  int harq_pid = ulsch_pdu->pusch_data.harq_process_id;
  ulsch->rnti = ulsch_pdu->rnti;
  //ulsch->rnti_type;
  ulsch->harq_mask |= 1<<harq_pid;
  ulsch->harq_process_id[slot] = harq_pid;

  ulsch->harq_processes[harq_pid]->frame=frame;
  ulsch->harq_processes[harq_pid]->slot=slot;
  ulsch->harq_processes[harq_pid]->handled= 0;
  ulsch->harq_processes[harq_pid]->status= NR_ACTIVE;
  memcpy((void*)&ulsch->harq_processes[harq_pid]->ulsch_pdu, (void*)ulsch_pdu, sizeof(nfapi_nr_pusch_pdu_t));

  //LOG_D(PHY,"Initializing nFAPI for ULSCH, UE %d, harq_pid %d\n",ulsch_id,harq_pid);

}

void nr_ulsch_unscrambling(int16_t* llr,
                           uint32_t size,
                           uint8_t q,
                           uint32_t Nid,
                           uint32_t n_RNTI) {

  uint8_t reset;
  uint32_t x1, x2, s=0;

  reset = 1;
  x2 = (n_RNTI<<15) + Nid;

  for (uint32_t i=0; i<size; i++) {
    if ((i&0x1f)==0) {
      s = lte_gold_generic(&x1, &x2, reset);
      reset = 0;
    }
    if (((s>>(i&0x1f))&1)==1)
      llr[i] = -llr[i];
  }
}

void nr_ulsch_unscrambling_optim(int16_t* llr,
				 uint32_t size,
				 uint8_t q,
				 uint32_t Nid,
				 uint32_t n_RNTI) {
  
#if defined(__x86_64__) || defined(__i386__)
  uint32_t x1, x2, s=0;

  x2 = (n_RNTI<<15) + Nid;

  uint8_t *s8=(uint8_t *)&s;
  __m128i *llr128 = (__m128i*)llr;
  int j=0;
  s = lte_gold_generic(&x1, &x2, 1);

  for (int i=0; i<((size>>5)+((size&0x1f) > 0 ? 1 : 0)); i++,j+=4) {
    llr128[j]   = _mm_mullo_epi16(llr128[j],byte2m128i[s8[0]]);
    llr128[j+1] = _mm_mullo_epi16(llr128[j+1],byte2m128i[s8[1]]);
    llr128[j+2] = _mm_mullo_epi16(llr128[j+2],byte2m128i[s8[2]]);
    llr128[j+3] = _mm_mullo_epi16(llr128[j+3],byte2m128i[s8[3]]);
    s = lte_gold_generic(&x1, &x2, 0);
  }
#else

    nr_ulsch_unscrambling(llr,
                          size,
                          q,
                          Nid,
                          n_RNTI);
#endif
}

void dump_pusch_stats(PHY_VARS_gNB *gNB) {

  for (int i=0;i<NUMBER_OF_NR_ULSCH_MAX;i++)
    if (gNB->ulsch_stats[i].rnti>0) 
      LOG_I(PHY,"ULSCH RNTI %x: round_trials %d(%1.1e):%d(%1.1e):%d(%1.1e):%d, current_Qm %d, current_RI %d, total_bytes RX/SCHED %d/%d\n",
	    gNB->ulsch_stats[i].rnti,
	    gNB->ulsch_stats[i].round_trials[0],
	    (double)gNB->ulsch_stats[i].round_trials[1]/gNB->ulsch_stats[i].round_trials[0],
	    gNB->ulsch_stats[i].round_trials[1],
	    (double)gNB->ulsch_stats[i].round_trials[2]/gNB->ulsch_stats[i].round_trials[0],
	    gNB->ulsch_stats[i].round_trials[2],
	    (double)gNB->ulsch_stats[i].round_trials[3]/gNB->ulsch_stats[i].round_trials[0],
	    gNB->ulsch_stats[i].round_trials[3],
	    gNB->ulsch_stats[i].current_Qm,
	    gNB->ulsch_stats[i].current_RI,
	    gNB->ulsch_stats[i].total_bytes_rx,
	    gNB->ulsch_stats[i].total_bytes_tx);
  
}

void clear_pusch_stats(PHY_VARS_gNB *gNB) {

  for (int i=0;i<NUMBER_OF_NR_ULSCH_MAX;i++)
    memset((void*)&gNB->ulsch_stats[i],0,sizeof(gNB->ulsch_stats[i]));
}
