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
//============================================================================================================================
// encoder interface
#ifndef __NRLDPC_DEFS__H__
#define __NRLDPC_DEFS__H__
#include "openair1/PHY/CODING/nrLDPC_decoder/nrLDPC_types.h"
/**
   \brief LDPC encoder
   \param 1 input
   \param 2 channel_input
   \param 3 int Zc
   \param 4 int Kb
   \param 5 short block_length
   \param 6 short BG
   \param 7 int n_segment
   \param 8 unsigned int macro_num
   \param 9-12 time_stats_t *tinput,*tprep, *tparity,*toutput
*/
typedef struct {
	int n_segments;          // optim8seg
	unsigned int macro_num; // optim8segmulti
	unsigned char gen_code; //orig
	time_stats_t *tinput;
	time_stats_t *tprep;
	time_stats_t *tparity;
	time_stats_t *toutput;
}encoder_implemparams_t;
#define INIT0_LDPCIMPLEMPARAMS {0,0,0,NULL,NULL,NULL,NULL}
typedef int(*nrLDPC_encoderfunc_t)(unsigned char **,unsigned char **,int,int,short, short, encoder_implemparams_t*);
//============================================================================================================================
// decoder interface
/**
   \brief LDPC decoder API type definition
   \param p_decParams LDPC decoder parameters
   \param p_llr Input LLRs
   \param p_llrOut Output vector
   \param p_profiler LDPC profiler statistics
*/
typedef int32_t(*nrLDPC_decoderfunc_t)(t_nrLDPC_dec_params* , int8_t*, int8_t* , t_nrLDPC_procBuf* , t_nrLDPC_time_stats* );
typedef void(*cudft_EnTx)(int16_t*, int16_t*, unsigned char);
typedef void(*cudft_load)(void);
#endif