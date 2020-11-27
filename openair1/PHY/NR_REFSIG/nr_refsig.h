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

/* Definitions for LTE Reference signals */
/* Author R. Knopp / EURECOM / OpenAirInterface.org */
#ifndef __NR_REFSIG__H__
#define __NR_REFSIG__H__

#include "PHY/defs_gNB.h"
#include "PHY/LTE_REFSIG/lte_refsig.h"
#include "PHY/sse_intrin.h"

/*!\brief This function generates the NR Gold sequence (38-211, Sec 5.2.1) for the PBCH DMRS.
@param PHY_VARS_gNB* gNB structure provides configuration, frame parameters and the pointers to the 32 bits sequence storage tables
 */
void nr_init_pbch_dmrs(PHY_VARS_gNB* gNB);
/*!\brief This function generates the NR Gold sequence (38-211, Sec 5.2.1) for the PDCCH DMRS.
@param PHY_VARS_gNB* gNB structure provides configuration, frame parameters and the pointers to the 32 bits sequence storage tables
@param Nid is used for the initialization of x2, Physical cell Id by default or upper layer configured pdcch_scrambling_ID
 */
void nr_init_pdcch_dmrs(PHY_VARS_gNB* gNB, uint32_t Nid);
void nr_init_pdsch_dmrs(PHY_VARS_gNB* gNB, uint32_t Nid);
void nr_init_csi_rs(PHY_VARS_gNB* gNB, uint32_t Nid);

void nr_gold_pusch(PHY_VARS_gNB* gNB, uint32_t *Nid);

int nr_pusch_dmrs_rx(PHY_VARS_gNB *gNB,
                     unsigned int Ns,
                     unsigned int *nr_gold_pusch,
                     int32_t *output,
                     unsigned short p,
                     unsigned char lp,
                     unsigned short nb_pusch_rb,
                     uint32_t re_offset,
                     uint8_t dmrs_type);

void init_scrambling_luts(void);
void nr_gen_ref_conj_symbols(uint32_t *in, uint32_t length, int16_t *output, uint16_t offset, int mod_order);
uint8_t get_next_dmrs_symbol_in_slot(uint16_t  ul_dmrs_symb_pos, uint8_t counter, uint8_t end_symbol);
uint8_t get_dmrs_symbols_in_slot(uint16_t l_prime_mask,  uint16_t nb_symb);

void nr_generate_modulation_table(void);

extern __m64 byte2m64_re[256];
extern __m64 byte2m64_im[256];
extern __m128i byte2m128i[256];

#endif
