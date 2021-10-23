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

/**********************************************************************
*
* FILENAME    :  pucch_uci_ue_nr.h
*
* MODULE      :  Packed Uplink Control Channel aka PUCCH
*                PUCCH is used to trasnmit Uplink Control Information UCI
*                which is composed of:
*                - SR Scheduling Request
*                - HARQ ACK/NACK
*                - CSI Channel State Information
*
* DESCRIPTION :  functions related to PUCCH management
*                TS 38.213 9  UE procedure for reporting control information
*
************************************************************************/

#ifndef PUCCH_UCI_UE_NR_H
#define PUCCH_UCI_UE_NR_H

/************** INCLUDE *******************************************/

#include "PHY/defs_nr_UE.h"
#include "RRC/NR_UE/rrc_proto.h"

#ifdef DEFINE_VARIABLES_PUCCH_UE_NR_H
#define EXTERN
#define INIT_VARIABLES_PUCCH_UE_NR_H
#else
#define EXTERN extern
#undef INIT_VARIABLES_PUCCH_UE_NR_H
#endif

/************** DEFINE ********************************************/

#define BITS_PER_SYMBOL_BPSK  (1)     /* 1 bit per symbol for bpsk modulation */
#define BITS_PER_SYMBOL_QPSK  (2)     /* 2 bits per symbol for bpsk modulation */


/*************** FUNCTIONS ****************************************/

void pucch_procedures_ue_nr(PHY_VARS_NR_UE *ue, 
                            uint8_t gNB_id,
                            UE_nr_rxtx_proc_t *proc);


void set_csi_nr(int csi_status, uint32_t csi_payload);

uint8_t get_nb_symbols_pucch(NR_PUCCH_Resource_t *pucch_resource, pucch_format_nr_t format_type);

uint16_t get_starting_symb_idx(NR_PUCCH_Resource_t *pucch_resource, pucch_format_nr_t format_type);

int get_ics_pucch(NR_PUCCH_Resource_t *pucch_resource, pucch_format_nr_t format_type);

NR_PUCCH_Resource_t *select_resource_by_id(int resource_id, NR_PUCCH_Config_t *pucch_config);
#endif /* PUCCH_UCI_UE_NR_H */
