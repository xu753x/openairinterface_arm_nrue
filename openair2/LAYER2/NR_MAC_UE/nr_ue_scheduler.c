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

/* \file        nr_ue_scheduler.c
 * \brief       Routines for UE scheduling
 * \author      Guido Casati
 * \date        Jan 2021
 * \version     0.1
 * \company     Fraunhofer IIS
 * \email       guido.casati@iis.fraunhofer.de
 */

#include <stdio.h>
#include <math.h>

/* exe */
#include <common/utils/nr/nr_common.h>

/* RRC*/
#include "RRC/NR_UE/rrc_proto.h"
#include "NR_RACH-ConfigCommon.h"
#include "NR_RACH-ConfigGeneric.h"
#include "NR_FrequencyInfoDL.h"
#include "NR_PDCCH-ConfigCommon.h"

/* MAC */
#include "NR_MAC_COMMON/nr_mac.h"
#include "NR_MAC_UE/mac_proto.h"
#include "NR_MAC_UE/mac_extern.h"

/* utils */
#include "assertions.h"
#include "asn1_conversions.h"
#include "SIMULATION/TOOLS/sim.h" // for taus

static prach_association_pattern_t prach_assoc_pattern;
static ssb_list_info_t ssb_list;

void fill_ul_config(fapi_nr_ul_config_request_t *ul_config, frame_t frame_tx, int slot_tx, uint8_t pdu_type){

  ul_config->ul_config_list[ul_config->number_pdus].pdu_type = pdu_type;
  ul_config->slot = slot_tx;
  ul_config->sfn = frame_tx;
  ul_config->number_pdus++;

  LOG_D(MAC, "In %s: Set config request for UL transmission in [%d.%d], number of UL PDUs: %d\n", __FUNCTION__, ul_config->sfn, ul_config->slot, ul_config->number_pdus);

}

void fill_scheduled_response(nr_scheduled_response_t *scheduled_response,
                             fapi_nr_dl_config_request_t *dl_config,
                             fapi_nr_ul_config_request_t *ul_config,
                             fapi_nr_tx_request_t *tx_request,
                             module_id_t mod_id,
                             int cc_id,
                             frame_t frame,
                             int slot,
                             int thread_id){

  scheduled_response->dl_config  = dl_config;
  scheduled_response->ul_config  = ul_config;
  scheduled_response->tx_request = tx_request;
  scheduled_response->module_id  = mod_id;
  scheduled_response->CC_id      = cc_id;
  scheduled_response->frame      = frame;
  scheduled_response->slot       = slot;
  scheduled_response->thread_id  = thread_id;

}

/*
 * This function returns the slot offset K2 corresponding to a given time domain
 * indication value from RRC configuration.
 */
long get_k2(NR_UE_MAC_INST_t *mac, uint8_t time_domain_ind) {
  long k2 = -1;
  // Get K2 from RRC configuration
  NR_PUSCH_Config_t *pusch_config=mac->ULbwp[0]->bwp_Dedicated->pusch_Config->choice.setup;
  NR_PUSCH_TimeDomainResourceAllocationList_t *pusch_TimeDomainAllocationList = NULL;
  if (pusch_config->pusch_TimeDomainAllocationList) {
    pusch_TimeDomainAllocationList = pusch_config->pusch_TimeDomainAllocationList->choice.setup;
  }
  else if (mac->ULbwp[0]->bwp_Common->pusch_ConfigCommon->choice.setup->pusch_TimeDomainAllocationList) {
    pusch_TimeDomainAllocationList = mac->ULbwp[0]->bwp_Common->pusch_ConfigCommon->choice.setup->pusch_TimeDomainAllocationList;
  }
  if (pusch_TimeDomainAllocationList) {
    if (time_domain_ind >= pusch_TimeDomainAllocationList->list.count) {
      LOG_E(MAC, "time_domain_ind %d >= pusch->TimeDomainAllocationList->list.count %d\n",
            time_domain_ind, pusch_TimeDomainAllocationList->list.count);
      return -1;
    }
    k2 = *pusch_TimeDomainAllocationList->list.array[time_domain_ind]->k2;
  }

  AssertFatal(k2 >= DURATION_RX_TO_TX,
              "Slot offset K2 (%ld) cannot be less than DURATION_RX_TO_TX (%d)\n",
              k2,DURATION_RX_TO_TX);

  LOG_D(MAC, "get_k2(): k2 is %ld\n", k2);
  return k2;
}

/*
 * This function returns the UL config corresponding to a given UL slot
 * from MAC instance .
 */
fapi_nr_ul_config_request_t *get_ul_config_request(NR_UE_MAC_INST_t *mac, int slot)
{
  //Check if request to access ul_config is for a UL slot
  if (is_nr_UL_slot(mac->scc, slot, mac->frame_type) == 0) {
    LOG_W(MAC, "Slot %d is not a UL slot. %s called for wrong slot!!!\n", slot, __FUNCTION__);
    return NULL;
  }

  // Calculate the index of the UL slot in mac->ul_config_request list. This is
  // based on the TDD pattern (slot configuration period) and number of UL+mixed
  // slots in the period. TS 38.213 Sec 11.1
  int mu = mac->ULbwp[0]->bwp_Common->genericParameters.subcarrierSpacing;
  NR_TDD_UL_DL_Pattern_t *tdd_pattern = &mac->scc->tdd_UL_DL_ConfigurationCommon->pattern1;
  const int num_slots_per_tdd = nr_slots_per_frame[mu] >> (7 - tdd_pattern->dl_UL_TransmissionPeriodicity);
  const int num_slots_ul = tdd_pattern->nrofUplinkSlots + (tdd_pattern->nrofUplinkSymbols!=0);
  int index = (slot + num_slots_ul - num_slots_per_tdd) % num_slots_per_tdd;

  LOG_D(MAC, "In %s slots per tdd %d, num_slots_ul %d, index %d\n", __FUNCTION__,
                num_slots_per_tdd,
                num_slots_ul,
                index);

  return &mac->ul_config_request[index];
}

void ul_layers_config(NR_UE_MAC_INST_t * mac, nfapi_nr_ue_pusch_pdu_t *pusch_config_pdu, dci_pdu_rel15_t *dci) {

  fapi_nr_pusch_config_dedicated_t *pusch_config_dedicated = &mac->phy_config.config_req.ul_bwp_dedicated.pusch_config_dedicated;
  NR_PUSCH_Config_t *pusch_Config = mac->ULbwp[0]->bwp_Dedicated->pusch_Config->choice.setup;

  /* PRECOD_NBR_LAYERS */
  if ((pusch_config_dedicated->tx_config == tx_config_nonCodebook));
  // 0 bits if the higher layer parameter txConfig = nonCodeBook

  if ((pusch_config_dedicated->tx_config == tx_config_codebook)){

    uint8_t n_antenna_port = 0; //FIXME!!!

    if (n_antenna_port == 1); // 1 antenna port and the higher layer parameter txConfig = codebook 0 bits

    if (n_antenna_port == 4){ // 4 antenna port and the higher layer parameter txConfig = codebook

      // Table 7.3.1.1.2-2: transformPrecoder=disabled and maxRank = 2 or 3 or 4
      if ((pusch_config_dedicated->transform_precoder == transform_precoder_disabled)
        && ((pusch_config_dedicated->max_rank == 2) ||
        (pusch_config_dedicated->max_rank == 3) ||
        (pusch_config_dedicated->max_rank == 4))){

        if (pusch_config_dedicated->codebook_subset == codebook_subset_fullyAndPartialAndNonCoherent) {
          pusch_config_pdu->nrOfLayers = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][0];
          pusch_config_pdu->transform_precoding = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][1];
        }

        if (pusch_config_dedicated->codebook_subset == codebook_subset_partialAndNonCoherent){
          pusch_config_pdu->nrOfLayers = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][2];
          pusch_config_pdu->transform_precoding = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][3];
        }

        if (pusch_config_dedicated->codebook_subset == codebook_subset_nonCoherent){
          pusch_config_pdu->nrOfLayers = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][4];
          pusch_config_pdu->transform_precoding = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][5];
        }
      }

      // Table 7.3.1.1.2-3: transformPrecoder= enabled, or transformPrecoder=disabled and maxRank = 1
      if (((pusch_config_dedicated->transform_precoder == transform_precoder_enabled)
        || (pusch_config_dedicated->transform_precoder == transform_precoder_disabled))
        && (pusch_config_dedicated->max_rank == 1)){

        if (pusch_config_dedicated->codebook_subset == codebook_subset_fullyAndPartialAndNonCoherent) {
          pusch_config_pdu->nrOfLayers = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][6];
          pusch_config_pdu->transform_precoding = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][7];
        }

        if (pusch_config_dedicated->codebook_subset == codebook_subset_partialAndNonCoherent){
          pusch_config_pdu->nrOfLayers = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][8];
          pusch_config_pdu->transform_precoding = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][9];
        }

        if (pusch_config_dedicated->codebook_subset == codebook_subset_nonCoherent){
          pusch_config_pdu->nrOfLayers = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][10];
          pusch_config_pdu->transform_precoding = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][11];
        }
      }
    }

    if (n_antenna_port == 4){ // 2 antenna port and the higher layer parameter txConfig = codebook
      // Table 7.3.1.1.2-4: transformPrecoder=disabled and maxRank = 2
      if ((pusch_config_dedicated->transform_precoder == transform_precoder_disabled) && (pusch_config_dedicated->max_rank == 2)){

        if (pusch_config_dedicated->codebook_subset == codebook_subset_fullyAndPartialAndNonCoherent) {
          pusch_config_pdu->nrOfLayers = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][12];
          pusch_config_pdu->transform_precoding = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][13];
        }

        if (pusch_config_dedicated->codebook_subset == codebook_subset_nonCoherent){
          pusch_config_pdu->nrOfLayers = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][14];
          pusch_config_pdu->transform_precoding = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][15];
        }

      }

      // Table 7.3.1.1.2-5: transformPrecoder= enabled, or transformPrecoder= disabled and maxRank = 1
      if (((pusch_config_dedicated->transform_precoder == transform_precoder_enabled)
        || (pusch_config_dedicated->transform_precoder == transform_precoder_disabled))
        && (pusch_config_dedicated->max_rank == 1)){

        if (pusch_config_dedicated->codebook_subset == codebook_subset_fullyAndPartialAndNonCoherent) {
          pusch_config_pdu->nrOfLayers = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][16];
          pusch_config_pdu->transform_precoding = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][17];
        }

        if (pusch_config_dedicated->codebook_subset == codebook_subset_nonCoherent){
          pusch_config_pdu->nrOfLayers = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][18];
          pusch_config_pdu->transform_precoding = table_7_3_1_1_2_2_3_4_5[dci->precoding_information.val][19];
        }

      }
    }
  }

  /*-------------------- Changed to enable Transform precoding in RF SIM------------------------------------------------*/

  if (pusch_config_pdu->transform_precoding == transform_precoder_enabled) {

    pusch_config_dedicated->transform_precoder = transform_precoder_enabled;

    if(pusch_Config->dmrs_UplinkForPUSCH_MappingTypeA != NULL) {

      NR_DMRS_UplinkConfig_t *NR_DMRS_ulconfig = pusch_Config->dmrs_UplinkForPUSCH_MappingTypeA->choice.setup;

      if (NR_DMRS_ulconfig->dmrs_Type == NULL)
        pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_a.dmrs_type = 1;
      if (NR_DMRS_ulconfig->maxLength == NULL)
        pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_a.max_length = 1;

    } else if(pusch_Config->dmrs_UplinkForPUSCH_MappingTypeB != NULL) {

      NR_DMRS_UplinkConfig_t *NR_DMRS_ulconfig = pusch_Config->dmrs_UplinkForPUSCH_MappingTypeB->choice.setup;

      if (NR_DMRS_ulconfig->dmrs_Type == NULL)
        pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_b.dmrs_type = 1;
      if (NR_DMRS_ulconfig->maxLength == NULL)
        pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_b.max_length = 1;

    }
  } else
    pusch_config_dedicated->transform_precoder = transform_precoder_disabled;

  // mapping type b configured from RRC. TBD: Mapping type b is not handled in this function.
  if ((pusch_config_dedicated->transform_precoder == transform_precoder_enabled) &&
      (pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_b.dmrs_type == 1) &&
      (pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_b.max_length == 1)) { // tables 7.3.1.1.2-6
    pusch_config_pdu->num_dmrs_cdm_grps_no_data = 2;
    pusch_config_pdu->dmrs_ports = dci->antenna_ports.val;
  }
}

// todo: this function shall be reviewed completely because of the many comments left by the author
void ul_ports_config(NR_UE_MAC_INST_t * mac, nfapi_nr_ue_pusch_pdu_t *pusch_config_pdu, dci_pdu_rel15_t *dci) {

  /* ANTENNA_PORTS */
  uint8_t rank = 0; // We need to initialize rank FIXME!!!
  fapi_nr_pusch_config_dedicated_t *pusch_config_dedicated = &mac->phy_config.config_req.ul_bwp_dedicated.pusch_config_dedicated;

  if ((pusch_config_dedicated->transform_precoder == transform_precoder_enabled) &&
    (pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_a.dmrs_type == 1) &&
    (pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_a.max_length == 1)) { // tables 7.3.1.1.2-6
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = 2; //TBC
      pusch_config_pdu->dmrs_ports = dci->antenna_ports.val; //TBC
  }

  if ((pusch_config_dedicated->transform_precoder == transform_precoder_enabled) &&
    (pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_a.dmrs_type == 1) &&
    (pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_a.max_length == 2)) { // tables 7.3.1.1.2-7

    pusch_config_pdu->num_dmrs_cdm_grps_no_data = 2; //TBC
    pusch_config_pdu->dmrs_ports = (dci->antenna_ports.val > 3)?(dci->antenna_ports.val-4):(dci->antenna_ports.val); //TBC
    //pusch_config_pdu->n_front_load_symb = (dci->antenna_ports > 3)?2:1; //FIXME
  }

  if ((pusch_config_dedicated->transform_precoder == transform_precoder_disabled) &&
    (pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_a.dmrs_type == 1) &&
    (pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_a.max_length == 1)) { // tables 7.3.1.1.2-8/9/10/11

    if (rank == 1) {
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = (dci->antenna_ports.val > 1)?2:1; //TBC
      pusch_config_pdu->dmrs_ports = (dci->antenna_ports.val > 1)?(dci->antenna_ports.val-2):(dci->antenna_ports.val); //TBC
    }

    if (rank == 2){
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = (dci->antenna_ports.val > 0)?2:1; //TBC
      pusch_config_pdu->dmrs_ports = 0; //FIXME
      //pusch_config_pdu->dmrs_ports[0] = (dci->antenna_ports > 1)?(dci->antenna_ports > 2 ?0:2):0;
      //pusch_config_pdu->dmrs_ports[1] = (dci->antenna_ports > 1)?(dci->antenna_ports > 2 ?2:3):1;
    }

    if (rank == 3){
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = 2; //TBC
      pusch_config_pdu->dmrs_ports = 0; //FIXME
      //pusch_config_pdu->dmrs_ports[0] = 0;
      //pusch_config_pdu->dmrs_ports[1] = 1;
      //pusch_config_pdu->dmrs_ports[2] = 2;
    }

    if (rank == 4){
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = 2; //TBC
      pusch_config_pdu->dmrs_ports = 0; //FIXME
      //pusch_config_pdu->dmrs_ports[0] = 0;
      //pusch_config_pdu->dmrs_ports[1] = 1;
      //pusch_config_pdu->dmrs_ports[2] = 2;
      //pusch_config_pdu->dmrs_ports[3] = 3;
    }
  }

  if ((pusch_config_dedicated->transform_precoder == transform_precoder_disabled) &&
    (pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_a.dmrs_type == 1) &&
    (pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_a.max_length == 2)) { // tables 7.3.1.1.2-12/13/14/15

    if (rank == 1){
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = (dci->antenna_ports.val > 1)?2:1; //TBC
      pusch_config_pdu->dmrs_ports = (dci->antenna_ports.val > 1)?(dci->antenna_ports.val > 5 ?(dci->antenna_ports.val-6):(dci->antenna_ports.val-2)):dci->antenna_ports.val; //TBC
      //pusch_config_pdu->n_front_load_symb = (dci->antenna_ports.val > 6)?2:1; //FIXME
    }

    if (rank == 2){
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = (dci->antenna_ports.val > 0)?2:1; //TBC
      pusch_config_pdu->dmrs_ports = 0; //FIXME
      //pusch_config_pdu->dmrs_ports[0] = table_7_3_1_1_2_13[dci->antenna_ports.val][1];
      //pusch_config_pdu->dmrs_ports[1] = table_7_3_1_1_2_13[dci->antenna_ports.val][2];
      //pusch_config_pdu->n_front_load_symb = (dci->antenna_ports.val > 3)?2:1; // FIXME
    }

    if (rank == 3){
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = 2; //TBC
      pusch_config_pdu->dmrs_ports = 0; //FIXME
      //pusch_config_pdu->dmrs_ports[0] = table_7_3_1_1_2_14[dci->antenna_ports.val][1];
      //pusch_config_pdu->dmrs_ports[1] = table_7_3_1_1_2_14[dci->antenna_ports.val][2];
      //pusch_config_pdu->dmrs_ports[2] = table_7_3_1_1_2_14[dci->antenna_ports.val][3];
      //pusch_config_pdu->n_front_load_symb = (dci->antenna_ports.val > 1)?2:1; //FIXME
    }

    if (rank == 4){
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = 2; //TBC
      pusch_config_pdu->dmrs_ports = 0; //FIXME
      //pusch_config_pdu->dmrs_ports[0] = table_7_3_1_1_2_15[dci->antenna_ports.val][1];
      //pusch_config_pdu->dmrs_ports[1] = table_7_3_1_1_2_15[dci->antenna_ports.val][2];
      //pusch_config_pdu->dmrs_ports[2] = table_7_3_1_1_2_15[dci->antenna_ports.val][3];
      //pusch_config_pdu->dmrs_ports[3] = table_7_3_1_1_2_15[dci->antenna_ports.val][4];
      //pusch_config_pdu->n_front_load_symb = (dci->antenna_ports.val > 1)?2:1; //FIXME
    }
  }

  if ((pusch_config_dedicated->transform_precoder == transform_precoder_disabled) &&
    (pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_a.dmrs_type == 2) &&
    (pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_a.max_length == 1)) { // tables 7.3.1.1.2-16/17/18/19

    if (rank == 1){
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = (dci->antenna_ports.val > 1)?((dci->antenna_ports.val > 5)?3:2):1; //TBC
      pusch_config_pdu->dmrs_ports = (dci->antenna_ports.val > 1)?(dci->antenna_ports.val > 5 ?(dci->antenna_ports.val-6):(dci->antenna_ports.val-2)):dci->antenna_ports.val; //TBC
    }

    if (rank == 2){
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = (dci->antenna_ports.val > 0)?((dci->antenna_ports.val > 2)?3:2):1; //TBC
      pusch_config_pdu->dmrs_ports = 0; //FIXME
      //pusch_config_pdu->dmrs_ports[0] = table_7_3_1_1_2_17[dci->antenna_ports.val][1];
      //pusch_config_pdu->dmrs_ports[1] = table_7_3_1_1_2_17[dci->antenna_ports.val][2];
    }

    if (rank == 3){
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = (dci->antenna_ports.val > 0)?3:2; //TBC
      pusch_config_pdu->dmrs_ports = 0; //FIXME
      //pusch_config_pdu->dmrs_ports[0] = table_7_3_1_1_2_18[dci->antenna_ports.val][1];
      //pusch_config_pdu->dmrs_ports[1] = table_7_3_1_1_2_18[dci->antenna_ports.val][2];
      //pusch_config_pdu->dmrs_ports[2] = table_7_3_1_1_2_18[dci->antenna_ports.val][3];
    }

    if (rank == 4){
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = dci->antenna_ports.val + 2; //TBC
      pusch_config_pdu->dmrs_ports = 0; //FIXME
      //pusch_config_pdu->dmrs_ports[0] = 0;
      //pusch_config_pdu->dmrs_ports[1] = 1;
      //pusch_config_pdu->dmrs_ports[2] = 2;
      //pusch_config_pdu->dmrs_ports[3] = 3;
    }
  }

  if ((pusch_config_dedicated->transform_precoder == transform_precoder_disabled) &&
    (pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_a.dmrs_type == 2) &&
    (pusch_config_dedicated->dmrs_ul_for_pusch_mapping_type_a.max_length == 2)) { // tables 7.3.1.1.2-20/21/22/23

    if (rank == 1){
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = table_7_3_1_1_2_20[dci->antenna_ports.val][0]; //TBC
      pusch_config_pdu->dmrs_ports = table_7_3_1_1_2_20[dci->antenna_ports.val][1]; //TBC
      //pusch_config_pdu->n_front_load_symb = table_7_3_1_1_2_20[dci->antenna_ports.val][2]; //FIXME
    }

    if (rank == 2){
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = table_7_3_1_1_2_21[dci->antenna_ports.val][0]; //TBC
      pusch_config_pdu->dmrs_ports = 0; //FIXME
      //pusch_config_pdu->dmrs_ports[0] = table_7_3_1_1_2_21[dci->antenna_ports.val][1];
      //pusch_config_pdu->dmrs_ports[1] = table_7_3_1_1_2_21[dci->antenna_ports.val][2];
      //pusch_config_pdu->n_front_load_symb = table_7_3_1_1_2_21[dci->antenna_ports.val][3]; //FIXME
      }

    if (rank == 3){
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = table_7_3_1_1_2_22[dci->antenna_ports.val][0]; //TBC
      pusch_config_pdu->dmrs_ports = 0; //FIXME
      //pusch_config_pdu->dmrs_ports[0] = table_7_3_1_1_2_22[dci->antenna_ports.val][1];
      //pusch_config_pdu->dmrs_ports[1] = table_7_3_1_1_2_22[dci->antenna_ports.val][2];
      //pusch_config_pdu->dmrs_ports[2] = table_7_3_1_1_2_22[dci->antenna_ports.val][3];
      //pusch_config_pdu->n_front_load_symb = table_7_3_1_1_2_22[dci->antenna_ports.val][4]; //FIXME
    }

    if (rank == 4){
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = table_7_3_1_1_2_23[dci->antenna_ports.val][0]; //TBC
      pusch_config_pdu->dmrs_ports = 0; //FIXME
      //pusch_config_pdu->dmrs_ports[0] = table_7_3_1_1_2_23[dci->antenna_ports.val][1];
      //pusch_config_pdu->dmrs_ports[1] = table_7_3_1_1_2_23[dci->antenna_ports.val][2];
      //pusch_config_pdu->dmrs_ports[2] = table_7_3_1_1_2_23[dci->antenna_ports.val][3];
      //pusch_config_pdu->dmrs_ports[3] = table_7_3_1_1_2_23[dci->antenna_ports.val][4];
      //pusch_config_pdu->n_front_load_symb = table_7_3_1_1_2_23[dci->antenna_ports.val][5]; //FIXME
    }
  }
}

// Configuration of Msg3 PDU according to clauses:
// - 8.3 of 3GPP TS 38.213 version 16.3.0 Release 16
// - 6.1.2.2 of TS 38.214
// - 6.1.3 of TS 38.214
// - 6.2.2 of TS 38.214
// - 6.1.4.2 of TS 38.214
// - 6.4.1.1.1 of TS 38.211
// - 6.3.1.7 of 38.211
int nr_config_pusch_pdu(NR_UE_MAC_INST_t *mac,
                        nfapi_nr_ue_pusch_pdu_t *pusch_config_pdu,
                        dci_pdu_rel15_t *dci,
                        RAR_grant_t *rar_grant,
                        uint16_t rnti,
                        uint8_t *dci_format){

  int f_alloc;
  int mask;
  int StartSymbolIndex;
  int NrOfSymbols;
  uint8_t nb_dmrs_re_per_rb;

  uint16_t        l_prime_mask = 1;
  uint16_t number_dmrs_symbols = 0;
  int                N_PRB_oh  = 0;

  NR_ServingCellConfigCommon_t *scc = mac->scc;
  int rnti_type = get_rnti_type(mac, rnti);

  // Common configuration
  pusch_config_pdu->dmrs_config_type = pusch_dmrs_type1;
  pusch_config_pdu->pdu_bit_map      = PUSCH_PDU_BITMAP_PUSCH_DATA;
  pusch_config_pdu->nrOfLayers       = 1;
  pusch_config_pdu->rnti             = rnti;

  if (rar_grant) {

    // Note: for Msg3 or MsgA PUSCH transmission the N_PRB_oh is always set to 0

    NR_BWP_Uplink_t *ubwp = mac->ULbwp[0];
    NR_BWP_UplinkDedicated_t *ibwp = mac->scg->spCellConfig->spCellConfigDedicated->uplinkConfig->initialUplinkBWP;
    NR_PUSCH_Config_t *pusch_Config = ibwp->pusch_Config->choice.setup;
    int startSymbolAndLength = ubwp->bwp_Common->pusch_ConfigCommon->choice.setup->pusch_TimeDomainAllocationList->list.array[rar_grant->Msg3_t_alloc]->startSymbolAndLength;

    // active BWP start
    int abwp_start = NRRIV2PRBOFFSET(ubwp->bwp_Common->genericParameters.locationAndBandwidth, MAX_BWP_SIZE);
    int abwp_size = NRRIV2BW(ubwp->bwp_Common->genericParameters.locationAndBandwidth, MAX_BWP_SIZE);

    // initial BWP start
    int ibwp_start = NRRIV2PRBOFFSET(scc->uplinkConfigCommon->initialUplinkBWP->genericParameters.locationAndBandwidth, MAX_BWP_SIZE);
    int ibwp_size = NRRIV2BW(scc->uplinkConfigCommon->initialUplinkBWP->genericParameters.locationAndBandwidth, MAX_BWP_SIZE);

    // BWP start selection according to 8.3 of TS 38.213
    pusch_config_pdu->bwp_size = ibwp_size;
    if ((ibwp_start < abwp_start) || (ibwp_size > abwp_size))
      pusch_config_pdu->bwp_start = abwp_start;
    else
      pusch_config_pdu->bwp_start = ibwp_start;

    //// Resource assignment from RAR
    // Frequency domain allocation according to 8.3 of TS 38.213
    if (ibwp_size < 180)
      mask = (1 << ((int) ceil(log2((ibwp_size*(ibwp_size+1))>>1)))) - 1;
    else
      mask = (1 << (28 - (int)(ceil(log2((ibwp_size*(ibwp_size+1))>>1))))) - 1;

    f_alloc = rar_grant->Msg3_f_alloc & mask;
    if (nr_ue_process_dci_freq_dom_resource_assignment(pusch_config_pdu, NULL, ibwp_size, 0, f_alloc) < 0)
      return -1;

    // virtual resource block to physical resource mapping for Msg3 PUSCH (6.3.1.7 in 38.211)
    pusch_config_pdu->rb_start += ibwp_start - abwp_start;

    // Time domain allocation
    SLIV2SL(startSymbolAndLength, &StartSymbolIndex, &NrOfSymbols);
    pusch_config_pdu->start_symbol_index = StartSymbolIndex;
    pusch_config_pdu->nr_of_symbols = NrOfSymbols;

    #ifdef DEBUG_MSG3
    LOG_D(MAC, "In %s BWP assignment (BWP (start %d, size %d) \n", __FUNCTION__, pusch_config_pdu->bwp_start, pusch_config_pdu->bwp_size);
    #endif

    // MCS
    pusch_config_pdu->mcs_index = rar_grant->mcs;
    // Frequency hopping
    pusch_config_pdu->frequency_hopping = rar_grant->freq_hopping;

    // DM-RS configuration according to 6.2.2 UE DM-RS transmission procedure in 38.214
    pusch_config_pdu->num_dmrs_cdm_grps_no_data = 2;
    pusch_config_pdu->dmrs_ports = 1;

    // DMRS sequence initialization [TS 38.211, sec 6.4.1.1.1].
    // Should match what is sent in DCI 0_1, otherwise set to 0.
    pusch_config_pdu->scid = 0;

    // Transform precoding according to 6.1.3 UE procedure for applying transform precoding on PUSCH in 38.214
    pusch_config_pdu->transform_precoding = get_transformPrecoding(scc, pusch_Config, NULL, NULL, NR_RNTI_RA, 0); // TBR fix rnti and take out

    // Resource allocation in frequency domain according to 6.1.2.2 in TS 38.214
    pusch_config_pdu->resource_alloc = pusch_Config->resourceAllocation;

    //// Completing PUSCH PDU
    pusch_config_pdu->mcs_table = 0;
    pusch_config_pdu->cyclic_prefix = 0;
    pusch_config_pdu->data_scrambling_id = *scc->physCellId;
    pusch_config_pdu->ul_dmrs_scrambling_id = *scc->physCellId;
    pusch_config_pdu->subcarrier_spacing = ubwp->bwp_Common->genericParameters.subcarrierSpacing;
    pusch_config_pdu->vrb_to_prb_mapping = 0;
    pusch_config_pdu->uplink_frequency_shift_7p5khz = 0;
    //Optional Data only included if indicated in pduBitmap
    pusch_config_pdu->pusch_data.rv_index = 0;  // 8.3 in 38.213
    pusch_config_pdu->pusch_data.harq_process_id = 0;
    pusch_config_pdu->pusch_data.new_data_indicator = 1; // new data
    pusch_config_pdu->pusch_data.num_cb = 0;

  } else if (dci) {

    int target_ss;
    bool valid_ptrs_setup = 0;
    uint16_t n_RB_ULBWP = NRRIV2BW(mac->ULbwp[0]->bwp_Common->genericParameters.locationAndBandwidth, MAX_BWP_SIZE);
    fapi_nr_pusch_config_dedicated_t *pusch_config_dedicated = &mac->phy_config.config_req.ul_bwp_dedicated.pusch_config_dedicated;
    NR_PUSCH_Config_t *pusch_Config = mac->ULbwp[0]->bwp_Dedicated->pusch_Config->choice.setup;

    // Basic sanity check for MCS value to check for a false or erroneous DCI
    if (dci->mcs > 28) {
      LOG_W(MAC, "MCS value %d out of bounds! Possibly due to false DCI. Ignoring DCI!\n", dci->mcs);
      return -1;
    }

    /* Transform precoding */
    if (rnti_type != NR_RNTI_CS || (rnti_type == NR_RNTI_CS && dci->ndi == 1)) {
      pusch_config_pdu->transform_precoding = get_transformPrecoding(scc, pusch_Config, NULL, dci_format, rnti_type, 0);
    }

    /*DCI format-related configuration*/
    if (*dci_format == NR_UL_DCI_FORMAT_0_0) {

      target_ss = NR_SearchSpace__searchSpaceType_PR_common;

    } else if (*dci_format == NR_UL_DCI_FORMAT_0_1) {

      /* BANDWIDTH_PART_IND */
      if (dci->bwp_indicator.val != 1) {
        LOG_W(MAC, "bwp_indicator != 1! Possibly due to false DCI. Ignoring DCI!\n");
        return -1;
      }
      config_bwp_ue(mac, &dci->bwp_indicator.val, dci_format);
      target_ss = NR_SearchSpace__searchSpaceType_PR_ue_Specific;
      ul_layers_config(mac, pusch_config_pdu, dci);
      ul_ports_config(mac, pusch_config_pdu, dci);

    } else {

      LOG_E(MAC, "In %s: UL grant from DCI format %d is not handled...\n", __FUNCTION__, *dci_format);
      return -1;

    }

    /* TRANSFORM PRECODING ------------------------------------------------------------------------------------------*/

    if (pusch_config_pdu->transform_precoding == transform_precoder_enabled) {

      pusch_config_pdu->num_dmrs_cdm_grps_no_data = 2;

      NR_DMRS_UplinkConfig_t *NR_DMRS_ulconfig = NULL;
      if(pusch_Config->dmrs_UplinkForPUSCH_MappingTypeA != NULL)
        NR_DMRS_ulconfig = pusch_Config->dmrs_UplinkForPUSCH_MappingTypeA->choice.setup;
      else
        NR_DMRS_ulconfig = pusch_Config->dmrs_UplinkForPUSCH_MappingTypeB->choice.setup;

      uint32_t n_RS_Id = 0;
      if (NR_DMRS_ulconfig->transformPrecodingEnabled->nPUSCH_Identity != NULL)
        n_RS_Id = *NR_DMRS_ulconfig->transformPrecodingEnabled->nPUSCH_Identity;
      else
        n_RS_Id = *mac->scc->physCellId;

      // U as specified in section 6.4.1.1.1.2 in 38.211, if sequence hopping and group hopping are disabled
      pusch_config_pdu->dfts_ofdm.low_papr_group_number = n_RS_Id % 30;

      // V as specified in section 6.4.1.1.1.2 in 38.211 V = 0 if sequence hopping and group hopping are disabled
      if ((NR_DMRS_ulconfig->transformPrecodingEnabled->sequenceGroupHopping == NULL) &&
            (NR_DMRS_ulconfig->transformPrecodingEnabled->sequenceHopping == NULL))
          pusch_config_pdu->dfts_ofdm.low_papr_sequence_number = 0;
      else
        AssertFatal(1==0,"SequenceGroupHopping or sequenceHopping are NOT Supported\n");

      LOG_D(MAC,"TRANSFORM PRECODING IS ENABLED. CDM groups: %d, U: %d \n", pusch_config_pdu->num_dmrs_cdm_grps_no_data,
                pusch_config_pdu->dfts_ofdm.low_papr_group_number);
    }

    /* TRANSFORM PRECODING --------------------------------------------------------------------------------------------------------*/

    /* IDENTIFIER_DCI_FORMATS */
    /* FREQ_DOM_RESOURCE_ASSIGNMENT_UL */
    if (nr_ue_process_dci_freq_dom_resource_assignment(pusch_config_pdu, NULL, n_RB_ULBWP, 0, dci->frequency_domain_assignment.val) < 0){
      return -1;
    }
    /* TIME_DOM_RESOURCE_ASSIGNMENT */
    if (nr_ue_process_dci_time_dom_resource_assignment(mac, pusch_config_pdu, NULL, dci->time_domain_assignment.val) < 0) {
      return -1;
    }

    /* FREQ_HOPPING_FLAG */
    if ((pusch_config_dedicated->resource_allocation != 0) && (pusch_config_dedicated->frequency_hopping !=0)){
      pusch_config_pdu->frequency_hopping = dci->frequency_hopping_flag.val;
    }

    /* MCS */
    pusch_config_pdu->mcs_index = dci->mcs;

    /* MCS TABLE */
    if (pusch_config_pdu->transform_precoding == transform_precoder_disabled) {
      pusch_config_pdu->mcs_table = get_pusch_mcs_table(pusch_Config->mcs_Table, 0, *dci_format, rnti_type, target_ss, false);
    } else {
      pusch_config_pdu->mcs_table = get_pusch_mcs_table(pusch_Config->mcs_TableTransformPrecoder, 1, *dci_format, rnti_type, target_ss, false);
    }

    /* NDI */
    pusch_config_pdu->pusch_data.new_data_indicator = dci->ndi;
    /* RV */
    pusch_config_pdu->pusch_data.rv_index = dci->rv;
    /* HARQ_PROCESS_NUMBER */
    pusch_config_pdu->pusch_data.harq_process_id = dci->harq_pid;
    /* TPC_PUSCH */
    // according to TS 38.213 Table Table 7.1.1-1
    if (dci->tpc == 0) {
      pusch_config_pdu->absolute_delta_PUSCH = -4;
    }
    if (dci->tpc == 1) {
      pusch_config_pdu->absolute_delta_PUSCH = -1;
    }
    if (dci->tpc == 2) {
      pusch_config_pdu->absolute_delta_PUSCH = 1;
    }
    if (dci->tpc == 3) {
      pusch_config_pdu->absolute_delta_PUSCH = 4;
    }

    /* DMRS */
    l_prime_mask = get_l_prime(pusch_config_pdu->nr_of_symbols, typeB, pusch_dmrs_pos0, pusch_len1);
    if (pusch_config_pdu->transform_precoding == transform_precoder_disabled)
      pusch_config_pdu->num_dmrs_cdm_grps_no_data = 1;

    // Num PRB Overhead from PUSCH-ServingCellConfig
    if (mac->scg->spCellConfig->spCellConfigDedicated->uplinkConfig->pusch_ServingCellConfig->choice.setup->xOverhead == NULL) {
      N_PRB_oh = 0;
    } else {
      N_PRB_oh = *mac->scg->spCellConfig->spCellConfigDedicated->uplinkConfig->pusch_ServingCellConfig->choice.setup->xOverhead;
    }

    /* PTRS */
    if (mac->ULbwp[0]->bwp_Dedicated->pusch_Config->choice.setup->dmrs_UplinkForPUSCH_MappingTypeB->choice.setup->phaseTrackingRS != NULL) {
      if (pusch_config_pdu->transform_precoding == transform_precoder_disabled) {
        nfapi_nr_ue_ptrs_ports_t ptrs_ports_list;
        pusch_config_pdu->pusch_ptrs.ptrs_ports_list = &ptrs_ports_list;
        valid_ptrs_setup = set_ul_ptrs_values(mac->ULbwp[0]->bwp_Dedicated->pusch_Config->choice.setup->dmrs_UplinkForPUSCH_MappingTypeB->choice.setup->phaseTrackingRS->choice.setup,
                                              pusch_config_pdu->rb_size, pusch_config_pdu->mcs_index, pusch_config_pdu->mcs_table,
                                              &pusch_config_pdu->pusch_ptrs.ptrs_freq_density,&pusch_config_pdu->pusch_ptrs.ptrs_time_density,
                                              &pusch_config_pdu->pusch_ptrs.ptrs_ports_list->ptrs_re_offset,&pusch_config_pdu->pusch_ptrs.num_ptrs_ports,
                                              &pusch_config_pdu->pusch_ptrs.ul_ptrs_power, pusch_config_pdu->nr_of_symbols);
        if(valid_ptrs_setup==true) {
          pusch_config_pdu->pdu_bit_map |= PUSCH_PDU_BITMAP_PUSCH_PTRS;
        }
        LOG_D(MAC, "UL PTRS values: PTRS time den: %d, PTRS freq den: %d\n", pusch_config_pdu->pusch_ptrs.ptrs_time_density, pusch_config_pdu->pusch_ptrs.ptrs_freq_density);
      }
    }

  }

  LOG_D(MAC, "In %s: received UL grant (rb_start %d, rb_size %d, start_symbol_index %d, nr_of_symbols %d) for RNTI type %s \n",
    __FUNCTION__,
    pusch_config_pdu->rb_start,
    pusch_config_pdu->rb_size,
    pusch_config_pdu->start_symbol_index,
    pusch_config_pdu->nr_of_symbols,
    rnti_types[rnti_type]);

  pusch_config_pdu->ul_dmrs_symb_pos = l_prime_mask << pusch_config_pdu->start_symbol_index;;
  pusch_config_pdu->target_code_rate = nr_get_code_rate_ul(pusch_config_pdu->mcs_index, pusch_config_pdu->mcs_table);
  pusch_config_pdu->qam_mod_order = nr_get_Qm_ul(pusch_config_pdu->mcs_index, pusch_config_pdu->mcs_table);

  if (pusch_config_pdu->target_code_rate == 0 || pusch_config_pdu->qam_mod_order == 0) {
    LOG_W(MAC, "In %s: Invalid code rate or Mod order, likely due to unexpected UL DCI. Ignoring DCI! \n", __FUNCTION__);
    return -1;
  }

  get_num_re_dmrs(pusch_config_pdu, &nb_dmrs_re_per_rb, &number_dmrs_symbols);

  // Compute TBS
  pusch_config_pdu->pusch_data.tb_size = nr_compute_tbs(pusch_config_pdu->qam_mod_order,
                                                        pusch_config_pdu->target_code_rate,
                                                        pusch_config_pdu->rb_size,
                                                        pusch_config_pdu->nr_of_symbols,
                                                        nb_dmrs_re_per_rb*number_dmrs_symbols,
                                                        N_PRB_oh,
                                                        0, // TBR to verify tb scaling
                                                        pusch_config_pdu->nrOfLayers)/8;

  return 0;

}

// Performs :
// 1. TODO: Call RRC for link status return to PHY
// 2. TODO: Perform SR/BSR procedures for scheduling feedback
// 3. TODO: Perform PHR procedures
NR_UE_L2_STATE_t nr_ue_scheduler(nr_downlink_indication_t *dl_info, nr_uplink_indication_t *ul_info){

  uint32_t search_space_mask = 0;

  if (dl_info){

    module_id_t mod_id    = dl_info->module_id;
    uint32_t gNB_index    = dl_info->gNB_index;
    int cc_id             = dl_info->cc_id;
    frame_t rx_frame      = dl_info->frame;
    slot_t rx_slot        = dl_info->slot;
    NR_UE_MAC_INST_t *mac = get_mac_inst(mod_id);

    fapi_nr_dl_config_request_t *dl_config = &mac->dl_config_request;
    nr_scheduled_response_t scheduled_response;
    nr_dcireq_t dcireq;

    // check type0 from 38.213 13 if we have no CellGroupConfig
    // TODO: implementation to be completed
    if (mac->scg == NULL) {

      if(dl_info->ssb_index != -1){

        if(mac->type0_pdcch_ss_mux_pattern == 1){
          //  38.213 chapter 13
          if((mac->type0_pdcch_ss_sfn_c == SFN_C_MOD_2_EQ_0) && !(rx_frame & 0x1) && (rx_slot == mac->type0_pdcch_ss_n_c)){
            search_space_mask = search_space_mask | type0_pdcch;
            mac->type0_pdcch_consecutive_slots = mac->type0_pdcch_dci_config.coreset.duration;
          }
          if((mac->type0_pdcch_ss_sfn_c == SFN_C_MOD_2_EQ_1) && (rx_frame & 0x1) && (rx_slot == mac->type0_pdcch_ss_n_c)){
            search_space_mask = search_space_mask | type0_pdcch;
            mac->type0_pdcch_consecutive_slots = mac->type0_pdcch_dci_config.coreset.duration;
          }
        }
        if(mac->type0_pdcch_ss_mux_pattern == 2){
          //  38.213 Table 13-13, 13-14
          if((rx_frame == get_ssb_frame(rx_frame)) && (rx_slot == mac->type0_pdcch_ss_n_c)){
            search_space_mask = search_space_mask | type0_pdcch;
            mac->type0_pdcch_consecutive_slots = mac->type0_pdcch_dci_config.coreset.duration;
          }
        }
        if(mac->type0_pdcch_ss_mux_pattern == 3){
          //  38.213 Table 13-15
          if((rx_frame == get_ssb_frame(rx_frame)) && (rx_slot == mac->type0_pdcch_ss_n_c)){
            search_space_mask = search_space_mask | type0_pdcch;
            mac->type0_pdcch_consecutive_slots = mac->type0_pdcch_dci_config.coreset.duration;
          }
        }
      } // ssb_index != -1

      // Type0 PDCCH search space
      if((search_space_mask & type0_pdcch) || ( mac->type0_pdcch_consecutive_slots != 0 )){
        mac->type0_pdcch_consecutive_slots = mac->type0_pdcch_consecutive_slots - 1;

        dl_config->dl_config_list[dl_config->number_pdus].dci_config_pdu.dci_config_rel15 = mac->type0_pdcch_dci_config;
        dl_config->dl_config_list[dl_config->number_pdus].pdu_type = FAPI_NR_DL_CONFIG_TYPE_DCI;

        /*
        dl_config->dl_config_list[dl_config->number_pdus].dci_config_pdu.dci_config_rel15.rnti = 0xaaaa;        //      to be set
        dl_config->dl_config_list[dl_config->number_pdus].dci_config_pdu.dci_config_rel15.N_RB_BWP = 106;       //      to be set

        LOG_I(MAC,"nr_ue_scheduler Type0 PDCCH with rnti %x, BWP %d\n",
        dl_config->dl_config_list[dl_config->number_pdus].dci_config_pdu.dci_config_rel15.rnti,
        dl_config->dl_config_list[dl_config->number_pdus].dci_config_pdu.dci_config_rel15.N_RB_BWP);
        */
        dl_config->number_pdus = dl_config->number_pdus + 1;

        fill_scheduled_response(&scheduled_response, dl_config, NULL, NULL, mod_id, cc_id, rx_frame, rx_slot, dl_info->thread_id);
        if(mac->if_module != NULL && mac->if_module->scheduled_response != NULL)
          mac->if_module->scheduled_response(&scheduled_response);
      }
    } else { // we have an scg

      dcireq.module_id = mod_id;
      dcireq.gNB_index = gNB_index;
      dcireq.cc_id     = cc_id;
      dcireq.frame     = rx_frame;
      dcireq.slot      = rx_slot;
      dcireq.dl_config_req.number_pdus = 0;
      nr_ue_dcireq(&dcireq); //to be replaced with function pointer later

      fill_scheduled_response(&scheduled_response, &dcireq.dl_config_req, NULL, NULL, mod_id, cc_id, rx_frame, rx_slot, dl_info->thread_id);
      if(mac->if_module != NULL && mac->if_module->scheduled_response != NULL){
        mac->if_module->scheduled_response(&scheduled_response);
      }

      /*
        if(search_space_mask & type0a_pdcch){
        }
        
        if(search_space_mask & type1_pdcch){
        }

        if(search_space_mask & type2_pdcch){
        }

        if(search_space_mask & type3_pdcch){
        }
      */
    }
  } else if (ul_info) {

    int cc_id             = ul_info->cc_id;
    frame_t rx_frame      = ul_info->frame_rx;
    slot_t rx_slot        = ul_info->slot_rx;
    frame_t frame_tx      = ul_info->frame_tx;
    slot_t slot_tx        = ul_info->slot_tx;
    module_id_t mod_id    = ul_info->module_id;
    uint8_t access_mode   = SCHEDULED_ACCESS;

    NR_UE_MAC_INST_t *mac = get_mac_inst(mod_id);
    RA_config_t *ra       = &mac->ra;

    fapi_nr_ul_config_request_t *ul_config = get_ul_config_request(mac, slot_tx);

    // Schedule ULSCH only if the current frame and slot match those in ul_config_req
    // AND if a UL grant (UL DCI or Msg3) has been received (as indicated by num_pdus)
    if ((ul_info->slot_tx == ul_config->slot && ul_info->frame_tx == ul_config->sfn) && ul_config->number_pdus > 0){

      LOG_D(MAC, "In %s:[%d.%d]: number of UL PDUs: %d with UL transmission in [%d.%d]\n", __FUNCTION__, frame_tx, slot_tx, ul_config->number_pdus, ul_config->sfn, ul_config->slot);

      uint8_t ulsch_input_buffer[MAX_ULSCH_PAYLOAD_BYTES];
      uint8_t data_existing = 0;
      nr_scheduled_response_t scheduled_response;
      fapi_nr_tx_request_t tx_req;

      for (int j = 0; j < ul_config->number_pdus; j++) {

        fapi_nr_ul_config_request_pdu_t *ulcfg_pdu = &ul_config->ul_config_list[j];

        if (ulcfg_pdu->pdu_type == FAPI_NR_UL_CONFIG_TYPE_PUSCH) {

          uint16_t TBS_bytes = ulcfg_pdu->pusch_config_pdu.pusch_data.tb_size;

          if (ra->ra_state == WAIT_RAR){
            memcpy(ulsch_input_buffer, mac->ulsch_pdu.payload, TBS_bytes);
            LOG_D(NR_MAC,"[RAPROC] Msg3 to be transmitted:\n");
            for (int k = 0; k < TBS_bytes; k++) {
              LOG_D(NR_MAC,"(%i): 0x%x\n",k,mac->ulsch_pdu.payload[k]);
            }
          } else {
            if (IS_SOFTMODEM_NOS1 && (mac->UL_ndi[ulcfg_pdu->pusch_config_pdu.pusch_data.harq_process_id] != ulcfg_pdu->pusch_config_pdu.pusch_data.new_data_indicator)){
              // Getting IP traffic to be transmitted
              data_existing = nr_ue_get_sdu(mod_id,
                                            cc_id,
                                            frame_tx,
                                            slot_tx,
                                            0,
                                            ulsch_input_buffer,
                                            TBS_bytes,
                                            &access_mode);
            }

            mac->UL_ndi[ulcfg_pdu->pusch_config_pdu.pusch_data.harq_process_id] = ulcfg_pdu->pusch_config_pdu.pusch_data.new_data_indicator;
            //Random traffic to be transmitted if there is no IP traffic available for this Tx opportunity
            if (!IS_SOFTMODEM_NOS1 || !data_existing) {
              //Use zeros for the header bytes in noS1 mode, in order to make sure that the LCID is not valid
              //and block this traffic from being forwarded to the upper layers at the gNB
              LOG_D(PHY, "In %s: Random data to be transmitted: TBS_bytes %d \n", __FUNCTION__, TBS_bytes);

              //Give the first byte a dummy value (a value not corresponding to any valid LCID based on 38.321, Table 6.2.1-2)
              //in order to distinguish the PHY random packets at the MAC layer of the gNB receiver from the normal packets that should
              //have a valid LCID (nr_process_mac_pdu function)
              ulsch_input_buffer[0] = 0x31;

              for (int i = 1; i < TBS_bytes; i++) {
                ulsch_input_buffer[i] = (unsigned char) rand();
              }
            }
          }

          #ifdef DEBUG_MAC_PDU
          LOG_D(PHY, "Is data existing ?: %d \n", data_existing);
          LOG_I(PHY, "Printing MAC PDU to be encoded, TBS is: %d \n", TBS_bytes);
          for (i = 0; i < TBS_bytes; i++) {
            printf("%02x", ulsch_input_buffer[i]);
          }
          printf("\n");
          #endif

          // Config UL TX PDU
          tx_req.slot = slot_tx;
          tx_req.sfn = frame_tx;
          tx_req.number_of_pdus++;
          tx_req.tx_request_body[0].pdu_length = TBS_bytes;
          tx_req.tx_request_body[0].pdu_index = j;
          tx_req.tx_request_body[0].pdu = ulsch_input_buffer;

          if (ra->ra_state != RA_SUCCEEDED && !ra->cfra){
            nr_Msg3_transmitted(ul_info->module_id, ul_info->cc_id, ul_info->frame_tx, ul_info->gNB_index);
          }

        }
      }

      fill_scheduled_response(&scheduled_response, NULL, ul_config, &tx_req, mod_id, cc_id, rx_frame, rx_slot, ul_info->thread_id);
      if(mac->if_module != NULL && mac->if_module->scheduled_response != NULL){
        mac->if_module->scheduled_response(&scheduled_response);
      }
    }
  }

  return UE_CONNECTION_OK;

}

// PUSCH scheduler:
// - Calculate the slot in which ULSCH should be scheduled. This is current slot + K2,
// - where K2 is the offset between the slot in which UL DCI is received and the slot
// - in which ULSCH should be scheduled. K2 is configured in RRC configuration.  
// PUSCH Msg3 scheduler:
// - scheduled by RAR UL grant according to 8.3 of TS 38.213
// Note: Msg3 tx in the uplink symbols of mixed slot
int nr_ue_pusch_scheduler(NR_UE_MAC_INST_t *mac,
                          uint8_t is_Msg3,
                          frame_t current_frame,
                          int current_slot,
                          frame_t *frame_tx,
                          int *slot_tx,
                          uint8_t tda_id){

  int delta = 0;
  NR_BWP_Uplink_t *ubwp = mac->ULbwp[0];
  // Get the numerology to calculate the Tx frame and slot
  int mu = ubwp->bwp_Common->genericParameters.subcarrierSpacing;
  struct NR_PUSCH_TimeDomainResourceAllocationList *pusch_TimeDomainAllocationList = ubwp->bwp_Common->pusch_ConfigCommon->choice.setup->pusch_TimeDomainAllocationList;
  // k2 as per 3GPP TS 38.214 version 15.9.0 Release 15 ch 6.1.2.1.1
  // PUSCH time domain resource allocation is higher layer configured from uschTimeDomainAllocationList in either pusch-ConfigCommon
  uint8_t k2;

  if (is_Msg3) {
    k2 = *pusch_TimeDomainAllocationList->list.array[tda_id]->k2;

    switch (mu) {
      case 0:
        delta = 2;
        break;
      case 1:
        delta = 3;
        break;
      case 2:
        delta = 4;
        break;
      case 3:
        delta = 6;
        break;
    }

    AssertFatal((k2+delta) >= DURATION_RX_TO_TX,
                "Slot offset (%d) for Msg3 cannot be less than DURATION_RX_TO_TX (%d)\n",
                k2+delta,DURATION_RX_TO_TX);

    *slot_tx = (current_slot + k2 + delta) % nr_slots_per_frame[mu];
    if (current_slot + k2 + delta > nr_slots_per_frame[mu]){
      *frame_tx = (current_frame + 1) % 1024;
    } else {
      *frame_tx = current_frame;
    }

  } else {

    // Get slot offset K2 which will be used to calculate TX slot
    k2 = get_k2(mac, tda_id);
    if (k2 < 0) { // This can happen when a false DCI is received
      return -1;
    }

    // Calculate TX slot and frame
    *slot_tx = (current_slot + k2) % nr_slots_per_frame[mu];
    *frame_tx = ((current_slot + k2) > nr_slots_per_frame[mu]) ? (current_frame + 1) % 1024 : current_frame;

  }

  LOG_D(MAC, "In %s: currently at [%d.%d] UL transmission in [%d.%d] (k2 %d delta %d)\n", __FUNCTION__, current_frame, current_slot, *frame_tx, *slot_tx, k2, delta);

  return 0;

}

// Build the list of all the valid RACH occasions in the maximum association pattern period according to the PRACH config
static void build_ro_list(NR_ServingCellConfigCommon_t *scc, uint8_t unpaired) {

  int x,y; // PRACH Configuration Index table variables used to compute the valid frame numbers
  int y2;  // PRACH Configuration Index table additional variable used to compute the valid frame numbers
  uint8_t slot_shift_for_map;
  uint8_t map_shift;
  boolean_t even_slot_invalid;
  int64_t s_map;
  uint8_t prach_conf_start_symbol; // Starting symbol of the PRACH occasions in the PRACH slot
  uint8_t N_t_slot; // Number of PRACH occasions in a 14-symbols PRACH slot
  uint8_t N_dur; // Duration of a PRACH occasion (nb of symbols)
  uint8_t frame; // Maximum is NB_FRAMES_IN_MAX_ASSOCIATION_PATTERN_PERIOD
  uint8_t slot; // Maximum is the number of slots in a frame @ SCS 240kHz
  uint16_t format = 0xffff;
  uint8_t format2 = 0xff;
  int nb_fdm;

  uint8_t config_index, mu;
  uint32_t pointa;
  int msg1_FDM;

  uint8_t prach_conf_period_idx;
  uint8_t nb_of_frames_per_prach_conf_period;
  uint8_t prach_conf_period_frame_idx;
  int64_t *prach_config_info_p;

  NR_RACH_ConfigCommon_t *setup = scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup;
  NR_FrequencyInfoDL_t *frequencyInfoDL = scc->downlinkConfigCommon->frequencyInfoDL;
  NR_RACH_ConfigGeneric_t *rach_ConfigGeneric = &setup->rach_ConfigGeneric;

  config_index = rach_ConfigGeneric->prach_ConfigurationIndex;

  if (setup->msg1_SubcarrierSpacing)
    mu = *setup->msg1_SubcarrierSpacing;
  else
    mu = frequencyInfoDL->scs_SpecificCarrierList.list.array[0]->subcarrierSpacing;

  pointa = frequencyInfoDL->absoluteFrequencyPointA;
  msg1_FDM = rach_ConfigGeneric->msg1_FDM;

  switch (msg1_FDM){
    case 0:
    case 1:
    case 2:
    case 3:
      nb_fdm = 1 << msg1_FDM;
      break;
    default:
      AssertFatal(1 == 0, "Unknown msg1_FDM from rach_ConfigGeneric %d\n", msg1_FDM);
  }

  // Create the PRACH occasions map
  // ==============================
  // WIP: For now assume no rejected PRACH occasions because of conflict with SSB or TDD_UL_DL_ConfigurationCommon schedule

  // Identify the proper PRACH Configuration Index table according to the operating frequency
  LOG_D(MAC,"Pointa %u, mu = %u, PRACH config index  = %u, unpaired = %u\n", pointa, mu, config_index, unpaired);

  prach_config_info_p = get_prach_config_info(pointa, config_index, unpaired);

  if (pointa > 2016666) { //FR2

    x = prach_config_info_p[2];
    y = prach_config_info_p[3];
    y2 = prach_config_info_p[4];

    s_map = prach_config_info_p[5];

    prach_conf_start_symbol = prach_config_info_p[6];
    N_t_slot = prach_config_info_p[8];
    N_dur = prach_config_info_p[9];
    if (prach_config_info_p[1] != -1)
      format2 = (uint8_t) prach_config_info_p[1];
    format = ((uint8_t) prach_config_info_p[0]) | (format2<<8);

    slot_shift_for_map = mu-2;
    if ( (mu == 3) && (prach_config_info_p[7] == 1) )
      even_slot_invalid = true;
    else
      even_slot_invalid = false;
  }
  else { // FR1
    x = prach_config_info_p[2];
    y = prach_config_info_p[3];
    y2 = y;

    s_map = prach_config_info_p[4];

    prach_conf_start_symbol = prach_config_info_p[5];
    N_t_slot = prach_config_info_p[7];
    N_dur = prach_config_info_p[8];
    if (prach_config_info_p[1] != -1)
      format2 = (uint8_t) prach_config_info_p[1];
    format = ((uint8_t) prach_config_info_p[0]) | (format2<<8);

    slot_shift_for_map = mu;
    if ( (mu == 1) && (prach_config_info_p[6] <= 1) )
      // no prach in even slots @ 30kHz for 1 prach per subframe
      even_slot_invalid = true;
    else
      even_slot_invalid = false;
  } // FR2 / FR1

  prach_assoc_pattern.nb_of_prach_conf_period_in_max_period = MAX_NB_PRACH_CONF_PERIOD_IN_ASSOCIATION_PATTERN_PERIOD / x;
  nb_of_frames_per_prach_conf_period = x;

  LOG_D(MAC,"nb_of_prach_conf_period_in_max_period %d\n", prach_assoc_pattern.nb_of_prach_conf_period_in_max_period);

  // Fill in the PRACH occasions table for every slot in every frame in every PRACH configuration periods in the maximum association pattern period
  // ----------------------------------------------------------------------------------------------------------------------------------------------
  // ----------------------------------------------------------------------------------------------------------------------------------------------
  // For every PRACH configuration periods
  // -------------------------------------
  for (prach_conf_period_idx=0; prach_conf_period_idx<prach_assoc_pattern.nb_of_prach_conf_period_in_max_period; prach_conf_period_idx++) {
    prach_assoc_pattern.prach_conf_period_list[prach_conf_period_idx].nb_of_prach_occasion = 0;
    prach_assoc_pattern.prach_conf_period_list[prach_conf_period_idx].nb_of_frame = nb_of_frames_per_prach_conf_period;
    prach_assoc_pattern.prach_conf_period_list[prach_conf_period_idx].nb_of_slot = nr_slots_per_frame[mu];

    LOG_D(MAC,"PRACH Conf Period Idx %d\n", prach_conf_period_idx);

    // For every frames in a PRACH configuration period
    // ------------------------------------------------
    for (prach_conf_period_frame_idx=0; prach_conf_period_frame_idx<nb_of_frames_per_prach_conf_period; prach_conf_period_frame_idx++) {
      frame = (prach_conf_period_idx * nb_of_frames_per_prach_conf_period) + prach_conf_period_frame_idx;

      LOG_D(MAC,"PRACH Conf Period Frame Idx %d - Frame %d\n", prach_conf_period_frame_idx, frame);
      // Is it a valid frame for this PRACH configuration index? (n_sfn mod x = y)
      if ( (frame%x)==y || (frame%x)==y2 ) {

        // For every slot in a frame
        // -------------------------
        for (slot=0; slot<nr_slots_per_frame[mu]; slot++) {
          // Is it a valid slot?
          map_shift = slot >> slot_shift_for_map; // in PRACH configuration index table slots are numbered wrt 60kHz
          if ( (s_map>>map_shift)&0x01 ) {
            // Valid slot

            // Additionally, for 30kHz/120kHz, we must check for the n_RA_Slot param also
            if ( even_slot_invalid && (slot%2 == 0) )
                continue; // no prach in even slots @ 30kHz/120kHz for 1 prach per 60khz slot/subframe

            // We're good: valid frame and valid slot
            // Compute all the PRACH occasions in the slot

            uint8_t n_prach_occ_in_time;
            uint8_t n_prach_occ_in_freq;

            prach_assoc_pattern.prach_conf_period_list[prach_conf_period_idx].prach_occasion_slot_map[prach_conf_period_frame_idx][slot].nb_of_prach_occasion_in_time = N_t_slot;
            prach_assoc_pattern.prach_conf_period_list[prach_conf_period_idx].prach_occasion_slot_map[prach_conf_period_frame_idx][slot].nb_of_prach_occasion_in_freq = nb_fdm;

            for (n_prach_occ_in_time=0; n_prach_occ_in_time<N_t_slot; n_prach_occ_in_time++) {
              uint8_t start_symbol = prach_conf_start_symbol + n_prach_occ_in_time * N_dur;
              LOG_D(MAC,"PRACH Occ in time %d\n", n_prach_occ_in_time);

              for (n_prach_occ_in_freq=0; n_prach_occ_in_freq<nb_fdm; n_prach_occ_in_freq++) {
                prach_occasion_info_t *prach_occasion_p = &prach_assoc_pattern.prach_conf_period_list[prach_conf_period_idx].prach_occasion_slot_map[prach_conf_period_frame_idx][slot].prach_occasion[n_prach_occ_in_time][n_prach_occ_in_freq];

                prach_occasion_p->start_symbol = start_symbol;
                prach_occasion_p->fdm = n_prach_occ_in_freq;
                prach_occasion_p->frame = frame;
                prach_occasion_p->slot = slot;
                prach_occasion_p->format = format;
                prach_assoc_pattern.prach_conf_period_list[prach_conf_period_idx].nb_of_prach_occasion++;

                LOG_D(MAC,"Adding a PRACH occasion: frame %u, slot-symbol %d-%d, occ_in_time-occ_in-freq %d-%d, nb ROs in conf period %d, for this slot: RO# in time %d, RO# in freq %d\n",
                    frame, slot, start_symbol, n_prach_occ_in_time, n_prach_occ_in_freq, prach_assoc_pattern.prach_conf_period_list[prach_conf_period_idx].nb_of_prach_occasion,
                    prach_assoc_pattern.prach_conf_period_list[prach_conf_period_idx].prach_occasion_slot_map[prach_conf_period_frame_idx][slot].nb_of_prach_occasion_in_time,
                    prach_assoc_pattern.prach_conf_period_list[prach_conf_period_idx].prach_occasion_slot_map[prach_conf_period_frame_idx][slot].nb_of_prach_occasion_in_freq);
              } // For every freq in the slot
            } // For every time occasions in the slot
          } // Valid slot?
        } // For every slots in a frame
      } // Valid frame?
    } // For every frames in a prach configuration period
  } // For every prach configuration periods in the maximum association pattern period (160ms)
}

// Build the list of all the valid/transmitted SSBs according to the config
static void build_ssb_list(NR_ServingCellConfigCommon_t *scc) {

  // Create the list of transmitted SSBs
  // ===================================
  BIT_STRING_t *ssb_bitmap;
  uint64_t ssb_positionsInBurst;
  uint8_t ssb_idx = 0;

  switch (scc->ssb_PositionsInBurst->present) {
    case NR_ServingCellConfigCommon__ssb_PositionsInBurst_PR_shortBitmap:
      ssb_bitmap = &scc->ssb_PositionsInBurst->choice.shortBitmap;

      ssb_positionsInBurst = BIT_STRING_to_uint8(ssb_bitmap);
      LOG_D(MAC,"SSB config: SSB_positions_in_burst 0x%lx\n", ssb_positionsInBurst);

      for (uint8_t bit_nb=3; bit_nb<=3; bit_nb--) {
        // If SSB is transmitted
        if ((ssb_positionsInBurst>>bit_nb) & 0x01) {
          ssb_list.nb_tx_ssb++;
          ssb_list.tx_ssb[ssb_idx].transmitted = true;
          LOG_D(MAC,"SSB idx %d transmitted\n", ssb_idx);
        }
        ssb_idx++;
      }
      break;
    case NR_ServingCellConfigCommon__ssb_PositionsInBurst_PR_mediumBitmap:
      ssb_bitmap = &scc->ssb_PositionsInBurst->choice.mediumBitmap;

      ssb_positionsInBurst = BIT_STRING_to_uint8(ssb_bitmap);
      LOG_D(MAC,"SSB config: SSB_positions_in_burst 0x%lx\n", ssb_positionsInBurst);

      for (uint8_t bit_nb=7; bit_nb<=7; bit_nb--) {
        // If SSB is transmitted
        if ((ssb_positionsInBurst>>bit_nb) & 0x01) {
          ssb_list.nb_tx_ssb++;
          ssb_list.tx_ssb[ssb_idx].transmitted = true;
          LOG_D(MAC,"SSB idx %d transmitted\n", ssb_idx);
        }
        ssb_idx++;
      }
      break;
    case NR_ServingCellConfigCommon__ssb_PositionsInBurst_PR_longBitmap:
      ssb_bitmap = &scc->ssb_PositionsInBurst->choice.longBitmap;

      ssb_positionsInBurst = BIT_STRING_to_uint64(ssb_bitmap);
      LOG_D(MAC,"SSB config: SSB_positions_in_burst 0x%lx\n", ssb_positionsInBurst);

      for (uint8_t bit_nb=63; bit_nb<=63; bit_nb--) {
        // If SSB is transmitted
        if ((ssb_positionsInBurst>>bit_nb) & 0x01) {
          ssb_list.nb_tx_ssb++;
          ssb_list.tx_ssb[ssb_idx].transmitted = true;
          LOG_D(MAC,"SSB idx %d transmitted\n", ssb_idx);
        }
        ssb_idx++;
      }
      break;
    default:
      AssertFatal(false,"ssb_PositionsInBurst not present\n");
      break;
  }
}

// Map the transmitted SSBs to the ROs and create the association pattern according to the config
static void map_ssb_to_ro(NR_ServingCellConfigCommon_t *scc) {

  // Map SSBs to PRACH occasions
  // ===========================
  // WIP: Assumption: No PRACH occasion is rejected because of a conflict with SSBs or TDD_UL_DL_ConfigurationCommon schedule
  NR_RACH_ConfigCommon_t *setup = scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup;
  NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR ssb_perRACH_config = setup->ssb_perRACH_OccasionAndCB_PreamblesPerSSB->present;

  boolean_t multiple_ssb_per_ro; // true if more than one or exactly one SSB per RACH occasion, false if more than one RO per SSB
  uint8_t ssb_rach_ratio; // Nb of SSBs per RACH or RACHs per SSB
  uint16_t required_nb_of_prach_occasion; // Nb of RACH occasions required to map all the SSBs
  uint8_t required_nb_of_prach_conf_period; // Nb of PRACH configuration periods required to map all the SSBs

  // Determine the SSB to RACH mapping ratio
  // =======================================
  switch (ssb_perRACH_config){
    case NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR_oneEighth:
      multiple_ssb_per_ro = false;
      ssb_rach_ratio = 8;
      break;
    case NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR_oneFourth:
      multiple_ssb_per_ro = false;
      ssb_rach_ratio = 4;
      break;
    case NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR_oneHalf:
      multiple_ssb_per_ro = false;
      ssb_rach_ratio = 2;
      break;
    case NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR_one:
      multiple_ssb_per_ro = true;
      ssb_rach_ratio = 1;
      break;
    case NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR_two:
      multiple_ssb_per_ro = true;
      ssb_rach_ratio = 2;
      break;
    case NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR_four:
      multiple_ssb_per_ro = true;
      ssb_rach_ratio = 4;
      break;
    case NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR_eight:
      multiple_ssb_per_ro = true;
      ssb_rach_ratio = 8;
      break;
    case NR_RACH_ConfigCommon__ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR_sixteen:
      multiple_ssb_per_ro = true;
      ssb_rach_ratio = 16;
      break;
    default:
      AssertFatal(1 == 0, "Unsupported ssb_perRACH_config %d\n", ssb_perRACH_config);
      break;
  }
  LOG_D(MAC,"SSB rach ratio %d, Multiple SSB per RO %d\n", ssb_rach_ratio, multiple_ssb_per_ro);

  // Evaluate the number of PRACH configuration periods required to map all the SSBs and set the association period
  // ==============================================================================================================
  // WIP: Assumption for now is that all the PRACH configuration periods within a maximum association pattern period have the same number of PRACH occasions
  //      (No PRACH occasions are conflicting with SSBs nor TDD_UL_DL_ConfigurationCommon schedule)
  //      There is only one possible association period which can contain up to 16 PRACH configuration periods
  LOG_D(MAC,"Evaluate the number of PRACH configuration periods required to map all the SSBs and set the association period\n");
  if (true == multiple_ssb_per_ro) {
    required_nb_of_prach_occasion = ((ssb_list.nb_tx_ssb-1) + ssb_rach_ratio) / ssb_rach_ratio;
  }
  else {
    required_nb_of_prach_occasion = ssb_list.nb_tx_ssb * ssb_rach_ratio;
  }

  required_nb_of_prach_conf_period = ((required_nb_of_prach_occasion-1) + prach_assoc_pattern.prach_conf_period_list[0].nb_of_prach_occasion) / prach_assoc_pattern.prach_conf_period_list[0].nb_of_prach_occasion;

  if (required_nb_of_prach_conf_period == 1) {
    prach_assoc_pattern.prach_association_period_list[0].nb_of_prach_conf_period = 1;
  }
  else if (required_nb_of_prach_conf_period == 2) {
    prach_assoc_pattern.prach_association_period_list[0].nb_of_prach_conf_period = 2;
  }
  else if (required_nb_of_prach_conf_period <= 4) {
    prach_assoc_pattern.prach_association_period_list[0].nb_of_prach_conf_period = 4;
  }
  else if (required_nb_of_prach_conf_period <= 8) {
    prach_assoc_pattern.prach_association_period_list[0].nb_of_prach_conf_period = 8;
  }
  else if (required_nb_of_prach_conf_period <= 16) {
    prach_assoc_pattern.prach_association_period_list[0].nb_of_prach_conf_period = 16;
  }
  else {
    AssertFatal(1 == 0, "Invalid number of PRACH config periods within an association period %d\n", required_nb_of_prach_conf_period);
  }

  prach_assoc_pattern.nb_of_assoc_period = 1; // WIP: only one possible association period
  prach_assoc_pattern.prach_association_period_list[0].nb_of_frame = prach_assoc_pattern.prach_association_period_list[0].nb_of_prach_conf_period * prach_assoc_pattern.prach_conf_period_list[0].nb_of_frame;
  prach_assoc_pattern.nb_of_frame = prach_assoc_pattern.prach_association_period_list[0].nb_of_frame;

  LOG_D(MAC,"Assoc period %d, Nb of frames in assoc period %d\n",
        prach_assoc_pattern.prach_association_period_list[0].nb_of_prach_conf_period,
        prach_assoc_pattern.prach_association_period_list[0].nb_of_frame);

  // Proceed to the SSB to RO mapping
  // ================================
  uint8_t association_period_idx; // Association period index within the association pattern
  uint8_t ssb_idx = 0;
  uint8_t prach_configuration_period_idx; // PRACH Configuration period index within the association pattern
  prach_conf_period_t *prach_conf_period_p;

  // Map all the association periods within the association pattern period
  LOG_D(MAC,"Proceed to the SSB to RO mapping\n");
  for (association_period_idx=0; association_period_idx<prach_assoc_pattern.nb_of_assoc_period; association_period_idx++) {
    uint8_t n_prach_conf=0; // PRACH Configuration period index within the association period
    uint8_t frame=0;
    uint8_t slot=0;
    uint8_t ro_in_time=0;
    uint8_t ro_in_freq=0;

    // Set the starting PRACH Configuration period index in the association_pattern map for this particular association period
    prach_configuration_period_idx = 0;  // WIP: only one possible association period so the starting PRACH configuration period is automatically 0

    // Check if we need to map multiple SSBs per RO or multiple ROs per SSB
    if (true == multiple_ssb_per_ro) {
      // --------------------
      // --------------------
      // Multiple SSBs per RO
      // --------------------
      // --------------------

      // WIP: For the moment, only map each SSB idx once per association period if configuration is multiple SSBs per RO
      //      this is true if no PRACH occasions are conflicting with SSBs nor TDD_UL_DL_ConfigurationCommon schedule
      ssb_idx = 0;

      // Go through the list of PRACH config periods within this association period
      for (n_prach_conf=0; n_prach_conf<prach_assoc_pattern.prach_association_period_list[association_period_idx].nb_of_prach_conf_period; n_prach_conf++, prach_configuration_period_idx++) {
        // Build the association period with its association PRACH Configuration indexes
        prach_conf_period_p = &prach_assoc_pattern.prach_conf_period_list[prach_configuration_period_idx];
        prach_assoc_pattern.prach_association_period_list[association_period_idx].prach_conf_period_list[n_prach_conf] = prach_conf_period_p;

        // Go through all the ROs within the PRACH config period
        for (frame=0; frame<prach_conf_period_p->nb_of_frame; frame++) {
          for (slot=0; slot<prach_conf_period_p->nb_of_slot; slot++) {
            for (ro_in_time=0; ro_in_time<prach_conf_period_p->prach_occasion_slot_map[frame][slot].nb_of_prach_occasion_in_time; ro_in_time++) {
              for (ro_in_freq=0; ro_in_freq<prach_conf_period_p->prach_occasion_slot_map[frame][slot].nb_of_prach_occasion_in_freq; ro_in_freq++) {
                prach_occasion_info_t *ro_p = &prach_conf_period_p->prach_occasion_slot_map[frame][slot].prach_occasion[ro_in_time][ro_in_freq];

                // Go through the list of transmitted SSBs and map the required amount of SSBs to this RO
                // WIP: For the moment, only map each SSB idx once per association period if configuration is multiple SSBs per RO
                //      this is true if no PRACH occasions are conflicting with SSBs nor TDD_UL_DL_ConfigurationCommon schedule
                for (; ssb_idx<MAX_NB_SSB; ssb_idx++) {
                  // Map only the transmitted ssb_idx
                  if (true == ssb_list.tx_ssb[ssb_idx].transmitted) {
                    ro_p->mapped_ssb_idx[ro_p->nb_mapped_ssb] = ssb_idx;
                    ro_p->nb_mapped_ssb++;
                    ssb_list.tx_ssb[ssb_idx].mapped_ro[ssb_list.tx_ssb[ssb_idx].nb_mapped_ro] = ro_p;
                    ssb_list.tx_ssb[ssb_idx].nb_mapped_ro++;
                    AssertFatal(MAX_NB_RO_PER_SSB_IN_ASSOCIATION_PATTERN > ssb_list.tx_ssb[ssb_idx].nb_mapped_ro,"Too many mapped ROs (%d) to a single SSB\n", ssb_list.tx_ssb[ssb_idx].nb_mapped_ro);

                    LOG_D(MAC,"Mapped ssb_idx %u to RO slot-symbol %u-%u, %u-%u-%u/%u\n", ssb_idx, ro_p->slot, ro_p->start_symbol, slot, ro_in_time, ro_in_freq, prach_conf_period_p->prach_occasion_slot_map[frame][slot].nb_of_prach_occasion_in_freq);
                    LOG_D(MAC,"Nb mapped ROs for this ssb idx: in the association period only %u\n", ssb_list.tx_ssb[ssb_idx].nb_mapped_ro);

                    // If all the required SSBs are mapped to this RO, exit the loop of SSBs
                    if (ro_p->nb_mapped_ssb == ssb_rach_ratio) {
                      ssb_idx++;
                      break;
                    }
                  } // if ssb_idx is transmitted
                } // for ssb_idx

                // Exit the loop of ROs if there is no more SSB to map
                if (MAX_NB_SSB == ssb_idx) break;
              } // for ro_in_freq

              // Exit the loop of ROs if there is no more SSB to map
              if (MAX_NB_SSB == ssb_idx) break;
            } // for ro_in_time

            // Exit the loop of slots if there is no more SSB to map
            if (MAX_NB_SSB == ssb_idx) break;
          } // for slot

          // Exit the loop frames if there is no more SSB to map
          if (MAX_NB_SSB == ssb_idx) break;
        } // for frame

        // Exit the loop of PRACH configurations if there is no more SSB to map
        if (MAX_NB_SSB == ssb_idx) break;
      } // for n_prach_conf

      // WIP: note that there is no re-mapping of the SSBs within the association period since there is no invalid ROs in the PRACH config periods that would create this situation

    } // if multiple_ssbs_per_ro

    else {
      // --------------------
      // --------------------
      // Multiple ROs per SSB
      // --------------------
      // --------------------

      n_prach_conf = 0;

      // Go through the list of transmitted SSBs
      for (ssb_idx=0; ssb_idx<MAX_NB_SSB; ssb_idx++) {
        uint8_t nb_mapped_ro_in_association_period=0; // Reset the nb of mapped ROs for the new SSB index

        // Map only the transmitted ssb_idx
        if (true == ssb_list.tx_ssb[ssb_idx].transmitted) {

          // Map all the required ROs to this SSB
          // Go through the list of PRACH config periods within this association period
          for (; n_prach_conf<prach_assoc_pattern.prach_association_period_list[association_period_idx].nb_of_prach_conf_period; n_prach_conf++, prach_configuration_period_idx++) {

            // Build the association period with its association PRACH Configuration indexes
            prach_conf_period_p = &prach_assoc_pattern.prach_conf_period_list[prach_configuration_period_idx];
            prach_assoc_pattern.prach_association_period_list[association_period_idx].prach_conf_period_list[n_prach_conf] = prach_conf_period_p;

            for (; frame<prach_conf_period_p->nb_of_frame; frame++) {
              for (; slot<prach_conf_period_p->nb_of_slot; slot++) {
                for (; ro_in_time<prach_conf_period_p->prach_occasion_slot_map[frame][slot].nb_of_prach_occasion_in_time; ro_in_time++) {
                  for (; ro_in_freq<prach_conf_period_p->prach_occasion_slot_map[frame][slot].nb_of_prach_occasion_in_freq; ro_in_freq++) {
                    prach_occasion_info_t *ro_p = &prach_conf_period_p->prach_occasion_slot_map[frame][slot].prach_occasion[ro_in_time][ro_in_freq];

                    ro_p->mapped_ssb_idx[0] = ssb_idx;
                    ro_p->nb_mapped_ssb = 1;
                    ssb_list.tx_ssb[ssb_idx].mapped_ro[ssb_list.tx_ssb[ssb_idx].nb_mapped_ro] = ro_p;
                    ssb_list.tx_ssb[ssb_idx].nb_mapped_ro++;
                    AssertFatal(MAX_NB_RO_PER_SSB_IN_ASSOCIATION_PATTERN > ssb_list.tx_ssb[ssb_idx].nb_mapped_ro,"Too many mapped ROs (%d) to a single SSB\n", ssb_list.tx_ssb[ssb_idx].nb_mapped_ro);
                    nb_mapped_ro_in_association_period++;

                    LOG_D(MAC,"Mapped ssb_idx %u to RO slot-symbol %u-%u, %u-%u-%u/%u\n", ssb_idx, ro_p->slot, ro_p->start_symbol, slot, ro_in_time, ro_in_freq, prach_conf_period_p->prach_occasion_slot_map[frame][slot].nb_of_prach_occasion_in_freq);
                    LOG_D(MAC,"Nb mapped ROs for this ssb idx: in the association period only %u / total %u\n", ssb_list.tx_ssb[ssb_idx].nb_mapped_ro, nb_mapped_ro_in_association_period);

                    // Exit the loop if this SSB has been mapped to all the required ROs
                    // WIP: Assuming that ssb_rach_ratio equals the maximum nb of times a given ssb_idx is mapped within an association period:
                    //      this is true if no PRACH occasions are conflicting with SSBs nor TDD_UL_DL_ConfigurationCommon schedule
                    if (nb_mapped_ro_in_association_period == ssb_rach_ratio) {
                      ro_in_freq++;
                      break;
                    }
                  } // for ro_in_freq

                  // Exit the loop if this SSB has been mapped to all the required ROs
                  if (nb_mapped_ro_in_association_period == ssb_rach_ratio) {
                    break;
                  }
                  else ro_in_freq = 0; // else go to the next time symbol in that slot and reset the freq index
                } // for ro_in_time

                // Exit the loop if this SSB has been mapped to all the required ROs
                if (nb_mapped_ro_in_association_period == ssb_rach_ratio) {
                  break;
                }
                else ro_in_time = 0; // else go to the next slot in that PRACH config period and reset the symbol index
              } // for slot

              // Exit the loop if this SSB has been mapped to all the required ROs
              if (nb_mapped_ro_in_association_period == ssb_rach_ratio) {
                break;
              }
              else slot = 0; // else go to the next frame in that PRACH config period and reset the slot index
            } // for frame

            // Exit the loop if this SSB has been mapped to all the required ROs
            if (nb_mapped_ro_in_association_period == ssb_rach_ratio) {
              break;
            }
            else frame = 0; // else go to the next PRACH config period in that association period and reset the frame index
          } // for n_prach_conf

        } // if ssb_idx is transmitted
      } // for ssb_idx
    } // else if multiple_ssbs_per_ro

  } // for association_period_index
}

// Returns a RACH occasion if any matches the SSB idx, the frame and the slot
static int get_nr_prach_info_from_ssb_index(uint8_t ssb_idx,
                                            int frame,
                                            int slot,
                                            prach_occasion_info_t **prach_occasion_info_pp) {

  ssb_info_t *ssb_info_p;
  prach_occasion_slot_t *prach_occasion_slot_p = NULL;

  *prach_occasion_info_pp = NULL;

  // Search for a matching RO slot in the SSB_to_RO map
  // A valid RO slot will match:
  //      - ssb_idx mapped to one of the ROs in that RO slot
  //      - exact slot number
  //      - frame offset
  ssb_info_p = &ssb_list.tx_ssb[ssb_idx];
  for (uint8_t n_mapped_ro=0; n_mapped_ro<ssb_info_p->nb_mapped_ro; n_mapped_ro++) {
    if ((slot == ssb_info_p->mapped_ro[n_mapped_ro]->slot) &&
        (ssb_info_p->mapped_ro[n_mapped_ro]->frame == (frame % prach_assoc_pattern.nb_of_frame))) {

      uint8_t prach_config_period_nb = ssb_info_p->mapped_ro[n_mapped_ro]->frame / prach_assoc_pattern.prach_conf_period_list[0].nb_of_frame;
      uint8_t frame_nb_in_prach_config_period = ssb_info_p->mapped_ro[n_mapped_ro]->frame % prach_assoc_pattern.prach_conf_period_list[0].nb_of_frame;
      prach_occasion_slot_p = &prach_assoc_pattern.prach_conf_period_list[prach_config_period_nb].prach_occasion_slot_map[frame_nb_in_prach_config_period][slot];
    }
  }

  // If there is a matching RO slot in the SSB_to_RO map
  if (NULL != prach_occasion_slot_p)
  {
    // A random RO mapped to the SSB index should be selected in the slot

    // First count the number of times the SSB index is found in that RO
    uint8_t nb_mapped_ssb = 0;

    for (int ro_in_time=0; ro_in_time < prach_occasion_slot_p->nb_of_prach_occasion_in_time; ro_in_time++) {
      for (int ro_in_freq=0; ro_in_freq < prach_occasion_slot_p->nb_of_prach_occasion_in_freq; ro_in_freq++) {
        prach_occasion_info_t *prach_occasion_info_p = &prach_occasion_slot_p->prach_occasion[ro_in_time][ro_in_freq];

        for (uint8_t ssb_nb=0; ssb_nb<prach_occasion_info_p->nb_mapped_ssb; ssb_nb++) {
          if (prach_occasion_info_p->mapped_ssb_idx[ssb_nb] == ssb_idx) {
            nb_mapped_ssb++;
          }
        }
      }
    }

    // Choose a random SSB nb
    uint8_t random_ssb_nb = 0;

    random_ssb_nb = ((taus()) % nb_mapped_ssb);

    // Select the RO according to the chosen random SSB nb
    nb_mapped_ssb=0;
    for (int ro_in_time=0; ro_in_time < prach_occasion_slot_p->nb_of_prach_occasion_in_time; ro_in_time++) {
      for (int ro_in_freq=0; ro_in_freq < prach_occasion_slot_p->nb_of_prach_occasion_in_freq; ro_in_freq++) {
        prach_occasion_info_t *prach_occasion_info_p = &prach_occasion_slot_p->prach_occasion[ro_in_time][ro_in_freq];

        for (uint8_t ssb_nb=0; ssb_nb<prach_occasion_info_p->nb_mapped_ssb; ssb_nb++) {
          if (prach_occasion_info_p->mapped_ssb_idx[ssb_nb] == ssb_idx) {
            if (nb_mapped_ssb == random_ssb_nb) {
              *prach_occasion_info_pp = prach_occasion_info_p;
              return 1;
            }
            else {
              nb_mapped_ssb++;
            }
          }
        }
      }
    }
  }

  return 0;
}

// Build the SSB to RO mapping upon RRC configuration update
void build_ssb_to_ro_map(NR_ServingCellConfigCommon_t *scc, uint8_t unpaired){

  // Clear all the lists and maps
  memset(&prach_assoc_pattern, 0, sizeof(prach_association_pattern_t));
  memset(&ssb_list, 0, sizeof(ssb_list_info_t));

  // Build the list of all the valid RACH occasions in the maximum association pattern period according to the PRACH config
  LOG_D(MAC,"Build RO list\n");
  build_ro_list(scc, unpaired);

  // Build the list of all the valid/transmitted SSBs according to the config
  LOG_D(MAC,"Build SSB list\n");
  build_ssb_list(scc);

  // Map the transmitted SSBs to the ROs and create the association pattern according to the config
  LOG_D(MAC,"Map SSB to RO\n");
  map_ssb_to_ro(scc);
  LOG_D(MAC,"Map SSB to RO done\n");
}

// This function schedules the PRACH according to prach_ConfigurationIndex and TS 38.211, tables 6.3.3.2.x
// PRACH formats 9, 10, 11 are corresponding to dual PRACH format configurations A1/B1, A2/B2, A3/B3.
// - todo:
// - Partial configuration is actually already stored in (fapi_nr_prach_config_t) &mac->phy_config.config_req->prach_config
void nr_ue_prach_scheduler(module_id_t module_idP, frame_t frameP, sub_frame_t slotP, int thread_id) {

  uint16_t format, format0, format1, ncs;
  int is_nr_prach_slot;
  prach_occasion_info_t *prach_occasion_info_p;

  NR_UE_MAC_INST_t *mac = get_mac_inst(module_idP);
  RA_config_t *ra = &mac->ra;

  //fapi_nr_ul_config_request_t *ul_config = get_ul_config_request(mac, slotP);
  fapi_nr_ul_config_request_t *ul_config = &mac->ul_config_request[0];
  fapi_nr_ul_config_prach_pdu *prach_config_pdu;
  fapi_nr_config_request_t *cfg = &mac->phy_config.config_req;
  fapi_nr_prach_config_t *prach_config = &cfg->prach_config;
  nr_scheduled_response_t scheduled_response;

  NR_ServingCellConfigCommon_t *scc = mac->scc;
  NR_RACH_ConfigCommon_t *setup = scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup;
  NR_RACH_ConfigGeneric_t *rach_ConfigGeneric = &setup->rach_ConfigGeneric;

  ra->RA_offset = 2; // to compensate the rx frame offset at the gNB
  ra->generate_nr_prach = 0; // Reset flag for PRACH generation

  if (is_nr_UL_slot(scc, slotP, mac->frame_type)) {

    // WIP Need to get the proper selected ssb_idx
    //     Initial beam selection functionality is not available yet
    uint8_t selected_gnb_ssb_idx = mac->mib_ssb;

    // Get any valid PRACH occasion in the current slot for the selected SSB index
    is_nr_prach_slot = get_nr_prach_info_from_ssb_index(selected_gnb_ssb_idx,
                                                       (int)frameP,
                                                       (int)slotP,
                                                        &prach_occasion_info_p);

    if (is_nr_prach_slot && ra->ra_state == RA_UE_IDLE) {
      AssertFatal(NULL != prach_occasion_info_p,"PRACH Occasion Info not returned in a valid NR Prach Slot\n");

      ra->generate_nr_prach = 1;

      format = prach_occasion_info_p->format;
      format0 = format & 0xff;        // single PRACH format
      format1 = (format >> 8) & 0xff; // dual PRACH format

      prach_config_pdu = &ul_config->ul_config_list[ul_config->number_pdus].prach_config_pdu;
      memset(prach_config_pdu, 0, sizeof(fapi_nr_ul_config_prach_pdu));

      fill_ul_config(ul_config, frameP, slotP, FAPI_NR_UL_CONFIG_TYPE_PRACH);

      LOG_D(PHY, "In %s: (%p) %d UL PDUs:\n", __FUNCTION__, ul_config, ul_config->number_pdus);

      ncs = get_NCS(rach_ConfigGeneric->zeroCorrelationZoneConfig, format0, setup->restrictedSetConfig);

      prach_config_pdu->phys_cell_id = *scc->physCellId;
      prach_config_pdu->num_prach_ocas = 1;
      prach_config_pdu->prach_slot = prach_occasion_info_p->slot;
      prach_config_pdu->prach_start_symbol = prach_occasion_info_p->start_symbol;
      prach_config_pdu->num_ra = prach_occasion_info_p->fdm;

      prach_config_pdu->num_cs = ncs;
      prach_config_pdu->root_seq_id = prach_config->num_prach_fd_occasions_list[prach_occasion_info_p->fdm].prach_root_sequence_index;
      prach_config_pdu->restricted_set = prach_config->restricted_set_config;
      prach_config_pdu->freq_msg1 = prach_config->num_prach_fd_occasions_list[prach_occasion_info_p->fdm].k1;

      LOG_D(MAC,"Selected RO Frame %u, Slot %u, Symbol %u, Fdm %u\n", frameP, prach_config_pdu->prach_slot, prach_config_pdu->prach_start_symbol, prach_config_pdu->num_ra);

      // Search which SSB is mapped in the RO (among all the SSBs mapped to this RO)
      for (prach_config_pdu->ssb_nb_in_ro=0; prach_config_pdu->ssb_nb_in_ro<prach_occasion_info_p->nb_mapped_ssb; prach_config_pdu->ssb_nb_in_ro++) {
        if (prach_occasion_info_p->mapped_ssb_idx[prach_config_pdu->ssb_nb_in_ro] == selected_gnb_ssb_idx)
          break;
      }
      AssertFatal(prach_config_pdu->ssb_nb_in_ro<prach_occasion_info_p->nb_mapped_ssb, "%u not found in the mapped SSBs to the PRACH occasion", selected_gnb_ssb_idx);

      if (format1 != 0xff) {
        switch(format0) { // dual PRACH format
          case 0xa1:
            prach_config_pdu->prach_format = 11;
            break;
          case 0xa2:
            prach_config_pdu->prach_format = 12;
            break;
          case 0xa3:
            prach_config_pdu->prach_format = 13;
            break;
        default:
          AssertFatal(1 == 0, "Only formats A1/B1 A2/B2 A3/B3 are valid for dual format");
        }
      } else {
        switch(format0) { // single PRACH format
          case 0:
            prach_config_pdu->prach_format = 0;
            break;
          case 1:
            prach_config_pdu->prach_format = 1;
            break;
          case 2:
            prach_config_pdu->prach_format = 2;
            break;
          case 3:
            prach_config_pdu->prach_format = 3;
            break;
          case 0xa1:
            prach_config_pdu->prach_format = 4;
            break;
          case 0xa2:
            prach_config_pdu->prach_format = 5;
            break;
          case 0xa3:
            prach_config_pdu->prach_format = 6;
            break;
          case 0xb1:
            prach_config_pdu->prach_format = 7;
            break;
          case 0xb4:
            prach_config_pdu->prach_format = 8;
            break;
          case 0xc0:
            prach_config_pdu->prach_format = 9;
            break;
          case 0xc2:
            prach_config_pdu->prach_format = 10;
            break;
          default:
            AssertFatal(1 == 0, "Invalid PRACH format");
        }
      } // if format1
      fill_scheduled_response(&scheduled_response, NULL, ul_config, NULL, module_idP, 0 /*TBR fix*/, frameP, slotP, thread_id);
      if(mac->if_module != NULL && mac->if_module->scheduled_response != NULL)
        mac->if_module->scheduled_response(&scheduled_response);
    } // is_nr_prach_slot
  } // if is_nr_UL_slot
}

uint8_t
nr_ue_get_sdu(module_id_t module_idP, int CC_id, frame_t frameP,
           sub_frame_t subframe, uint8_t eNB_index,
           uint8_t *ulsch_buffer, uint16_t buflen, uint8_t *access_mode) {
  uint8_t total_rlc_pdu_header_len = 0;
  int16_t buflen_remain = 0;
  uint8_t lcid = 0;
  uint16_t sdu_lengths[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  uint8_t sdu_lcids[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  uint16_t payload_offset = 0, num_sdus = 0;
  uint8_t ulsch_sdus[MAX_ULSCH_PAYLOAD_BYTES];
  uint16_t sdu_length_total = 0;
  //unsigned short post_padding = 0;
  NR_UE_MAC_INST_t *mac = get_mac_inst(module_idP);

  rlc_buffer_occupancy_t lcid_buffer_occupancy_old =
    0, lcid_buffer_occupancy_new = 0;
  LOG_D(MAC,
        "[UE %d] MAC PROCESS UL TRANSPORT BLOCK at frame%d subframe %d TBS=%d\n",
        module_idP, frameP, subframe, buflen);
  AssertFatal(CC_id == 0,
              "Transmission on secondary CCs is not supported yet\n");

  // Check for DCCH first
  // TO DO: Multiplex in the order defined by the logical channel prioritization
  for (lcid = UL_SCH_LCID_SRB1;
       lcid < NR_MAX_NUM_LCID; lcid++) {

      lcid_buffer_occupancy_old = mac_rlc_get_buffer_occupancy_ind(module_idP, mac->crnti, eNB_index, frameP, subframe, ENB_FLAG_NO, lcid);
      lcid_buffer_occupancy_new = lcid_buffer_occupancy_old;

      if(lcid_buffer_occupancy_new){

        buflen_remain =
          buflen - (total_rlc_pdu_header_len + sdu_length_total + MAX_RLC_SDU_SUBHEADER_SIZE);
        LOG_D(MAC,
              "[UE %d] Frame %d : UL-DXCH -> ULSCH, RLC %d has %d bytes to "
              "send (Transport Block size %d SDU Length Total %d , mac header len %d, buflen_remain %d )\n", //BSR byte before Tx=%d
              module_idP, frameP, lcid, lcid_buffer_occupancy_new,
              buflen, sdu_length_total,
              total_rlc_pdu_header_len, buflen_remain); // ,nr_ue_mac_inst->scheduling_info.BSR_bytes[nr_ue_mac_inst->scheduling_info.LCGID[lcid]]

        while(buflen_remain > 0 && lcid_buffer_occupancy_new){

        sdu_lengths[num_sdus] = mac_rlc_data_req(module_idP,
                                mac->crnti,
                                eNB_index,
                                frameP,
                                ENB_FLAG_NO,
                                MBMS_FLAG_NO,
                                lcid,
                                buflen_remain,
                                (char *)&ulsch_sdus[sdu_length_total],0,
                                0);

        AssertFatal(buflen_remain >= sdu_lengths[num_sdus],
                    "LCID=%d RLC has segmented %d bytes but MAC has max=%d\n",
                    lcid, sdu_lengths[num_sdus], buflen_remain);

        if (sdu_lengths[num_sdus]) {
          sdu_length_total += sdu_lengths[num_sdus];
          sdu_lcids[num_sdus] = lcid;

          total_rlc_pdu_header_len += MAX_RLC_SDU_SUBHEADER_SIZE; //rlc_pdu_header_len_last;

          //Update number of SDU
          num_sdus++;
        }

        /* Get updated BO after multiplexing this PDU */
        lcid_buffer_occupancy_new = mac_rlc_get_buffer_occupancy_ind(module_idP,
                                                                     mac->crnti,
                                                                     eNB_index,
                                                                     frameP,
                                                                     subframe,
                                                                     ENB_FLAG_NO,
                                                                     lcid);
        buflen_remain =
                  buflen - (total_rlc_pdu_header_len + sdu_length_total + MAX_RLC_SDU_SUBHEADER_SIZE);
        }
  }

}

  // Generate ULSCH PDU
  if (num_sdus>0) {
  payload_offset = nr_generate_ulsch_pdu(ulsch_sdus,
                                         ulsch_buffer,  // mac header
                                         num_sdus,  // num sdus
                                         sdu_lengths, // sdu length
                                         sdu_lcids, // sdu lcid
                                         0, // power_headroom
                                         mac->crnti, // crnti
                                         0, // truncated_bsr
                                         0, // short_bsr
                                         0, // long_bsr
                                         0, // post_padding 
                                         buflen);  // TBS in bytes
  }
  else
          return 0;

  // Padding: fill remainder of ULSCH with 0
  if (buflen - payload_offset > 0){
          for (int j = payload_offset; j < buflen; j++)
                  ulsch_buffer[j] = 0;
  }

#if defined(ENABLE_MAC_PAYLOAD_DEBUG)
  LOG_I(MAC, "Printing UL MAC payload UE side, payload_offset: %d \n", payload_offset);
  for (int i = 0; i < buflen ; i++) {
          //harq_process_ul_ue->a[i] = (unsigned char) rand();
          //printf("a[%d]=0x%02x\n",i,harq_process_ul_ue->a[i]);
          printf("%02x ",(unsigned char)ulsch_buffer[i]);
  }
  printf("\n");
#endif

  return 1;
}
