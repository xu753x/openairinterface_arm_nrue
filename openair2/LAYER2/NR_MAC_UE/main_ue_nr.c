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

/* \file main_ue_nr.c
 * \brief top init of Layer 2
 * \author R. Knopp, K.H. HSU
 * \date 2018
 * \version 0.1
 * \company Eurecom / NTUST
 * \email: knopp@eurecom.fr, kai-hsiang.hsu@eurecom.fr
 * \note
 * \warning
 */

//#include "defs.h"
#include "mac_proto.h"
#include "../../ARCH/COMMON/common_lib.h"
//#undef MALLOC
#include "assertions.h"
#include "PHY/types.h"
#include "PHY/defs_UE.h"
#include "openair2/LAYER2/RLC/rlc.h"
#include "openair2/LAYER2/PDCP_v10.1.0/pdcp.h"
#include "openair2/LAYER2/nr_pdcp/nr_pdcp_entity.h"
#include "executables/softmodem-common.h"

static NR_UE_MAC_INST_t *nr_ue_mac_inst; 

NR_UE_MAC_INST_t * nr_l2_init_ue(NR_UE_RRC_INST_t* rrc_inst)
{
    //LOG_I(MAC, "[MAIN] MAC_INIT_GLOBAL_PARAM IN...\n");

    //LOG_I(MAC, "[MAIN] init UE MAC functions \n");
    
    //init mac here
    nr_ue_mac_inst = (NR_UE_MAC_INST_t *)calloc(sizeof(NR_UE_MAC_INST_t),NB_NR_UE_MAC_INST);
    if (rrc_inst) {
      nr_rrc_mac_config_req_ue(0,0,0,NULL,rrc_inst->cell_group_config);
      
      if (IS_SOFTMODEM_NOS1){
        if (rlc_module_init(0) != 0) {
          LOG_I(RLC, "Problem at RLC initiation \n");
        }
        pdcp_layer_init();
        nr_DRB_preconfiguration();
      }

      // Allocate memory for ul_config_request in the mac instance. This is now a pointer and will
      // point to a list of structures (one for each UL slot) to store PUSCH scheduling parameters
      // received from UL DCI.
      if (nr_ue_mac_inst->scc) {
        int num_slots_ul = nr_ue_mac_inst->scc->tdd_UL_DL_ConfigurationCommon->pattern1.nrofUplinkSlots;
        if (nr_ue_mac_inst->scc->tdd_UL_DL_ConfigurationCommon->pattern1.nrofUplinkSymbols>0)
          num_slots_ul++;
        LOG_D(MAC, "Initializing ul_config_request. num_slots_ul = %d\n", num_slots_ul);
        nr_ue_mac_inst->ul_config_request = (fapi_nr_ul_config_request_t *)calloc(num_slots_ul, sizeof(fapi_nr_ul_config_request_t));
      }
    }
    else LOG_I(MAC,"Running without RRC instance\n");

    return (nr_ue_mac_inst);
}

NR_UE_MAC_INST_t *get_mac_inst(module_id_t module_id){
    return &nr_ue_mac_inst[(int)module_id];
}
