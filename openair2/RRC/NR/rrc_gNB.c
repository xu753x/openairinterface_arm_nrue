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

/*! \file rrc_gNB.c
 * \brief rrc procedures for gNB
 * \author Navid Nikaein and  Raymond Knopp , WEI-TAI CHEN
 * \date 2011 - 2014 , 2018
 * \version 1.0
 * \company Eurecom, NTUST
 * \email: navid.nikaein@eurecom.fr and raymond.knopp@eurecom.fr, kroempa@gmail.com
 */
#define RRC_GNB_C
#define RRC_GNB_C

#include "nr_rrc_config.h"
#include "nr_rrc_defs.h"
#include "nr_rrc_extern.h"
#include "assertions.h"
#include "common/ran_context.h"
#include "asn1_conversions.h"

#include "RRC/L2_INTERFACE/openair_rrc_L2_interface.h"
#include "LAYER2/RLC/rlc.h"
#include "LAYER2/NR_MAC_gNB/mac_proto.h"
#include "common/utils/LOG/log.h"
#include "COMMON/mac_rrc_primitives.h"
#include "RRC/NR/MESSAGES/asn1_msg.h"

#include "NR_BCCH-BCH-Message.h"
#include "NR_UL-DCCH-Message.h"
#include "NR_DL-DCCH-Message.h"
#include "NR_CellGroupConfig.h"
#include "NR_MeasResults.h"
#include "LTE_UECapabilityInformation.h"
#include "LTE_UL-DCCH-Message.h"
#include "NR_UL-CCCH-Message.h"
#include "NR_RRCSetupRequest-IEs.h"
#include "NR_RRCSetupComplete-IEs.h"
#include "NR_RRCReestablishmentRequest-IEs.h"

#include "rlc.h"
#include "rrc_eNB_UE_context.h"
#include "platform_types.h"
#include "msc.h"
#include "common/utils/LOG/vcd_signal_dumper.h"

#include "T.h"

#include "RRC/NAS/nas_config.h"
#include "RRC/NAS/rb_config.h"
#include "OCG.h"
#include "OCG_extern.h"

#include "UTIL/OSA/osa_defs.h"

#include "rrc_eNB_S1AP.h"
#include "rrc_gNB_NGAP.h"

#include "rrc_eNB_GTPV1U.h"

#include "nr_pdcp/nr_pdcp_entity.h"
#include "pdcp.h"
#include "gtpv1u_eNB_task.h"


#include "intertask_interface.h"
#include "SIMULATION/TOOLS/sim.h" // for taus

#include "executables/softmodem-common.h"
#include <openair2/RRC/NR/rrc_gNB_UE_context.h>
#include <openair2/X2AP/x2ap_eNB.h>

#include "BIT_STRING.h"
#include "assertions.h"

//#define XER_PRINT

extern RAN_CONTEXT_t RC;

extern boolean_t nr_rrc_pdcp_config_asn1_req(
    const protocol_ctxt_t *const  ctxt_pP,
    NR_SRB_ToAddModList_t  *const srb2add_list,
    NR_DRB_ToAddModList_t  *const drb2add_list,
    NR_DRB_ToReleaseList_t *const drb2release_list,
    const uint8_t                   security_modeP,
    uint8_t                  *const kRRCenc,
    uint8_t                  *const kRRCint,
    uint8_t                  *const kUPenc,
    uint8_t                  *const kUPint
  #if (LTE_RRC_VERSION >= MAKE_VERSION(9, 0, 0))
    ,LTE_PMCH_InfoList_r9_t  *pmch_InfoList_r9
  #endif
    ,rb_id_t                 *const defaultDRB,
    struct NR_CellGroupConfig__rlc_BearerToAddModList *rlc_bearer2add_list);

extern rlc_op_status_t nr_rrc_rlc_config_asn1_req (const protocol_ctxt_t   * const ctxt_pP,
    const NR_SRB_ToAddModList_t   * const srb2add_listP,
    const NR_DRB_ToAddModList_t   * const drb2add_listP,
    const NR_DRB_ToReleaseList_t  * const drb2release_listP,
    const LTE_PMCH_InfoList_r9_t * const pmch_InfoList_r9_pP,
    struct NR_CellGroupConfig__rlc_BearerToAddModList *rlc_bearer2add_list);

static inline uint64_t bitStr_to_uint64(BIT_STRING_t *asn);

mui_t                               rrc_gNB_mui = 0;
uint8_t first_rrcreconfiguration = 0;

///---------------------------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------------------------///

void openair_nr_rrc_on(const protocol_ctxt_t *const ctxt_pP) {
  LOG_I(NR_RRC, PROTOCOL_NR_RRC_CTXT_FMT" gNB:OPENAIR NR RRC IN....\n",PROTOCOL_NR_RRC_CTXT_ARGS(ctxt_pP));
  rrc_config_nr_buffer (&RC.nrrrc[ctxt_pP->module_id]->carrier.SI, BCCH, 1);
  RC.nrrrc[ctxt_pP->module_id]->carrier.SI.Active = 1;
  rrc_config_nr_buffer (&RC.nrrrc[ctxt_pP->module_id]->carrier.Srb0, CCCH, 1);
  RC.nrrrc[ctxt_pP->module_id]->carrier.Srb0.Active = 1;
}

///---------------------------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------------------------///

void rrc_gNB_process_SgNBAdditionRequest(
  const protocol_ctxt_t  *const ctxt_pP,
  rrc_gNB_ue_context_t   *ue_context_pP
) {
  rrc_gNB_generate_SgNBAdditionRequestAcknowledge(ctxt_pP,ue_context_pP);
}

void rrc_gNB_generate_SgNBAdditionRequestAcknowledge(
  const protocol_ctxt_t  *const ctxt_pP,
  rrc_gNB_ue_context_t   *const ue_context_pP) {
  //uint8_t size;
  //uint8_t buffer[100];
  //int     CC_id = ue_context_pP->ue_context.primaryCC_id;
  //OCTET_STRING_t                                      *secondaryCellGroup;
  NR_CellGroupConfig_t                                *cellGroupconfig;
  struct NR_CellGroupConfig__rlc_BearerToAddModList   *rlc_BearerToAddModList;
  struct NR_MAC_CellGroupConfig                       *mac_CellGroupConfig;
  struct NR_PhysicalCellGroupConfig                   *physicalCellGroupConfig;
  struct NR_SpCellConfig                              *spCellConfig;
  //struct NR_CellGroupConfig__sCellToAddModList        *sCellToAddModList;
  cellGroupconfig                           = CALLOC(1,sizeof(NR_CellGroupConfig_t));
  cellGroupconfig->rlc_BearerToAddModList   = CALLOC(1,sizeof(struct NR_CellGroupConfig__rlc_BearerToAddModList));
  cellGroupconfig->mac_CellGroupConfig      = CALLOC(1,sizeof(struct NR_MAC_CellGroupConfig));
  cellGroupconfig->physicalCellGroupConfig  = CALLOC(1,sizeof(struct NR_PhysicalCellGroupConfig));
  cellGroupconfig->spCellConfig             = CALLOC(1,sizeof(struct NR_SpCellConfig));
  //cellGroupconfig->sCellToAddModList        = CALLOC(1,sizeof(struct NR_CellGroupConfig__sCellToAddModList));
  rlc_BearerToAddModList   = cellGroupconfig->rlc_BearerToAddModList;
  mac_CellGroupConfig      = cellGroupconfig->mac_CellGroupConfig;
  physicalCellGroupConfig  = cellGroupconfig->physicalCellGroupConfig;
  spCellConfig             = cellGroupconfig->spCellConfig;
  //sCellToAddModList        = cellGroupconfig->sCellToAddModList;
  rlc_bearer_config_t *rlc_config;
  rlc_config = CALLOC(1,sizeof(rlc_bearer_config_t));
  //Fill rlc_bearer config value
  rrc_config_rlc_bearer(ctxt_pP->module_id,
                        ue_context_pP->ue_context.primaryCC_id,
                        rlc_config
                       );
  //Fill rlc_bearer config to structure
  do_RLC_BEARER(ctxt_pP->module_id,
                ue_context_pP->ue_context.primaryCC_id,
                rlc_BearerToAddModList,
                rlc_config);
  mac_cellgroup_t *mac_cellgroup_config;
  mac_cellgroup_config = CALLOC(1,sizeof(mac_cellgroup_t));
  //Fill mac_cellgroup_config config value
  rrc_config_mac_cellgroup(ctxt_pP->module_id,
                           ue_context_pP->ue_context.primaryCC_id,
                           mac_cellgroup_config
                          );
  //Fill mac_cellgroup config to structure
  do_MAC_CELLGROUP(ctxt_pP->module_id,
                   ue_context_pP->ue_context.primaryCC_id,
                   mac_CellGroupConfig,
                   mac_cellgroup_config);
  physicalcellgroup_t *physicalcellgroup_config;
  physicalcellgroup_config = CALLOC(1,sizeof(physicalcellgroup_t));
  //Fill physicalcellgroup_config config value
  rrc_config_physicalcellgroup(ctxt_pP->module_id,
                               ue_context_pP->ue_context.primaryCC_id,
                               physicalcellgroup_config
                              );
  //Fill physicalcellgroup config to structure
  do_PHYSICALCELLGROUP(ctxt_pP->module_id,
                       ue_context_pP->ue_context.primaryCC_id,
                       physicalCellGroupConfig,
                       physicalcellgroup_config);
  do_SpCellConfig(RC.nrrrc[ctxt_pP->module_id],
                  spCellConfig);
}

///---------------------------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------------------------///

static void init_NR_SI(gNB_RRC_INST *rrc, gNB_RrcConfigurationReq *configuration) {
  LOG_D(RRC,"%s()\n\n\n\n",__FUNCTION__);
  rrc->carrier.MIB             = (uint8_t *) malloc16(4);
  rrc->carrier.sizeof_MIB      = do_MIB_NR(rrc,0);

  if(get_softmodem_params()->sa) {
    rrc->carrier.sizeof_SIB1 = do_SIB1_NR(&rrc->carrier,configuration);
  }

  LOG_I(NR_RRC,"Done init_NR_SI\n");
  rrc_mac_config_req_gNB(rrc->module_id,
                         rrc->carrier.ssb_SubcarrierOffset,
                         rrc->carrier.pdsch_AntennaPorts,
                         rrc->carrier.pusch_AntennaPorts,
                         rrc->carrier.pusch_TargetSNRx10,
                         rrc->carrier.pucch_TargetSNRx10,
                         (NR_ServingCellConfigCommon_t *)rrc->carrier.servingcellconfigcommon,
                         0,
                         0, // WIP hardcoded rnti
                         (NR_CellGroupConfig_t *)NULL
                        );

  if (get_softmodem_params()->phy_test > 0 || get_softmodem_params()->do_ra > 0 || get_softmodem_params()->sa > 0) {
    // This is for phytest only, emulate first X2 message if uecap.raw file is present
    FILE *fd;
    fd = fopen("uecap.raw","r");

    if (fd != NULL) {
      char buffer[4096];
      int msg_len=fread(buffer,1,4096,fd);
      LOG_I(RRC,"Read in %d bytes for uecap\n",msg_len);
      LTE_UL_DCCH_Message_t *LTE_UL_DCCH_Message;
      asn_dec_rval_t dec_rval = uper_decode_complete( NULL,
                                &asn_DEF_LTE_UL_DCCH_Message,
                                (void **)&LTE_UL_DCCH_Message,
                                (uint8_t *)buffer,
                                msg_len);

      if ((dec_rval.code != RC_OK) && (dec_rval.consumed == 0)) {
        AssertFatal(1==0,"NR_UL_DCCH_MESSAGE decode error\n");
        // free the memory
        SEQUENCE_free( &asn_DEF_LTE_UL_DCCH_Message, LTE_UL_DCCH_Message, 1 );
        return;
      }

      fclose(fd);
      xer_fprint(stdout,&asn_DEF_LTE_UL_DCCH_Message, LTE_UL_DCCH_Message);
      // recreate enough of X2 EN-DC Container
      AssertFatal(LTE_UL_DCCH_Message->message.choice.c1.present == LTE_UL_DCCH_MessageType__c1_PR_ueCapabilityInformation,
                  "ueCapabilityInformation not present\n");
      NR_CG_ConfigInfo_t *CG_ConfigInfo = calloc(1,sizeof(*CG_ConfigInfo));
      CG_ConfigInfo->criticalExtensions.present = NR_CG_ConfigInfo__criticalExtensions_PR_c1;
      CG_ConfigInfo->criticalExtensions.choice.c1 = calloc(1,sizeof(*CG_ConfigInfo->criticalExtensions.choice.c1));
      CG_ConfigInfo->criticalExtensions.choice.c1->present = NR_CG_ConfigInfo__criticalExtensions__c1_PR_cg_ConfigInfo;
      CG_ConfigInfo->criticalExtensions.choice.c1->choice.cg_ConfigInfo = calloc(1,sizeof(*CG_ConfigInfo->criticalExtensions.choice.c1->choice.cg_ConfigInfo));
      NR_CG_ConfigInfo_IEs_t *cg_ConfigInfo = CG_ConfigInfo->criticalExtensions.choice.c1->choice.cg_ConfigInfo;
      cg_ConfigInfo->ue_CapabilityInfo = calloc(1,sizeof(*cg_ConfigInfo->ue_CapabilityInfo));
      asn_enc_rval_t enc_rval = uper_encode_to_buffer(&asn_DEF_LTE_UE_CapabilityRAT_ContainerList,NULL,
                                (void *)&LTE_UL_DCCH_Message->message.choice.c1.choice.ueCapabilityInformation.criticalExtensions.choice.c1.choice.ueCapabilityInformation_r8.ue_CapabilityRAT_ContainerList,buffer,4096);
      AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %jd)!\n",
                   enc_rval.failed_type->name, enc_rval.encoded);
      OCTET_STRING_fromBuf(cg_ConfigInfo->ue_CapabilityInfo,
                           (const char *)buffer,
                           (enc_rval.encoded+7)>>3);
      parse_CG_ConfigInfo(rrc,CG_ConfigInfo,NULL);
    } else {
      struct rrc_gNB_ue_context_s *ue_context_p = rrc_gNB_allocate_new_UE_context(rrc);
      ue_context_p->ue_context.spCellConfig = calloc(1, sizeof(struct NR_SpCellConfig));
      ue_context_p->ue_context.spCellConfig->spCellConfigDedicated = configuration->scd;
      LOG_I(NR_RRC,"Adding new user (%p)\n",ue_context_p);
      rrc_add_nsa_user(rrc,ue_context_p,NULL);
    }
  }
}


char openair_rrc_gNB_configuration(const module_id_t gnb_mod_idP, gNB_RrcConfigurationReq *configuration) {
  protocol_ctxt_t      ctxt;
  gNB_RRC_INST         *rrc=RC.nrrrc[gnb_mod_idP];
  PROTOCOL_CTXT_SET_BY_MODULE_ID(&ctxt, gnb_mod_idP, GNB_FLAG_YES, NOT_A_RNTI, 0, 0,gnb_mod_idP);
  LOG_I(NR_RRC,
        PROTOCOL_NR_RRC_CTXT_FMT" Init...\n",
        PROTOCOL_NR_RRC_CTXT_ARGS(&ctxt));

  AssertFatal(rrc != NULL, "RC.nrrrc not initialized!");
  AssertFatal(NUMBER_OF_UE_MAX < (module_id_t)0xFFFFFFFFFFFFFFFF, " variable overflow");
  AssertFatal(configuration!=NULL,"configuration input is null\n");
  rrc->module_id = gnb_mod_idP;
  rrc->Nb_ue = 0;
  rrc->carrier.Srb0.Active = 0;
  nr_uid_linear_allocator_init(&rrc->uid_allocator);
  RB_INIT(&rrc->rrc_ue_head);
  rrc->initial_id2_s1ap_ids = hashtable_create (NUMBER_OF_UE_MAX * 2, NULL, NULL);
  rrc->s1ap_id2_s1ap_ids    = hashtable_create (NUMBER_OF_UE_MAX * 2, NULL, NULL);
  rrc->initial_id2_ngap_ids = hashtable_create (NUMBER_OF_UE_MAX * 2, NULL, NULL);
  rrc->ngap_id2_ngap_ids    = hashtable_create (NUMBER_OF_UE_MAX * 2, NULL, NULL);
  rrc->carrier.servingcellconfigcommon = configuration->scc;
  rrc->carrier.ssb_SubcarrierOffset = configuration->ssb_SubcarrierOffset;
  rrc->carrier.pdsch_AntennaPorts = configuration->pdsch_AntennaPorts;
  rrc->carrier.pusch_AntennaPorts = configuration->pusch_AntennaPorts;
  rrc->carrier.pusch_TargetSNRx10 = configuration->pusch_TargetSNRx10;
  rrc->carrier.pucch_TargetSNRx10 = configuration->pucch_TargetSNRx10;
  /// System Information INIT
  LOG_I(NR_RRC, PROTOCOL_NR_RRC_CTXT_FMT" Checking release \n",PROTOCOL_NR_RRC_CTXT_ARGS(&ctxt));
  init_NR_SI(rrc, configuration);
  rrc_init_nr_global_param();
  openair_nr_rrc_on(&ctxt);
  return 0;
}//END openair_rrc_gNB_configuration


void rrc_gNB_process_AdditionRequestInformation(const module_id_t gnb_mod_idP, x2ap_ENDC_sgnb_addition_req_t *m) {
  struct NR_CG_ConfigInfo *cg_configinfo = NULL;
  asn_dec_rval_t dec_rval = uper_decode_complete(NULL,
                            &asn_DEF_NR_CG_ConfigInfo,
                            (void **)&cg_configinfo,
                            (uint8_t *)m->rrc_buffer,
                            (int) m->rrc_buffer_size);//m->rrc_buffer_size);
  gNB_RRC_INST         *rrc=RC.nrrrc[gnb_mod_idP];

  if ((dec_rval.code != RC_OK) && (dec_rval.consumed == 0)) {
    AssertFatal(1==0,"NR_UL_DCCH_MESSAGE decode error\n");
    // free the memory
    SEQUENCE_free(&asn_DEF_NR_CG_ConfigInfo, cg_configinfo, 1);
    return;
  }

  xer_fprint(stdout,&asn_DEF_NR_CG_ConfigInfo, cg_configinfo);
  // recreate enough of X2 EN-DC Container
  AssertFatal(cg_configinfo->criticalExtensions.choice.c1->present == NR_CG_ConfigInfo__criticalExtensions__c1_PR_cg_ConfigInfo,
              "ueCapabilityInformation not present\n");
  parse_CG_ConfigInfo(rrc,cg_configinfo,m);
}


//-----------------------------------------------------------------------------
uint8_t
rrc_gNB_get_next_transaction_identifier(
    module_id_t gnb_mod_idP
)
//-----------------------------------------------------------------------------
{
  static uint8_t                      nr_rrc_transaction_identifier[NUMBER_OF_gNB_MAX];
  nr_rrc_transaction_identifier[gnb_mod_idP] = (nr_rrc_transaction_identifier[gnb_mod_idP] + 1) % NR_RRC_TRANSACTION_IDENTIFIER_NUMBER;
  LOG_T(NR_RRC, "generated xid is %d\n", nr_rrc_transaction_identifier[gnb_mod_idP]);
  return nr_rrc_transaction_identifier[gnb_mod_idP];
}

//-----------------------------------------------------------------------------
void
rrc_gNB_generate_RRCSetup(
    const protocol_ctxt_t    *const ctxt_pP,
    rrc_gNB_ue_context_t     *const ue_context_pP,
    const int                CC_id
)
//-----------------------------------------------------------------------------
{
  LOG_I(NR_RRC, "rrc_gNB_generate_RRCSetup \n");
  NR_SRB_ToAddModList_t        **SRB_configList = NULL;

  // T(T_GNB_RRC_SETUP,
  //   T_INT(ctxt_pP->module_id),
  //   T_INT(ctxt_pP->frame),
  //   T_INT(ctxt_pP->subframe),
  //   T_INT(ctxt_pP->rnti));
  gNB_RRC_UE_t *ue_p = &ue_context_pP->ue_context;
  SRB_configList = &ue_p->SRB_configList;
  ue_p->Srb0.Tx_buffer.payload_size = do_RRCSetup(ctxt_pP,
              ue_context_pP,
              CC_id,
              (uint8_t *) ue_p->Srb0.Tx_buffer.Payload,
              rrc_gNB_get_next_transaction_identifier(ctxt_pP->module_id),
              SRB_configList);

  LOG_DUMPMSG(NR_RRC, DEBUG_RRC,
              (char *)(ue_p->Srb0.Tx_buffer.Payload),
              ue_p->Srb0.Tx_buffer.payload_size,
              "[MSG] RRC Setup\n");

  LOG_D(NR_RRC,
      PROTOCOL_NR_RRC_CTXT_UE_FMT" RRC_gNB --- MAC_CONFIG_REQ  (SRB1) ---> MAC_gNB\n",
      PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP));

  // rrc_mac_config_req_eNB

  MSC_LOG_TX_MESSAGE(
      MSC_RRC_GNB,
      MSC_RRC_UE,
      ue_p->Srb0.Tx_buffer.Header, // LG WARNING
      ue_p->Srb0.Tx_buffer.payload_size,
      MSC_AS_TIME_FMT" RRCSetup UE %x size %u",
      MSC_AS_TIME_ARGS(ctxt_pP),
      ue_context_pP->ue_context.rnti,
      ue_p->Srb0.Tx_buffer.payload_size);
  LOG_I(NR_RRC,
      PROTOCOL_NR_RRC_CTXT_UE_FMT" [RAPROC] Logical Channel DL-CCCH, Generating RRCSetup (bytes %d)\n",
      PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
      ue_p->Srb0.Tx_buffer.payload_size);
  // activate release timer, if RRCSetupComplete not received after 100 frames, remove UE
  ue_context_pP->ue_context.ue_release_timer = 1;
  // remove UE after 10 frames after RRCConnectionRelease is triggered
  ue_context_pP->ue_context.ue_release_timer_thres = 1000;
  /* init timers */
  //   ue_context_pP->ue_context.ue_rrc_inactivity_timer = 0;
#ifdef ITTI_SIM
  MessageDef *message_p;
  uint8_t *message_buffer;
  message_buffer = itti_malloc (TASK_RRC_GNB, TASK_RRC_UE_SIM,
            ue_p->Srb0.Tx_buffer.payload_size);
  memcpy (message_buffer, (uint8_t*)ue_p->Srb0.Tx_buffer.Payload, ue_p->Srb0.Tx_buffer.payload_size);
  message_p = itti_alloc_new_message (TASK_RRC_GNB, 0, GNB_RRC_CCCH_DATA_IND);
  GNB_RRC_CCCH_DATA_IND (message_p).sdu = message_buffer;
  GNB_RRC_CCCH_DATA_IND (message_p).size  = ue_p->Srb0.Tx_buffer.payload_size;
  itti_send_msg_to_task (TASK_RRC_UE_SIM, ctxt_pP->instance, message_p);
#endif
}

//-----------------------------------------------------------------------------
void
rrc_gNB_generate_RRCSetup_for_RRCReestablishmentRequest(
    const protocol_ctxt_t    *const ctxt_pP,
    const int                CC_id
)
//-----------------------------------------------------------------------------
{
  LOG_I(NR_RRC, "generate RRCSetup for RRCReestablishmentRequest \n");
  NR_SRB_ToAddModList_t        **SRB_configList = NULL;
  rrc_gNB_ue_context_t         *ue_context_pP   = NULL;
  gNB_RRC_INST                 *rrc_instance_p = RC.nrrrc[ctxt_pP->module_id];

  ue_context_pP = rrc_gNB_get_next_free_ue_context(ctxt_pP, rrc_instance_p, 0);

  gNB_RRC_UE_t *ue_p = &ue_context_pP->ue_context;
  SRB_configList = &ue_p->SRB_configList;
  ue_p->Srb0.Tx_buffer.payload_size = do_RRCSetup(ctxt_pP,
              ue_context_pP,
              CC_id,
              (uint8_t *) ue_p->Srb0.Tx_buffer.Payload,
              rrc_gNB_get_next_transaction_identifier(ctxt_pP->module_id),
              SRB_configList);

  LOG_DUMPMSG(NR_RRC, DEBUG_RRC,
              (char *)(ue_p->Srb0.Tx_buffer.Payload),
              ue_p->Srb0.Tx_buffer.payload_size,
              "[MSG] RRC Setup\n");

  LOG_D(NR_RRC,
          PROTOCOL_NR_RRC_CTXT_UE_FMT" RRC_gNB --- MAC_CONFIG_REQ  (SRB1) ---> MAC_gNB\n",
          PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP));

  rrc_mac_config_req_gNB(rrc_instance_p->module_id,
                         rrc_instance_p->carrier.ssb_SubcarrierOffset,
                         rrc_instance_p->carrier.pdsch_AntennaPorts,
                         rrc_instance_p->carrier.pusch_AntennaPorts,
                         rrc_instance_p->carrier.pusch_TargetSNRx10,
                         rrc_instance_p->carrier.pucch_TargetSNRx10,
                         (NR_ServingCellConfigCommon_t *)rrc_instance_p->carrier.servingcellconfigcommon,
                         0,
                         ue_context_pP->ue_context.rnti,
                         (NR_CellGroupConfig_t *)NULL
                        );

    MSC_LOG_TX_MESSAGE(
        MSC_RRC_GNB,
        MSC_RRC_UE,
        ue_p->Srb0.Tx_buffer.Header, // LG WARNING
        ue_p->Srb0.Tx_buffer.payload_size,
        MSC_AS_TIME_FMT" RRCSetup UE %x size %u",
        MSC_AS_TIME_ARGS(ctxt_pP),
        ue_context_pP->ue_context.rnti,
        ue_p->Srb0.Tx_buffer.payload_size);
    LOG_I(NR_RRC,
        PROTOCOL_NR_RRC_CTXT_UE_FMT" [RAPROC] Logical Channel DL-CCCH, Generating RRCSetup (bytes %d)\n",
        PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
        ue_p->Srb0.Tx_buffer.payload_size);
    // activate release timer, if RRCSetupComplete not received after 100 frames, remove UE
    ue_context_pP->ue_context.ue_release_timer = 1;
    // remove UE after 10 frames after RRCConnectionRelease is triggered
    ue_context_pP->ue_context.ue_release_timer_thres = 1000;
    /* init timers */
    //   ue_context_pP->ue_context.ue_rrc_inactivity_timer = 0;
#ifdef ITTI_SIM
    MessageDef *message_p;
    uint8_t *message_buffer;
    message_buffer = itti_malloc (TASK_RRC_GNB, TASK_RRC_UE_SIM,
              ue_p->Srb0.Tx_buffer.payload_size);
    memcpy (message_buffer, (uint8_t*)ue_p->Srb0.Tx_buffer.Payload, ue_p->Srb0.Tx_buffer.payload_size);
    message_p = itti_alloc_new_message (TASK_RRC_GNB, 0, GNB_RRC_CCCH_DATA_IND);
    GNB_RRC_CCCH_DATA_IND (message_p).sdu = message_buffer;
    GNB_RRC_CCCH_DATA_IND (message_p).size  = ue_p->Srb0.Tx_buffer.payload_size;
    itti_send_msg_to_task (TASK_RRC_UE_SIM, ctxt_pP->instance, message_p);
#endif
}

void
rrc_gNB_generate_RRCReject(
    const protocol_ctxt_t    *const ctxt_pP,
    rrc_gNB_ue_context_t     *const ue_context_pP,
    const int                CC_id
)
//-----------------------------------------------------------------------------
{
  LOG_I(NR_RRC, "rrc_gNB_generate_RRCReject \n");
  gNB_RRC_UE_t *ue_p = &ue_context_pP->ue_context;
  ue_p->Srb0.Tx_buffer.payload_size = do_RRCReject(ctxt_pP->module_id,
                                                  (uint8_t *)ue_p->Srb0.Tx_buffer.Payload);
  LOG_DUMPMSG(NR_RRC, DEBUG_RRC,
              (char *)(ue_p->Srb0.Tx_buffer.Payload),
              ue_p->Srb0.Tx_buffer.payload_size,
              "[MSG] RRCReject \n");
  MSC_LOG_TX_MESSAGE(MSC_RRC_GNB,
                    MSC_RRC_UE,
                    ue_p->Srb0.Tx_buffer.Header,
                    ue_p->Srb0.Tx_buffer.payload_size,
                    MSC_AS_TIME_FMT" NR_RRCReject UE %x size %u",
                    MSC_AS_TIME_ARGS(ctxt_pP),
                    ue_context_pP == NULL ? -1 : ue_context_pP->ue_context.rnti,
                    ue_p->Srb0.Tx_buffer.payload_size);
  LOG_I(NR_RRC,
      PROTOCOL_NR_RRC_CTXT_UE_FMT" [RAPROC] Logical Channel DL-CCCH, Generating NR_RRCReject (bytes %d)\n",
      PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
      ue_p->Srb0.Tx_buffer.payload_size);

#ifdef ITTI_SIM
  MessageDef *message_p;
  uint8_t *message_buffer;
  message_buffer = itti_malloc (TASK_RRC_GNB, TASK_RRC_UE_SIM,
            ue_p->Srb0.Tx_buffer.payload_size);
  memcpy (message_buffer, (uint8_t*)ue_p->Srb0.Tx_buffer.Payload, ue_p->Srb0.Tx_buffer.payload_size);
  message_p = itti_alloc_new_message (TASK_RRC_GNB, 0, GNB_RRC_CCCH_DATA_IND);
  GNB_RRC_CCCH_DATA_IND (message_p).sdu = message_buffer;
  GNB_RRC_CCCH_DATA_IND (message_p).size  = ue_p->Srb0.Tx_buffer.payload_size;
  itti_send_msg_to_task (TASK_RRC_UE_SIM, ctxt_pP->instance, message_p);
#endif
}

//-----------------------------------------------------------------------------
/*
* Process the rrc setup complete message from UE (SRB1 Active)
*/
void
rrc_gNB_process_RRCSetupComplete(
  const protocol_ctxt_t     *const ctxt_pP,
  rrc_gNB_ue_context_t      *ue_context_pP,
  NR_RRCSetupComplete_IEs_t *rrcSetupComplete
)
//-----------------------------------------------------------------------------
{
  LOG_I(NR_RRC, PROTOCOL_NR_RRC_CTXT_UE_FMT" [RAPROC] Logical Channel UL-DCCH, " "processing NR_RRCSetupComplete from UE (SRB1 Active)\n",
      PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP));
  ue_context_pP->ue_context.Srb1.Active = 1;
  ue_context_pP->ue_context.Status = NR_RRC_CONNECTED;

  if (AMF_MODE_ENABLED) {
    rrc_gNB_send_NGAP_NAS_FIRST_REQ(ctxt_pP, ue_context_pP, rrcSetupComplete);
  } else {
    rrc_gNB_generate_SecurityModeCommand(ctxt_pP, ue_context_pP);
  }
}

//-----------------------------------------------------------------------------
void 
rrc_gNB_generate_defaultRRCReconfiguration(
  const protocol_ctxt_t     *const ctxt_pP,
  rrc_gNB_ue_context_t      *ue_context_pP
)
//-----------------------------------------------------------------------------
{
  uint8_t                       buffer[RRC_BUF_SIZE];
  uint16_t                      size;
  NR_SRB_ToAddModList_t        **SRB_configList2 = NULL;
  NR_SRB_ToAddModList_t        *SRB_configList  = ue_context_pP->ue_context.SRB_configList;
  NR_DRB_ToAddModList_t        **DRB_configList  = NULL;
  NR_DRB_ToAddModList_t        **DRB_configList2 = NULL;
  NR_SRB_ToAddMod_t            *SRB2_config     = NULL;
  NR_DRB_ToAddMod_t            *DRB_config      = NULL;
  NR_SDAP_Config_t             *sdap_config     = NULL;
  struct NR_RRCReconfiguration_v1530_IEs__dedicatedNAS_MessageList
                               *dedicatedNAS_MessageList = NULL;
  NR_DedicatedNAS_Message_t    *dedicatedNAS_Message = NULL;

  uint8_t xid = rrc_gNB_get_next_transaction_identifier(ctxt_pP->module_id);

  /******************** Radio Bearer Config ********************/
  /* Configure SRB2 */
  SRB_configList2 = &ue_context_pP->ue_context.SRB_configList2[xid];
  if (*SRB_configList2) {
    free(*SRB_configList2);
  }
  *SRB_configList2 = CALLOC(1, sizeof(**SRB_configList2));
  memset(*SRB_configList2, 0, sizeof(**SRB_configList2));
  SRB2_config = CALLOC(1, sizeof(*SRB2_config));
  SRB2_config->srb_Identity = 2;
  ASN_SEQUENCE_ADD(&(*SRB_configList2)->list, SRB2_config);
  ASN_SEQUENCE_ADD(&SRB_configList->list, SRB2_config);

  /* Configure DRB */
  DRB_configList = &ue_context_pP->ue_context.DRB_configList;
  if (*DRB_configList) {
      free(*DRB_configList);
  }
  *DRB_configList = CALLOC(1, sizeof(**DRB_configList));
  memset(*DRB_configList, 0, sizeof(**DRB_configList));

  DRB_configList2 = &ue_context_pP->ue_context.DRB_configList2[xid];
  if (*DRB_configList2) {
      free(*DRB_configList2);
  }
  *DRB_configList2 = CALLOC(1, sizeof(**DRB_configList2));
  memset(*DRB_configList2, 0, sizeof(**DRB_configList2));

  DRB_config = CALLOC(1, sizeof(*DRB_config));
  DRB_config->drb_Identity = 1;
  DRB_config->cnAssociation = CALLOC(1, sizeof(*DRB_config->cnAssociation));
  DRB_config->cnAssociation->present = NR_DRB_ToAddMod__cnAssociation_PR_sdap_Config;
  // TODO sdap_Config
  sdap_config = CALLOC(1, sizeof(NR_SDAP_Config_t));
  memset(sdap_config, 0, sizeof(NR_SDAP_Config_t));
  DRB_config->cnAssociation->choice.sdap_Config = sdap_config;
  // TODO pdcp_Config
  DRB_config->reestablishPDCP = NULL;
  DRB_config->recoverPDCP = NULL;
  DRB_config->pdcp_Config = calloc(1, sizeof(*DRB_config->pdcp_Config));
  DRB_config->pdcp_Config->drb = calloc(1,sizeof(*DRB_config->pdcp_Config->drb));
  DRB_config->pdcp_Config->drb->discardTimer = calloc(1, sizeof(*DRB_config->pdcp_Config->drb->discardTimer));
  *DRB_config->pdcp_Config->drb->discardTimer = NR_PDCP_Config__drb__discardTimer_infinity;
  DRB_config->pdcp_Config->drb->pdcp_SN_SizeUL = calloc(1, sizeof(*DRB_config->pdcp_Config->drb->pdcp_SN_SizeUL));
  *DRB_config->pdcp_Config->drb->pdcp_SN_SizeUL = NR_PDCP_Config__drb__pdcp_SN_SizeUL_len18bits;
  DRB_config->pdcp_Config->drb->pdcp_SN_SizeDL = calloc(1, sizeof(*DRB_config->pdcp_Config->drb->pdcp_SN_SizeDL));
  *DRB_config->pdcp_Config->drb->pdcp_SN_SizeDL = NR_PDCP_Config__drb__pdcp_SN_SizeDL_len18bits;
  DRB_config->pdcp_Config->drb->headerCompression.present = NR_PDCP_Config__drb__headerCompression_PR_notUsed;
  DRB_config->pdcp_Config->drb->headerCompression.choice.notUsed = 0;

  DRB_config->pdcp_Config->drb->integrityProtection = NULL;
  DRB_config->pdcp_Config->drb->statusReportRequired = NULL;
  DRB_config->pdcp_Config->drb->outOfOrderDelivery = NULL;
  DRB_config->pdcp_Config->moreThanOneRLC = NULL;

  DRB_config->pdcp_Config->t_Reordering = calloc(1, sizeof(*DRB_config->pdcp_Config->t_Reordering));
  *DRB_config->pdcp_Config->t_Reordering = NR_PDCP_Config__t_Reordering_ms0;
  DRB_config->pdcp_Config->ext1 = NULL;

  ASN_SEQUENCE_ADD(&(*DRB_configList)->list, DRB_config);
  ASN_SEQUENCE_ADD(&(*DRB_configList2)->list, DRB_config);

  dedicatedNAS_MessageList = CALLOC(1, sizeof(struct NR_RRCReconfiguration_v1530_IEs__dedicatedNAS_MessageList));

  if (ue_context_pP->ue_context.nas_pdu_flag == 1) {
    dedicatedNAS_Message = CALLOC(1, sizeof(NR_DedicatedNAS_Message_t));
    memset(dedicatedNAS_Message, 0, sizeof(OCTET_STRING_t));
    OCTET_STRING_fromBuf(dedicatedNAS_Message,
                          (char *)ue_context_pP->ue_context.nas_pdu.buffer,
                          ue_context_pP->ue_context.nas_pdu.length);
    ASN_SEQUENCE_ADD(&dedicatedNAS_MessageList->list, dedicatedNAS_Message);
  }

  /* If list is empty free the list and reset the address */
  if (dedicatedNAS_MessageList->list.count == 0) {
    free(dedicatedNAS_MessageList);
    dedicatedNAS_MessageList = NULL;
  }

  memset(buffer, 0, RRC_BUF_SIZE);
  size = do_RRCReconfiguration(ctxt_pP, buffer,
                                xid,
                                *SRB_configList2,
                                *DRB_configList,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                dedicatedNAS_MessageList,
                                NULL);

  free(ue_context_pP->ue_context.nas_pdu.buffer);

  LOG_DUMPMSG(NR_RRC, DEBUG_RRC,(char *)buffer, size, "[MSG] RRC Reconfiguration\n");
  LOG_I(NR_RRC, "[gNB %d] Frame %d, Logical Channel DL-DCCH, Generate NR_RRCReconfiguration (bytes %d, UE id %x)\n",
          ctxt_pP->module_id,
          ctxt_pP->frame,
          size,
          ue_context_pP->ue_context.rnti);
  LOG_D(NR_RRC, "[FRAME %05d][RRC_gNB][MOD %u][][--- PDCP_DATA_REQ/%d Bytes (rrcReconfiguration to UE %x MUI %d) --->][PDCP][MOD %u][RB %u]\n",
          ctxt_pP->frame,
          ctxt_pP->module_id,
          size,
          ue_context_pP->ue_context.rnti,
          rrc_gNB_mui,
          ctxt_pP->module_id,
          DCCH);
  MSC_LOG_TX_MESSAGE(MSC_RRC_GNB,
          MSC_RRC_UE,
          buffer,
          size,
          MSC_AS_TIME_FMT" NR_RRCReconfiguration UE %x MUI %d size %u",
          MSC_AS_TIME_ARGS(ctxt_pP),
          ue_context_pP->ue_context.rnti,
          rrc_gNB_mui,
          size);
#ifdef ITTI_SIM
    MessageDef *message_p;
    uint8_t *message_buffer;
    message_buffer = itti_malloc (TASK_RRC_GNB, TASK_RRC_UE_SIM, size);
    memcpy (message_buffer, buffer, size);
    message_p = itti_alloc_new_message (TASK_RRC_GNB, 0, GNB_RRC_DCCH_DATA_IND);
    GNB_RRC_DCCH_DATA_IND (message_p).rbid = DCCH;
    GNB_RRC_DCCH_DATA_IND (message_p).sdu = message_buffer;
    GNB_RRC_DCCH_DATA_IND (message_p).size	= size;
    itti_send_msg_to_task (TASK_RRC_UE_SIM, ctxt_pP->instance, message_p);
#else
  nr_rrc_data_req(ctxt_pP,
              DCCH,
              rrc_gNB_mui++,
              SDU_CONFIRM_NO,
              size,
              buffer,
              PDCP_TRANSMISSION_MODE_CONTROL);
#endif
  // rrc_pdcp_config_asn1_req
  // rrc_rlc_config_asn1_req
}

//-----------------------------------------------------------------------------
void
rrc_gNB_generate_dedicatedRRCReconfiguration_release(
    const protocol_ctxt_t   *const ctxt_pP,
    rrc_gNB_ue_context_t    *const ue_context_pP,
    uint8_t                  xid,
    uint32_t                 nas_length,
    uint8_t                 *nas_buffer)
//-----------------------------------------------------------------------------
{
  uint8_t                             buffer[RRC_BUF_SIZE];
  int                                 i;
  uint16_t                            size  = 0;
  NR_DRB_ToReleaseList_t             **DRB_Release_configList2 = NULL;
  NR_DRB_Identity_t                  *DRB_release;
  struct NR_RRCReconfiguration_v1530_IEs__dedicatedNAS_MessageList
                                     *dedicatedNAS_MessageList = NULL;
  NR_DedicatedNAS_Message_t          *dedicatedNAS_Message     = NULL;

  DRB_Release_configList2 = &ue_context_pP->ue_context.DRB_Release_configList2[xid];
  if (*DRB_Release_configList2) {
    free(*DRB_Release_configList2);
  }

  *DRB_Release_configList2 = CALLOC(1, sizeof(**DRB_Release_configList2));
  for(i = 0; i < NB_RB_MAX; i++) {
    if((ue_context_pP->ue_context.pdusession[i].status == PDU_SESSION_STATUS_TORELEASE) && ue_context_pP->ue_context.pdusession[i].xid == xid) {
      DRB_release = CALLOC(1, sizeof(NR_DRB_Identity_t));
      *DRB_release = i+1;
      ASN_SEQUENCE_ADD(&(*DRB_Release_configList2)->list, DRB_release);
    }
  }

  /* If list is empty free the list and reset the address */
  if (nas_length > 0) {
    dedicatedNAS_MessageList = CALLOC(1, sizeof(struct NR_RRCReconfiguration_v1530_IEs__dedicatedNAS_MessageList));
    dedicatedNAS_Message = CALLOC(1, sizeof(NR_DedicatedNAS_Message_t));
    memset(dedicatedNAS_Message, 0, sizeof(OCTET_STRING_t));
    OCTET_STRING_fromBuf(dedicatedNAS_Message,
                         (char *)nas_buffer,
                         nas_length);
    ASN_SEQUENCE_ADD(&dedicatedNAS_MessageList->list, dedicatedNAS_Message);
    LOG_I(NR_RRC,"add NAS info with size %d\n", nas_length);
  } else {
    LOG_W(NR_RRC,"dedlicated NAS list is empty\n");
  }

  memset(buffer, 0, RRC_BUF_SIZE);
  size = do_RRCReconfiguration(ctxt_pP, buffer, xid,
                               NULL,
                               NULL,
                               *DRB_Release_configList2,
                               NULL,
                               NULL,
                               NULL,
                               dedicatedNAS_MessageList,
                               NULL);

  ue_context_pP->ue_context.pdu_session_release_command_flag = 1;
  LOG_DUMPMSG(NR_RRC,DEBUG_RRC,(char *)buffer,size,
              "[MSG] RRC Reconfiguration\n");

  /* Free all NAS PDUs */
  if (nas_length > 0) {
    /* Free the NAS PDU buffer and invalidate it */
    free(nas_buffer);
  }

  LOG_I(NR_RRC,
        "[gNB %d] Frame %d, Logical Channel DL-DCCH, Generate NR_RRCReconfiguration (bytes %d, UE RNTI %x)\n",
        ctxt_pP->module_id, ctxt_pP->frame, size, ue_context_pP->ue_context.rnti);
  LOG_D(NR_RRC,
        "[FRAME %05d][RRC_gNB][MOD %u][][--- PDCP_DATA_REQ/%d Bytes (rrcReconfiguration to UE %x MUI %d) --->][PDCP][MOD %u][RB %u]\n",
        ctxt_pP->frame, ctxt_pP->module_id, size, ue_context_pP->ue_context.rnti, rrc_gNB_mui, ctxt_pP->module_id, DCCH);
  MSC_LOG_TX_MESSAGE(
    MSC_RRC_GNB,
    MSC_RRC_UE,
    buffer,
    size,
    MSC_AS_TIME_FMT" dedicated NR_RRCReconfiguration UE %x MUI %d size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti,
    rrc_gNB_mui,
    size);
#ifdef ITTI_SIM
  MessageDef *message_p;
  uint8_t *message_buffer;
  message_buffer = itti_malloc (TASK_RRC_GNB, TASK_RRC_UE_SIM, size);
  memcpy (message_buffer, buffer, size);
  message_p = itti_alloc_new_message (TASK_RRC_GNB, 0, GNB_RRC_DCCH_DATA_IND);
  GNB_RRC_DCCH_DATA_IND (message_p).rbid = DCCH;
  GNB_RRC_DCCH_DATA_IND (message_p).sdu = message_buffer;
  GNB_RRC_DCCH_DATA_IND (message_p).size	= size;
  itti_send_msg_to_task (TASK_RRC_UE_SIM, ctxt_pP->instance, message_p);
#else
  nr_rrc_data_req(
    ctxt_pP,
    DCCH,
    rrc_gNB_mui++,
    SDU_CONFIRM_NO,
    size,
    buffer,
    PDCP_TRANSMISSION_MODE_CONTROL);
#endif
}

//-----------------------------------------------------------------------------
/*
* Process the RRC Reconfiguration Complete from the UE
*/
void
rrc_gNB_process_RRCReconfigurationComplete(
    const protocol_ctxt_t *const ctxt_pP,
    rrc_gNB_ue_context_t  *ue_context_pP,
    const uint8_t xid
)
{
  int                                 drb_id;
  uint8_t                            *kRRCenc = NULL;
  uint8_t                            *kRRCint = NULL;
  uint8_t                            *kUPenc = NULL;
  NR_DRB_ToAddModList_t              *DRB_configList = ue_context_pP->ue_context.DRB_configList2[xid];
  NR_SRB_ToAddModList_t              *SRB_configList = ue_context_pP->ue_context.SRB_configList2[xid];
  NR_DRB_ToReleaseList_t             *DRB_Release_configList2 = ue_context_pP->ue_context.DRB_Release_configList2[xid];
  NR_DRB_Identity_t                  *drb_id_p      = NULL;
//  uint8_t                             nr_DRB2LCHAN[8];

  ue_context_pP->ue_context.ue_reestablishment_timer = 0;

#ifndef PHYSIM
  /* Derive the keys from kgnb */
  if (DRB_configList != NULL) {
      derive_key_up_enc(ue_context_pP->ue_context.ciphering_algorithm,
                      ue_context_pP->ue_context.kgnb,
                      &kUPenc);
  }

  derive_key_rrc_enc(ue_context_pP->ue_context.ciphering_algorithm,
                      ue_context_pP->ue_context.kgnb,
                      &kRRCenc);
  derive_key_rrc_int(ue_context_pP->ue_context.integrity_algorithm,
                      ue_context_pP->ue_context.kgnb,
                      &kRRCint);
#endif
  /* Refresh SRBs/DRBs */
  MSC_LOG_TX_MESSAGE(MSC_RRC_GNB, MSC_PDCP_ENB, NULL, 0, MSC_AS_TIME_FMT" CONFIG_REQ UE %x DRB (security unchanged)",
                   MSC_AS_TIME_ARGS(ctxt_pP),
                   ue_context_pP->ue_context.rnti);

#ifndef ITTI_SIM
  nr_rrc_pdcp_config_asn1_req(ctxt_pP,
                              SRB_configList, // NULL,
                              DRB_configList,
                              DRB_Release_configList2,
                              0xff, // already configured during the securitymodecommand
                              kRRCenc,
                              kRRCint,
                              kUPenc,
                              NULL,
                              NULL,
                              NULL,
                              NULL);
  /* Refresh SRBs/DRBs */
  nr_rrc_rlc_config_asn1_req(ctxt_pP,
                          SRB_configList, // NULL,
                          DRB_configList,
                          DRB_Release_configList2,
                          NULL,
                          NULL);
#endif

  /* Set the SRB active in UE context */
  if (SRB_configList != NULL) {
    for (int i = 0; (i < SRB_configList->list.count) && (i < 3); i++) {
      if (SRB_configList->list.array[i]->srb_Identity == 1) {
        ue_context_pP->ue_context.Srb1.Active = 1;
      } else if (SRB_configList->list.array[i]->srb_Identity == 2) {
        ue_context_pP->ue_context.Srb2.Active = 1;
        ue_context_pP->ue_context.Srb2.Srb_info.Srb_id = 2;
        LOG_I(NR_RRC,"[gNB %d] Frame      %d CC %d : SRB2 is now active\n",
              ctxt_pP->module_id,
              ctxt_pP->frame,
              ue_context_pP->ue_context.primaryCC_id);
      } else {
        LOG_W(NR_RRC,"[gNB %d] Frame      %d CC %d : invalide SRB identity %ld\n",
              ctxt_pP->module_id,
              ctxt_pP->frame,
              ue_context_pP->ue_context.primaryCC_id,
              SRB_configList->list.array[i]->srb_Identity);
      }
    }

    free(SRB_configList);
    ue_context_pP->ue_context.SRB_configList2[xid] = NULL;
  }

  /* Loop through DRBs and establish if necessary */
  if (DRB_configList != NULL) {
    for (int i = 0; i < DRB_configList->list.count; i++) {
      if (DRB_configList->list.array[i]) {
        drb_id = (int)DRB_configList->list.array[i]->drb_Identity;
        LOG_I(NR_RRC, "[gNB %d] Frame  %d : Logical Channel UL-DCCH, Received NR_RRCReconfigurationComplete from UE rnti %x, reconfiguring DRB %d\n",
            ctxt_pP->module_id,
            ctxt_pP->frame,
            ctxt_pP->rnti,
            (int)DRB_configList->list.array[i]->drb_Identity);
            //(int)*DRB_configList->list.array[i]->pdcp_Config->moreThanOneRLC->primaryPath.logicalChannel);

        if (ue_context_pP->ue_context.DRB_active[drb_id] == 0) {
          ue_context_pP->ue_context.DRB_active[drb_id] = 1;
          LOG_D(NR_RRC, "[gNB %d] Frame %d: Establish RLC UM Bidirectional, DRB %d Active\n",
                  ctxt_pP->module_id, ctxt_pP->frame, (int)DRB_configList->list.array[i]->drb_Identity);

          LOG_D(NR_RRC,
                  PROTOCOL_NR_RRC_CTXT_UE_FMT" RRC_gNB --- MAC_CONFIG_REQ  (DRB) ---> MAC_gNB\n",
                  PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP));

          //if (DRB_configList->list.array[i]->pdcp_Config->moreThanOneRLC->primaryPath.logicalChannel) {
          //  nr_DRB2LCHAN[i] = (uint8_t) * DRB_configList->list.array[i]->pdcp_Config->moreThanOneRLC->primaryPath.logicalChannel;
          //}

            // rrc_mac_config_req_eNB
        } else {        // remove LCHAN from MAC/PHY
          if (ue_context_pP->ue_context.DRB_active[drb_id] == 1) {
            /* TODO : It may be needed if gNB goes into full stack working. */
            // DRB has just been removed so remove RLC + PDCP for DRB
            /*      rrc_pdcp_config_req (ctxt_pP->module_id, frameP, 1, CONFIG_ACTION_REMOVE,
            (ue_mod_idP * NB_RB_MAX) + DRB2LCHAN[i],UNDEF_SECURITY_MODE);
            */
            /*rrc_rlc_config_req(ctxt_pP,
                                SRB_FLAG_NO,
                                MBMS_FLAG_NO,
                                CONFIG_ACTION_REMOVE,
                                nr_DRB2LCHAN[i],
                                Rlc_info_um);*/
          }

          ue_context_pP->ue_context.DRB_active[drb_id] = 0;
          LOG_D(NR_RRC, PROTOCOL_NR_RRC_CTXT_UE_FMT" RRC_eNB --- MAC_CONFIG_REQ  (DRB) ---> MAC_eNB\n",
                  PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP));

          // rrc_mac_config_req_eNB

        } // end else of if (ue_context_pP->ue_context.DRB_active[drb_id] == 0)
      } // end if (DRB_configList->list.array[i])
    } // end for (int i = 0; i < DRB_configList->list.count; i++)

    free(DRB_configList);
    ue_context_pP->ue_context.DRB_configList2[xid] = NULL;
  } // end if DRB_configList != NULL

  if(DRB_Release_configList2 != NULL) {
    for (int i = 0; i < DRB_Release_configList2->list.count; i++) {
      if (DRB_Release_configList2->list.array[i]) {
        drb_id_p = DRB_Release_configList2->list.array[i];
        drb_id = *drb_id_p;

        if (ue_context_pP->ue_context.DRB_active[drb_id] == 1) {
        ue_context_pP->ue_context.DRB_active[drb_id] = 0;
        }
      }
    }

    free(DRB_Release_configList2);
    ue_context_pP->ue_context.DRB_Release_configList2[xid] = NULL;
  }
}

//-----------------------------------------------------------------------------
void
rrc_gNB_generate_RRCReestablishment(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_gNB_ue_context_t  *const ue_context_pP,
  const int             CC_id)
//-----------------------------------------------------------------------------
{
    // int UE_id = -1;
    //NR_LogicalChannelConfig_t  *SRB1_logicalChannelConfig = NULL;
    NR_SRB_ToAddModList_t      **SRB_configList;
    // NR_SRB_ToAddMod_t          *SRB1_config = NULL;
    //rrc_gNB_carrier_data_t     *carrier = NULL;
    gNB_RRC_UE_t               *ue_context = NULL;
    module_id_t                 module_id = ctxt_pP->module_id;
    // uint16_t                    rnti = ctxt_pP->rnti;
    
    SRB_configList = &(ue_context_pP->ue_context.SRB_configList);
    //carrier = &(RC.nrrrc[ctxt_pP->module_id]->carrier);
    ue_context = &(ue_context_pP->ue_context);
    ue_context->Srb0.Tx_buffer.payload_size = do_RRCReestablishment(ctxt_pP,
        ue_context_pP,
        CC_id,
        (uint8_t *) ue_context->Srb0.Tx_buffer.Payload,
        //(uint8_t) carrier->p_gNB, // at this point we do not have the UE capability information, so it can only be TM1 or TM2
        rrc_gNB_get_next_transaction_identifier(module_id),
        SRB_configList
        //&(ue_context->physicalConfigDedicated)
        );

    /* Configure SRB1 for UE */
    if (*SRB_configList != NULL) {
      for (int cnt = 0; cnt < (*SRB_configList)->list.count; cnt++) {
        if ((*SRB_configList)->list.array[cnt]->srb_Identity == 1) {
          // SRB1_config = (*SRB_configList)->list.array[cnt];
        }

        LOG_D(NR_RRC, PROTOCOL_NR_RRC_CTXT_UE_FMT" RRC_gNB --- MAC_CONFIG_REQ  (SRB1) ---> MAC_gNB\n",
              PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP));

        // rrc_mac_config_req_eNB
      }
    }  // if (*SRB_configList != NULL)
    
    MSC_LOG_TX_MESSAGE(MSC_RRC_GNB,
                       MSC_RRC_UE,
                       ue_context->Srb0.Tx_buffer.Header,
                       ue_context->Srb0.Tx_buffer.payload_size,
                       MSC_AS_TIME_FMT" NR_RRCReestablishment UE %x size %u",
                       MSC_AS_TIME_ARGS(ctxt_pP),
                       ue_context->rnti,
                       ue_context->Srb0.Tx_buffer.payload_size);
    LOG_I(NR_RRC, PROTOCOL_NR_RRC_CTXT_UE_FMT" [RAPROC] Logical Channel DL-DCCH, Generating NR_RRCReestablishment (bytes %d)\n",
          PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
          ue_context->Srb0.Tx_buffer.payload_size);
#if(0)
    /* TODO : It may be needed if gNB goes into full stack working. */
    UE_id = find_nr_UE_id(module_id, rnti);
    if (UE_id != -1) {
      /* Activate reject timer, if RRCComplete not received after 10 frames, reject UE */
      RC.nrmac[module_id]->UE_info.UE_sched_ctrl[UE_id].ue_reestablishment_reject_timer = 1;
      /* Reject UE after 10 frames, LTE_RRCConnectionReestablishmentReject is triggered */
      RC.nrmac[module_id]->UE_info.UE_sched_ctrl[UE_id].ue_reestablishment_reject_timer_thres = 100;
    } else {
      LOG_E(NR_RRC, PROTOCOL_NR_RRC_CTXT_UE_FMT" Generating NR_RRCReestablishment without UE_id(MAC) rnti %x\n",
            PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
            rnti);
    }
#endif
#ifdef ITTI_SIM
        MessageDef *message_p;
        uint8_t *message_buffer;
        message_buffer = itti_malloc (TASK_RRC_GNB, TASK_RRC_UE_SIM, ue_context->Srb0.Tx_buffer.payload_size);
        memcpy (message_buffer, (uint8_t *) ue_context->Srb0.Tx_buffer.Payload, ue_context->Srb0.Tx_buffer.payload_size);
        message_p = itti_alloc_new_message (TASK_RRC_GNB, 0, GNB_RRC_DCCH_DATA_IND);
        GNB_RRC_DCCH_DATA_IND (message_p).rbid = DCCH;
        GNB_RRC_DCCH_DATA_IND (message_p).sdu = message_buffer;
        GNB_RRC_DCCH_DATA_IND (message_p).size  = ue_context->Srb0.Tx_buffer.payload_size;
        itti_send_msg_to_task (TASK_RRC_UE_SIM, ctxt_pP->instance, message_p);
#endif

}

//-----------------------------------------------------------------------------
void
rrc_gNB_process_RRCConnectionReestablishmentComplete(
  const protocol_ctxt_t *const ctxt_pP,
  const rnti_t reestablish_rnti,
  rrc_gNB_ue_context_t         *ue_context_pP,
  const uint8_t xid
)
//-----------------------------------------------------------------------------
{
  LOG_I(NR_RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" [RAPROC] Logical Channel UL-DCCH, processing NR_RRCConnectionReestablishmentComplete from UE (SRB1 Active)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));

  NR_DRB_ToAddModList_t                 *DRB_configList = ue_context_pP->ue_context.DRB_configList;
  NR_SRB_ToAddModList_t                 *SRB_configList = ue_context_pP->ue_context.SRB_configList;
  NR_SRB_ToAddModList_t                **SRB_configList2 = NULL;
  NR_DRB_ToAddModList_t                **DRB_configList2 = NULL;
  NR_SRB_ToAddMod_t                     *SRB2_config     = NULL;
  NR_DRB_ToAddMod_t                     *DRB_config      = NULL;
  //NR_SDAP_Config_t                      *sdap_config     = NULL;
  int i = 0;
  uint8_t                             buffer[RRC_BUF_SIZE];
  uint16_t                            size;

  uint8_t next_xid = rrc_gNB_get_next_transaction_identifier(ctxt_pP->module_id);
  int ret = 0;
  ue_context_pP->ue_context.Status = NR_RRC_CONNECTED;
  ue_context_pP->ue_context.ue_rrc_inactivity_timer = 1; // set rrc inactivity when UE goes into RRC_CONNECTED
  ue_context_pP->ue_context.reestablishment_xid = next_xid;
  SRB_configList2 = &ue_context_pP->ue_context.SRB_configList2[xid];

  // get old configuration of SRB2
  if (*SRB_configList2 != NULL) {
    if((*SRB_configList2)->list.count!=0) {
      LOG_D(NR_RRC, "SRB_configList2(%p) count is %d\n           SRB_configList2->list.array[0] addr is %p",
            SRB_configList2, (*SRB_configList2)->list.count,  (*SRB_configList2)->list.array[0]);
    }

    for (i = 0; (i < (*SRB_configList2)->list.count) && (i < 3); i++) {
      if ((*SRB_configList2)->list.array[i]->srb_Identity == 2 ) {
        LOG_D(NR_RRC, "get SRB2_config from (ue_context_pP->ue_context.SRB_configList2[%d])\n", xid);
        SRB2_config = (*SRB_configList2)->list.array[i];
        break;
      }
    }
  }

  // SRB2_config = CALLOC(1, sizeof(*SRB2_config));
  // SRB2_config->srb_Identity = 2;

  SRB_configList2 = &(ue_context_pP->ue_context.SRB_configList2[next_xid]);
  DRB_configList2 = &(ue_context_pP->ue_context.DRB_configList2[next_xid]);

  if (*SRB_configList2) {
    free(*SRB_configList2);
    LOG_D(NR_RRC, "free(ue_context_pP->ue_context.SRB_configList2[%d])\n", next_xid);
  }

  *SRB_configList2 = CALLOC(1, sizeof(**SRB_configList2));

  if (SRB2_config != NULL) {
    // Add SRB2 to SRB configuration list
    ASN_SEQUENCE_ADD(&SRB_configList->list, SRB2_config);
    ASN_SEQUENCE_ADD(&(*SRB_configList2)->list, SRB2_config);
    LOG_D(NR_RRC, "Add SRB2_config (srb_Identity:%ld) to ue_context_pP->ue_context.SRB_configList\n",
          SRB2_config->srb_Identity);
    LOG_D(NR_RRC, "Add SRB2_config (srb_Identity:%ld) to ue_context_pP->ue_context.SRB_configList2[%d]\n",
          SRB2_config->srb_Identity, next_xid);
  } else {
    // SRB configuration list only contains SRB1.
    LOG_W(NR_RRC,"SRB2 configuration does not exist in SRB configuration list\n");
  }

  if (*DRB_configList2) {
    free(*DRB_configList2);
    LOG_D(NR_RRC, "free(ue_context_pP->ue_context.DRB_configList2[%d])\n", next_xid);
  }

  *DRB_configList2 = CALLOC(1, sizeof(**DRB_configList2));

  if (DRB_configList != NULL) {
    LOG_D(NR_RRC, "get DRB_config from (ue_context_pP->ue_context.DRB_configList)\n");

    for (i = 0; (i < DRB_configList->list.count) && (i < 3); i++) {
      DRB_config = DRB_configList->list.array[i];
      // Add DRB to DRB configuration list, for LTE_RRCConnectionReconfigurationComplete
      ASN_SEQUENCE_ADD(&(*DRB_configList2)->list, DRB_config);
    }
  }

  ue_context_pP->ue_context.Srb1.Active = 1;
  //ue_context_pP->ue_context.Srb2.Srb_info.Srb_id = 2;

  if (AMF_MODE_ENABLED) {
    hashtable_rc_t    h_rc;
    int               j;
    rrc_ue_ngap_ids_t *rrc_ue_ngap_ids_p = NULL;
    uint16_t ue_initial_id = ue_context_pP->ue_context.ue_initial_id;
    uint32_t gNB_ue_ngap_id = ue_context_pP->ue_context.gNB_ue_ngap_id;
    gNB_RRC_INST *rrc_instance_p = RC.nrrrc[GNB_INSTANCE_TO_MODULE_ID(ctxt_pP->instance)];

    if (gNB_ue_ngap_id > 0) {
      h_rc = hashtable_get(rrc_instance_p->ngap_id2_ngap_ids, (hash_key_t)gNB_ue_ngap_id, (void **)&rrc_ue_ngap_ids_p);

      if  (h_rc == HASH_TABLE_OK) {
    	  rrc_ue_ngap_ids_p->ue_rnti = ctxt_pP->rnti;
      }
    }

    if (ue_initial_id != 0) {
      h_rc = hashtable_get(rrc_instance_p->initial_id2_ngap_ids, (hash_key_t)ue_initial_id, (void **)&rrc_ue_ngap_ids_p);

      if  (h_rc == HASH_TABLE_OK) {
    	  rrc_ue_ngap_ids_p->ue_rnti = ctxt_pP->rnti;
      }
    }

    gtpv1u_gnb_create_tunnel_req_t  create_tunnel_req;
    /* Save e RAB information for later */
    memset(&create_tunnel_req, 0, sizeof(create_tunnel_req));

    for ( j = 0, i = 0; i < NB_RB_MAX; i++) {
      if (ue_context_pP->ue_context.pdusession[i].status == PDU_SESSION_STATUS_ESTABLISHED || ue_context_pP->ue_context.pdusession[i].status == PDU_SESSION_STATUS_DONE) {
        create_tunnel_req.pdusession_id[j]   = ue_context_pP->ue_context.pdusession[i].param.pdusession_id;
        create_tunnel_req.upf_NGu_teid[j]  = ue_context_pP->ue_context.pdusession[i].param.gtp_teid;
        memcpy(create_tunnel_req.upf_addr[j].buffer,
               ue_context_pP->ue_context.pdusession[i].param.upf_addr.buffer,
                sizeof(uint8_t)*20);
        create_tunnel_req.upf_addr[j].length = ue_context_pP->ue_context.pdusession[i].param.upf_addr.length;
        j++;
      }
    }

    create_tunnel_req.rnti       = ctxt_pP->rnti; // warning put zero above
    create_tunnel_req.num_tunnels    = j;
    ret = gtpv1u_update_ngu_tunnel(
            ctxt_pP->instance,
            &create_tunnel_req,
            reestablish_rnti);

    if ( ret != 0 ) {
      LOG_E(NR_RRC,"gtpv1u_update_ngu_tunnel failed,start to release UE %x\n",reestablish_rnti);

      // update s1u tunnel failed,reset rnti?
      if (gNB_ue_ngap_id > 0) {
        h_rc = hashtable_get(rrc_instance_p->ngap_id2_ngap_ids, (hash_key_t)gNB_ue_ngap_id, (void **)&rrc_ue_ngap_ids_p);

        if (h_rc == HASH_TABLE_OK ) {
        	rrc_ue_ngap_ids_p->ue_rnti = reestablish_rnti;
        }
      }

      if (ue_initial_id != 0) {
        h_rc = hashtable_get(rrc_instance_p->initial_id2_ngap_ids, (hash_key_t)ue_initial_id, (void **)&rrc_ue_ngap_ids_p);

        if (h_rc == HASH_TABLE_OK ) {
          rrc_ue_ngap_ids_p->ue_rnti = reestablish_rnti;
        }
      }

      ue_context_pP->ue_context.ue_release_timer_s1 = 1;
      ue_context_pP->ue_context.ue_release_timer_thres_s1 = 100;
      ue_context_pP->ue_context.ue_release_timer = 0;
      ue_context_pP->ue_context.ue_reestablishment_timer = 0;
      ue_context_pP->ue_context.ul_failure_timer = 20000; // set ul_failure to 20000 for triggering rrc_eNB_send_S1AP_UE_CONTEXT_RELEASE_REQ
      ue_context_pP->ue_context.ul_failure_timer = 0;
      return;
    }
  } /* AMF_MODE_ENABLED */

  /* Update RNTI in ue_context */
  ue_context_pP->ue_id_rnti                    = ctxt_pP->rnti; // here ue_id_rnti is just a key, may be something else
  ue_context_pP->ue_context.rnti               = ctxt_pP->rnti;

  if (AMF_MODE_ENABLED) {
    uint8_t send_security_mode_command = FALSE;
    nr_rrc_pdcp_config_security(
      ctxt_pP,
      ue_context_pP,
      send_security_mode_command);
    LOG_D(NR_RRC, "set security successfully \n");
  }

  /* Add all NAS PDUs to the list */
  for (i = 0; i < ue_context_pP->ue_context.nb_of_pdusessions; i++) {

    /* TODO parameters yet to process ... */
    /* TODO should test if pdu session are Ok before! */
    ue_context_pP->ue_context.pdusession[i].status = PDU_SESSION_STATUS_DONE;
    ue_context_pP->ue_context.pdusession[i].xid    = xid;
    LOG_D(NR_RRC, "setting the status for the default DRB (index %d) to (%d,%s)\n",
          i, ue_context_pP->ue_context.pdusession[i].status, "PDU_SESSION_STATUS_DONE");
  }

  memset(buffer, 0, RRC_BUF_SIZE);

  size = do_RRCReconfiguration(ctxt_pP, buffer,
                                xid,
                               *SRB_configList2,
                                DRB_configList,
                                NULL,
                                NULL,
                                NULL,
                                NULL, // MeasObj_list,
                                NULL,
                                NULL);
  LOG_DUMPMSG(NR_RRC,DEBUG_RRC,(char *)buffer,size,
              "[MSG] RRC Reconfiguration\n");

  /* Free all NAS PDUs */
  for (i = 0; i < ue_context_pP->ue_context.nb_of_pdusessions; i++) {
    if (ue_context_pP->ue_context.pdusession[i].param.nas_pdu.buffer != NULL) {
      /* Free the NAS PDU buffer and invalidate it */
      free(ue_context_pP->ue_context.pdusession[i].param.nas_pdu.buffer);
      ue_context_pP->ue_context.pdusession[i].param.nas_pdu.buffer = NULL;
    }
  }

  if(size==65535) {
    LOG_E(NR_RRC,"RRC decode err!!! do_RRCReconfiguration\n");
    return;
  } else {
    LOG_I(NR_RRC,
          "[gNB %d] Frame %d, Logical Channel DL-DCCH, Generate NR_RRCConnectionReconfiguration (bytes %d, UE id %x)\n",
          ctxt_pP->module_id, ctxt_pP->frame, size, ue_context_pP->ue_context.rnti);
    LOG_D(NR_RRC,
          "[FRAME %05d][RRC_gNB][MOD %u][][--- PDCP_DATA_REQ/%d Bytes (rrcConnectionReconfiguration to UE %x MUI %d) --->][PDCP][MOD %u][RB %u]\n",
          ctxt_pP->frame, ctxt_pP->module_id, size, ue_context_pP->ue_context.rnti, rrc_gNB_mui, ctxt_pP->module_id, DCCH);
    MSC_LOG_TX_MESSAGE(
      MSC_RRC_GNB,
      MSC_RRC_UE,
      buffer,
      size,
      MSC_AS_TIME_FMT" LTE_RRCConnectionReconfiguration UE %x MUI %d size %u",
      MSC_AS_TIME_ARGS(ctxt_pP),
      ue_context_pP->ue_context.rnti,
      rrc_gNB_mui,
      size);
#ifdef ITTI_SIM
    MessageDef *message_p;
    uint8_t *message_buffer;
    message_buffer = itti_malloc (TASK_RRC_GNB, TASK_RRC_UE_SIM, size);
    memcpy (message_buffer, buffer, size);
    message_p = itti_alloc_new_message (TASK_RRC_GNB, 0, GNB_RRC_DCCH_DATA_IND);
    GNB_RRC_DCCH_DATA_IND (message_p).rbid = DCCH;
    GNB_RRC_DCCH_DATA_IND (message_p).sdu = message_buffer;
    GNB_RRC_DCCH_DATA_IND (message_p).size	= size;
    itti_send_msg_to_task (TASK_RRC_UE_SIM, ctxt_pP->instance, message_p);
#else
    nr_rrc_data_req(
      ctxt_pP,
      DCCH,
      rrc_gNB_mui++,
      SDU_CONFIRM_NO,
      size,
      buffer,
      PDCP_TRANSMISSION_MODE_CONTROL);
#endif
  }
}
//-----------------------------------------------------------------------------

/*------------------------------------------------------------------------------*/
int nr_rrc_gNB_decode_ccch(protocol_ctxt_t    *const ctxt_pP,
                           const uint8_t      *buffer,
                           int                buffer_length,
                           const int          CC_id)
{
  module_id_t                                       Idx;
  asn_dec_rval_t                                    dec_rval;
  NR_UL_CCCH_Message_t                             *ul_ccch_msg = NULL;
  struct rrc_gNB_ue_context_s                      *ue_context_p = NULL;
  gNB_RRC_INST                                     *gnb_rrc_inst = RC.nrrrc[ctxt_pP->module_id];
  NR_RRCSetupRequest_IEs_t                         *rrcSetupRequest = NULL;
  NR_RRCReestablishmentRequest_IEs_t                rrcReestablishmentRequest;
  uint64_t                                          random_value = 0;
  int i;

    dec_rval = uper_decode( NULL,
                            &asn_DEF_NR_UL_CCCH_Message,
                            (void **)&ul_ccch_msg,
                            (uint8_t *) buffer,
                            100,
                            0,
                            0);

    if (dec_rval.consumed == 0) {
        /* TODO */
        LOG_E(NR_RRC, PROTOCOL_NR_RRC_CTXT_UE_FMT" FATAL Error in receiving CCCH\n",
                   PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP));
        return -1;
    }

    if (ul_ccch_msg->message.present == NR_UL_CCCH_MessageType_PR_c1) {
     switch (ul_ccch_msg->message.choice.c1->present) {
      case NR_UL_CCCH_MessageType__c1_PR_NOTHING:
        /* TODO */
        LOG_I(NR_RRC,
            PROTOCOL_NR_RRC_CTXT_FMT" Received PR_NOTHING on UL-CCCH-Message\n",
            PROTOCOL_NR_RRC_CTXT_ARGS(ctxt_pP));
        break;

      case NR_UL_CCCH_MessageType__c1_PR_rrcSetupRequest:
        LOG_I(NR_RRC, "Received RRCSetupRequest on UL-CCCH-Message (UE rnti %x)\n", ctxt_pP->rnti);
        ue_context_p = rrc_gNB_get_ue_context(gnb_rrc_inst, ctxt_pP->rnti);
        if (ue_context_p != NULL) {
          rrc_gNB_free_mem_UE_context(ctxt_pP, ue_context_p);
          MSC_LOG_RX_DISCARDED_MESSAGE(
              MSC_RRC_GNB,
              MSC_RRC_UE,
              buffer,
              dec_rval.consumed,
              MSC_AS_TIME_FMT" NR_RRCSetupRequest UE %x size %u (UE already in context)",
              MSC_AS_TIME_ARGS(ctxt_pP),
              ue_context_p->ue_context.rnti,
              dec_rval.consumed);
        } else {
          rrcSetupRequest = &ul_ccch_msg->message.choice.c1->choice.rrcSetupRequest->rrcSetupRequest;
          if (NR_InitialUE_Identity_PR_randomValue == rrcSetupRequest->ue_Identity.present) {
            /* randomValue                         BIT STRING (SIZE (39)) */
            if (rrcSetupRequest->ue_Identity.choice.randomValue.size != 5) { // 39-bit random value
              LOG_E(NR_RRC, "wrong InitialUE-Identity randomValue size, expected 5, provided %lu",
                          (long unsigned int)rrcSetupRequest->ue_Identity.choice.randomValue.size);
              return -1;
            }

            memcpy(((uint8_t *) & random_value) + 3,
                    rrcSetupRequest->ue_Identity.choice.randomValue.buf,
                    rrcSetupRequest->ue_Identity.choice.randomValue.size);

            /* if there is already a registered UE (with another RNTI) with this random_value,
            * the current one must be removed from MAC/PHY (zombie UE)
            */
            if ((ue_context_p = rrc_gNB_ue_context_random_exist(RC.nrrrc[ctxt_pP->module_id], random_value))) {
              LOG_W(NR_RRC, "new UE rnti %x (coming with random value) is already there as UE %x, removing %x from MAC/PHY\n",
                      ctxt_pP->rnti, ue_context_p->ue_context.rnti, ue_context_p->ue_context.rnti);
              ue_context_p->ue_context.ul_failure_timer = 20000;
            }

            ue_context_p = rrc_gNB_get_next_free_ue_context(ctxt_pP, RC.nrrrc[ctxt_pP->module_id], random_value);
            ue_context_p->ue_context.Srb0.Srb_id = 0;
            ue_context_p->ue_context.Srb0.Active = 1;
            memcpy(ue_context_p->ue_context.Srb0.Rx_buffer.Payload,
                    buffer,
                    buffer_length);
            ue_context_p->ue_context.Srb0.Rx_buffer.payload_size = buffer_length;
          } else if (NR_InitialUE_Identity_PR_ng_5G_S_TMSI_Part1 == rrcSetupRequest->ue_Identity.present) {
            /* TODO */
            /* <5G-S-TMSI> = <AMF Set ID><AMF Pointer><5G-TMSI> 48-bit */
            /* ng-5G-S-TMSI-Part1                  BIT STRING (SIZE (39)) */
            if (rrcSetupRequest->ue_Identity.choice.ng_5G_S_TMSI_Part1.size != 5) {
              LOG_E(NR_RRC, "wrong ng_5G_S_TMSI_Part1 size, expected 5, provided %lu \n",
                          (long unsigned int)rrcSetupRequest->ue_Identity.choice.ng_5G_S_TMSI_Part1.size);
              return -1;
            }

            uint64_t s_tmsi_part1 = bitStr_to_uint64(&rrcSetupRequest->ue_Identity.choice.ng_5G_S_TMSI_Part1);

            // memcpy(((uint8_t *) & random_value) + 3,
            //         rrcSetupRequest->ue_Identity.choice.ng_5G_S_TMSI_Part1.buf,
            //         rrcSetupRequest->ue_Identity.choice.ng_5G_S_TMSI_Part1.size);

            if ((ue_context_p = rrc_gNB_ue_context_5g_s_tmsi_exist(RC.nrrrc[ctxt_pP->module_id], s_tmsi_part1))) {
              LOG_I(NR_RRC, " 5G-S-TMSI-Part1 exists, ue_context_p %p, old rnti %x => %x\n",ue_context_p, ue_context_p->ue_context.rnti, ctxt_pP->rnti);

              nr_rrc_mac_remove_ue(ctxt_pP->module_id, ue_context_p->ue_context.rnti);

              /* replace rnti in the context */
              /* for that, remove the context from the RB tree */
              RB_REMOVE(rrc_nr_ue_tree_s, &RC.nrrrc[ctxt_pP->module_id]->rrc_ue_head, ue_context_p);
              /* and insert again, after changing rnti everywhere it has to be changed */
              ue_context_p->ue_id_rnti = ctxt_pP->rnti;
              ue_context_p->ue_context.rnti = ctxt_pP->rnti;
              RB_INSERT(rrc_nr_ue_tree_s, &RC.nrrrc[ctxt_pP->module_id]->rrc_ue_head, ue_context_p);
              /* reset timers */
              ue_context_p->ue_context.ul_failure_timer = 0;
              ue_context_p->ue_context.ue_release_timer = 0;
              ue_context_p->ue_context.ue_reestablishment_timer = 0;
              ue_context_p->ue_context.ue_release_timer_s1 = 0;
              ue_context_p->ue_context.ue_release_timer_rrc = 0;
            } else {
              LOG_I(NR_RRC, " 5G-S-TMSI-Part1 doesn't exist, setting ng_5G_S_TMSI_Part1 to %p => %ld\n",
                              ue_context_p, s_tmsi_part1);

              ue_context_p = rrc_gNB_get_next_free_ue_context(ctxt_pP, RC.nrrrc[ctxt_pP->module_id], s_tmsi_part1);

              if (ue_context_p == NULL) {
                  LOG_E(RRC, "%s:%d:%s: rrc_gNB_get_next_free_ue_context returned NULL\n", __FILE__, __LINE__, __FUNCTION__);
              }

       if (ue_context_p != NULL) {
                  ue_context_p->ue_context.Initialue_identity_5g_s_TMSI.presence = TRUE;
                  ue_context_p->ue_context.ng_5G_S_TMSI_Part1 = s_tmsi_part1;
              }
            }
          } else {
            /* TODO */
            memcpy(((uint8_t *) & random_value) + 3,
                    rrcSetupRequest->ue_Identity.choice.randomValue.buf,
                    rrcSetupRequest->ue_Identity.choice.randomValue.size);

            rrc_gNB_get_next_free_ue_context(ctxt_pP, RC.nrrrc[ctxt_pP->module_id], random_value);
            LOG_E(NR_RRC,
                    PROTOCOL_NR_RRC_CTXT_UE_FMT" RRCSetupRequest without random UE identity or S-TMSI not supported, let's reject the UE\n",
                    PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP));
            rrc_gNB_generate_RRCReject(ctxt_pP,
                                       rrc_gNB_get_ue_context(gnb_rrc_inst, ctxt_pP->rnti),
                                       CC_id);
            break;
          }
        }

        if (ue_context_p != NULL) {
          ue_context_p->ue_context.establishment_cause = rrcSetupRequest->establishmentCause;
        }

        rrc_gNB_generate_RRCSetup(ctxt_pP,
                                  rrc_gNB_get_ue_context(gnb_rrc_inst, ctxt_pP->rnti),
                                  CC_id);
        break;

      case NR_UL_CCCH_MessageType__c1_PR_rrcResumeRequest:
                LOG_I(NR_RRC, "receive rrcResumeRequest message \n");
        break;

      case NR_UL_CCCH_MessageType__c1_PR_rrcReestablishmentRequest:
        LOG_I(NR_RRC, "receive rrcReestablishmentRequest message \n");
        LOG_DUMPMSG(NR_RRC, DEBUG_RRC,(char *)(buffer), buffer_length,
              "[MSG] RRC Reestablishment Request\n");
        LOG_D(NR_RRC,
              PROTOCOL_NR_RRC_CTXT_UE_FMT"MAC_gNB--- MAC_DATA_IND (rrcReestablishmentRequest on SRB0) --> RRC_gNB\n",
              PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP));

        rrcReestablishmentRequest = ul_ccch_msg->message.choice.c1->choice.rrcReestablishmentRequest->rrcReestablishmentRequest;
        LOG_I(NR_RRC,
          PROTOCOL_NR_RRC_CTXT_UE_FMT" NR_RRCReestablishmentRequest cause %s\n",
          PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
          ((rrcReestablishmentRequest.reestablishmentCause == NR_ReestablishmentCause_otherFailure) ?   "Other Failure" :
          (rrcReestablishmentRequest.reestablishmentCause == NR_ReestablishmentCause_handoverFailure) ? "Handover Failure" :
          "reconfigurationFailure"));
        {
          uint16_t                          c_rnti = 0;
          if (rrcReestablishmentRequest.ue_Identity.physCellId != RC.nrrrc[ctxt_pP->module_id]->carrier.physCellId) {
            /* UE was moving from previous cell so quickly that RRCReestablishment for previous cell was recieved in this cell */
            LOG_E(NR_RRC,
                  PROTOCOL_NR_RRC_CTXT_UE_FMT" NR_RRCReestablishmentRequest ue_Identity.physCellId(%ld) is not equal to current physCellId(%d), fallback to RRC establishment\n",
                  PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
                  rrcReestablishmentRequest.ue_Identity.physCellId,
                  RC.nrrrc[ctxt_pP->module_id]->carrier.physCellId);
            rrc_gNB_generate_RRCSetup_for_RRCReestablishmentRequest(ctxt_pP, CC_id);
            break;
          }

          LOG_D(NR_RRC, "physCellId is %ld\n", rrcReestablishmentRequest.ue_Identity.physCellId);

          for (i = 0; i < rrcReestablishmentRequest.ue_Identity.shortMAC_I.size; i++) {
            LOG_D(NR_RRC, "rrcReestablishmentRequest.ue_Identity.shortMAC_I.buf[%d] = %x\n",
                  i, rrcReestablishmentRequest.ue_Identity.shortMAC_I.buf[i]);
          }

          if (rrcReestablishmentRequest.ue_Identity.c_RNTI < 0 ||
              rrcReestablishmentRequest.ue_Identity.c_RNTI > 65535) {
            /* c_RNTI range error should not happen */
            LOG_E(NR_RRC,
                  PROTOCOL_NR_RRC_CTXT_UE_FMT" NR_RRCReestablishmentRequest c_RNTI range error, fallback to RRC establishment\n",
                  PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP));
            rrc_gNB_generate_RRCSetup_for_RRCReestablishmentRequest(ctxt_pP, CC_id);
            break;
          }

          c_rnti = rrcReestablishmentRequest.ue_Identity.c_RNTI;
          LOG_I(NR_RRC, "c_rnti is %x\n", c_rnti);
          ue_context_p = rrc_gNB_get_ue_context(gnb_rrc_inst, c_rnti);

          if (ue_context_p == NULL) {
            LOG_E(NR_RRC,
                  PROTOCOL_NR_RRC_CTXT_UE_FMT" NR_RRCReestablishmentRequest without UE context, fallback to RRC establishment\n",
                  PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP));
            rrc_gNB_generate_RRCSetup_for_RRCReestablishmentRequest(ctxt_pP, CC_id);
            break;
          }
#if(0)
          /* TODO : It may be needed if gNB goes into full stack working. */
          int UE_id = find_nr_UE_id(ctxt_pP->module_id, c_rnti);

          if(UE_id == -1) {
            LOG_E(NR_RRC,
                  PROTOCOL_NR_RRC_CTXT_UE_FMT" NR_RRCReestablishmentRequest without UE_id(MAC) rnti %x, fallback to RRC establishment\n",
                  PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),c_rnti);
            rrc_gNB_generate_RRCSetup_for_RRCReestablishmentRequest(ctxt_pP, CC_id);
            break;
          }

          //previous rnti
          rnti_t previous_rnti = 0;

          for (i = 0; i < MAX_MOBILES_PER_ENB; i++) {
            if (reestablish_rnti_map[i][1] == c_rnti) {
              previous_rnti = reestablish_rnti_map[i][0];
              break;
            }
          }

          if(previous_rnti != 0) {
            UE_id = find_nr_UE_id(ctxt_pP->module_id, previous_rnti);

            if(UE_id == -1) {
              LOG_E(NR_RRC,
                    PROTOCOL_NR_RRC_CTXT_UE_FMT" RRCReestablishmentRequest without UE_id(MAC) previous rnti %x, fallback to RRC establishment\n",
                    PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),previous_rnti);
              rrc_gNB_generate_RRCSetup_for_RRCReestablishmentRequest(ctxt_pP, CC_id);
              break;
            }
          }
#endif
          //c-plane not end
          if((ue_context_p->ue_context.Status != NR_RRC_RECONFIGURED) && (ue_context_p->ue_context.reestablishment_cause == NR_ReestablishmentCause_spare1)) {
            LOG_E(NR_RRC,
                  PROTOCOL_NR_RRC_CTXT_UE_FMT" NR_RRCReestablishmentRequest (UE %x c-plane is not end), RRC establishment failed \n",
                  PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),c_rnti);
            /* TODO RRC Release ? */
            break;
          }

          if(ue_context_p->ue_context.ue_reestablishment_timer > 0) {
            LOG_E(NR_RRC,
                  PROTOCOL_NR_RRC_CTXT_UE_FMT" RRRCReconfigurationComplete(Previous) don't receive, delete the Previous UE,\nprevious Status %d, new Status NR_RRC_RECONFIGURED\n",
                  PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
                  ue_context_p->ue_context.Status
                  );
            ue_context_p->ue_context.Status = NR_RRC_RECONFIGURED;
            protocol_ctxt_t  ctxt_old_p;
            PROTOCOL_CTXT_SET_BY_INSTANCE(&ctxt_old_p,
                                          ctxt_pP->instance,
                                          GNB_FLAG_YES,
                                          c_rnti,
                                          ctxt_pP->frame,
                                          ctxt_pP->subframe);
            rrc_gNB_process_RRCReconfigurationComplete(&ctxt_old_p,
                ue_context_p,
                ue_context_p->ue_context.reestablishment_xid);

            for (uint8_t pdusessionid = 0; pdusessionid < ue_context_p->ue_context.nb_of_pdusessions; pdusessionid++) {
              if (ue_context_p->ue_context.pdusession[pdusessionid].status == PDU_SESSION_STATUS_DONE) {
                ue_context_p->ue_context.pdusession[pdusessionid].status = PDU_SESSION_STATUS_ESTABLISHED;
              } else {
                ue_context_p->ue_context.pdusession[pdusessionid].status = PDU_SESSION_STATUS_FAILED;
              }
            }
          }

          LOG_D(NR_RRC,
                PROTOCOL_NR_RRC_CTXT_UE_FMT" UE context: %p\n",
                PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
                ue_context_p);
          /* reset timers */
          ue_context_p->ue_context.ul_failure_timer = 0;
          ue_context_p->ue_context.ue_release_timer = 0;
          ue_context_p->ue_context.ue_reestablishment_timer = 0;
          // ue_context_p->ue_context.ue_release_timer_s1 = 0;
          ue_context_p->ue_context.ue_release_timer_rrc = 0;
          ue_context_p->ue_context.reestablishment_xid = -1;

          // insert C-RNTI to map
          for (i = 0; i < MAX_MOBILES_PER_ENB; i++) {
            if (reestablish_rnti_map[i][0] == 0) {
              reestablish_rnti_map[i][0] = ctxt_pP->rnti;
              reestablish_rnti_map[i][1] = c_rnti;
              LOG_D(NR_RRC, "reestablish_rnti_map[%d] [0] %x, [1] %x\n",
                    i, reestablish_rnti_map[i][0], reestablish_rnti_map[i][1]);
              break;
            }
          }

          ue_context_p->ue_context.reestablishment_cause = rrcReestablishmentRequest.reestablishmentCause;
          LOG_D(NR_RRC, PROTOCOL_NR_RRC_CTXT_UE_FMT" Accept reestablishment request from UE physCellId %ld cause %ld\n",
                PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
                rrcReestablishmentRequest.ue_Identity.physCellId,
                ue_context_p->ue_context.reestablishment_cause);

          ue_context_p->ue_context.primaryCC_id = CC_id;
          //LG COMMENT Idx = (ue_mod_idP * NB_RB_MAX) + DCCH;
          Idx = DCCH;
          // SRB1
          ue_context_p->ue_context.Srb1.Active = 1;
          ue_context_p->ue_context.Srb1.Srb_info.Srb_id = Idx;
          memcpy(&ue_context_p->ue_context.Srb1.Srb_info.Lchan_desc[0],
                &DCCH_LCHAN_DESC,
                LCHAN_DESC_SIZE);
          memcpy(&ue_context_p->ue_context.Srb1.Srb_info.Lchan_desc[1],
                &DCCH_LCHAN_DESC,
                LCHAN_DESC_SIZE);
          // SRB2: set  it to go through SRB1 with id 1 (DCCH)
          ue_context_p->ue_context.Srb2.Active = 1;
          ue_context_p->ue_context.Srb2.Srb_info.Srb_id = Idx;
          memcpy(&ue_context_p->ue_context.Srb2.Srb_info.Lchan_desc[0],
                &DCCH_LCHAN_DESC,
                LCHAN_DESC_SIZE);
          memcpy(&ue_context_p->ue_context.Srb2.Srb_info.Lchan_desc[1],
                &DCCH_LCHAN_DESC,
                LCHAN_DESC_SIZE);

          rrc_gNB_generate_RRCReestablishment(ctxt_pP, ue_context_p, CC_id);

          LOG_I(NR_RRC, PROTOCOL_NR_RRC_CTXT_UE_FMT"CALLING RLC CONFIG SRB1 (rbid %d)\n",
                PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
                Idx);
          MSC_LOG_TX_MESSAGE(MSC_RRC_GNB,
                            MSC_PDCP_GNB,
                            NULL,
                            0,
                            MSC_AS_TIME_FMT" CONFIG_REQ UE %x SRB",
                            MSC_AS_TIME_ARGS(ctxt_pP),
                            ue_context_p->ue_context.rnti);
          // nr_rrc_pdcp_config_asn1_req(ctxt_pP,
          //                         ue_context_p->ue_context.SRB_configList,
          //                         NULL,
          //                         NULL,
          //                         0xff,
          //                         NULL,
          //                         NULL,
          //                         NULL,
          //                         NULL,
          //                         NULL,
          //                         NULL,
          //                         NULL);

          // if (!NODE_IS_CU(RC.nrrrc[ctxt_pP->module_id]->node_type)) {
            // nr_rrc_rlc_config_asn1_req(ctxt_pP,
            //                         ue_context_p->ue_context.SRB_configList,
            //                         NULL,
            //                         NULL,
            //                         NULL,
            //                         NULL);
          // }
        }
        break;

      case NR_UL_CCCH_MessageType__c1_PR_rrcSystemInfoRequest:
        LOG_I(NR_RRC, "receive rrcSystemInfoRequest message \n");
        /* TODO */
        break;

      default:
        LOG_E(NR_RRC, PROTOCOL_NR_RRC_CTXT_UE_FMT" Unknown message\n",
                   PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP));
        break;
    }
  }
  return 0;
}

/*! \fn uint64_t bitStr_to_uint64(BIT_STRING_t *)
 *\brief  This function extract at most a 64 bits value from a BIT_STRING_t object, the exact bits number depend on the BIT_STRING_t contents.
 *\param[in] pointer to the BIT_STRING_t object.
 *\return the extracted value.
 */
static inline uint64_t bitStr_to_uint64(BIT_STRING_t *asn) {
  uint64_t result = 0;
  int index;
  int shift;

  DevCheck ((asn->size > 0) && (asn->size <= 8), asn->size, 0, 0);

  shift = ((asn->size - 1) * 8) - asn->bits_unused;
  for (index = 0; index < (asn->size - 1); index++) {
    result |= (uint64_t)asn->buf[index] << shift;
    shift -= 8;
  }

  result |= asn->buf[index] >> asn->bits_unused;

  return result;
}

//-----------------------------------------------------------------------------
int
rrc_gNB_decode_dcch(
  const protocol_ctxt_t *const ctxt_pP,
  const rb_id_t                Srb_id,
  const uint8_t    *const      Rx_sdu,
  const sdu_size_t             sdu_sizeP
)
//-----------------------------------------------------------------------------
{
  asn_dec_rval_t                      dec_rval;
  NR_UL_DCCH_Message_t                *ul_dcch_msg  = NULL;
  struct rrc_gNB_ue_context_s         *ue_context_p = NULL;
  MessageDef                         *msg_delete_tunnels_p = NULL;
  uint8_t                             xid;

  int i;

  if ((Srb_id != 1) && (Srb_id != 2)) {
    LOG_E(NR_RRC, PROTOCOL_NR_RRC_CTXT_UE_FMT" Received message on SRB%ld, should not have ...\n",
          PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
          Srb_id);
  } else {
    LOG_D(NR_RRC, PROTOCOL_NR_RRC_CTXT_UE_FMT" Received message on SRB%ld\n",
              PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
              Srb_id);
  }

  LOG_D(NR_RRC, PROTOCOL_NR_RRC_CTXT_UE_FMT" Decoding UL-DCCH Message\n",
                  PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP));
  dec_rval = uper_decode(
                  NULL,
                  &asn_DEF_NR_UL_DCCH_Message,
                  (void **)&ul_dcch_msg,
                  Rx_sdu,
                  sdu_sizeP,
                  0,
                  0);
  // xer_fprint(stdout, &asn_DEF_NR_UL_DCCH_Message, (void *)&ul_dcch_msg);

  {
    for (i = 0; i < sdu_sizeP; i++) {
      LOG_T(NR_RRC, "%x.", Rx_sdu[i]);
    }

    LOG_T(NR_RRC, "\n");
  }

  if ((dec_rval.code != RC_OK) && (dec_rval.consumed == 0)) {
    LOG_E(NR_RRC, PROTOCOL_NR_RRC_CTXT_UE_FMT" Failed to decode UL-DCCH (%zu bytes)\n",
        PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
        dec_rval.consumed);
    return -1;
  }

  ue_context_p = rrc_gNB_get_ue_context(RC.nrrrc[ctxt_pP->module_id],
                                          ctxt_pP->rnti);

  if (ul_dcch_msg->message.present == NR_UL_DCCH_MessageType_PR_c1) {
    switch (ul_dcch_msg->message.choice.c1->present) {
      case NR_UL_DCCH_MessageType__c1_PR_NOTHING:
        LOG_I(NR_RRC,
            PROTOCOL_NR_RRC_CTXT_FMT" Received PR_NOTHING on UL-DCCH-Message\n",
            PROTOCOL_NR_RRC_CTXT_ARGS(ctxt_pP));
        break;

      case NR_UL_DCCH_MessageType__c1_PR_rrcReconfigurationComplete:
        LOG_I(NR_RRC, "Receive RRC Reconfiguration Complete message UE %x\n", ctxt_pP->rnti);
        if(!ue_context_p) {
          LOG_E(NR_RRC, "Processing NR_RRCReconfigurationComplete UE %x, ue_context_p is NULL\n", ctxt_pP->rnti);
          break;
        }

        LOG_DUMPMSG(NR_RRC, DEBUG_RRC, (char *)(Rx_sdu), sdu_sizeP,
                    "[MSG] RRC Connection Reconfiguration Complete\n");
        MSC_LOG_RX_MESSAGE(
        MSC_RRC_GNB,
        MSC_RRC_UE,
        Rx_sdu,
        sdu_sizeP,
        MSC_AS_TIME_FMT" NR_RRCReconfigurationComplete UE %x size %u",
        MSC_AS_TIME_ARGS(ctxt_pP),
        ue_context_p->ue_context.rnti,
        sdu_sizeP);
        LOG_D(NR_RRC,
            PROTOCOL_NR_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
            "(RRCReconfigurationComplete) ---> RRC_gNB]\n",
            PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
            DCCH,
            sdu_sizeP);

        if (ul_dcch_msg->message.choice.c1->present == NR_UL_DCCH_MessageType__c1_PR_rrcReconfigurationComplete) {
          if (ul_dcch_msg->message.choice.c1->choice.rrcReconfigurationComplete->criticalExtensions.present ==
            NR_RRCReconfigurationComplete__criticalExtensions_PR_rrcReconfigurationComplete)
            rrc_gNB_process_RRCReconfigurationComplete(
                ctxt_pP,
                ue_context_p,
                ul_dcch_msg->message.choice.c1->choice.rrcReconfigurationComplete->rrc_TransactionIdentifier);
        }

        if (AMF_MODE_ENABLED) {
          if(ue_context_p->ue_context.pdu_session_release_command_flag == 1) {
            xid = ul_dcch_msg->message.choice.c1->choice.rrcReconfigurationComplete->rrc_TransactionIdentifier;
            ue_context_p->ue_context.pdu_session_release_command_flag = 0;
            //gtp tunnel delete
            msg_delete_tunnels_p = itti_alloc_new_message(TASK_RRC_GNB, 0, GTPV1U_GNB_DELETE_TUNNEL_REQ);
            memset(&GTPV1U_GNB_DELETE_TUNNEL_REQ(msg_delete_tunnels_p), 0, sizeof(GTPV1U_GNB_DELETE_TUNNEL_REQ(msg_delete_tunnels_p)));
            GTPV1U_GNB_DELETE_TUNNEL_REQ(msg_delete_tunnels_p).rnti = ue_context_p->ue_context.rnti;

            for(i = 0; i < NB_RB_MAX; i++) {
              if(xid == ue_context_p->ue_context.pdusession[i].xid) {
                GTPV1U_GNB_DELETE_TUNNEL_REQ(msg_delete_tunnels_p).pdusession_id[GTPV1U_GNB_DELETE_TUNNEL_REQ(msg_delete_tunnels_p).num_pdusession++] =
                  ue_context_p->ue_context.gnb_gtp_psi[i];
                ue_context_p->ue_context.gnb_gtp_teid[i] = 0;
                memset(&ue_context_p->ue_context.gnb_gtp_addrs[i], 0, sizeof(ue_context_p->ue_context.gnb_gtp_addrs[i]));
                ue_context_p->ue_context.gnb_gtp_psi[i]  = 0;
              }
            }

            itti_send_msg_to_task(TASK_GTPV1_U, ctxt_pP->instance, msg_delete_tunnels_p);
            //NGAP_PDUSESSION_RELEASE_RESPONSE
            rrc_gNB_send_NGAP_PDUSESSION_RELEASE_RESPONSE(ctxt_pP, ue_context_p, xid);
          } else {
            rrc_gNB_send_NGAP_PDUSESSION_SETUP_RESP(ctxt_pP,
                                               ue_context_p,
                                               ul_dcch_msg->message.choice.c1->choice.rrcReconfigurationComplete->rrc_TransactionIdentifier);
          }
        }

        if (first_rrcreconfiguration == 0){
          first_rrcreconfiguration = 1;
          rrc_gNB_send_NGAP_INITIAL_CONTEXT_SETUP_RESP(ctxt_pP, ue_context_p);
        }

        break;

      case NR_UL_DCCH_MessageType__c1_PR_rrcSetupComplete:
        if(!ue_context_p) {
          LOG_I(NR_RRC, "Processing NR_RRCSetupComplete UE %x, ue_context_p is NULL\n", ctxt_pP->rnti);
          break;
        }

        LOG_DUMPMSG(NR_RRC, DEBUG_RRC,(char *)Rx_sdu,sdu_sizeP,
                    "[MSG] RRC SetupComplete\n");
        MSC_LOG_RX_MESSAGE(
                MSC_RRC_GNB,
                MSC_RRC_UE,
                Rx_sdu,
                sdu_sizeP,
                MSC_AS_TIME_FMT" NR_RRCSetupComplete UE %x size %u",
                MSC_AS_TIME_ARGS(ctxt_pP),
                ue_context_p->ue_context.rnti,
                sdu_sizeP);
        LOG_D(NR_RRC,
                PROTOCOL_NR_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
                "(RRCSetupComplete) ---> RRC_gNB\n",
                PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
                DCCH,
                sdu_sizeP);

        if (ul_dcch_msg->message.choice.c1->choice.rrcSetupComplete->criticalExtensions.present ==
              NR_RRCSetupComplete__criticalExtensions_PR_rrcSetupComplete) {
          if (ul_dcch_msg->message.choice.c1->choice.rrcSetupComplete->criticalExtensions.choice.
            rrcSetupComplete->ng_5G_S_TMSI_Value != NULL) {
            if (ul_dcch_msg->message.choice.c1->choice.rrcSetupComplete->criticalExtensions.choice.
            rrcSetupComplete->ng_5G_S_TMSI_Value->present == NR_RRCSetupComplete_IEs__ng_5G_S_TMSI_Value_PR_ng_5G_S_TMSI_Part2) {
            // ng-5G-S-TMSI-Part2                  BIT STRING (SIZE (9))
              if (ul_dcch_msg->message.choice.c1->choice.rrcSetupComplete->criticalExtensions.choice.
                rrcSetupComplete->ng_5G_S_TMSI_Value->choice.ng_5G_S_TMSI_Part2.size != 2) {
                LOG_E(NR_RRC, "wrong ng_5G_S_TMSI_Part2 size, expected 2, provided %lu",
                            (long unsigned int)ul_dcch_msg->message.choice.c1->choice.rrcSetupComplete->
                            criticalExtensions.choice.rrcSetupComplete->
                            ng_5G_S_TMSI_Value->choice.ng_5G_S_TMSI_Part2.size);
                return -1;
              }

              if (ue_context_p->ue_context.ng_5G_S_TMSI_Part1 != 0) {
                ue_context_p->ue_context.ng_5G_S_TMSI_Part2 =
                                BIT_STRING_to_uint16(&ul_dcch_msg->message.choice.c1->choice.rrcSetupComplete->
                                    criticalExtensions.choice.rrcSetupComplete->
                                    ng_5G_S_TMSI_Value->choice.ng_5G_S_TMSI_Part2);
              }

            /* TODO */
            } else if (ul_dcch_msg->message.choice.c1->choice.rrcSetupComplete->criticalExtensions.choice.
              rrcSetupComplete->ng_5G_S_TMSI_Value->present == NR_RRCSetupComplete_IEs__ng_5G_S_TMSI_Value_PR_ng_5G_S_TMSI) {
              // NG-5G-S-TMSI ::=                         BIT STRING (SIZE (48))
              if (ul_dcch_msg->message.choice.c1->choice.rrcSetupComplete->criticalExtensions.choice.
                rrcSetupComplete->ng_5G_S_TMSI_Value->choice.ng_5G_S_TMSI.size != 6) {
                LOG_E(NR_RRC, "wrong ng_5G_S_TMSI size, expected 6, provided %lu",
                            (long unsigned int)ul_dcch_msg->message.choice.c1->choice.rrcSetupComplete->
                            criticalExtensions.choice.rrcSetupComplete->
                            ng_5G_S_TMSI_Value->choice.ng_5G_S_TMSI.size);
                return -1;
              }

              uint64_t fiveg_s_TMSI = bitStr_to_uint64(&ul_dcch_msg->message.choice.c1->choice.rrcSetupComplete->
                  criticalExtensions.choice.rrcSetupComplete->ng_5G_S_TMSI_Value->choice.ng_5G_S_TMSI);
              LOG_I(NR_RRC, "Received rrcSetupComplete, 5g_s_TMSI: 0x%lX, amf_set_id: 0x%lX(%ld), amf_pointer: 0x%lX(%ld), 5g TMSI: 0x%X \n",
                  fiveg_s_TMSI, fiveg_s_TMSI >> 38, fiveg_s_TMSI >> 38,
                  (fiveg_s_TMSI >> 32) & 0x3F, (fiveg_s_TMSI >> 32) & 0x3F,
                  (uint32_t)fiveg_s_TMSI);
              if (ue_context_p->ue_context.Initialue_identity_5g_s_TMSI.presence == TRUE) {
                  ue_context_p->ue_context.Initialue_identity_5g_s_TMSI.amf_set_id = fiveg_s_TMSI >> 38;
                  ue_context_p->ue_context.Initialue_identity_5g_s_TMSI.amf_pointer = (fiveg_s_TMSI >> 32) & 0x3F;
                  ue_context_p->ue_context.Initialue_identity_5g_s_TMSI.fiveg_tmsi = (uint32_t)fiveg_s_TMSI;
              }
            }
          }

          rrc_gNB_process_RRCSetupComplete(
                  ctxt_pP,
                  ue_context_p,
                  ul_dcch_msg->message.choice.c1->choice.rrcSetupComplete->criticalExtensions.choice.rrcSetupComplete);
          LOG_I(NR_RRC, PROTOCOL_NR_RRC_CTXT_UE_FMT" UE State = NR_RRC_CONNECTED \n",
              PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP));
        }

        ue_context_p->ue_context.ue_release_timer = 0;
        break;

        case NR_UL_DCCH_MessageType__c1_PR_ulInformationTransfer:
            LOG_I(NR_RRC,"Recived RRC GNB UL Information Transfer \n");
            if(!ue_context_p) {
                LOG_I(NR_RRC, "Processing ulInformationTransfer UE %x, ue_context_p is NULL\n", ctxt_pP->rnti);
                break;
            }

            LOG_D(NR_RRC,"[MSG] RRC UL Information Transfer \n");
            LOG_DUMPMSG(RRC,DEBUG_RRC,(char *)Rx_sdu,sdu_sizeP,
                        "[MSG] RRC UL Information Transfer \n");
            MSC_LOG_RX_MESSAGE(
              MSC_RRC_GNB,
              MSC_RRC_UE,
              Rx_sdu,
              sdu_sizeP,
              MSC_AS_TIME_FMT" ulInformationTransfer UE %x size %u",
              MSC_AS_TIME_ARGS(ctxt_pP),
              ue_context_p->ue_context.rnti,
              sdu_sizeP);

            if (AMF_MODE_ENABLED == 1) {
                rrc_gNB_send_NGAP_UPLINK_NAS(ctxt_pP,
                                          ue_context_p,
                                          ul_dcch_msg);
            }
            break;

        case NR_UL_DCCH_MessageType__c1_PR_securityModeComplete:
        // to avoid segmentation fault
           if(!ue_context_p) {
              LOG_I(NR_RRC, "Processing securityModeComplete UE %x, ue_context_p is NULL\n", ctxt_pP->rnti);
              break;
           }

        LOG_I(NR_RRC,
              PROTOCOL_NR_RRC_CTXT_UE_FMT" received securityModeComplete on UL-DCCH %d from UE\n",
              PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
              DCCH);
        LOG_D(NR_RRC,
              PROTOCOL_NR_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
              "(securityModeComplete) ---> RRC_eNB\n",
              PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
              DCCH,
              sdu_sizeP);

        if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
          xer_fprint(stdout, &asn_DEF_NR_UL_DCCH_Message, (void *)ul_dcch_msg);
        }

        rrc_gNB_generate_UECapabilityEnquiry(ctxt_pP, ue_context_p);
        //rrc_gNB_generate_defaultRRCReconfiguration(ctxt_pP, ue_context_p);
        break;
        case NR_UL_DCCH_MessageType__c1_PR_securityModeFailure:
            LOG_DUMPMSG(NR_RRC,DEBUG_RRC,(char *)Rx_sdu,sdu_sizeP,
                       "[MSG] NR RRC Security Mode Failure\n");
            MSC_LOG_RX_MESSAGE(
                MSC_RRC_GNB,
                MSC_RRC_UE,
                Rx_sdu,
                sdu_sizeP,
                MSC_AS_TIME_FMT" securityModeFailure UE %x size %u",
                MSC_AS_TIME_ARGS(ctxt_pP),
                ue_context_p->ue_context.rnti,
                sdu_sizeP);
            LOG_W(NR_RRC,
                  PROTOCOL_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
                  "(securityModeFailure) ---> RRC_gNB\n",
                  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
                  DCCH,
                  sdu_sizeP);
            
            if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
              xer_fprint(stdout, &asn_DEF_NR_UL_DCCH_Message, (void *)ul_dcch_msg);
            }
            
            rrc_gNB_generate_UECapabilityEnquiry(ctxt_pP, ue_context_p);
            break;

      case NR_UL_DCCH_MessageType__c1_PR_ueCapabilityInformation:
        if(!ue_context_p) {
          LOG_I(NR_RRC, "Processing ueCapabilityInformation UE %x, ue_context_p is NULL\n", ctxt_pP->rnti);
          break;
        }

        LOG_DUMPMSG(NR_RRC,DEBUG_RRC,(char *)Rx_sdu,sdu_sizeP,
                "[MSG] NR_RRC UECapablility Information\n");
            MSC_LOG_RX_MESSAGE(
            MSC_RRC_GNB,
            MSC_RRC_UE,
            Rx_sdu,
            sdu_sizeP,
            MSC_AS_TIME_FMT" ueCapabilityInformation UE %x size %u",
            MSC_AS_TIME_ARGS(ctxt_pP),
            ue_context_p->ue_context.rnti,
            sdu_sizeP);
        LOG_I(NR_RRC,
            PROTOCOL_NR_RRC_CTXT_UE_FMT" received ueCapabilityInformation on UL-DCCH %d from UE\n",
            PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
            DCCH);
        LOG_D(RRC,
        PROTOCOL_NR_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
        "(UECapabilityInformation) ---> RRC_eNB\n",
        PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
        DCCH,
        sdu_sizeP);
        if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
            xer_fprint(stdout, &asn_DEF_NR_UL_DCCH_Message, (void *)ul_dcch_msg);
        }
        LOG_I(NR_RRC, "got UE capabilities for UE %x\n", ctxt_pP->rnti);
        int eutra_index = -1;

        if( ul_dcch_msg->message.choice.c1->choice.ueCapabilityInformation->criticalExtensions.present ==
        NR_UECapabilityInformation__criticalExtensions_PR_ueCapabilityInformation ) {
          for(i = 0;i < ul_dcch_msg->message.choice.c1->choice.ueCapabilityInformation->criticalExtensions.choice.ueCapabilityInformation->ue_CapabilityRAT_ContainerList->list.count; i++){
            if(ul_dcch_msg->message.choice.c1->choice.ueCapabilityInformation->criticalExtensions.choice.ueCapabilityInformation->ue_CapabilityRAT_ContainerList->list.array[i]->rat_Type ==
              NR_RAT_Type_nr){
              if(ue_context_p->ue_context.UE_Capability_nr){
                ASN_STRUCT_FREE(asn_DEF_NR_UE_NR_Capability,ue_context_p->ue_context.UE_Capability_nr);
                ue_context_p->ue_context.UE_Capability_nr = 0;
              }

              dec_rval = uper_decode(NULL,
                                      &asn_DEF_NR_UE_NR_Capability,
                                      (void**)&ue_context_p->ue_context.UE_Capability_nr,
                                      ul_dcch_msg->message.choice.c1->choice.ueCapabilityInformation->criticalExtensions.choice.ueCapabilityInformation->ue_CapabilityRAT_ContainerList->list.array[i]->ue_CapabilityRAT_Container.buf,
                                      ul_dcch_msg->message.choice.c1->choice.ueCapabilityInformation->criticalExtensions.choice.ueCapabilityInformation->ue_CapabilityRAT_ContainerList->list.array[i]->ue_CapabilityRAT_Container.size,
                                      0,0);
              if(LOG_DEBUGFLAG(DEBUG_ASN1)){
                xer_fprint(stdout,&asn_DEF_NR_UE_NR_Capability,ue_context_p->ue_context.UE_Capability_nr);
              }

              if((dec_rval.code != RC_OK) && (dec_rval.consumed == 0)){
                LOG_E(NR_RRC,PROTOCOL_NR_RRC_CTXT_UE_FMT" Failed to decode nr UE capabilities (%zu bytes)\n",
                PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),dec_rval.consumed);
                ASN_STRUCT_FREE(asn_DEF_NR_UE_NR_Capability,ue_context_p->ue_context.UE_Capability_nr);
                ue_context_p->ue_context.UE_Capability_nr = 0;
              }

              ue_context_p->ue_context.UE_Capability_size =
              ul_dcch_msg->message.choice.c1->choice.ueCapabilityInformation->criticalExtensions.choice.ueCapabilityInformation->ue_CapabilityRAT_ContainerList->list.array[i]->ue_CapabilityRAT_Container.size;
              if(eutra_index != -1){
                LOG_E(NR_RRC,"fatal: more than 1 eutra capability\n");
                exit(1);
              }
              eutra_index = i;
            }

            if(ul_dcch_msg->message.choice.c1->choice.ueCapabilityInformation->criticalExtensions.choice.ueCapabilityInformation->ue_CapabilityRAT_ContainerList->list.array[i]->rat_Type ==
            NR_RAT_Type_eutra_nr){
            if(ue_context_p->ue_context.UE_Capability_MRDC){
              ASN_STRUCT_FREE(asn_DEF_NR_UE_MRDC_Capability,ue_context_p->ue_context.UE_Capability_MRDC);
              ue_context_p->ue_context.UE_Capability_MRDC = 0;
            }
            dec_rval = uper_decode(NULL,
                                    &asn_DEF_NR_UE_MRDC_Capability,
                                    (void**)&ue_context_p->ue_context.UE_Capability_MRDC,
                                    ul_dcch_msg->message.choice.c1->choice.ueCapabilityInformation->criticalExtensions.choice.ueCapabilityInformation->ue_CapabilityRAT_ContainerList->list.array[i]->ue_CapabilityRAT_Container.buf,
                                    ul_dcch_msg->message.choice.c1->choice.ueCapabilityInformation->criticalExtensions.choice.ueCapabilityInformation->ue_CapabilityRAT_ContainerList->list.array[i]->ue_CapabilityRAT_Container.size,
                                    0,0);

            if(LOG_DEBUGFLAG(DEBUG_ASN1)){
              xer_fprint(stdout,&asn_DEF_NR_UE_MRDC_Capability,ue_context_p->ue_context.UE_Capability_MRDC);
            }

            if((dec_rval.code != RC_OK) && (dec_rval.consumed == 0)){
              LOG_E(NR_RRC,PROTOCOL_NR_RRC_CTXT_FMT" Failed to decode nr UE capabilities (%zu bytes)\n",
                  PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),dec_rval.consumed);
              ASN_STRUCT_FREE(asn_DEF_NR_UE_MRDC_Capability,ue_context_p->ue_context.UE_Capability_MRDC);
              ue_context_p->ue_context.UE_Capability_MRDC = 0;
            }
              ue_context_p->ue_context.UE_MRDC_Capability_size =
              ul_dcch_msg->message.choice.c1->choice.ueCapabilityInformation->criticalExtensions.choice.ueCapabilityInformation->ue_CapabilityRAT_ContainerList->list.array[i]->ue_CapabilityRAT_Container.size;
            }

            if(ul_dcch_msg->message.choice.c1->choice.ueCapabilityInformation->criticalExtensions.choice.ueCapabilityInformation->ue_CapabilityRAT_ContainerList->list.array[i]->rat_Type ==
            NR_RAT_Type_eutra){
              //TODO
            }
          }

          if(eutra_index == -1)
          break;
      }
      if (AMF_MODE_ENABLED == 1) {
          rrc_gNB_send_NGAP_UE_CAPABILITIES_IND(ctxt_pP,
                                    ue_context_p,
                                    ul_dcch_msg);
      }
      rrc_gNB_generate_defaultRRCReconfiguration(ctxt_pP, ue_context_p);
      break;

            case NR_UL_DCCH_MessageType__c1_PR_rrcReestablishmentComplete:
              LOG_DUMPMSG(NR_RRC,DEBUG_RRC,(char *)Rx_sdu,sdu_sizeP,
                          "[MSG] NR_RRC Connection Reestablishment Complete\n");
              MSC_LOG_RX_MESSAGE(
                MSC_RRC_GNB,
                MSC_RRC_UE,
                Rx_sdu,
                sdu_sizeP,
                MSC_AS_TIME_FMT" NR_RRCConnectionReestablishmentComplete UE %x size %u",
                MSC_AS_TIME_ARGS(ctxt_pP),
                ue_context_p->ue_context.rnti,
                sdu_sizeP);
              LOG_I(NR_RRC,
                    PROTOCOL_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
                    "(rrcConnectionReestablishmentComplete) ---> RRC_gNB\n",
                    PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
                    DCCH,
                    sdu_sizeP);
              {
                rnti_t reestablish_rnti = 0;

                // select C-RNTI from map
                for (i = 0; i < MAX_MOBILES_PER_ENB; i++) {
                  if (reestablish_rnti_map[i][0] == ctxt_pP->rnti) {
                    reestablish_rnti = reestablish_rnti_map[i][1];
                    ue_context_p = rrc_gNB_get_ue_context(
                                     RC.nrrrc[ctxt_pP->module_id],
                                     reestablish_rnti);
                    // clear currentC-RNTI from map
                    reestablish_rnti_map[i][0] = 0;
                    reestablish_rnti_map[i][1] = 0;
                    LOG_D(NR_RRC, "reestablish_rnti_map[%d] [0] %x, [1] %x\n",
                          i, reestablish_rnti_map[i][0], reestablish_rnti_map[i][1]);
                    break;
                  }
                }

                if (!ue_context_p) {
                  LOG_E(NR_RRC,
                        PROTOCOL_NR_RRC_CTXT_UE_FMT" NR_RRCConnectionReestablishmentComplete without UE context, falt\n",
                        PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP));
                  break;
                }

#if(0)
                /* TODO : It may be needed if gNB goes into full stack working. */
                //clear
                int UE_id = find_nr_UE_id(ctxt_pP->module_id, ctxt_pP->rnti);

                if(UE_id == -1) {
                  LOG_E(NR_RRC,
                        PROTOCOL_RRC_CTXT_UE_FMT" NR_RRCConnectionReestablishmentComplete without UE_id(MAC) rnti %x, fault\n",
                        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),ctxt_pP->rnti);
                  break;
                }

                RC.nrmac[ctxt_pP->module_id]->UE_info.UE_sched_ctrl[UE_id].ue_reestablishment_reject_timer = 0;
#endif
                ue_context_p->ue_context.reestablishment_xid = -1;

                if (ul_dcch_msg->message.choice.c1->choice.rrcReestablishmentComplete->criticalExtensions.present ==
                    NR_RRCReestablishmentComplete__criticalExtensions_PR_rrcReestablishmentComplete) {
                  rrc_gNB_process_RRCConnectionReestablishmentComplete(ctxt_pP, reestablish_rnti, ue_context_p,
                      ul_dcch_msg->message.choice.c1->choice.rrcReestablishmentComplete->rrc_TransactionIdentifier);

                }

                //ue_context_p->ue_context.ue_release_timer = 0;
                ue_context_p->ue_context.ue_reestablishment_timer = 1;
                // remove UE after 100 frames after LTE_RRCConnectionReestablishmentRelease is triggered
                ue_context_p->ue_context.ue_reestablishment_timer_thres = 1000;
              }
              break;
      default:
        break;
    }
  }
  return 0;
}

void rrc_gNB_process_release_request(const module_id_t gnb_mod_idP, x2ap_ENDC_sgnb_release_request_t *m)
{
  gNB_RRC_INST *rrc = RC.nrrrc[gnb_mod_idP];
  rrc_remove_nsa_user(rrc, m->rnti);
}

void rrc_gNB_process_dc_overall_timeout(const module_id_t gnb_mod_idP, x2ap_ENDC_dc_overall_timeout_t *m)
{
  gNB_RRC_INST *rrc = RC.nrrrc[gnb_mod_idP];
  rrc_remove_nsa_user(rrc, m->rnti);
}

void nr_rrc_subframe_process(protocol_ctxt_t *const ctxt_pP, const int CC_id) {
  MessageDef *msg;

  /* send a tick to x2ap */
  if (is_x2ap_enabled()){
    msg = itti_alloc_new_message(TASK_RRC_ENB, 0, X2AP_SUBFRAME_PROCESS);
    itti_send_msg_to_task(TASK_X2AP, ctxt_pP->module_id, msg);
  }
}

///---------------------------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------------------------///
void *rrc_gnb_task(void *args_p) {
  MessageDef                         *msg_p;
  const char                         *msg_name_p;
  instance_t                         instance;
  int                                result;
  //SRB_INFO                           *srb_info_p;
  //int                                CC_id;
  protocol_ctxt_t                    ctxt;
  itti_mark_task_ready(TASK_RRC_GNB);
  LOG_I(NR_RRC,"Entering main loop of NR_RRC message task\n");

  while (1) {
    // Wait for a message
    itti_receive_msg(TASK_RRC_GNB, &msg_p);
    msg_name_p = ITTI_MSG_NAME(msg_p);
    instance = ITTI_MSG_DESTINATION_INSTANCE(msg_p);

    /* RRC_SUBFRAME_PROCESS is sent every subframe, do not log it */
    if (ITTI_MSG_ID(msg_p) != RRC_SUBFRAME_PROCESS)
      LOG_I(NR_RRC,"Received message %s\n",msg_name_p);

    switch (ITTI_MSG_ID(msg_p)) {
      case TERMINATE_MESSAGE:
        LOG_W(NR_RRC, " *** Exiting NR_RRC thread\n");
        itti_exit_task();
        break;

      case MESSAGE_TEST:
        LOG_I(NR_RRC, "[gNB %ld] Received %s\n", instance, msg_name_p);
        break;

      case RRC_SUBFRAME_PROCESS:
        nr_rrc_subframe_process(&RRC_SUBFRAME_PROCESS(msg_p).ctxt, RRC_SUBFRAME_PROCESS(msg_p).CC_id);
        break;

      /* Messages from MAC */
      case NR_RRC_MAC_CCCH_DATA_IND:
      PROTOCOL_CTXT_SET_BY_INSTANCE(&ctxt,
                                    NR_RRC_MAC_CCCH_DATA_IND(msg_p).gnb_index,
                                    GNB_FLAG_YES,
                                    NR_RRC_MAC_CCCH_DATA_IND(msg_p).rnti,
                                    msg_p->ittiMsgHeader.lte_time.frame,
                                    msg_p->ittiMsgHeader.lte_time.slot);
        LOG_I(NR_RRC,"Decoding CCCH : inst %ld, CC_id %d, ctxt %p, sib_info_p->Rx_buffer.payload_size %d\n",
                instance,
                NR_RRC_MAC_CCCH_DATA_IND(msg_p).CC_id,
                &ctxt,
                NR_RRC_MAC_CCCH_DATA_IND(msg_p).sdu_size);

        if (NR_RRC_MAC_CCCH_DATA_IND(msg_p).sdu_size >= CCCH_SDU_SIZE) {
          LOG_I(NR_RRC, "CCCH message has size %d > %d\n",
              NR_RRC_MAC_CCCH_DATA_IND(msg_p).sdu_size,CCCH_SDU_SIZE);
          break;
      }

      nr_rrc_gNB_decode_ccch(&ctxt,
                          (uint8_t *)NR_RRC_MAC_CCCH_DATA_IND(msg_p).sdu,
                          NR_RRC_MAC_CCCH_DATA_IND(msg_p).sdu_size,
                          NR_RRC_MAC_CCCH_DATA_IND(msg_p).CC_id);
      break;

      /* Messages from PDCP */
      case NR_RRC_DCCH_DATA_IND:
        PROTOCOL_CTXT_SET_BY_INSTANCE(&ctxt,
                                      instance,
                                      GNB_FLAG_YES,
                                      NR_RRC_DCCH_DATA_IND(msg_p).rnti,
                                      msg_p->ittiMsgHeader.lte_time.frame,
                                      msg_p->ittiMsgHeader.lte_time.slot);
        LOG_D(NR_RRC, PROTOCOL_NR_RRC_CTXT_UE_FMT" Received on DCCH %d %s\n",
                PROTOCOL_NR_RRC_CTXT_UE_ARGS(&ctxt),
                NR_RRC_DCCH_DATA_IND(msg_p).dcch_index,
                msg_name_p);
        rrc_gNB_decode_dcch(&ctxt,
                            NR_RRC_DCCH_DATA_IND(msg_p).dcch_index,
                            NR_RRC_DCCH_DATA_IND(msg_p).sdu_p,
                            NR_RRC_DCCH_DATA_IND(msg_p).sdu_size);
        break;

      case NGAP_DOWNLINK_NAS:
        rrc_gNB_process_NGAP_DOWNLINK_NAS(msg_p, msg_name_p, instance, &rrc_gNB_mui);
        break;

      case NGAP_PDUSESSION_SETUP_REQ:
        rrc_gNB_process_NGAP_PDUSESSION_SETUP_REQ(msg_p, msg_name_p, instance);
        break;

      case NGAP_PDUSESSION_RELEASE_COMMAND:
        rrc_gNB_process_NGAP_PDUSESSION_RELEASE_COMMAND(msg_p, msg_name_p, instance);
        break;

      case GTPV1U_GNB_DELETE_TUNNEL_RESP:
        break;

      /*
      #if defined(ENABLE_USE_MME)

            // Messages from S1AP
          case S1AP_DOWNLINK_NAS:
            rrc_eNB_process_S1AP_DOWNLINK_NAS(msg_p, msg_name_p, instance, &rrc_gNB_mui);
            break;

          case S1AP_INITIAL_CONTEXT_SETUP_REQ:
            rrc_eNB_process_S1AP_INITIAL_CONTEXT_SETUP_REQ(msg_p, msg_name_p, instance);
            break;

          case S1AP_UE_CTXT_MODIFICATION_REQ:
            rrc_eNB_process_S1AP_UE_CTXT_MODIFICATION_REQ(msg_p, msg_name_p, instance);
            break;

          case S1AP_PAGING_IND:
            LOG_D(RRC, "[eNB %d] Received Paging message from S1AP: %s\n", instance, msg_name_p);
            rrc_eNB_process_PAGING_IND(msg_p, msg_name_p, instance);
            break;

          case S1AP_E_RAB_SETUP_REQ:
            rrc_eNB_process_S1AP_E_RAB_SETUP_REQ(msg_p, msg_name_p, instance);
            LOG_D(RRC, "[eNB %d] Received the message %s\n", instance, msg_name_p);
            break;

          case S1AP_E_RAB_MODIFY_REQ:
            rrc_eNB_process_S1AP_E_RAB_MODIFY_REQ(msg_p, msg_name_p, instance);
            break;

          case S1AP_E_RAB_RELEASE_COMMAND:
            rrc_eNB_process_S1AP_E_RAB_RELEASE_COMMAND(msg_p, msg_name_p, instance);
            break;

          case S1AP_UE_CONTEXT_RELEASE_REQ:
            rrc_eNB_process_S1AP_UE_CONTEXT_RELEASE_REQ(msg_p, msg_name_p, instance);
            break;

          case S1AP_UE_CONTEXT_RELEASE_COMMAND:
            rrc_eNB_process_S1AP_UE_CONTEXT_RELEASE_COMMAND(msg_p, msg_name_p, instance);
            break;

          case GTPV1U_ENB_DELETE_TUNNEL_RESP:
            ///Nothing to do. Apparently everything is done in S1AP processing
            //LOG_I(RRC, "[eNB %d] Received message %s, not processed because procedure not synched\n",
            //instance, msg_name_p);
            if (rrc_eNB_get_ue_context(RC.nrrrc[instance], GTPV1U_ENB_DELETE_TUNNEL_RESP(msg_p).rnti)
                && rrc_eNB_get_ue_context(RC.nrrrc[instance], GTPV1U_ENB_DELETE_TUNNEL_RESP(msg_p).rnti)->ue_context.ue_release_timer_rrc > 0) {
              rrc_eNB_get_ue_context(RC.nrrrc[instance], GTPV1U_ENB_DELETE_TUNNEL_RESP(msg_p).rnti)->ue_context.ue_release_timer_rrc =
              rrc_eNB_get_ue_context(RC.nrrrc[instance], GTPV1U_ENB_DELETE_TUNNEL_RESP(msg_p).rnti)->ue_context.ue_release_timer_thres_rrc;
            }
            break;

      #endif
      */
      /* Messages from gNB app */
      case NRRRC_CONFIGURATION_REQ:
        LOG_I(NR_RRC, "[gNB %ld] Received %s : %p\n", instance, msg_name_p,&NRRRC_CONFIGURATION_REQ(msg_p));
        openair_rrc_gNB_configuration(GNB_INSTANCE_TO_MODULE_ID(instance), &NRRRC_CONFIGURATION_REQ(msg_p));
        break;

      /* Messages from X2AP */
      case X2AP_ENDC_SGNB_ADDITION_REQ:
        LOG_I(NR_RRC, "Received ENDC sgNB addition request from X2AP \n");
        rrc_gNB_process_AdditionRequestInformation(GNB_INSTANCE_TO_MODULE_ID(instance), &X2AP_ENDC_SGNB_ADDITION_REQ(msg_p));
        break;

      case X2AP_ENDC_SGNB_RECONF_COMPLETE:
        LOG_I(NR_RRC, "Handling of reconfiguration complete message at RRC gNB is pending \n");
        break;

      case NGAP_INITIAL_CONTEXT_SETUP_REQ:
        rrc_gNB_process_NGAP_INITIAL_CONTEXT_SETUP_REQ(msg_p, msg_name_p, instance);
        break;

      case X2AP_ENDC_SGNB_RELEASE_REQUEST:
        LOG_I(NR_RRC, "Received ENDC sgNB release request from X2AP \n");
        rrc_gNB_process_release_request(GNB_INSTANCE_TO_MODULE_ID(instance), &X2AP_ENDC_SGNB_RELEASE_REQUEST(msg_p));
        break;

      case X2AP_ENDC_DC_OVERALL_TIMEOUT:
        rrc_gNB_process_dc_overall_timeout(GNB_INSTANCE_TO_MODULE_ID(instance), &X2AP_ENDC_DC_OVERALL_TIMEOUT(msg_p));
        break;

      /* Messages from GTP */
      case GTPV1U_ENB_DELETE_TUNNEL_RESP:
        /* nothing to do? */
        break;

      case NGAP_UE_CONTEXT_RELEASE_REQ:
        rrc_gNB_process_NGAP_UE_CONTEXT_RELEASE_REQ(msg_p, msg_name_p, instance);
        break;

      case NGAP_UE_CONTEXT_RELEASE_COMMAND:
        rrc_gNB_process_NGAP_UE_CONTEXT_RELEASE_COMMAND(msg_p, msg_name_p, instance);
        break;

      default:
        LOG_E(NR_RRC, "[gNB %ld] Received unexpected message %s\n", instance, msg_name_p);
        break;
    }

    result = itti_free(ITTI_MSG_ORIGIN_ID(msg_p), msg_p);
    AssertFatal(result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
    msg_p = NULL;
  }
}

//-----------------------------------------------------------------------------
void
rrc_gNB_generate_SecurityModeCommand(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_gNB_ue_context_t          *const ue_context_pP
)
//-----------------------------------------------------------------------------
{
  uint8_t                             buffer[100];
  uint8_t                             size;
  T(T_ENB_RRC_SECURITY_MODE_COMMAND, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
  NR_IntegrityProtAlgorithm_t integrity_algorithm = (NR_IntegrityProtAlgorithm_t)ue_context_pP->ue_context.integrity_algorithm;
  size = do_NR_SecurityModeCommand(
           ctxt_pP,
           buffer,
           rrc_gNB_get_next_transaction_identifier(ctxt_pP->module_id),
           ue_context_pP->ue_context.ciphering_algorithm,
           &integrity_algorithm);
  LOG_DUMPMSG(NR_RRC,DEBUG_RRC,(char *)buffer,size,"[MSG] RRC Security Mode Command\n");
  LOG_I(NR_RRC,
        PROTOCOL_NR_RRC_CTXT_UE_FMT" Logical Channel DL-DCCH, Generate SecurityModeCommand (bytes %d)\n",
        PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
        size);
  LOG_D(NR_RRC,
        PROTOCOL_NR_RRC_CTXT_UE_FMT" --- PDCP_DATA_REQ/%d Bytes (securityModeCommand to UE MUI %d) --->[PDCP][RB %02d]\n",
        PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
        size,
        rrc_gNB_mui,
        DCCH);
  MSC_LOG_TX_MESSAGE(
    MSC_RRC_GNB,
    MSC_RRC_UE,
    buffer,
    size,
    MSC_AS_TIME_FMT" securityModeCommand UE %x MUI %d size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti,
    rrc_gNB_mui,
    size);

  LOG_I(NR_RRC,"calling rrc_data_req :securityModeCommand\n");
#ifdef ITTI_SIM
  MessageDef *message_p;
  uint8_t *message_buffer;
  message_buffer = itti_malloc (TASK_RRC_GNB, TASK_RRC_UE_SIM,size);
  memcpy (message_buffer, buffer, size);
  message_p = itti_alloc_new_message (TASK_RRC_GNB, 0, GNB_RRC_DCCH_DATA_IND);
  GNB_RRC_DCCH_DATA_IND (message_p).rbid = DCCH;
  GNB_RRC_DCCH_DATA_IND (message_p).sdu = message_buffer;
  GNB_RRC_DCCH_DATA_IND (message_p).size	= size;
  itti_send_msg_to_task (TASK_RRC_UE_SIM, ctxt_pP->instance, message_p);
#else
  nr_rrc_data_req(ctxt_pP,
               DCCH,
               rrc_gNB_mui++,
               SDU_CONFIRM_NO,
               size,
               buffer,
               PDCP_TRANSMISSION_MODE_CONTROL);
#endif

}

void
rrc_gNB_generate_UECapabilityEnquiry(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_gNB_ue_context_t          *const ue_context_pP
)
//-----------------------------------------------------------------------------
{
  uint8_t                             buffer[100];
  uint8_t                             size;
  T(T_ENB_RRC_UE_CAPABILITY_ENQUIRY, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
  size = do_NR_SA_UECapabilityEnquiry(
           ctxt_pP,
           buffer,
           rrc_gNB_get_next_transaction_identifier(ctxt_pP->module_id));
  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" Logical Channel DL-DCCH, Generate NR UECapabilityEnquiry (bytes %d)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        size);
  LOG_D(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" --- PDCP_DATA_REQ/%d Bytes (NR UECapabilityEnquiry MUI %d) --->[PDCP][RB %02d]\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        size,
        rrc_gNB_mui,
        DCCH);
  MSC_LOG_TX_MESSAGE(
    MSC_RRC_GNB,
    MSC_RRC_UE,
    buffer,
    size,
    MSC_AS_TIME_FMT" rrcNRUECapabilityEnquiry UE %x MUI %d size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti,
    rrc_gNB_mui,
    size);
#ifdef ITTI_SIM
  MessageDef *message_p;
  uint8_t *message_buffer;
  message_buffer = itti_malloc (TASK_RRC_GNB, TASK_RRC_UE_SIM, size);
  memcpy (message_buffer, buffer, size);
  message_p = itti_alloc_new_message (TASK_RRC_GNB, 0, GNB_RRC_DCCH_DATA_IND);
  GNB_RRC_DCCH_DATA_IND (message_p).rbid = DCCH;
  GNB_RRC_DCCH_DATA_IND (message_p).sdu = message_buffer;
  GNB_RRC_DCCH_DATA_IND (message_p).size  = size;
  itti_send_msg_to_task (TASK_RRC_UE_SIM, ctxt_pP->instance, message_p);
#else
  nr_rrc_data_req(
    ctxt_pP,
    DCCH,
    rrc_gNB_mui++,
    SDU_CONFIRM_NO,
    size,
    buffer,
    PDCP_TRANSMISSION_MODE_CONTROL);
#endif
}

//-----------------------------------------------------------------------------
/*
* Generate the RRC Connection Release to UE.
* If received, UE should switch to RRC_IDLE mode.
*/
void
rrc_gNB_generate_RRCRelease(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_gNB_ue_context_t  *const ue_context_pP
)
//-----------------------------------------------------------------------------
{
  uint8_t buffer[RRC_BUF_SIZE];
  uint16_t size = 0;

  memset(buffer, 0, RRC_BUF_SIZE);

  size = do_NR_RRCRelease(buffer,rrc_gNB_get_next_transaction_identifier(ctxt_pP->module_id));
  ue_context_pP->ue_context.ue_reestablishment_timer = 0;
  ue_context_pP->ue_context.ue_release_timer = 0;
  LOG_I(NR_RRC,
        PROTOCOL_NR_RRC_CTXT_UE_FMT" Logical Channel DL-DCCH, Generate RRCRelease (bytes %d)\n",
        PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
        size);
  LOG_D(NR_RRC,
        PROTOCOL_NR_RRC_CTXT_UE_FMT" --- PDCP_DATA_REQ/%d Bytes (rrcRelease MUI %d) --->[PDCP][RB %u]\n",
        PROTOCOL_NR_RRC_CTXT_UE_ARGS(ctxt_pP),
        size,
        rrc_gNB_mui,
        DCCH);
  MSC_LOG_TX_MESSAGE(
    MSC_RRC_GNB,
    MSC_RRC_UE,
    buffer,
    size,
    MSC_AS_TIME_FMT" NR_RRCRelease UE %x MUI %d size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti,
    rrc_gNB_mui,
    size);

#ifdef ITTI_SIM
    MessageDef *message_p;
    uint8_t *message_buffer;
    message_buffer = itti_malloc (TASK_RRC_GNB, TASK_RRC_UE_SIM, size);
    memcpy (message_buffer, buffer, size);
    message_p = itti_alloc_new_message (TASK_RRC_GNB, 0, GNB_RRC_DCCH_DATA_IND);
    GNB_RRC_DCCH_DATA_IND (message_p).rbid = DCCH;
    GNB_RRC_DCCH_DATA_IND (message_p).sdu = message_buffer;
    GNB_RRC_DCCH_DATA_IND (message_p).size  = size;
    itti_send_msg_to_task (TASK_RRC_UE_SIM, ctxt_pP->instance, message_p);
#else
  if (NODE_IS_CU(RC.rrc[ctxt_pP->module_id]->node_type)) {
    MessageDef *m = itti_alloc_new_message(TASK_RRC_GNB, 0, F1AP_UE_CONTEXT_RELEASE_CMD);
    F1AP_UE_CONTEXT_RELEASE_CMD(m).rnti = ctxt_pP->rnti;
    F1AP_UE_CONTEXT_RELEASE_CMD(m).cause = F1AP_CAUSE_RADIO_NETWORK;
    F1AP_UE_CONTEXT_RELEASE_CMD(m).cause_value = 10; // 10 = F1AP_CauseRadioNetwork_normal_release
    F1AP_UE_CONTEXT_RELEASE_CMD(m).rrc_container = buffer;
    F1AP_UE_CONTEXT_RELEASE_CMD(m).rrc_container_length = size;
    itti_send_msg_to_task(TASK_CU_F1, ctxt_pP->module_id, m);
  } else {
    nr_rrc_data_req(ctxt_pP,
                 DCCH,
                 rrc_gNB_mui++,
                 SDU_CONFIRM_NO,
                 size,
                 buffer,
                 PDCP_TRANSMISSION_MODE_CONTROL);
  }
#endif
}
void nr_rrc_trigger(protocol_ctxt_t *ctxt, int CC_id, int frame, int subframe)
{
  MessageDef *message_p;
  message_p = itti_alloc_new_message(TASK_RRC_GNB, 0, RRC_SUBFRAME_PROCESS);
  RRC_SUBFRAME_PROCESS(message_p).ctxt  = *ctxt;
  RRC_SUBFRAME_PROCESS(message_p).CC_id = CC_id;
  itti_send_msg_to_task(TASK_RRC_GNB, ctxt->module_id, message_p);
}
