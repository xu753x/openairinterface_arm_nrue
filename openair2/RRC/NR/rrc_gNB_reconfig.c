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

/*! \file rrc_gNB_reconfig.c
 * \brief rrc gNB RRCreconfiguration support routines
 * \author Raymond Knopp
 * \date 2019
 * \version 1.0
 * \company Eurecom
 * \email: raymond.knopp@eurecom.fr
 */
#ifndef RRC_GNB_NSA_C
#define RRC_GNB_NSA_C

#include "NR_ServingCellConfigCommon.h"
#include "NR_RRCReconfiguration.h"
#include "NR_RRCReconfiguration-IEs.h"
#include "NR_CellGroupConfig.h"
#include "NR_MAC-CellGroupConfig.h"
#include "NR_BSR-Config.h"
#include "NR_PDSCH-ServingCellConfig.h"
#include "NR_RLC-BearerConfig.h"
#include "BOOLEAN.h"
#include "assertions.h"
#include "common/utils/nr/nr_common.h"
#include "SIMULATION/TOOLS/sim.h"
#include "executables/softmodem-common.h"

#define false 0
#define true 1


void fill_default_secondaryCellGroup(NR_ServingCellConfigCommon_t *servingcellconfigcommon,
                                     NR_CellGroupConfig_t *secondaryCellGroup,
                                     int scg_id,
                                     int servCellIndex,
                                     int n_physical_antenna_ports,
                                     int initial_csi_index) {
  AssertFatal(servingcellconfigcommon!=NULL,"servingcellconfigcommon is null\n");
  AssertFatal(secondaryCellGroup!=NULL,"secondaryCellGroup is null\n");

  if(servingcellconfigcommon->ssb_PositionsInBurst->present !=2)
    AssertFatal(1==0,"Currenrly implemented only for medium size SSB bitmap\n");
  uint8_t bitmap = servingcellconfigcommon->ssb_PositionsInBurst->choice.mediumBitmap.buf[0];

  memset(secondaryCellGroup,0,sizeof(NR_CellGroupConfig_t));
  secondaryCellGroup->cellGroupId = scg_id;
  NR_RLC_BearerConfig_t *RLC_BearerConfig = calloc(1,sizeof(*RLC_BearerConfig));
  RLC_BearerConfig->logicalChannelIdentity = 4;
  RLC_BearerConfig->servedRadioBearer = calloc(1,sizeof(*RLC_BearerConfig->servedRadioBearer));
  RLC_BearerConfig->servedRadioBearer->present =  NR_RLC_BearerConfig__servedRadioBearer_PR_drb_Identity;
  RLC_BearerConfig->servedRadioBearer->choice.drb_Identity=1;
  RLC_BearerConfig->reestablishRLC=calloc(1,sizeof(*RLC_BearerConfig->reestablishRLC));
  *RLC_BearerConfig->reestablishRLC=NR_RLC_BearerConfig__reestablishRLC_true;
  RLC_BearerConfig->rlc_Config=calloc(1,sizeof(*RLC_BearerConfig->rlc_Config));

#if 0

  // RLC UM Bi-directional Bearer configuration 
  RLC_BearerConfig->rlc_Config->present = NR_RLC_Config_PR_um_Bi_Directional;
  RLC_BearerConfig->rlc_Config->choice.um_Bi_Directional = calloc(1,sizeof(*RLC_BearerConfig->rlc_Config->choice.um_Bi_Directional));
  RLC_BearerConfig->rlc_Config->choice.um_Bi_Directional->ul_UM_RLC.sn_FieldLength = calloc(1,sizeof(*RLC_BearerConfig->rlc_Config->choice.um_Bi_Directional->ul_UM_RLC.sn_FieldLength));
  *RLC_BearerConfig->rlc_Config->choice.um_Bi_Directional->ul_UM_RLC.sn_FieldLength   =    NR_SN_FieldLengthUM_size12;

  RLC_BearerConfig->rlc_Config->choice.um_Bi_Directional->dl_UM_RLC.sn_FieldLength = calloc(1,sizeof(*RLC_BearerConfig->rlc_Config->choice.um_Bi_Directional->dl_UM_RLC.sn_FieldLength));
  *RLC_BearerConfig->rlc_Config->choice.um_Bi_Directional->dl_UM_RLC.sn_FieldLength   =    NR_SN_FieldLengthUM_size12;
  RLC_BearerConfig->rlc_Config->choice.um_Bi_Directional->dl_UM_RLC.t_Reassembly = NR_T_Reassembly_ms15;

#else

  // RLC AM Bearer configuration
  RLC_BearerConfig->rlc_Config->present = NR_RLC_Config_PR_am;
  RLC_BearerConfig->rlc_Config->choice.am = calloc(1,sizeof(*RLC_BearerConfig->rlc_Config->choice.am));
  RLC_BearerConfig->rlc_Config->choice.am->ul_AM_RLC.sn_FieldLength = calloc(1,sizeof(*RLC_BearerConfig->rlc_Config->choice.am->ul_AM_RLC.sn_FieldLength));
  *RLC_BearerConfig->rlc_Config->choice.am->ul_AM_RLC.sn_FieldLength   =  NR_SN_FieldLengthAM_size18;
  RLC_BearerConfig->rlc_Config->choice.am->ul_AM_RLC.t_PollRetransmit = NR_T_PollRetransmit_ms45;
  RLC_BearerConfig->rlc_Config->choice.am->ul_AM_RLC.pollPDU          = NR_PollPDU_p64;
  RLC_BearerConfig->rlc_Config->choice.am->ul_AM_RLC.pollByte         = NR_PollByte_kB500;
  RLC_BearerConfig->rlc_Config->choice.am->ul_AM_RLC.maxRetxThreshold = NR_UL_AM_RLC__maxRetxThreshold_t32;
  RLC_BearerConfig->rlc_Config->choice.am->dl_AM_RLC.sn_FieldLength = calloc(1,sizeof(*RLC_BearerConfig->rlc_Config->choice.am->dl_AM_RLC.sn_FieldLength));
  *RLC_BearerConfig->rlc_Config->choice.am->dl_AM_RLC.sn_FieldLength = NR_SN_FieldLengthAM_size18;
  RLC_BearerConfig->rlc_Config->choice.am->dl_AM_RLC.t_Reassembly   = NR_T_Reassembly_ms15;
  RLC_BearerConfig->rlc_Config->choice.am->dl_AM_RLC.t_StatusProhibit = NR_T_StatusProhibit_ms15;

#endif

  RLC_BearerConfig->mac_LogicalChannelConfig = calloc(1,sizeof(*RLC_BearerConfig->mac_LogicalChannelConfig));
  RLC_BearerConfig->mac_LogicalChannelConfig->ul_SpecificParameters = calloc(1,sizeof(*RLC_BearerConfig->mac_LogicalChannelConfig->ul_SpecificParameters));
  RLC_BearerConfig->mac_LogicalChannelConfig->ul_SpecificParameters->priority            = 1;
  RLC_BearerConfig->mac_LogicalChannelConfig->ul_SpecificParameters->prioritisedBitRate  = NR_LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_infinity;
  RLC_BearerConfig->mac_LogicalChannelConfig->ul_SpecificParameters->bucketSizeDuration  = NR_LogicalChannelConfig__ul_SpecificParameters__bucketSizeDuration_ms50;
  RLC_BearerConfig->mac_LogicalChannelConfig->ul_SpecificParameters->allowedServingCells = NULL;
  RLC_BearerConfig->mac_LogicalChannelConfig->ul_SpecificParameters->allowedSCS_List     = NULL;
  RLC_BearerConfig->mac_LogicalChannelConfig->ul_SpecificParameters->maxPUSCH_Duration   = NULL;
  RLC_BearerConfig->mac_LogicalChannelConfig->ul_SpecificParameters->configuredGrantType1Allowed = NULL;
  RLC_BearerConfig->mac_LogicalChannelConfig->ul_SpecificParameters->logicalChannelGroup   = calloc(1,sizeof(*RLC_BearerConfig->mac_LogicalChannelConfig->ul_SpecificParameters->logicalChannelGroup));
  *RLC_BearerConfig->mac_LogicalChannelConfig->ul_SpecificParameters->logicalChannelGroup  = 1;
  RLC_BearerConfig->mac_LogicalChannelConfig->ul_SpecificParameters->schedulingRequestID   = NULL;
  RLC_BearerConfig->mac_LogicalChannelConfig->ul_SpecificParameters->logicalChannelSR_Mask = false;
  RLC_BearerConfig->mac_LogicalChannelConfig->ul_SpecificParameters->logicalChannelSR_DelayTimerApplied = false;
  RLC_BearerConfig->mac_LogicalChannelConfig->ul_SpecificParameters->bitRateQueryProhibitTimer   = NULL;
  secondaryCellGroup->rlc_BearerToAddModList = calloc(1,sizeof(*secondaryCellGroup->rlc_BearerToAddModList));
  ASN_SEQUENCE_ADD(&secondaryCellGroup->rlc_BearerToAddModList->list, RLC_BearerConfig);
  secondaryCellGroup->mac_CellGroupConfig=calloc(1,sizeof(*secondaryCellGroup->mac_CellGroupConfig));
  secondaryCellGroup->mac_CellGroupConfig->drx_Config = NULL;
  secondaryCellGroup->mac_CellGroupConfig->schedulingRequestConfig = NULL;
  secondaryCellGroup->mac_CellGroupConfig->bsr_Config=calloc(1,sizeof(*secondaryCellGroup->mac_CellGroupConfig->bsr_Config));
  secondaryCellGroup->mac_CellGroupConfig->bsr_Config->periodicBSR_Timer = NR_BSR_Config__periodicBSR_Timer_sf80;
  secondaryCellGroup->mac_CellGroupConfig->bsr_Config->retxBSR_Timer     = NR_BSR_Config__retxBSR_Timer_sf320;
  secondaryCellGroup->mac_CellGroupConfig->tag_Config=calloc(1,sizeof(*secondaryCellGroup->mac_CellGroupConfig->tag_Config));
  secondaryCellGroup->mac_CellGroupConfig->tag_Config->tag_ToReleaseList = NULL;
  secondaryCellGroup->mac_CellGroupConfig->tag_Config->tag_ToAddModList  = calloc(1,sizeof(*secondaryCellGroup->mac_CellGroupConfig->tag_Config->tag_ToAddModList));
  struct NR_TAG *tag=calloc(1,sizeof(*tag));
  tag->tag_Id             = 0;
  tag->timeAlignmentTimer = NR_TimeAlignmentTimer_infinity;
  ASN_SEQUENCE_ADD(&secondaryCellGroup->mac_CellGroupConfig->tag_Config->tag_ToAddModList->list,tag);
  secondaryCellGroup->mac_CellGroupConfig->phr_Config  = calloc(1,sizeof(*secondaryCellGroup->mac_CellGroupConfig->phr_Config));
  secondaryCellGroup->mac_CellGroupConfig->phr_Config->present = NR_SetupRelease_PHR_Config_PR_setup;
  secondaryCellGroup->mac_CellGroupConfig->phr_Config->choice.setup  = calloc(1,sizeof(*secondaryCellGroup->mac_CellGroupConfig->phr_Config->choice.setup));
  secondaryCellGroup->mac_CellGroupConfig->phr_Config->choice.setup->phr_PeriodicTimer = NR_PHR_Config__phr_PeriodicTimer_sf20;
  secondaryCellGroup->mac_CellGroupConfig->phr_Config->choice.setup->phr_ProhibitTimer = NR_PHR_Config__phr_ProhibitTimer_sf0;
  secondaryCellGroup->mac_CellGroupConfig->phr_Config->choice.setup->phr_Tx_PowerFactorChange = NR_PHR_Config__phr_Tx_PowerFactorChange_dB3;
  secondaryCellGroup->mac_CellGroupConfig->phr_Config->choice.setup->multiplePHR=false;
  secondaryCellGroup->mac_CellGroupConfig->phr_Config->choice.setup->dummy=false;
  secondaryCellGroup->mac_CellGroupConfig->phr_Config->choice.setup->phr_Type2OtherCell = false;
  secondaryCellGroup->mac_CellGroupConfig->phr_Config->choice.setup->phr_ModeOtherCG = NR_PHR_Config__phr_ModeOtherCG_real;
  secondaryCellGroup->mac_CellGroupConfig->skipUplinkTxDynamic=false;
  secondaryCellGroup->mac_CellGroupConfig->ext1 = NULL;
  secondaryCellGroup->physicalCellGroupConfig = calloc(1,sizeof(*secondaryCellGroup->physicalCellGroupConfig));
  secondaryCellGroup->physicalCellGroupConfig->harq_ACK_SpatialBundlingPUCCH=NULL;
  secondaryCellGroup->physicalCellGroupConfig->harq_ACK_SpatialBundlingPUSCH=NULL;
  secondaryCellGroup->physicalCellGroupConfig->p_NR_FR1=calloc(1,sizeof(*secondaryCellGroup->physicalCellGroupConfig->p_NR_FR1));
  *secondaryCellGroup->physicalCellGroupConfig->p_NR_FR1=20;
  secondaryCellGroup->physicalCellGroupConfig->pdsch_HARQ_ACK_Codebook=NR_PhysicalCellGroupConfig__pdsch_HARQ_ACK_Codebook_dynamic;
  secondaryCellGroup->physicalCellGroupConfig->tpc_SRS_RNTI=NULL;
  secondaryCellGroup->physicalCellGroupConfig->tpc_PUCCH_RNTI=NULL;
  secondaryCellGroup->physicalCellGroupConfig->tpc_PUSCH_RNTI=NULL;
  secondaryCellGroup->physicalCellGroupConfig->sp_CSI_RNTI=NULL;
  secondaryCellGroup->physicalCellGroupConfig->cs_RNTI=NULL;
  secondaryCellGroup->physicalCellGroupConfig->ext1=NULL;
  secondaryCellGroup->spCellConfig = calloc(1,sizeof(*secondaryCellGroup->spCellConfig));
  secondaryCellGroup->spCellConfig->servCellIndex = calloc(1,sizeof(*secondaryCellGroup->spCellConfig->servCellIndex));
  *secondaryCellGroup->spCellConfig->servCellIndex = servCellIndex;
  secondaryCellGroup->spCellConfig->reconfigurationWithSync=calloc(1,sizeof(*secondaryCellGroup->spCellConfig->reconfigurationWithSync));
  secondaryCellGroup->spCellConfig->reconfigurationWithSync->spCellConfigCommon=servingcellconfigcommon;
  secondaryCellGroup->spCellConfig->reconfigurationWithSync->newUE_Identity=(get_softmodem_params()->phy_test==1) ? 0x1234 : (taus()&0xffff);
  secondaryCellGroup->spCellConfig->reconfigurationWithSync->t304=NR_ReconfigurationWithSync__t304_ms2000;
  secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated = NULL;
  secondaryCellGroup->spCellConfig->reconfigurationWithSync->ext1                 = NULL;
  /* contention free ra */
  secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated = calloc(1,sizeof(struct NR_ReconfigurationWithSync__rach_ConfigDedicated));
  secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->present= NR_ReconfigurationWithSync__rach_ConfigDedicated_PR_uplink;
  secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink= calloc(1,sizeof(struct NR_RACH_ConfigDedicated));
  secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->cfra= calloc(1,sizeof(struct NR_CFRA));
  secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->ra_Prioritization= NULL;
  secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->cfra->occasions= calloc(1,sizeof(struct NR_CFRA__occasions));
  memcpy(&secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->cfra->occasions->rach_ConfigGeneric,
         &servingcellconfigcommon->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->rach_ConfigGeneric, sizeof(NR_RACH_ConfigGeneric_t));
  secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->cfra->occasions->ssb_perRACH_Occasion= calloc(1,sizeof(long));
  *secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->cfra->occasions->ssb_perRACH_Occasion = NR_CFRA__occasions__ssb_perRACH_Occasion_one;
  secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->cfra->resources.present = NR_CFRA__resources_PR_ssb;
  secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->cfra->resources.choice.ssb = calloc(1,sizeof(struct NR_CFRA__resources__ssb));
  secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->cfra->resources.choice.ssb->ra_ssb_OccasionMaskIndex = 0;
  struct NR_CFRA_SSB_Resource *ssbElem[8];
  ssbElem[0] = calloc(1,sizeof(struct NR_CFRA_SSB_Resource));
  ssbElem[0]->ssb = 0;
  ssbElem[0]->ra_PreambleIndex = 63;
  ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->cfra->resources.choice.ssb->ssb_ResourceList.list,ssbElem[0]);
#if 0
  ssbElem[1] = calloc(1,sizeof(struct NR_CFRA_SSB_Resource));
  ssbElem[1]->ssb = 1;
  ssbElem[1]->ra_PreambleIndex = 62;
  ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->cfra->resources.choice.ssb->ssb_ResourceList.list,ssbElem[1]);
  ssbElem[2] = calloc(1,sizeof(struct NR_CFRA_SSB_Resource));
  ssbElem[2]->ssb = 2;
  ssbElem[2]->ra_PreambleIndex = 63;
  ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->cfra->resources.choice.ssb->ssb_ResourceList.list,ssbElem[2]);
  ssbElem[3] = calloc(1,sizeof(struct NR_CFRA_SSB_Resource));
  ssbElem[3]->ssb = 3;
  ssbElem[3]->ra_PreambleIndex = 63;
  ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->cfra->resources.choice.ssb->ssb_ResourceList.list,ssbElem[3]);
  ssbElem[4] = calloc(1,sizeof(struct NR_CFRA_SSB_Resource));
  ssbElem[4]->ssb = 4;
  ssbElem[4]->ra_PreambleIndex = 63;
  ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->cfra->resources.choice.ssb->ssb_ResourceList.list,ssbElem[4]);
  ssbElem[5] = calloc(1,sizeof(struct NR_CFRA_SSB_Resource));
  ssbElem[5]->ssb = 5;
  ssbElem[5]->ra_PreambleIndex = 63;
  ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->cfra->resources.choice.ssb->ssb_ResourceList.list,ssbElem[5]);
  ssbElem[6] = calloc(1,sizeof(struct NR_CFRA_SSB_Resource));
  ssbElem[6]->ssb = 6;
  ssbElem[6]->ra_PreambleIndex = 63;
  ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->cfra->resources.choice.ssb->ssb_ResourceList.list,ssbElem[6]);
  ssbElem[7] = calloc(1,sizeof(struct NR_CFRA_SSB_Resource));
  ssbElem[7]->ssb = 7;
  ssbElem[7]->ra_PreambleIndex = 63;
  ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->cfra->resources.choice.ssb->ssb_ResourceList.list,ssbElem[7]);
#endif
  secondaryCellGroup->spCellConfig->reconfigurationWithSync->rach_ConfigDedicated->choice.uplink->cfra->ext1 = NULL;
  secondaryCellGroup->spCellConfig->rlf_TimersAndConstants = calloc(1,sizeof(*secondaryCellGroup->spCellConfig->rlf_TimersAndConstants));
  secondaryCellGroup->spCellConfig->rlf_TimersAndConstants->present = NR_SetupRelease_RLF_TimersAndConstants_PR_setup;
  secondaryCellGroup->spCellConfig->rlf_TimersAndConstants->choice.setup=calloc(1,sizeof(*secondaryCellGroup->spCellConfig->rlf_TimersAndConstants->choice.setup));
  secondaryCellGroup->spCellConfig->rlf_TimersAndConstants->choice.setup->t310 = NR_RLF_TimersAndConstants__t310_ms2000;
  secondaryCellGroup->spCellConfig->rlf_TimersAndConstants->choice.setup->n310 = NR_RLF_TimersAndConstants__n310_n10;
  secondaryCellGroup->spCellConfig->rlf_TimersAndConstants->choice.setup->n311 = NR_RLF_TimersAndConstants__n311_n1;
  secondaryCellGroup->spCellConfig->rlf_TimersAndConstants->choice.setup->ext1 = calloc(1,sizeof(*secondaryCellGroup->spCellConfig->rlf_TimersAndConstants->choice.setup->ext1));
  secondaryCellGroup->spCellConfig->rlf_TimersAndConstants->choice.setup->ext1->t311 = NR_RLF_TimersAndConstants__ext1__t311_ms30000;
  secondaryCellGroup->spCellConfig->rlmInSyncOutOfSyncThreshold                   = NULL;
  secondaryCellGroup->spCellConfig->spCellConfigDedicated = calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated));
  secondaryCellGroup->spCellConfig->spCellConfigDedicated->tdd_UL_DL_ConfigurationDedicated = NULL;
  secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP = calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP));
  secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdcch_Config=NULL;
  secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config=calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config));
  secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->present = NR_SetupRelease_PDSCH_Config_PR_setup;
  secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup = calloc(1,
      sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup));
  secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->dataScramblingIdentityPDSCH = NULL;
  secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA = calloc(1,
      sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA));
  secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->present= NR_SetupRelease_DMRS_DownlinkConfig_PR_setup;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup = calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup));

 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->dmrs_Type=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->maxLength=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->scramblingID0=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->scramblingID1=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->phaseTrackingRS=NULL;

 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->dmrs_AdditionalPosition = calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->dmrs_AdditionalPosition));
 *secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->dmrs_AdditionalPosition = NR_DMRS_DownlinkConfig__dmrs_AdditionalPosition_pos0;


 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->tci_StatesToAddModList=calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->tci_StatesToAddModList));

 int n_ssb = 0;
 NR_TCI_State_t *tcic[8];
 for (int i=0;i<8;i++) {
   if ((bitmap>>(7-i))&0x01){
     tcic[i]=calloc(1,sizeof(*tcic[i]));
     tcic[i]->tci_StateId=n_ssb;
     tcic[i]->qcl_Type1.cell=NULL;
     tcic[i]->qcl_Type1.bwp_Id=calloc(1,sizeof(*tcic[i]->qcl_Type1.bwp_Id));
     *tcic[i]->qcl_Type1.bwp_Id=1;
     tcic[i]->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_ssb;
     tcic[i]->qcl_Type1.referenceSignal.choice.ssb = i;
     tcic[i]->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeC;
     ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tcic[i]);
     n_ssb++;
   }
 }


#if 0

 NR_TCI_State_t*tci0=calloc(1,sizeof(*tci0));
 tci0->tci_StateId=0;
 tci0->qcl_Type1.cell=NULL;
 tci0->qcl_Type1.bwp_Id=calloc(1,sizeof(*tci0->qcl_Type1.bwp_Id));
 *tci0->qcl_Type1.bwp_Id=1;
 tci0->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_csi_rs;
 tci0->qcl_Type1.referenceSignal.choice.csi_rs = 2;
 tci0->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeA;
 ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tci0);

 NR_TCI_State_t*tci1=calloc(1,sizeof(*tci1));
 tci1->tci_StateId=1;
 tci1->qcl_Type1.cell=NULL;
 tci1->qcl_Type1.bwp_Id=calloc(1,sizeof(*tci1->qcl_Type1.bwp_Id));
 *tci1->qcl_Type1.bwp_Id=1;
 tci1->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_csi_rs;
 tci1->qcl_Type1.referenceSignal.choice.csi_rs = 6;
 tci1->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeA;
 ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tci1);

 NR_TCI_State_t*tci2=calloc(1,sizeof(*tci2));
 tci2->tci_StateId=2;
 tci2->qcl_Type1.cell=NULL;
 tci2->qcl_Type1.bwp_Id=calloc(1,sizeof(*tci2->qcl_Type1.bwp_Id));
 *tci2->qcl_Type1.bwp_Id=1;
 tci2->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_csi_rs;
 tci2->qcl_Type1.referenceSignal.choice.csi_rs = 10;
 tci2->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeA;
 ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tci2);

 NR_TCI_State_t *tci3=calloc(1,sizeof(*tci3));
 tci3->tci_StateId=3;
 tci3->qcl_Type1.cell=NULL;
 tci3->qcl_Type1.bwp_Id=calloc(1,sizeof(*tci3->qcl_Type1.bwp_Id));
 *tci3->qcl_Type1.bwp_Id=1;
 tci3->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_csi_rs;
 tci3->qcl_Type1.referenceSignal.choice.csi_rs = 14;
 tci3->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeA;
 ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tci3);

 NR_TCI_State_t*tci4=calloc(1,sizeof(*tci4));
 tci4->tci_StateId=4;
 tci4->qcl_Type1.cell=NULL;
 tci4->qcl_Type1.bwp_Id=calloc(1,sizeof(*tci4->qcl_Type1.bwp_Id));
 *tci4->qcl_Type1.bwp_Id=1;
 tci4->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_csi_rs;
 tci4->qcl_Type1.referenceSignal.choice.csi_rs = 18;
 tci4->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeA;
 ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tci4);

 NR_TCI_State_t*tci5=calloc(1,sizeof(*tci5));
 tci5->tci_StateId=5;
 tci5->qcl_Type1.cell=NULL;
 tci5->qcl_Type1.bwp_Id=calloc(1,sizeof(*tci5->qcl_Type1.bwp_Id));
 *tci5->qcl_Type1.bwp_Id=1;
 tci5->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_csi_rs;
 tci5->qcl_Type1.referenceSignal.choice.csi_rs = 22;
 tci5->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeA;
 ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tci5);

 NR_TCI_State_t*tci6=calloc(1,sizeof(*tci6));
 tci6->tci_StateId=6;
 tci6->qcl_Type1.cell=NULL;
 tci6->qcl_Type1.bwp_Id=calloc(1,sizeof(*tci6->qcl_Type1.bwp_Id));
 *tci6->qcl_Type1.bwp_Id=1;
 tci6->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_csi_rs;
 tci6->qcl_Type1.referenceSignal.choice.csi_rs = 26;
 tci6->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeA;
 ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tci6);

 NR_TCI_State_t*tci7=calloc(1,sizeof(*tci7));
 tci7->tci_StateId=7;
 tci7->qcl_Type1.cell=NULL;
 tci7->qcl_Type1.bwp_Id=calloc(1,sizeof(*tci7->qcl_Type1.bwp_Id));
 *tci7->qcl_Type1.bwp_Id=1;
 tci7->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_csi_rs;
 tci7->qcl_Type1.referenceSignal.choice.csi_rs = 30;
 tci7->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeA;
 ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tci7);

#endif

 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->tci_StatesToReleaseList=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->vrb_ToPRB_Interleaver=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->resourceAllocation=NR_PDSCH_Config__resourceAllocation_resourceAllocationType1;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->pdsch_TimeDomainAllocationList=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->pdsch_AggregationFactor=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->rateMatchPatternToAddModList=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->rateMatchPatternToReleaseList=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->rateMatchPatternGroup1=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->rateMatchPatternGroup2=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->rbg_Size=NR_PDSCH_Config__rbg_Size_config1;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->mcs_Table=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->maxNrofCodeWordsScheduledByDCI = calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->maxNrofCodeWordsScheduledByDCI));
 *secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->maxNrofCodeWordsScheduledByDCI = NR_PDSCH_Config__maxNrofCodeWordsScheduledByDCI_n1;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->prb_BundlingType.present = NR_PDSCH_Config__prb_BundlingType_PR_staticBundling;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->prb_BundlingType.choice.staticBundling = calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->prb_BundlingType.choice.staticBundling));
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->prb_BundlingType.choice.staticBundling->bundleSize =
   calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->prb_BundlingType.choice.staticBundling->bundleSize));
 *secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->prb_BundlingType.choice.staticBundling->bundleSize = NR_PDSCH_Config__prb_BundlingType__staticBundling__bundleSize_wideband;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->zp_CSI_RS_ResourceToAddModList=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->zp_CSI_RS_ResourceToReleaseList=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->aperiodic_ZP_CSI_RS_ResourceSetsToAddModList=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->aperiodic_ZP_CSI_RS_ResourceSetsToReleaseList=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->sp_ZP_CSI_RS_ResourceSetsToAddModList=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->sp_ZP_CSI_RS_ResourceSetsToReleaseList=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->pdsch_Config->choice.setup->p_ZP_CSI_RS_ResourceSet=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->sps_Config = NULL; //calloc(1,sizeof(struct NR_SetupRelease_SPS_Config));

 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->radioLinkMonitoringConfig = NULL;
#if 0
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->radioLinkMonitoringConfig = calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->radioLinkMonitoringConfig));
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->radioLinkMonitoringConfig->present = NR_SetupRelease_RadioLinkMonitoringConfig_PR_setup;

 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->radioLinkMonitoringConfig->choice.setup = calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->radioLinkMonitoringConfig->choice.setup));
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->radioLinkMonitoringConfig->choice.setup->failureDetectionResourcesToAddModList=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->radioLinkMonitoringConfig->choice.setup->failureDetectionResourcesToReleaseList=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->radioLinkMonitoringConfig->choice.setup->beamFailureInstanceMaxCount = calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->radioLinkMonitoringConfig->choice.setup->beamFailureInstanceMaxCount));
 *secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->radioLinkMonitoringConfig->choice.setup->beamFailureInstanceMaxCount = NR_RadioLinkMonitoringConfig__beamFailureInstanceMaxCount_n3;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->radioLinkMonitoringConfig->choice.setup->beamFailureDetectionTimer = calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->radioLinkMonitoringConfig->choice.setup->beamFailureDetectionTimer));
 *secondaryCellGroup->spCellConfig->spCellConfigDedicated->initialDownlinkBWP->radioLinkMonitoringConfig->choice.setup->beamFailureDetectionTimer = NR_RadioLinkMonitoringConfig__beamFailureDetectionTimer_pbfd2;
#endif

 secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToReleaseList= NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList = calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList));

 NR_BWP_Downlink_t *bwp=calloc(1,sizeof(*bwp));
 bwp->bwp_Id=1;
 bwp->bwp_Common=calloc(1,sizeof(*bwp->bwp_Common));
 // copy common BWP size from initial BWP except for bandwdith
 memcpy((void*)&bwp->bwp_Common->genericParameters,
	&servingcellconfigcommon->downlinkConfigCommon->initialDownlinkBWP->genericParameters,
	sizeof(bwp->bwp_Common->genericParameters));
 bwp->bwp_Common->genericParameters.locationAndBandwidth=PRBalloc_to_locationandbandwidth(servingcellconfigcommon->downlinkConfigCommon->frequencyInfoDL->scs_SpecificCarrierList.list.array[0]->carrierBandwidth,0);

 
 bwp->bwp_Common->pdcch_ConfigCommon=calloc(1,sizeof(*bwp->bwp_Common->pdcch_ConfigCommon));
 bwp->bwp_Common->pdcch_ConfigCommon->present = NR_SetupRelease_PDCCH_ConfigCommon_PR_setup;
 bwp->bwp_Common->pdcch_ConfigCommon->choice.setup = calloc(1,sizeof(*bwp->bwp_Common->pdcch_ConfigCommon->choice.setup));
 bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->controlResourceSetZero=NULL;
 bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->commonControlResourceSet=calloc(1,sizeof(*bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->commonControlResourceSet));

 NR_ControlResourceSet_t *coreset = calloc(1,sizeof(*coreset));
 coreset->controlResourceSetId=1;
 // frequencyDomainResources '11111111 11111111 00000000 00000000 00000000 00000'B,
 coreset->frequencyDomainResources.buf = calloc(1,6);
 coreset->frequencyDomainResources.buf[0] = 0xff;
 coreset->frequencyDomainResources.buf[1] = 0xff;
 coreset->frequencyDomainResources.buf[2] = 0;
 coreset->frequencyDomainResources.buf[3] = 0;
 coreset->frequencyDomainResources.buf[4] = 0;
 coreset->frequencyDomainResources.buf[5] = 0;
 coreset->frequencyDomainResources.size = 6;
 coreset->frequencyDomainResources.bits_unused = 3;
 coreset->duration=1;
 coreset->cce_REG_MappingType.present = NR_ControlResourceSet__cce_REG_MappingType_PR_nonInterleaved;
 coreset->precoderGranularity = NR_ControlResourceSet__precoderGranularity_sameAsREG_bundle;

 coreset->tci_StatesPDCCH_ToAddList=calloc(1,sizeof(*coreset->tci_StatesPDCCH_ToAddList));
 NR_TCI_StateId_t *tci[8];
 for (int i=0;i<8;i++) {
   if ((bitmap>>(7-i))&0x01){
     tci[i]=calloc(1,sizeof(*tci[i]));
     *tci[i] = i;
     ASN_SEQUENCE_ADD(&coreset->tci_StatesPDCCH_ToAddList->list,tci[i]);
   }
 }
 coreset->tci_StatesPDCCH_ToReleaseList = NULL;
 coreset->tci_PresentInDCI = NULL;
 coreset->pdcch_DMRS_ScramblingID = NULL;

 bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->commonControlResourceSet = coreset;

 bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->searchSpaceZero=NULL;
 bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->commonSearchSpaceList=NULL;
 bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->commonSearchSpaceList=calloc(1,sizeof(*bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->commonSearchSpaceList));

 NR_SearchSpace_t *ss=calloc(1,sizeof(*ss));
 ss->searchSpaceId = 1;
 ss->controlResourceSetId=calloc(1,sizeof(*ss->controlResourceSetId));
 *ss->controlResourceSetId=1;
 ss->monitoringSlotPeriodicityAndOffset = calloc(1,sizeof(*ss->monitoringSlotPeriodicityAndOffset));
 ss->monitoringSlotPeriodicityAndOffset->present = NR_SearchSpace__monitoringSlotPeriodicityAndOffset_PR_sl1;
 ss->duration=NULL; 
 ss->monitoringSymbolsWithinSlot = calloc(1,sizeof(*ss->monitoringSymbolsWithinSlot));
 ss->monitoringSymbolsWithinSlot->buf = calloc(1,2);
 // should be '1100 0000 0000 00'B (LSB first!), first two symols in slot, adjust if needed
 ss->monitoringSymbolsWithinSlot->buf[1] = 0;
 ss->monitoringSymbolsWithinSlot->buf[0] = (1<<7) | (1<<6);
 ss->monitoringSymbolsWithinSlot->size = 2;
 ss->monitoringSymbolsWithinSlot->bits_unused = 2;
 ss->nrofCandidates = calloc(1,sizeof(*ss->nrofCandidates));
 ss->nrofCandidates->aggregationLevel1 = NR_SearchSpace__nrofCandidates__aggregationLevel1_n0;
 ss->nrofCandidates->aggregationLevel2 = NR_SearchSpace__nrofCandidates__aggregationLevel2_n0;
 ss->nrofCandidates->aggregationLevel4 = NR_SearchSpace__nrofCandidates__aggregationLevel4_n1;
 ss->nrofCandidates->aggregationLevel8 = NR_SearchSpace__nrofCandidates__aggregationLevel8_n0;
 ss->nrofCandidates->aggregationLevel16 = NR_SearchSpace__nrofCandidates__aggregationLevel16_n0;
 ss->searchSpaceType = calloc(1,sizeof(*ss->searchSpaceType));
 ss->searchSpaceType->present = NR_SearchSpace__searchSpaceType_PR_common;
 ss->searchSpaceType->choice.common=calloc(1,sizeof(*ss->searchSpaceType->choice.common));
 ss->searchSpaceType->choice.common->dci_Format0_0_AndFormat1_0 = calloc(1,sizeof(*ss->searchSpaceType->choice.common->dci_Format0_0_AndFormat1_0));

 ASN_SEQUENCE_ADD(&bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->commonSearchSpaceList->list,ss);
 

 bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->searchSpaceSIB1=NULL;
 bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->searchSpaceOtherSystemInformation=NULL;
 bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->pagingSearchSpace=NULL;
 bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->ra_SearchSpace=calloc(1,sizeof(*bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->ra_SearchSpace));
 *bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->ra_SearchSpace=1;
 bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->ext1=NULL;

 bwp->bwp_Common->pdsch_ConfigCommon=calloc(1,sizeof(*bwp->bwp_Common->pdsch_ConfigCommon));
 bwp->bwp_Common->pdsch_ConfigCommon->present = NR_SetupRelease_PDSCH_ConfigCommon_PR_setup;
 bwp->bwp_Common->pdsch_ConfigCommon->choice.setup = calloc(1,sizeof(*bwp->bwp_Common->pdsch_ConfigCommon->choice.setup));
 bwp->bwp_Common->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList = calloc(1,sizeof(*bwp->bwp_Common->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList));

 // copy PDSCH TimeDomainResourceAllocation from InitialBWP
 
 NR_PDSCH_TimeDomainResourceAllocation_t *pdschi;
 for (int i=0;i<servingcellconfigcommon->downlinkConfigCommon->initialDownlinkBWP->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList->list.count;i++) {
   pdschi= calloc(1,sizeof(*pdschi));
   AssertFatal(servingcellconfigcommon->downlinkConfigCommon->initialDownlinkBWP->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList->list.array[i]->k0!=NULL,"element %d is null\n",i);
 
   pdschi->k0 = calloc(1,sizeof(*pdschi->k0));
   *pdschi->k0 = *servingcellconfigcommon->downlinkConfigCommon->initialDownlinkBWP->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList->list.array[i]->k0;
   pdschi->mappingType = servingcellconfigcommon->downlinkConfigCommon->initialDownlinkBWP->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList->list.array[i]->mappingType;
   pdschi->startSymbolAndLength = servingcellconfigcommon->downlinkConfigCommon->initialDownlinkBWP->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList->list.array[i]->startSymbolAndLength;
   ASN_SEQUENCE_ADD(&bwp->bwp_Common->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList->list,pdschi);
 }

 bwp->bwp_Dedicated=calloc(1,sizeof(*bwp->bwp_Dedicated));

 bwp->bwp_Dedicated->pdcch_Config=calloc(1,sizeof(*bwp->bwp_Dedicated->pdcch_Config));
 bwp->bwp_Dedicated->pdcch_Config->present = NR_SetupRelease_PDCCH_Config_PR_setup;
 bwp->bwp_Dedicated->pdcch_Config->choice.setup = calloc(1,sizeof(*bwp->bwp_Dedicated->pdcch_Config->choice.setup));
 bwp->bwp_Dedicated->pdcch_Config->choice.setup->controlResourceSetToAddModList = calloc(1,sizeof(*bwp->bwp_Dedicated->pdcch_Config->choice.setup->controlResourceSetToAddModList));

 ASN_SEQUENCE_ADD(&bwp->bwp_Dedicated->pdcch_Config->choice.setup->controlResourceSetToAddModList->list,
		  coreset);

 bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToAddModList = calloc(1,sizeof(*bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToAddModList));

 NR_SearchSpace_t *ss2 = calloc(1,sizeof(*ss2));

 ss2->searchSpaceId=2;
 ss2->controlResourceSetId=calloc(1,sizeof(*ss2->controlResourceSetId));
 *ss2->controlResourceSetId=1;
 ss2->monitoringSlotPeriodicityAndOffset=calloc(1,sizeof(*ss2->monitoringSlotPeriodicityAndOffset));
 ss2->monitoringSlotPeriodicityAndOffset->present = NR_SearchSpace__monitoringSlotPeriodicityAndOffset_PR_sl1;
 ss2->monitoringSlotPeriodicityAndOffset->choice.sl1=(NULL_t)0;
 ss2->duration=NULL;
 ss2->monitoringSymbolsWithinSlot = calloc(1,sizeof(*ss2->monitoringSymbolsWithinSlot));
 ss2->monitoringSymbolsWithinSlot->buf = calloc(1,2);
 ss2->monitoringSymbolsWithinSlot->size = 2;
 ss2->monitoringSymbolsWithinSlot->bits_unused = 2;
 ss2->monitoringSymbolsWithinSlot->buf[0]=0xc0;
 ss2->monitoringSymbolsWithinSlot->buf[1]=0x0;
 ss2->nrofCandidates=calloc(1,sizeof(*ss2->nrofCandidates));
 ss2->nrofCandidates->aggregationLevel1 = NR_SearchSpace__nrofCandidates__aggregationLevel1_n0;
 ss2->nrofCandidates->aggregationLevel2 = NR_SearchSpace__nrofCandidates__aggregationLevel2_n0;
 ss2->nrofCandidates->aggregationLevel4 = NR_SearchSpace__nrofCandidates__aggregationLevel4_n4;
 ss2->nrofCandidates->aggregationLevel8 = NR_SearchSpace__nrofCandidates__aggregationLevel8_n0;
 ss2->nrofCandidates->aggregationLevel16 = NR_SearchSpace__nrofCandidates__aggregationLevel16_n0;
 ss2->searchSpaceType=calloc(1,sizeof(*ss2->searchSpaceType));
 ss2->searchSpaceType->present = NR_SearchSpace__searchSpaceType_PR_ue_Specific;
 ss2->searchSpaceType->choice.ue_Specific = calloc(1,sizeof(*ss2->searchSpaceType->choice.ue_Specific));
 ss2->searchSpaceType->choice.ue_Specific->dci_Formats=NR_SearchSpace__searchSpaceType__ue_Specific__dci_Formats_formats0_1_And_1_1;

 ASN_SEQUENCE_ADD(&bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToAddModList->list,
		  ss2);


 bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToReleaseList = NULL;

 bwp->bwp_Dedicated->pdsch_Config = calloc(1,sizeof(*bwp->bwp_Dedicated->pdsch_Config));

  bwp->bwp_Dedicated->pdsch_Config->present = NR_SetupRelease_PDSCH_Config_PR_setup;
  bwp->bwp_Dedicated->pdsch_Config->choice.setup = calloc(1,sizeof(*bwp->bwp_Dedicated->pdsch_Config->choice.setup));
  bwp->bwp_Dedicated->pdsch_Config->choice.setup->dataScramblingIdentityPDSCH = NULL;
  bwp->bwp_Dedicated->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA = calloc(1,sizeof(*bwp->bwp_Dedicated->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA));
  bwp->bwp_Dedicated->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->present= NR_SetupRelease_DMRS_DownlinkConfig_PR_setup;

 bwp->bwp_Dedicated->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup = calloc(1,sizeof(*bwp->bwp_Dedicated->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup));

 bwp->bwp_Dedicated->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->dmrs_Type=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->maxLength=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->scramblingID0=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->scramblingID1=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->phaseTrackingRS=NULL;

 bwp->bwp_Dedicated->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->dmrs_AdditionalPosition = calloc(1,sizeof(*bwp->bwp_Dedicated->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->dmrs_AdditionalPosition));
 *bwp->bwp_Dedicated->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->dmrs_AdditionalPosition = NR_DMRS_DownlinkConfig__dmrs_AdditionalPosition_pos0;

 //bwp->bwp_Dedicated->pdsch_Config->choice.setup->tci_StatesToAddModList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->tci_StatesToAddModList=calloc(1,sizeof(*bwp->bwp_Dedicated->pdsch_Config->choice.setup->tci_StatesToAddModList));

#if 0
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->tci_StatesToAddModList=calloc(1,sizeof(*bwp->bwp_Dedicated->pdsch_Config->choice.setup->tci_StatesToAddModList));

 NR_TCI_State_t*tcid0=calloc(1,sizeof(*tcid0));
 tcid0->tci_StateId=0;
 tcid0->qcl_Type1.cell=NULL;
 tcid0->qcl_Type1.bwp_Id=calloc(1,sizeof(*tcid0->qcl_Type1.bwp_Id));
 *tcid0->qcl_Type1.bwp_Id=1;
 tcid0->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_csi_rs;
 tcid0->qcl_Type1.referenceSignal.choice.csi_rs = 2;
 tcid0->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeA;
 ASN_SEQUENCE_ADD(&bwp->bwp_Dedicated->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tcid0);

 NR_TCI_State_t*tcid1=calloc(1,sizeof(*tcid1));
 tcid1->tci_StateId=0;
 tcid1->qcl_Type1.cell=NULL;
 tcid1->qcl_Type1.bwp_Id=calloc(1,sizeof(*tcid1->qcl_Type1.bwp_Id));
 *tcid1->qcl_Type1.bwp_Id=1;
 tcid1->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_csi_rs;
 tcid1->qcl_Type1.referenceSignal.choice.csi_rs = 6;
 tcid1->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeA;
 ASN_SEQUENCE_ADD(&bwp->bwp_Dedicated->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tcid1);

 NR_TCI_State_t*tcid2=calloc(1,sizeof(*tcid2));
 tcid2->tci_StateId=2;
 tcid2->qcl_Type1.cell=NULL;
 tcid2->qcl_Type1.bwp_Id=calloc(1,sizeof(*tcid2->qcl_Type1.bwp_Id));
 *tcid2->qcl_Type1.bwp_Id=1;
 tcid2->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_csi_rs;
 tcid2->qcl_Type1.referenceSignal.choice.csi_rs = 10;
 tcid2->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeA;
 ASN_SEQUENCE_ADD(&bwp->bwp_Dedicated->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tcid2);

 NR_TCI_State_t*tcid3=calloc(1,sizeof(*tcid3));
 tcid3->tci_StateId=3;
 tcid3->qcl_Type1.cell=NULL;
 tcid3->qcl_Type1.bwp_Id=calloc(1,sizeof(*tcid3->qcl_Type1.bwp_Id));
 *tcid3->qcl_Type1.bwp_Id=1;
 tcid3->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_csi_rs;
 tcid3->qcl_Type1.referenceSignal.choice.csi_rs = 14;
 tcid3->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeA;
 ASN_SEQUENCE_ADD(&bwp->bwp_Dedicated->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tcid3);

 NR_TCI_State_t*tcid4=calloc(1,sizeof(*tcid4));
 tcid4->tci_StateId=4;
 tcid4->qcl_Type1.cell=NULL;
 tcid4->qcl_Type1.bwp_Id=calloc(1,sizeof(*tcid4->qcl_Type1.bwp_Id));
 *tcid4->qcl_Type1.bwp_Id=1;
 tcid4->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_csi_rs;
 tcid4->qcl_Type1.referenceSignal.choice.csi_rs = 18;
 tcid4->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeA;
 ASN_SEQUENCE_ADD(&bwp->bwp_Dedicated->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tcid4);

 NR_TCI_State_t*tcid5=calloc(1,sizeof(*tcid5));
 tcid5->tci_StateId=5;
 tcid5->qcl_Type1.cell=NULL;
 tcid5->qcl_Type1.bwp_Id=calloc(1,sizeof(*tcid5->qcl_Type1.bwp_Id));
 *tcid5->qcl_Type1.bwp_Id=1;
 tcid5->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_csi_rs;
 tcid5->qcl_Type1.referenceSignal.choice.csi_rs = 22;
 tcid5->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeA;
 ASN_SEQUENCE_ADD(&bwp->bwp_Dedicated->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tcid5);

 NR_TCI_State_t*tcid6=calloc(1,sizeof(*tcid6));
 tcid6->tci_StateId=6;
 tcid6->qcl_Type1.cell=NULL;
 tcid6->qcl_Type1.bwp_Id=calloc(1,sizeof(*tcid6->qcl_Type1.bwp_Id));
 *tcid6->qcl_Type1.bwp_Id=1;
 tcid6->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_csi_rs;
 tcid6->qcl_Type1.referenceSignal.choice.csi_rs = 26;
 tcid6->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeA;
 ASN_SEQUENCE_ADD(&bwp->bwp_Dedicated->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tcid6);

 NR_TCI_State_t*tcid7=calloc(1,sizeof(*tcid7));
 tcid7->tci_StateId=7;
 tcid7->qcl_Type1.cell=NULL;
 tcid7->qcl_Type1.bwp_Id=calloc(1,sizeof(*tcid7->qcl_Type1.bwp_Id));
 *tcid7->qcl_Type1.bwp_Id=1;
 tcid7->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_csi_rs;
 tcid7->qcl_Type1.referenceSignal.choice.csi_rs = 30;
 tcid7->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeA;
 ASN_SEQUENCE_ADD(&bwp->bwp_Dedicated->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tcid7);
#endif


 n_ssb = 0;
 NR_TCI_State_t *tcid[8];
 for (int i=0;i<8;i++) {
   if ((bitmap>>(7-i))&0x01){
     tcid[i]=calloc(1,sizeof(*tcid[i]));
     tcid[i]->tci_StateId=n_ssb;
     tcid[i]->qcl_Type1.cell=NULL;
     tcid[i]->qcl_Type1.bwp_Id=calloc(1,sizeof(*tcid[i]->qcl_Type1.bwp_Id));
     *tcid[i]->qcl_Type1.bwp_Id=1;
     tcid[i]->qcl_Type1.referenceSignal.present = NR_QCL_Info__referenceSignal_PR_ssb;
     tcid[i]->qcl_Type1.referenceSignal.choice.ssb = i;
     tcid[i]->qcl_Type1.qcl_Type=NR_QCL_Info__qcl_Type_typeC;
     ASN_SEQUENCE_ADD(&bwp->bwp_Dedicated->pdsch_Config->choice.setup->tci_StatesToAddModList->list,tcid[i]);
     n_ssb++;
   }
 }


 bwp->bwp_Dedicated->pdsch_Config->choice.setup->tci_StatesToReleaseList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->vrb_ToPRB_Interleaver=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->resourceAllocation=NR_PDSCH_Config__resourceAllocation_resourceAllocationType1;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->pdsch_TimeDomainAllocationList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->pdsch_AggregationFactor=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->rateMatchPatternToAddModList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->rateMatchPatternToReleaseList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->rateMatchPatternGroup1=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->rateMatchPatternGroup2=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->rbg_Size=NR_PDSCH_Config__rbg_Size_config1;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->mcs_Table=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->maxNrofCodeWordsScheduledByDCI = calloc(1,sizeof(*bwp->bwp_Dedicated->pdsch_Config->choice.setup->maxNrofCodeWordsScheduledByDCI));
 *bwp->bwp_Dedicated->pdsch_Config->choice.setup->maxNrofCodeWordsScheduledByDCI = NR_PDSCH_Config__maxNrofCodeWordsScheduledByDCI_n1;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->prb_BundlingType.present = NR_PDSCH_Config__prb_BundlingType_PR_staticBundling;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->prb_BundlingType.choice.staticBundling = calloc(1,sizeof(*bwp->bwp_Dedicated->pdsch_Config->choice.setup->prb_BundlingType.choice.staticBundling));
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->prb_BundlingType.choice.staticBundling->bundleSize =
   calloc(1,sizeof(*bwp->bwp_Dedicated->pdsch_Config->choice.setup->prb_BundlingType.choice.staticBundling->bundleSize));
 *bwp->bwp_Dedicated->pdsch_Config->choice.setup->prb_BundlingType.choice.staticBundling->bundleSize = NR_PDSCH_Config__prb_BundlingType__staticBundling__bundleSize_wideband;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->zp_CSI_RS_ResourceToAddModList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->zp_CSI_RS_ResourceToReleaseList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->aperiodic_ZP_CSI_RS_ResourceSetsToAddModList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->aperiodic_ZP_CSI_RS_ResourceSetsToReleaseList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->sp_ZP_CSI_RS_ResourceSetsToAddModList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->sp_ZP_CSI_RS_ResourceSetsToReleaseList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->p_ZP_CSI_RS_ResourceSet=NULL;

 bwp->bwp_Dedicated->pdsch_Config->choice.setup->tci_StatesToReleaseList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->vrb_ToPRB_Interleaver=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->resourceAllocation=NR_PDSCH_Config__resourceAllocation_resourceAllocationType1;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->pdsch_TimeDomainAllocationList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->pdsch_AggregationFactor=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->rateMatchPatternToAddModList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->rateMatchPatternToReleaseList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->rateMatchPatternGroup1=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->rateMatchPatternGroup2=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->rbg_Size=NR_PDSCH_Config__rbg_Size_config1;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->mcs_Table=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->maxNrofCodeWordsScheduledByDCI = calloc(1,sizeof(*bwp->bwp_Dedicated->pdsch_Config->choice.setup->maxNrofCodeWordsScheduledByDCI));
 *bwp->bwp_Dedicated->pdsch_Config->choice.setup->maxNrofCodeWordsScheduledByDCI = NR_PDSCH_Config__maxNrofCodeWordsScheduledByDCI_n1;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->prb_BundlingType.present = NR_PDSCH_Config__prb_BundlingType_PR_staticBundling;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->prb_BundlingType.choice.staticBundling = calloc(1,sizeof(*bwp->bwp_Dedicated->pdsch_Config->choice.setup->prb_BundlingType.choice.staticBundling));
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->prb_BundlingType.choice.staticBundling->bundleSize =
   calloc(1,sizeof(*bwp->bwp_Dedicated->pdsch_Config->choice.setup->prb_BundlingType.choice.staticBundling->bundleSize));
 *bwp->bwp_Dedicated->pdsch_Config->choice.setup->prb_BundlingType.choice.staticBundling->bundleSize = NR_PDSCH_Config__prb_BundlingType__staticBundling__bundleSize_wideband;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->zp_CSI_RS_ResourceToAddModList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->zp_CSI_RS_ResourceToReleaseList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->aperiodic_ZP_CSI_RS_ResourceSetsToAddModList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->aperiodic_ZP_CSI_RS_ResourceSetsToReleaseList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->sp_ZP_CSI_RS_ResourceSetsToAddModList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->sp_ZP_CSI_RS_ResourceSetsToReleaseList=NULL;
 bwp->bwp_Dedicated->pdsch_Config->choice.setup->p_ZP_CSI_RS_ResourceSet=NULL;
 bwp->bwp_Dedicated->sps_Config = NULL; //calloc(1,sizeof(struct NR_SetupRelease_SPS_Config));

 bwp->bwp_Dedicated->radioLinkMonitoringConfig = NULL;
#if 0
 bwp->bwp_Dedicated->radioLinkMonitoringConfig = calloc(1,sizeof(*bwp->bwp_Dedicated->radioLinkMonitoringConfig));
 bwp->bwp_Dedicated->radioLinkMonitoringConfig->present = NR_SetupRelease_RadioLinkMonitoringConfig_PR_setup;

 bwp->bwp_Dedicated->radioLinkMonitoringConfig->choice.setup = calloc(1,sizeof(*bwp->bwp_Dedicated->radioLinkMonitoringConfig->choice.setup));
 bwp->bwp_Dedicated->radioLinkMonitoringConfig->choice.setup->failureDetectionResourcesToAddModList=NULL;
 bwp->bwp_Dedicated->radioLinkMonitoringConfig->choice.setup->failureDetectionResourcesToReleaseList=NULL;
 bwp->bwp_Dedicated->radioLinkMonitoringConfig->choice.setup->beamFailureInstanceMaxCount = calloc(1,sizeof(*bwp->bwp_Dedicated->radioLinkMonitoringConfig->choice.setup->beamFailureInstanceMaxCount));
 *bwp->bwp_Dedicated->radioLinkMonitoringConfig->choice.setup->beamFailureInstanceMaxCount = NR_RadioLinkMonitoringConfig__beamFailureInstanceMaxCount_n3;
 bwp->bwp_Dedicated->radioLinkMonitoringConfig->choice.setup->beamFailureDetectionTimer = calloc(1,sizeof(*bwp->bwp_Dedicated->radioLinkMonitoringConfig->choice.setup->beamFailureDetectionTimer));
 *bwp->bwp_Dedicated->radioLinkMonitoringConfig->choice.setup->beamFailureDetectionTimer = NR_RadioLinkMonitoringConfig__beamFailureDetectionTimer_pbfd2;
#endif
 
 ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list,bwp);

 secondaryCellGroup->spCellConfig->spCellConfigDedicated->firstActiveDownlinkBWP_Id=calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->firstActiveDownlinkBWP_Id));
 
 *secondaryCellGroup->spCellConfig->spCellConfigDedicated->firstActiveDownlinkBWP_Id=1;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->bwp_InactivityTimer = NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->defaultDownlinkBWP_Id = NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->defaultDownlinkBWP_Id = calloc(1, sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->defaultDownlinkBWP_Id));
 *secondaryCellGroup->spCellConfig->spCellConfigDedicated->defaultDownlinkBWP_Id = 1;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig=calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig));

 NR_BWP_UplinkDedicated_t *initialUplinkBWP = calloc(1,sizeof(*initialUplinkBWP));
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->initialUplinkBWP = initialUplinkBWP;
 initialUplinkBWP->pucch_Config = NULL;
 initialUplinkBWP->pusch_Config = calloc(1,sizeof(*initialUplinkBWP->pusch_Config));
 initialUplinkBWP->pusch_Config->present = NR_SetupRelease_PUSCH_Config_PR_setup;
 NR_PUSCH_Config_t *pusch_Config = calloc(1,sizeof(*pusch_Config));
 initialUplinkBWP->pusch_Config->choice.setup = pusch_Config;
 pusch_Config->txConfig=calloc(1,sizeof(*pusch_Config->txConfig));
 *pusch_Config->txConfig= NR_PUSCH_Config__txConfig_codebook;
 pusch_Config->dmrs_UplinkForPUSCH_MappingTypeA = NULL;
 pusch_Config->dmrs_UplinkForPUSCH_MappingTypeB = calloc(1,sizeof(*pusch_Config->dmrs_UplinkForPUSCH_MappingTypeB));
 pusch_Config->dmrs_UplinkForPUSCH_MappingTypeB->present = NR_SetupRelease_DMRS_UplinkConfig_PR_setup;
 pusch_Config->dmrs_UplinkForPUSCH_MappingTypeB->choice.setup = calloc(1,sizeof(*pusch_Config->dmrs_UplinkForPUSCH_MappingTypeB->choice.setup));
 NR_DMRS_UplinkConfig_t *NR_DMRS_UplinkConfig = pusch_Config->dmrs_UplinkForPUSCH_MappingTypeB->choice.setup;
 NR_DMRS_UplinkConfig->dmrs_Type = NULL;
 NR_DMRS_UplinkConfig->dmrs_AdditionalPosition = calloc(1,sizeof(*NR_DMRS_UplinkConfig->dmrs_AdditionalPosition));
 *NR_DMRS_UplinkConfig->dmrs_AdditionalPosition = NR_DMRS_UplinkConfig__dmrs_AdditionalPosition_pos0;
 NR_DMRS_UplinkConfig->phaseTrackingRS=NULL;
 NR_DMRS_UplinkConfig->maxLength=NULL;
 NR_DMRS_UplinkConfig->transformPrecodingDisabled = calloc(1,sizeof(*NR_DMRS_UplinkConfig->transformPrecodingDisabled));
 NR_DMRS_UplinkConfig->transformPrecodingDisabled->scramblingID0 = NULL;
 NR_DMRS_UplinkConfig->transformPrecodingDisabled->scramblingID1 = NULL;
 NR_DMRS_UplinkConfig->transformPrecodingEnabled = NULL;
 pusch_Config->pusch_PowerControl = calloc(1,sizeof(*pusch_Config->pusch_PowerControl));
 pusch_Config->pusch_PowerControl->tpc_Accumulation = NULL;
 pusch_Config->pusch_PowerControl->msg3_Alpha = calloc(1,sizeof(*pusch_Config->pusch_PowerControl->msg3_Alpha));
 *pusch_Config->pusch_PowerControl->msg3_Alpha = NR_Alpha_alpha1;
 pusch_Config->pusch_PowerControl->p0_NominalWithoutGrant = NULL;
 pusch_Config->pusch_PowerControl->p0_AlphaSets = calloc(1,sizeof(*pusch_Config->pusch_PowerControl->p0_AlphaSets));
 NR_P0_PUSCH_AlphaSet_t *aset = calloc(1,sizeof(*aset));
 aset->p0_PUSCH_AlphaSetId=0;
 aset->p0=calloc(1,sizeof(*aset->p0));
 *aset->p0 = 0;
 aset->alpha=calloc(1,sizeof(*aset->alpha));
 *aset->alpha=NR_Alpha_alpha1;
 ASN_SEQUENCE_ADD(&pusch_Config->pusch_PowerControl->p0_AlphaSets->list,aset);
 pusch_Config->pusch_PowerControl->pathlossReferenceRSToAddModList = NULL;
 pusch_Config->pusch_PowerControl->pathlossReferenceRSToReleaseList = NULL;
 pusch_Config->pusch_PowerControl->twoPUSCH_PC_AdjustmentStates = NULL;
 pusch_Config->pusch_PowerControl->deltaMCS = NULL;
 pusch_Config->pusch_PowerControl->sri_PUSCH_MappingToAddModList = NULL;
 pusch_Config->pusch_PowerControl->sri_PUSCH_MappingToReleaseList = NULL;
 pusch_Config->frequencyHopping=NULL;
 pusch_Config->frequencyHoppingOffsetLists=NULL;
 pusch_Config->resourceAllocation = NR_PUSCH_Config__resourceAllocation_resourceAllocationType1;
 pusch_Config->pusch_TimeDomainAllocationList = NULL;
 pusch_Config->pusch_AggregationFactor=NULL;
 pusch_Config->mcs_Table=NULL;
 pusch_Config->mcs_TableTransformPrecoder=NULL;
 pusch_Config->transformPrecoder=calloc(1,sizeof(*pusch_Config->transformPrecoder));
 *pusch_Config->transformPrecoder = NR_PUSCH_Config__transformPrecoder_disabled;
 pusch_Config->codebookSubset=calloc(1,sizeof(*pusch_Config->codebookSubset));
 *pusch_Config->codebookSubset = NR_PUSCH_Config__codebookSubset_nonCoherent;
 pusch_Config->maxRank=calloc(1,sizeof(*pusch_Config->maxRank));
 *pusch_Config->maxRank= 1;
 pusch_Config->rbg_Size=NULL;
 pusch_Config->uci_OnPUSCH=NULL;
 pusch_Config->tp_pi2BPSK=NULL;

 initialUplinkBWP->srs_Config = calloc(1,sizeof(*initialUplinkBWP->srs_Config));
 initialUplinkBWP->srs_Config->present = NR_SetupRelease_SRS_Config_PR_setup;
 NR_SRS_Config_t *srs_Config = calloc(1,sizeof(*srs_Config));
 initialUplinkBWP->srs_Config->choice.setup=srs_Config;
 srs_Config->srs_ResourceSetToReleaseList=NULL;
 srs_Config->srs_ResourceSetToAddModList=calloc(1,sizeof(*srs_Config->srs_ResourceSetToAddModList));
 NR_SRS_ResourceSet_t *srs_resset0=calloc(1,sizeof(*srs_resset0));
 srs_resset0->srs_ResourceSetId = 0;
 srs_resset0->srs_ResourceIdList=calloc(1,sizeof(*srs_resset0->srs_ResourceIdList));
 NR_SRS_ResourceId_t *srs_resset0_id=calloc(1,sizeof(*srs_resset0_id));
 *srs_resset0_id=0;
 ASN_SEQUENCE_ADD(&srs_resset0->srs_ResourceIdList->list,srs_resset0_id);
 srs_Config->srs_ResourceToReleaseList=NULL;
 srs_resset0->resourceType.present =  NR_SRS_ResourceSet__resourceType_PR_aperiodic;
 srs_resset0->resourceType.choice.aperiodic = calloc(1,sizeof(*srs_resset0->resourceType.choice.aperiodic));
 srs_resset0->resourceType.choice.aperiodic->aperiodicSRS_ResourceTrigger=1;
 srs_resset0->resourceType.choice.aperiodic->csi_RS=NULL;
 srs_resset0->resourceType.choice.aperiodic->slotOffset= calloc(1,sizeof(*srs_resset0->resourceType.choice.aperiodic->slotOffset));
 *srs_resset0->resourceType.choice.aperiodic->slotOffset=2;
 srs_resset0->resourceType.choice.aperiodic->ext1=NULL;
 srs_resset0->usage=NR_SRS_ResourceSet__usage_codebook;
 srs_resset0->alpha = calloc(1,sizeof(*srs_resset0->alpha));
 *srs_resset0->alpha = NR_Alpha_alpha1;
 srs_resset0->p0=calloc(1,sizeof(*srs_resset0->p0));
 *srs_resset0->p0=-80;
 srs_resset0->pathlossReferenceRS=NULL;
 srs_resset0->srs_PowerControlAdjustmentStates=NULL;
 ASN_SEQUENCE_ADD(&srs_Config->srs_ResourceSetToAddModList->list,srs_resset0);
 srs_Config->srs_ResourceToReleaseList=NULL;
 srs_Config->srs_ResourceToAddModList=calloc(1,sizeof(*srs_Config->srs_ResourceToAddModList));
 NR_SRS_Resource_t *srs_res0=calloc(1,sizeof(*srs_res0));
 srs_res0->srs_ResourceId=0;
 srs_res0->nrofSRS_Ports=NR_SRS_Resource__nrofSRS_Ports_port1;
 srs_res0->ptrs_PortIndex=NULL;
 srs_res0->transmissionComb.present=NR_SRS_Resource__transmissionComb_PR_n2; 
 srs_res0->transmissionComb.choice.n2=calloc(1,sizeof(*srs_res0->transmissionComb.choice.n2));
 srs_res0->transmissionComb.choice.n2->combOffset_n2=0;
 srs_res0->transmissionComb.choice.n2->cyclicShift_n2=0;
 srs_res0->resourceMapping.startPosition=2;
 srs_res0->resourceMapping.nrofSymbols=NR_SRS_Resource__resourceMapping__nrofSymbols_n1;
 srs_res0->resourceMapping.repetitionFactor=NR_SRS_Resource__resourceMapping__repetitionFactor_n1;
 srs_res0->freqDomainPosition=0;
 srs_res0->freqDomainShift=0;
 srs_res0->freqHopping.c_SRS = 0;
 srs_res0->freqHopping.b_SRS=0;
 srs_res0->freqHopping.b_hop=0;
 srs_res0->groupOrSequenceHopping=NR_SRS_Resource__groupOrSequenceHopping_neither;
 srs_res0->resourceType.present= NR_SRS_Resource__resourceType_PR_aperiodic;
 srs_res0->resourceType.choice.aperiodic=calloc(1,sizeof(*srs_res0->resourceType.choice.aperiodic));
 srs_res0->sequenceId=40;
 srs_res0->spatialRelationInfo=calloc(1,sizeof(*srs_res0->spatialRelationInfo));
 srs_res0->spatialRelationInfo->servingCellId=NULL;
 srs_res0->spatialRelationInfo->referenceSignal.present=NR_SRS_SpatialRelationInfo__referenceSignal_PR_csi_RS_Index;
 srs_res0->spatialRelationInfo->referenceSignal.choice.csi_RS_Index=0;
 ASN_SEQUENCE_ADD(&srs_Config->srs_ResourceToAddModList->list,srs_res0);

 secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->uplinkBWP_ToReleaseList = NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->uplinkBWP_ToAddModList = calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->uplinkBWP_ToAddModList));
 NR_BWP_Uplink_t *ubwp = calloc(1,sizeof(*ubwp));
 ubwp->bwp_Id=1;
 ubwp->bwp_Common = calloc(1,sizeof(*ubwp->bwp_Common));
 // copy bwp_Common from Initial UL BWP except for bandwidth
 memcpy((void*)&ubwp->bwp_Common->genericParameters,
	(void*)&servingcellconfigcommon->uplinkConfigCommon->initialUplinkBWP->genericParameters,
	sizeof(servingcellconfigcommon->uplinkConfigCommon->initialUplinkBWP->genericParameters));
 ubwp->bwp_Common->genericParameters.locationAndBandwidth=PRBalloc_to_locationandbandwidth(servingcellconfigcommon->uplinkConfigCommon->frequencyInfoUL->scs_SpecificCarrierList.list.array[0]->carrierBandwidth,0);

 ubwp->bwp_Common->rach_ConfigCommon  = servingcellconfigcommon->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon;
 ubwp->bwp_Common->pusch_ConfigCommon = servingcellconfigcommon->uplinkConfigCommon->initialUplinkBWP->pusch_ConfigCommon;
 ubwp->bwp_Common->pucch_ConfigCommon = servingcellconfigcommon->uplinkConfigCommon->initialUplinkBWP->pucch_ConfigCommon;
 
 ubwp->bwp_Dedicated = calloc(1,sizeof(*ubwp->bwp_Dedicated));
 ubwp->bwp_Dedicated->pucch_Config = calloc(1,sizeof(*ubwp->bwp_Dedicated->pucch_Config));
 ubwp->bwp_Dedicated->pucch_Config->present = NR_SetupRelease_PUCCH_Config_PR_setup;
 NR_PUCCH_Config_t *pucch_Config = calloc(1,sizeof(*pucch_Config));
 ubwp->bwp_Dedicated->pucch_Config->choice.setup=pucch_Config;
 pucch_Config->resourceSetToAddModList = calloc(1,sizeof(*pucch_Config->resourceSetToAddModList));
 pucch_Config->resourceSetToReleaseList = NULL;
 NR_PUCCH_ResourceSet_t *pucchresset0=calloc(1,sizeof(*pucchresset0));
 NR_PUCCH_ResourceSet_t *pucchresset1=calloc(1,sizeof(*pucchresset1));
 pucchresset0->pucch_ResourceSetId = 0;
 NR_PUCCH_ResourceId_t *pucchresset0id0=calloc(1,sizeof(*pucchresset0id0));
 NR_PUCCH_ResourceId_t *pucchresset0id1=calloc(1,sizeof(*pucchresset0id1));
 *pucchresset0id0=1;
 ASN_SEQUENCE_ADD(&pucchresset0->resourceList.list,pucchresset0id0);
 *pucchresset0id1=2;
 ASN_SEQUENCE_ADD(&pucchresset0->resourceList.list,pucchresset0id1);
 pucchresset0->maxPayloadSize=NULL;

 ASN_SEQUENCE_ADD(&pucch_Config->resourceSetToAddModList->list,pucchresset0);

 pucchresset1->pucch_ResourceSetId = 1;
 NR_PUCCH_ResourceId_t *pucchresset1id0=calloc(1,sizeof(*pucchresset1id0));
 NR_PUCCH_ResourceId_t *pucchresset1id1=calloc(1,sizeof(*pucchresset1id1));
 *pucchresset1id0=3;
 ASN_SEQUENCE_ADD(&pucchresset1->resourceList.list,pucchresset1id0);
 *pucchresset1id1=4;
 ASN_SEQUENCE_ADD(&pucchresset1->resourceList.list,pucchresset1id1);
 pucchresset1->maxPayloadSize=NULL;
 ASN_SEQUENCE_ADD(&pucch_Config->resourceSetToAddModList->list,pucchresset1);

 pucch_Config->resourceToAddModList = calloc(1,sizeof(*pucch_Config->resourceToAddModList));
 pucch_Config->resourceToReleaseList = NULL;
 NR_PUCCH_Resource_t *pucchres0=calloc(1,sizeof(*pucchres0));
 NR_PUCCH_Resource_t *pucchres1=calloc(1,sizeof(*pucchres1));
 NR_PUCCH_Resource_t *pucchres2=calloc(1,sizeof(*pucchres2));
 NR_PUCCH_Resource_t *pucchres3=calloc(1,sizeof(*pucchres3));
 pucchres0->pucch_ResourceId=1;
 pucchres0->startingPRB=48;
 pucchres0->intraSlotFrequencyHopping=NULL;
 pucchres0->secondHopPRB=NULL;
 pucchres0->format.present= NR_PUCCH_Resource__format_PR_format0;
 pucchres0->format.choice.format0=calloc(1,sizeof(*pucchres0->format.choice.format0));
 pucchres0->format.choice.format0->initialCyclicShift=0;
 pucchres0->format.choice.format0->nrofSymbols=1;
 pucchres0->format.choice.format0->startingSymbolIndex=13;
 ASN_SEQUENCE_ADD(&pucch_Config->resourceToAddModList->list,pucchres0);

 pucchres1->pucch_ResourceId=2;
 pucchres1->startingPRB=48;
 pucchres1->intraSlotFrequencyHopping=NULL;
 pucchres1->secondHopPRB=NULL;
 pucchres1->format.present= NR_PUCCH_Resource__format_PR_format0;
 pucchres1->format.choice.format0=calloc(1,sizeof(*pucchres1->format.choice.format0));
 pucchres1->format.choice.format0->initialCyclicShift=0;
 pucchres1->format.choice.format0->nrofSymbols=1;
 pucchres1->format.choice.format0->startingSymbolIndex=12;
 ASN_SEQUENCE_ADD(&pucch_Config->resourceToAddModList->list,pucchres1);

 pucchres2->pucch_ResourceId=3;
 pucchres2->startingPRB=40;
 pucchres2->intraSlotFrequencyHopping=NULL;
 pucchres2->secondHopPRB=NULL;
 pucchres2->format.present= NR_PUCCH_Resource__format_PR_format2;
 pucchres2->format.choice.format2=calloc(1,sizeof(*pucchres2->format.choice.format2));
 pucchres2->format.choice.format2->nrofPRBs=4;
 pucchres2->format.choice.format2->nrofSymbols=1;
 pucchres2->format.choice.format2->startingSymbolIndex=13;
 ASN_SEQUENCE_ADD(&pucch_Config->resourceToAddModList->list,pucchres2);

 pucchres3->pucch_ResourceId=4;
 pucchres3->startingPRB=40;
 pucchres3->intraSlotFrequencyHopping=NULL;
 pucchres3->secondHopPRB=NULL;
 pucchres3->format.present= NR_PUCCH_Resource__format_PR_format2;
 pucchres3->format.choice.format2=calloc(1,sizeof(*pucchres3->format.choice.format2));
 pucchres3->format.choice.format2->nrofPRBs=4;
 pucchres3->format.choice.format2->nrofSymbols=1;
 pucchres3->format.choice.format2->startingSymbolIndex=12;
 ASN_SEQUENCE_ADD(&pucch_Config->resourceToAddModList->list,pucchres3);

 pucch_Config->format2=calloc(1,sizeof(*pucch_Config->format2));
 pucch_Config->format2->present=NR_SetupRelease_PUCCH_FormatConfig_PR_setup;
 NR_PUCCH_FormatConfig_t *pucchfmt2 = calloc(1,sizeof(*pucchfmt2));
 pucch_Config->format2->choice.setup = pucchfmt2;
 pucchfmt2->interslotFrequencyHopping=NULL;
 pucchfmt2->additionalDMRS=NULL;
 pucchfmt2->maxCodeRate=calloc(1,sizeof(*pucchfmt2->maxCodeRate));
 *pucchfmt2->maxCodeRate=NR_PUCCH_MaxCodeRate_zeroDot25;
 pucchfmt2->nrofSlots=NULL;
 pucchfmt2->pi2BPSK=NULL;
 pucchfmt2->simultaneousHARQ_ACK_CSI=NULL;
 pucch_Config->schedulingRequestResourceToAddModList=NULL;
 pucch_Config->schedulingRequestResourceToReleaseList=NULL;
 pucch_Config->multi_CSI_PUCCH_ResourceList=NULL;
 pucch_Config->dl_DataToUL_ACK = calloc(1,sizeof(*pucch_Config->dl_DataToUL_ACK));
 long *delay[8];
 for (int i=0;i<8;i++) {
   delay[i] = calloc(1,sizeof(*delay[i]));
   *delay[i] = (i<6) ? (i+2) : 0;
   ASN_SEQUENCE_ADD(&pucch_Config->dl_DataToUL_ACK->list,delay[i]);
 }
 pucch_Config->spatialRelationInfoToAddModList = calloc(1,sizeof(*pucch_Config->spatialRelationInfoToAddModList));
 NR_PUCCH_SpatialRelationInfo_t *pucchspatial = calloc(1,sizeof(*pucchspatial));
 pucchspatial->pucch_SpatialRelationInfoId = 1;
 pucchspatial->servingCellId = NULL;
 pucchspatial->referenceSignal.present = NR_PUCCH_SpatialRelationInfo__referenceSignal_PR_csi_RS_Index;
 pucchspatial->referenceSignal.choice.csi_RS_Index = 0;
 pucchspatial->pucch_PathlossReferenceRS_Id = 0;
 pucchspatial->p0_PUCCH_Id = 1;
 pucchspatial->closedLoopIndex = NR_PUCCH_SpatialRelationInfo__closedLoopIndex_i0;
 ASN_SEQUENCE_ADD(&pucch_Config->spatialRelationInfoToAddModList->list,pucchspatial);
 pucch_Config->spatialRelationInfoToReleaseList=NULL;
 pucch_Config->pucch_PowerControl=calloc(1,sizeof(*pucch_Config->pucch_PowerControl));
 pucch_Config->pucch_PowerControl->deltaF_PUCCH_f0 = calloc(1,sizeof(*pucch_Config->pucch_PowerControl->deltaF_PUCCH_f0));
 *pucch_Config->pucch_PowerControl->deltaF_PUCCH_f0 = 0;
 pucch_Config->pucch_PowerControl->deltaF_PUCCH_f1 = calloc(1,sizeof(*pucch_Config->pucch_PowerControl->deltaF_PUCCH_f1));
 *pucch_Config->pucch_PowerControl->deltaF_PUCCH_f1 = 0;
 pucch_Config->pucch_PowerControl->deltaF_PUCCH_f2 = calloc(1,sizeof(*pucch_Config->pucch_PowerControl->deltaF_PUCCH_f2));
 *pucch_Config->pucch_PowerControl->deltaF_PUCCH_f2 = 0;
 pucch_Config->pucch_PowerControl->deltaF_PUCCH_f3 = calloc(1,sizeof(*pucch_Config->pucch_PowerControl->deltaF_PUCCH_f3));
 *pucch_Config->pucch_PowerControl->deltaF_PUCCH_f3 = 0;
 pucch_Config->pucch_PowerControl->deltaF_PUCCH_f4 = calloc(1,sizeof(*pucch_Config->pucch_PowerControl->deltaF_PUCCH_f4));
 *pucch_Config->pucch_PowerControl->deltaF_PUCCH_f4 = 0;
 pucch_Config->pucch_PowerControl->p0_Set = calloc(1,sizeof(*pucch_Config->pucch_PowerControl->p0_Set));
 NR_P0_PUCCH_t *p00 = calloc(1,sizeof(*p00));
 p00->p0_PUCCH_Id=1;
 p00->p0_PUCCH_Value = 0;
 ASN_SEQUENCE_ADD(&pucch_Config->pucch_PowerControl->p0_Set->list,p00);
 pucch_Config->pucch_PowerControl->pathlossReferenceRSs = NULL;

 // copy pusch_Config from dedicated initialBWP
 ubwp->bwp_Dedicated->pusch_Config = calloc(1,sizeof(*ubwp->bwp_Dedicated->pusch_Config));
 ubwp->bwp_Dedicated->pusch_Config->present = NR_SetupRelease_PUSCH_Config_PR_setup;
 ubwp->bwp_Dedicated->pusch_Config->choice.setup = pusch_Config;

 ubwp->bwp_Dedicated->configuredGrantConfig = NULL;
 ubwp->bwp_Dedicated->srs_Config = calloc(1,sizeof(*ubwp->bwp_Dedicated->srs_Config));
 ubwp->bwp_Dedicated->srs_Config->present = NR_SetupRelease_SRS_Config_PR_setup;
 ubwp->bwp_Dedicated->srs_Config->choice.setup = srs_Config;

 ubwp->bwp_Dedicated->beamFailureRecoveryConfig = NULL;

 ASN_SEQUENCE_ADD(&secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->uplinkBWP_ToAddModList->list,ubwp);

 secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->firstActiveUplinkBWP_Id = calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->firstActiveUplinkBWP_Id));
 *secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->firstActiveUplinkBWP_Id = 1;

 secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->pusch_ServingCellConfig = calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->pusch_ServingCellConfig));
 NR_PUSCH_ServingCellConfig_t *pusch_scc = calloc(1,sizeof(*pusch_scc));
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->pusch_ServingCellConfig->present = NR_SetupRelease_PUSCH_ServingCellConfig_PR_setup;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->pusch_ServingCellConfig->choice.setup = pusch_scc;
 pusch_scc->codeBlockGroupTransmission = NULL;
 pusch_scc->rateMatching = NULL;
 pusch_scc->xOverhead = NULL;
 pusch_scc->ext1=calloc(1,sizeof(*pusch_scc->ext1));
 pusch_scc->ext1->maxMIMO_Layers = calloc(1,sizeof(*pusch_scc->ext1->maxMIMO_Layers));
 *pusch_scc->ext1->maxMIMO_Layers = 1;
 pusch_scc->ext1->processingType2Enabled = NULL;

 secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->carrierSwitching = NULL;

 secondaryCellGroup->spCellConfig->spCellConfigDedicated->supplementaryUplink=NULL;

 secondaryCellGroup->spCellConfig->spCellConfigDedicated->pdcch_ServingCellConfig=NULL;

 secondaryCellGroup->spCellConfig->spCellConfigDedicated->pdsch_ServingCellConfig=calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->pdsch_ServingCellConfig));
 NR_PDSCH_ServingCellConfig_t *pdsch_servingcellconfig = calloc(1,sizeof(*pdsch_servingcellconfig));
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->pdsch_ServingCellConfig->present = NR_SetupRelease_PDSCH_ServingCellConfig_PR_setup;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->pdsch_ServingCellConfig->choice.setup = pdsch_servingcellconfig;
 pdsch_servingcellconfig->codeBlockGroupTransmission = NULL;
 pdsch_servingcellconfig->xOverhead = NULL;
 pdsch_servingcellconfig->nrofHARQ_ProcessesForPDSCH = NULL;
 pdsch_servingcellconfig->pucch_Cell= NULL;
 pdsch_servingcellconfig->ext1=calloc(1,sizeof(*pdsch_servingcellconfig->ext1));
 pdsch_servingcellconfig->ext1->maxMIMO_Layers = calloc(1,sizeof(*pdsch_servingcellconfig->ext1->maxMIMO_Layers));
 *pdsch_servingcellconfig->ext1->maxMIMO_Layers = 2;
 pdsch_servingcellconfig->ext1->processingType2Enabled = NULL;
 
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->csi_MeasConfig=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->csi_MeasConfig=calloc(1,sizeof(*secondaryCellGroup->spCellConfig->spCellConfigDedicated->csi_MeasConfig));
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->csi_MeasConfig->present = NR_SetupRelease_CSI_MeasConfig_PR_setup;

 NR_CSI_MeasConfig_t *csi_MeasConfig = calloc(1,sizeof(*csi_MeasConfig));
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->csi_MeasConfig->choice.setup = csi_MeasConfig;

 csi_MeasConfig->csi_IM_ResourceToAddModList = NULL;
 csi_MeasConfig->csi_IM_ResourceToReleaseList = NULL;
 csi_MeasConfig->csi_IM_ResourceSetToAddModList = NULL;
 csi_MeasConfig->csi_IM_ResourceSetToReleaseList = NULL;
 csi_MeasConfig->nzp_CSI_RS_ResourceSetToAddModList  = NULL;
 csi_MeasConfig->nzp_CSI_RS_ResourceSetToReleaseList = NULL;
 csi_MeasConfig->nzp_CSI_RS_ResourceToAddModList = NULL;

 csi_MeasConfig->csi_SSB_ResourceSetToAddModList = calloc(1,sizeof(*csi_MeasConfig->csi_SSB_ResourceSetToAddModList));
 csi_MeasConfig->csi_SSB_ResourceSetToReleaseList = NULL;

 NR_CSI_SSB_ResourceSet_t *ssbresset0 = calloc(1,sizeof(*ssbresset0));
 ssbresset0->csi_SSB_ResourceSetId=0;
 if ((bitmap>>7)&0x01){
   NR_SSB_Index_t *ssbresset00=calloc(1,sizeof(*ssbresset00));
   *ssbresset00=0;
   ASN_SEQUENCE_ADD(&ssbresset0->csi_SSB_ResourceList.list,ssbresset00);
 }
 if ((bitmap>>6)&0x01) {
   NR_SSB_Index_t *ssbresset01=calloc(1,sizeof(*ssbresset01));
   *ssbresset01=1;
   ASN_SEQUENCE_ADD(&ssbresset0->csi_SSB_ResourceList.list,ssbresset01);
 }
 if ((bitmap>>5)&0x01) {
   NR_SSB_Index_t *ssbresset02=calloc(1,sizeof(*ssbresset02));
   *ssbresset02=2;
   ASN_SEQUENCE_ADD(&ssbresset0->csi_SSB_ResourceList.list,ssbresset02);
 }
 if ((bitmap>>4)&0x01) {
   NR_SSB_Index_t *ssbresset03=calloc(1,sizeof(*ssbresset03));
   *ssbresset03=3;
   ASN_SEQUENCE_ADD(&ssbresset0->csi_SSB_ResourceList.list,ssbresset03);
 }
 if ((bitmap>>3)&0x01) {
   NR_SSB_Index_t *ssbresset04=calloc(1,sizeof(*ssbresset04));
   *ssbresset04=4;
   ASN_SEQUENCE_ADD(&ssbresset0->csi_SSB_ResourceList.list,ssbresset04);
 }
 if ((bitmap>>2)&0x01) {
   NR_SSB_Index_t *ssbresset05=calloc(1,sizeof(*ssbresset05));
   *ssbresset05=5;
   ASN_SEQUENCE_ADD(&ssbresset0->csi_SSB_ResourceList.list,ssbresset05);
 }
 if ((bitmap>>1)&0x01) {
   NR_SSB_Index_t *ssbresset06=calloc(1,sizeof(*ssbresset06));
   *ssbresset06=6;
   ASN_SEQUENCE_ADD(&ssbresset0->csi_SSB_ResourceList.list,ssbresset06);
 }
 if ((bitmap)&0x01) {
   NR_SSB_Index_t *ssbresset07=calloc(1,sizeof(*ssbresset07));
   *ssbresset07=7;
   ASN_SEQUENCE_ADD(&ssbresset0->csi_SSB_ResourceList.list,ssbresset07);
 }
 ASN_SEQUENCE_ADD(&csi_MeasConfig->csi_SSB_ResourceSetToAddModList->list,ssbresset0);

 csi_MeasConfig->csi_ResourceConfigToAddModList = calloc(1,sizeof(*csi_MeasConfig->csi_ResourceConfigToAddModList));
 csi_MeasConfig->csi_ResourceConfigToReleaseList = NULL;
 NR_CSI_ResourceConfig_t *csires0 = calloc(1,sizeof(*csires0));
 csires0->csi_ResourceConfigId=0;
 csires0->csi_RS_ResourceSetList.present = NR_CSI_ResourceConfig__csi_RS_ResourceSetList_PR_nzp_CSI_RS_SSB;
 csires0->csi_RS_ResourceSetList.choice.nzp_CSI_RS_SSB = calloc(1,sizeof(*csires0->csi_RS_ResourceSetList.choice.nzp_CSI_RS_SSB));
 csires0->csi_RS_ResourceSetList.choice.nzp_CSI_RS_SSB->nzp_CSI_RS_ResourceSetList = NULL;
 csires0->csi_RS_ResourceSetList.choice.nzp_CSI_RS_SSB->csi_SSB_ResourceSetList = calloc(1,sizeof(*csires0->csi_RS_ResourceSetList.choice.nzp_CSI_RS_SSB->csi_SSB_ResourceSetList));
 NR_CSI_SSB_ResourceSetId_t *ssbres00 = calloc(1,sizeof(*ssbres00));
 *ssbres00 = 0;
 ASN_SEQUENCE_ADD(&csires0->csi_RS_ResourceSetList.choice.nzp_CSI_RS_SSB->csi_SSB_ResourceSetList->list,ssbres00);
 csires0->bwp_Id = 1;
 csires0->resourceType = NR_CSI_ResourceConfig__resourceType_periodic;
 ASN_SEQUENCE_ADD(&csi_MeasConfig->csi_ResourceConfigToAddModList->list,csires0);

 csi_MeasConfig->csi_ReportConfigToAddModList = calloc(1,sizeof(*csi_MeasConfig->csi_ReportConfigToAddModList));
 csi_MeasConfig->csi_ReportConfigToReleaseList = NULL;
 NR_CSI_ReportConfig_t *csirep1 = calloc(1,sizeof(*csirep1));
 csirep1->reportConfigId=0;
 csirep1->carrier=NULL;
 csirep1->resourcesForChannelMeasurement=0;
 csirep1->csi_IM_ResourcesForInterference=NULL;
 csirep1->nzp_CSI_RS_ResourcesForInterference=NULL;
 csirep1->reportConfigType.present = NR_CSI_ReportConfig__reportConfigType_PR_periodic;
 csirep1->reportConfigType.choice.periodic = calloc(1,sizeof(*csirep1->reportConfigType.choice.periodic));
 csirep1->reportConfigType.choice.periodic->reportSlotConfig.present=NR_CSI_ReportPeriodicityAndOffset_PR_slots320;
 csirep1->reportConfigType.choice.periodic->reportSlotConfig.choice.slots320 = 49;
 NR_PUCCH_CSI_Resource_t *pucchcsires1 = calloc(1,sizeof(*pucchcsires1));
 pucchcsires1->uplinkBandwidthPartId=1;
 pucchcsires1->pucch_Resource=3;
 ASN_SEQUENCE_ADD(&csirep1->reportConfigType.choice.periodic->pucch_CSI_ResourceList.list,pucchcsires1);
 csirep1->reportQuantity.present = NR_CSI_ReportConfig__reportQuantity_PR_ssb_Index_RSRP;
 csirep1->reportQuantity.choice.ssb_Index_RSRP=(NULL_t)0;
 csirep1->reportFreqConfiguration = calloc(1,sizeof(*csirep1->reportFreqConfiguration));
 csirep1->reportFreqConfiguration->cqi_FormatIndicator=NR_CSI_ReportConfig__reportFreqConfiguration__cqi_FormatIndicator_widebandCQI;
 csirep1->reportFreqConfiguration->pmi_FormatIndicator=NR_CSI_ReportConfig__reportFreqConfiguration__pmi_FormatIndicator_widebandPMI;
 csirep1->reportFreqConfiguration->csi_ReportingBand=NULL;
 csirep1->timeRestrictionForChannelMeasurements= NR_CSI_ReportConfig__timeRestrictionForChannelMeasurements_configured;
 csirep1->timeRestrictionForInterferenceMeasurements=NR_CSI_ReportConfig__timeRestrictionForInterferenceMeasurements_configured;
 csirep1->codebookConfig=calloc(1,sizeof(*csirep1->codebookConfig));
 csirep1->codebookConfig->codebookType.present = NR_CodebookConfig__codebookType_PR_type1;
 csirep1->codebookConfig->codebookType.choice.type1 = calloc(1,sizeof(*csirep1->codebookConfig->codebookType.choice.type1));
 csirep1->codebookConfig->codebookType.choice.type1->subType.present=NR_CodebookConfig__codebookType__type1__subType_PR_typeI_SinglePanel;
 csirep1->codebookConfig->codebookType.choice.type1->subType.choice.typeI_SinglePanel=calloc(1,sizeof(*csirep1->codebookConfig->codebookType.choice.type1->subType.choice.typeI_SinglePanel));
 csirep1->codebookConfig->codebookType.choice.type1->subType.choice.typeI_SinglePanel->nrOfAntennaPorts.present=NR_CodebookConfig__codebookType__type1__subType__typeI_SinglePanel__nrOfAntennaPorts_PR_two;
 csirep1->codebookConfig->codebookType.choice.type1->subType.choice.typeI_SinglePanel->nrOfAntennaPorts.choice.two=
   calloc(1,sizeof(*csirep1->codebookConfig->codebookType.choice.type1->subType.choice.typeI_SinglePanel->nrOfAntennaPorts.choice.two));
 csirep1->codebookConfig->codebookType.choice.type1->subType.choice.typeI_SinglePanel->nrOfAntennaPorts.choice.two->twoTX_CodebookSubsetRestriction.size=1;
 csirep1->codebookConfig->codebookType.choice.type1->subType.choice.typeI_SinglePanel->nrOfAntennaPorts.choice.two->twoTX_CodebookSubsetRestriction.bits_unused=2;
 csirep1->codebookConfig->codebookType.choice.type1->subType.choice.typeI_SinglePanel->nrOfAntennaPorts.choice.two->twoTX_CodebookSubsetRestriction.buf=malloc(1);
 csirep1->codebookConfig->codebookType.choice.type1->subType.choice.typeI_SinglePanel->nrOfAntennaPorts.choice.two->twoTX_CodebookSubsetRestriction.buf[0]=0xfc;
 //'111111'B


 csirep1->codebookConfig->codebookType.choice.type1->subType.choice.typeI_SinglePanel->typeI_SinglePanel_ri_Restriction.size=1; 
 csirep1->codebookConfig->codebookType.choice.type1->subType.choice.typeI_SinglePanel->typeI_SinglePanel_ri_Restriction.bits_unused=0;
 csirep1->codebookConfig->codebookType.choice.type1->subType.choice.typeI_SinglePanel->typeI_SinglePanel_ri_Restriction.buf=malloc(1); 
 csirep1->codebookConfig->codebookType.choice.type1->subType.choice.typeI_SinglePanel->typeI_SinglePanel_ri_Restriction.buf[0]=0xc0; //'00000011'B

 csirep1->codebookConfig->codebookType.choice.type1->codebookMode=1;
 csirep1->dummy = NULL;
 csirep1->groupBasedBeamReporting.present = NR_CSI_ReportConfig__groupBasedBeamReporting_PR_disabled;
 csirep1->groupBasedBeamReporting.choice.disabled=calloc(1,sizeof(*csirep1->groupBasedBeamReporting.choice.disabled));
 csirep1->groupBasedBeamReporting.choice.disabled->nrofReportedRS = calloc(1,sizeof(*csirep1->groupBasedBeamReporting.choice.disabled->nrofReportedRS));
 *csirep1->groupBasedBeamReporting.choice.disabled->nrofReportedRS=NR_CSI_ReportConfig__groupBasedBeamReporting__disabled__nrofReportedRS_n1;
 // this corresponds to:
 //if the UE is configured with the higher layer parameter groupBasedBeamReporting set to 'disabled', the UE is not required to update measurements for more than 64 CSI-RS and/or SSB resources, and the UE shall report in a single report nrofReportedRS (higher layer configured) different CRI or SSBRI for each report setting. 

 csirep1->cqi_Table = calloc(1,sizeof(*csirep1->cqi_Table));
 *csirep1->cqi_Table = NR_CSI_ReportConfig__cqi_Table_table1;
 csirep1->subbandSize = NR_CSI_ReportConfig__subbandSize_value1;
 csirep1->non_PMI_PortIndication = NULL;
 csirep1->ext1 = NULL;

 ASN_SEQUENCE_ADD(&csi_MeasConfig->csi_ReportConfigToAddModList->list,csirep1);

 secondaryCellGroup->spCellConfig->spCellConfigDedicated->sCellDeactivationTimer=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->crossCarrierSchedulingConfig=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->tag_Id=0;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->dummy=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->pathlossReferenceLinking=NULL;
 secondaryCellGroup->spCellConfig->spCellConfigDedicated->servingCellMO=NULL;

}
void fill_default_reconfig(NR_ServingCellConfigCommon_t *servingcellconfigcommon,
                           NR_RRCReconfiguration_IEs_t *reconfig,
                           NR_CellGroupConfig_t *secondaryCellGroup,
                           int n_physical_antenna_ports,
                           int initial_csi_index) {
  AssertFatal(servingcellconfigcommon!=NULL,"servingcellconfigcommon is null\n");
  AssertFatal(reconfig!=NULL,"reconfig is null\n");
  AssertFatal(secondaryCellGroup!=NULL,"secondaryCellGroup is null\n");
  // radioBearerConfig
  reconfig->radioBearerConfig=NULL;
  // secondaryCellGroup
  fill_default_secondaryCellGroup(servingcellconfigcommon,secondaryCellGroup,1,1,n_physical_antenna_ports,initial_csi_index);
  xer_fprint(stdout, &asn_DEF_NR_CellGroupConfig, (const void*)secondaryCellGroup);

  char scg_buffer[1024];
  asn_enc_rval_t enc_rval = uper_encode_to_buffer(&asn_DEF_NR_CellGroupConfig, NULL, (void *)secondaryCellGroup, scg_buffer, 1024);
  AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %jd)!\n", enc_rval.failed_type->name, enc_rval.encoded);
  reconfig->secondaryCellGroup = calloc(1,sizeof(*reconfig->secondaryCellGroup));
  OCTET_STRING_fromBuf(reconfig->secondaryCellGroup,
                       (const char *)scg_buffer,
                       (enc_rval.encoded+7)>>3);
  // measConfig
  reconfig->measConfig=NULL;
  // lateNonCriticalExtension
  reconfig->lateNonCriticalExtension = NULL;
  // nonCriticalExtension
  reconfig->nonCriticalExtension = NULL;
}

void fill_default_rbconfig(NR_RadioBearerConfig_t *rbconfig) {

  rbconfig->srb_ToAddModList = NULL;
  rbconfig->srb3_ToRelease = NULL;
  rbconfig->drb_ToAddModList = calloc(1,sizeof(*rbconfig->drb_ToAddModList));
  NR_DRB_ToAddMod_t *drb_ToAddMod = calloc(1,sizeof(*drb_ToAddMod));
  drb_ToAddMod->cnAssociation = calloc(1,sizeof(*drb_ToAddMod->cnAssociation));
  drb_ToAddMod->cnAssociation->present = NR_DRB_ToAddMod__cnAssociation_PR_eps_BearerIdentity;
  drb_ToAddMod->cnAssociation->choice.eps_BearerIdentity= 5;
  drb_ToAddMod->drb_Identity = 1;
  drb_ToAddMod->reestablishPDCP = NULL;
  drb_ToAddMod->recoverPDCP = NULL;
  drb_ToAddMod->pdcp_Config = calloc(1,sizeof(*drb_ToAddMod->pdcp_Config));
  drb_ToAddMod->pdcp_Config->drb = calloc(1,sizeof(*drb_ToAddMod->pdcp_Config->drb));
  drb_ToAddMod->pdcp_Config->drb->discardTimer = calloc(1,sizeof(*drb_ToAddMod->pdcp_Config->drb->discardTimer));
  *drb_ToAddMod->pdcp_Config->drb->discardTimer=NR_PDCP_Config__drb__discardTimer_ms30;
  drb_ToAddMod->pdcp_Config->drb->pdcp_SN_SizeUL = calloc(1,sizeof(*drb_ToAddMod->pdcp_Config->drb->pdcp_SN_SizeUL));
  *drb_ToAddMod->pdcp_Config->drb->pdcp_SN_SizeUL = NR_PDCP_Config__drb__pdcp_SN_SizeUL_len18bits;
  drb_ToAddMod->pdcp_Config->drb->pdcp_SN_SizeDL = calloc(1,sizeof(*drb_ToAddMod->pdcp_Config->drb->pdcp_SN_SizeDL));
  *drb_ToAddMod->pdcp_Config->drb->pdcp_SN_SizeDL = NR_PDCP_Config__drb__pdcp_SN_SizeDL_len18bits;
  drb_ToAddMod->pdcp_Config->drb->headerCompression.present = NR_PDCP_Config__drb__headerCompression_PR_notUsed;
  drb_ToAddMod->pdcp_Config->drb->headerCompression.choice.notUsed = 0;

  drb_ToAddMod->pdcp_Config->drb->integrityProtection=NULL;
  drb_ToAddMod->pdcp_Config->drb->statusReportRequired=NULL;
  drb_ToAddMod->pdcp_Config->drb->outOfOrderDelivery=NULL;
  drb_ToAddMod->pdcp_Config->moreThanOneRLC = NULL;

  drb_ToAddMod->pdcp_Config->t_Reordering = calloc(1,sizeof(*drb_ToAddMod->pdcp_Config->t_Reordering));
  *drb_ToAddMod->pdcp_Config->t_Reordering = NR_PDCP_Config__t_Reordering_ms0;
  drb_ToAddMod->pdcp_Config->ext1 = NULL;

  ASN_SEQUENCE_ADD(&rbconfig->drb_ToAddModList->list,drb_ToAddMod);

  rbconfig->drb_ToReleaseList = NULL;

  rbconfig->securityConfig = calloc(1,sizeof(*rbconfig->securityConfig));
  rbconfig->securityConfig->securityAlgorithmConfig = calloc(1,sizeof(*rbconfig->securityConfig->securityAlgorithmConfig));
  rbconfig->securityConfig->securityAlgorithmConfig->cipheringAlgorithm = NR_CipheringAlgorithm_nea0;
  rbconfig->securityConfig->securityAlgorithmConfig->integrityProtAlgorithm=NULL;
  rbconfig->securityConfig->keyToUse = calloc(1,sizeof(*rbconfig->securityConfig->keyToUse));
  *rbconfig->securityConfig->keyToUse = NR_SecurityConfig__keyToUse_master;

  xer_fprint(stdout, &asn_DEF_NR_RadioBearerConfig, (const void*)rbconfig);
}

#endif
