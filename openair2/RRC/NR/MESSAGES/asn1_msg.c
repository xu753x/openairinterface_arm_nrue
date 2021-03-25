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

/*! \file asn1_msg.c
* \brief primitives to build the asn1 messages
* \author Raymond Knopp and Navid Nikaein, WEI-TAI CHEN
* \date 2011, 2018
* \version 1.0
* \company Eurecom, NTUST
* \email: {raymond.knopp, navid.nikaein}@eurecom.fr and kroempa@gmail.com
*/

#include <stdio.h>
#include <sys/types.h>
#include <stdlib.h> /* for atoi(3) */
#include <unistd.h> /* for getopt(3) */
#include <string.h> /* for strerror(3) */
#include <sysexits.h> /* for EX_* exit codes */
#include <errno.h>  /* for errno */
#include "common/utils/LOG/log.h"
#include <asn_application.h>
#include <asn_internal.h> /* for _ASN_DEFAULT_STACK_MAX */
#include <per_encoder.h>

#include "asn1_msg.h"
#include "../nr_rrc_proto.h"
#include "RRC/NR/nr_rrc_extern.h"
#include "NR_DL-CCCH-Message.h"
#include "NR_UL-CCCH-Message.h"
#include "NR_DL-DCCH-Message.h"
#include "NR_RRCReject.h"
#include "NR_RejectWaitTime.h"
#include "NR_RRCSetup.h"
#include "NR_RRCSetup-IEs.h"
#include "NR_SRB-ToAddModList.h"
#include "NR_CellGroupConfig.h"
#include "NR_RLC-BearerConfig.h"
#include "NR_RLC-Config.h"
#include "NR_LogicalChannelConfig.h"
#include "NR_PDCP-Config.h"
#include "NR_MAC-CellGroupConfig.h"
#include "NR_SecurityModeCommand.h"
#include "NR_CipheringAlgorithm.h"
#include "NR_RRCReconfiguration-IEs.h"
#include "NR_DRB-ToAddMod.h"
#include "NR_DRB-ToAddModList.h"
#include "NR_SecurityConfig.h"
#include "NR_RRCReconfiguration-v1530-IEs.h"
#include "NR_UL-DCCH-Message.h"
#include "NR_SDAP-Config.h"
#include "NR_RRCReconfigurationComplete.h"
#include "NR_RRCReconfigurationComplete-IEs.h"
#include "NR_DLInformationTransfer.h"
#include "NR_RRCReestablishmentRequest.h"
#include "PHY/defs_nr_common.h"
#if defined(NR_Rel16)
  #include "NR_SCS-SpecificCarrier.h"
  #include "NR_TDD-UL-DL-ConfigCommon.h"
  #include "NR_FrequencyInfoUL.h"
  #include "NR_FrequencyInfoDL.h"
  #include "NR_RACH-ConfigGeneric.h"
  #include "NR_RACH-ConfigCommon.h"
  #include "NR_PUSCH-TimeDomainResourceAllocation.h"
  #include "NR_PUSCH-ConfigCommon.h"
  #include "NR_PUCCH-ConfigCommon.h"
  #include "NR_PDSCH-TimeDomainResourceAllocation.h"
  #include "NR_PDSCH-ConfigCommon.h"
  #include "NR_RateMatchPattern.h"
  #include "NR_RateMatchPatternLTE-CRS.h"
  #include "NR_SearchSpace.h"
  #include "NR_ControlResourceSet.h"
  #include "NR_EUTRA-MBSFN-SubframeConfig.h"
  #include "NR_BWP-DownlinkCommon.h"
  #include "NR_BWP-DownlinkDedicated.h"
  #include "NR_UplinkConfigCommon.h"
  #include "NR_SetupRelease.h"
  #include "NR_PDCCH-ConfigCommon.h"
  #include "NR_BWP-UplinkCommon.h"

  #include "assertions.h"
  //#include "RRCConnectionRequest.h"
  //#include "UL-CCCH-Message.h"
  #include "NR_UL-DCCH-Message.h"
  //#include "DL-CCCH-Message.h"
  #include "NR_DL-DCCH-Message.h"
  //#include "EstablishmentCause.h"
  //#include "RRCConnectionSetup.h"
  #include "NR_SRB-ToAddModList.h"
  #include "NR_DRB-ToAddModList.h"
  //#include "MCCH-Message.h"
  //#define MRB1 1

  //#include "RRCConnectionSetupComplete.h"
  //#include "RRCConnectionReconfigurationComplete.h"
  //#include "RRCConnectionReconfiguration.h"
  #include "NR_MIB.h"
  //#include "SystemInformation.h"

  #include "NR_SIB1.h"
  #include "NR_ServingCellConfigCommon.h"
  //#include "SIB-Type.h"

  //#include "BCCH-DL-SCH-Message.h"

  //#include "PHY/defs.h"

  #include "NR_MeasObjectToAddModList.h"
  #include "NR_ReportConfigToAddModList.h"
  #include "NR_MeasIdToAddModList.h"
  #include "gnb_config.h"
#endif

#include "intertask_interface.h"

#include "common/ran_context.h"

//#include "PHY/defs.h"
/*#ifndef USER_MODE
#define msg printk
#ifndef errno
int errno;
#endif
#else
# if !defined (msg)
#   define msg printf
# endif
#endif*/

//#define XER_PRINT

typedef struct xer_sprint_string_s {
  char *string;
  size_t string_size;
  size_t string_index;
} xer_sprint_string_t;

//replace LTE
//extern unsigned char NB_eNB_INST;
extern unsigned char NB_gNB_INST;

extern RAN_CONTEXT_t RC;

/*
 * This is a helper function for xer_sprint, which directs all incoming data
 * into the provided string.
 */
static int xer__nr_print2s (const void *buffer, size_t size, void *app_key) {
  xer_sprint_string_t *string_buffer = (xer_sprint_string_t *) app_key;
  size_t string_remaining = string_buffer->string_size - string_buffer->string_index;

  if (string_remaining > 0) {
    if (size > string_remaining) {
      size = string_remaining;
    }

    memcpy(&string_buffer->string[string_buffer->string_index], buffer, size);
    string_buffer->string_index += size;
  }

  return 0;
}

int xer_nr_sprint (char *string, size_t string_size, asn_TYPE_descriptor_t *td, void *sptr) {
  asn_enc_rval_t er;
  xer_sprint_string_t string_buffer;
  string_buffer.string = string;
  string_buffer.string_size = string_size;
  string_buffer.string_index = 0;
  er = xer_encode(td, sptr, XER_F_BASIC, xer__nr_print2s, &string_buffer);

  if (er.encoded < 0) {
    LOG_E(RRC, "xer_sprint encoding error (%zd)!", er.encoded);
    er.encoded = string_buffer.string_size;
  } else {
    if (er.encoded > string_buffer.string_size) {
      LOG_E(RRC, "xer_sprint string buffer too small, got %zd need %zd!", string_buffer.string_size, er.encoded);
      er.encoded = string_buffer.string_size;
    }
  }

  return er.encoded;
}

//------------------------------------------------------------------------------

uint8_t do_MIB_NR(gNB_RRC_INST *rrc,uint32_t frame) { 

  asn_enc_rval_t enc_rval;
  rrc_gNB_carrier_data_t *carrier = &rrc->carrier;  

  NR_BCCH_BCH_Message_t *mib = &carrier->mib;
  NR_ServingCellConfigCommon_t *scc = carrier->servingcellconfigcommon;

  memset(mib,0,sizeof(NR_BCCH_BCH_Message_t));
  mib->message.present = NR_BCCH_BCH_MessageType_PR_mib;
  mib->message.choice.mib = CALLOC(1,sizeof(struct NR_MIB));
  memset(mib->message.choice.mib,0,sizeof(struct NR_MIB));
  //36.331 SFN BIT STRING (SIZE (8)  , 38.331 SFN BIT STRING (SIZE (6))
  uint8_t sfn_msb = (uint8_t)((frame>>4)&0x3f);
  mib->message.choice.mib->systemFrameNumber.buf = CALLOC(1,sizeof(uint8_t));
  mib->message.choice.mib->systemFrameNumber.buf[0] = sfn_msb << 2;
  mib->message.choice.mib->systemFrameNumber.size = 1;
  mib->message.choice.mib->systemFrameNumber.bits_unused=2;
  //38.331 spare BIT STRING (SIZE (1))
  uint16_t *spare= CALLOC(1, sizeof(uint16_t));

  if (spare == NULL) abort();

  mib->message.choice.mib->spare.buf = (uint8_t *)spare;
  mib->message.choice.mib->spare.size = 1;
  mib->message.choice.mib->spare.bits_unused = 7;  // This makes a spare of 1 bits

  mib->message.choice.mib->ssb_SubcarrierOffset = (carrier->ssb_SubcarrierOffset)&15;

  /*
  * The SIB1 will be sent in this allocation (Type0-PDCCH) : 38.213, 13-4 Table and 38.213 13-11 to 13-14 tables
  * the reverse allocation is in nr_ue_decode_mib()
  */
  if(rrc->carrier.pdcch_ConfigSIB1) {
    mib->message.choice.mib->pdcch_ConfigSIB1.controlResourceSetZero = rrc->carrier.pdcch_ConfigSIB1->controlResourceSetZero;
    mib->message.choice.mib->pdcch_ConfigSIB1.searchSpaceZero = rrc->carrier.pdcch_ConfigSIB1->searchSpaceZero;
  } else {
    mib->message.choice.mib->pdcch_ConfigSIB1.controlResourceSetZero = *scc->downlinkConfigCommon->initialDownlinkBWP->pdcch_ConfigCommon->choice.setup->controlResourceSetZero;
    mib->message.choice.mib->pdcch_ConfigSIB1.searchSpaceZero = *scc->downlinkConfigCommon->initialDownlinkBWP->pdcch_ConfigCommon->choice.setup->searchSpaceZero;
  }

  AssertFatal(scc->ssbSubcarrierSpacing != NULL, "scc->ssbSubcarrierSpacing is null\n");
  switch (*scc->ssbSubcarrierSpacing) {
  case NR_SubcarrierSpacing_kHz15:
    mib->message.choice.mib->subCarrierSpacingCommon = NR_MIB__subCarrierSpacingCommon_scs15or60;
    break;
    
  case NR_SubcarrierSpacing_kHz30:
    mib->message.choice.mib->subCarrierSpacingCommon = NR_MIB__subCarrierSpacingCommon_scs30or120;
    break;
    
  case NR_SubcarrierSpacing_kHz60:
    mib->message.choice.mib->subCarrierSpacingCommon = NR_MIB__subCarrierSpacingCommon_scs15or60;
    break;
    
  case NR_SubcarrierSpacing_kHz120:
    mib->message.choice.mib->subCarrierSpacingCommon = NR_MIB__subCarrierSpacingCommon_scs30or120;
    break;
    
  case NR_SubcarrierSpacing_kHz240:
    AssertFatal(1==0,"Unknown subCarrierSpacingCommon %d\n",(int)*scc->ssbSubcarrierSpacing);
    break;
    
  default:
      AssertFatal(1==0,"Unknown subCarrierSpacingCommon %d\n",(int)*scc->ssbSubcarrierSpacing);
  }

  switch (scc->dmrs_TypeA_Position) {
  case 	NR_ServingCellConfigCommon__dmrs_TypeA_Position_pos2:
    mib->message.choice.mib->dmrs_TypeA_Position = NR_MIB__dmrs_TypeA_Position_pos2;
    break;
    
  case 	NR_ServingCellConfigCommon__dmrs_TypeA_Position_pos3:
    mib->message.choice.mib->dmrs_TypeA_Position = NR_MIB__dmrs_TypeA_Position_pos3;
    break;
    
  default:
    AssertFatal(1==0,"Unknown dmrs_TypeA_Position %d\n",(int)scc->dmrs_TypeA_Position);
  }

  //  assign_enum
  mib->message.choice.mib->cellBarred = NR_MIB__cellBarred_notBarred;
  //  assign_enum
  mib->message.choice.mib->intraFreqReselection = NR_MIB__intraFreqReselection_notAllowed;
  //encode MIB to data
  enc_rval = uper_encode_to_buffer(&asn_DEF_NR_BCCH_BCH_Message,
                                   NULL,
                                   (void *)mib,
                                   carrier->MIB,
                                   24);
  AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %lu)!\n",
               enc_rval.failed_type->name, enc_rval.encoded);

  if (enc_rval.encoded==-1) {
    return(-1);
  }

  return((enc_rval.encoded+7)/8);
}

uint8_t do_SIB1_NR(rrc_gNB_carrier_data_t *carrier, 
	               gNB_RrcConfigurationReq *configuration
                  ) {
  asn_enc_rval_t enc_rval;

  // TODO : Add support for more than one PLMN
  int num_plmn = 1; // int num_plmn = configuration->num_plmn;
  struct NR_PLMN_Identity nr_plmn[num_plmn];
  NR_MCC_MNC_Digit_t nr_mcc_digit[num_plmn][3];
  NR_MCC_MNC_Digit_t nr_mnc_digit[num_plmn][3];
  memset(nr_plmn,0,sizeof(nr_plmn));
  memset(nr_mcc_digit,0,sizeof(nr_mcc_digit));
  memset(nr_mnc_digit,0,sizeof(nr_mnc_digit));

  NR_BCCH_DL_SCH_Message_t *sib1_message = CALLOC(1,sizeof(NR_BCCH_DL_SCH_Message_t));
  carrier->siblock1 = sib1_message;
  sib1_message->message.present = NR_BCCH_DL_SCH_MessageType_PR_c1;
  sib1_message->message.choice.c1 = CALLOC(1,sizeof(struct NR_BCCH_DL_SCH_MessageType__c1));
  sib1_message->message.choice.c1->present = NR_BCCH_DL_SCH_MessageType__c1_PR_systemInformationBlockType1;
  sib1_message->message.choice.c1->choice.systemInformationBlockType1 = CALLOC(1,sizeof(struct NR_SIB1));

  struct NR_SIB1 *sib1 = sib1_message->message.choice.c1->choice.systemInformationBlockType1;

  // cellSelectionInfo
  sib1->cellSelectionInfo = CALLOC(1,sizeof(struct NR_SIB1__cellSelectionInfo));
  sib1->cellSelectionInfo->q_RxLevMin = -50;

  // cellAccessRelatedInfo
  struct NR_PLMN_IdentityInfo *nr_plmn_info=CALLOC(1,sizeof(struct NR_PLMN_IdentityInfo));
  asn_set_empty(&nr_plmn_info->plmn_IdentityList.list);
  for (int i = 0; i < num_plmn; ++i) {
    nr_mcc_digit[i][0] = (configuration->mcc[i]/100)%10;
    nr_mcc_digit[i][1] = (configuration->mcc[i]/10)%10;
    nr_mcc_digit[i][2] = (configuration->mcc[i])%10;
    nr_plmn[i].mcc = CALLOC(1,sizeof(struct NR_MCC));
    asn_set_empty(&nr_plmn[i].mcc->list);
    ASN_SEQUENCE_ADD(&nr_plmn[i].mcc->list, &nr_mcc_digit[i][0]);
    ASN_SEQUENCE_ADD(&nr_plmn[i].mcc->list, &nr_mcc_digit[i][1]);
    ASN_SEQUENCE_ADD(&nr_plmn[i].mcc->list, &nr_mcc_digit[i][2]);
    if(configuration->mnc_digit_length[i] == 3) nr_mnc_digit[i][0] = (configuration->mnc[i]/100)%10;
    nr_mnc_digit[i][1] = (configuration->mnc[i]/10)%10;
    nr_mnc_digit[i][2] = (configuration->mnc[i])%10;
    nr_plmn[i].mnc.list.size=0;
    nr_plmn[i].mnc.list.count=0;
    if(configuration->mnc_digit_length[i] == 3) ASN_SEQUENCE_ADD(&nr_plmn[i].mnc.list, &nr_mnc_digit[i][0]);
    ASN_SEQUENCE_ADD(&nr_plmn[i].mnc.list, &nr_mnc_digit[i][1]);
    ASN_SEQUENCE_ADD(&nr_plmn[i].mnc.list, &nr_mnc_digit[i][2]);
    ASN_SEQUENCE_ADD(&nr_plmn_info->plmn_IdentityList.list, &nr_plmn[i]);
  }//end plmn loop

  nr_plmn_info->cellIdentity.buf = CALLOC(1,5);
  nr_plmn_info->cellIdentity.buf[0]= (configuration->cell_identity >> 28) & 0xff;
  nr_plmn_info->cellIdentity.buf[1]= (configuration->cell_identity >> 20) & 0xff;
  nr_plmn_info->cellIdentity.buf[2]= (configuration->cell_identity >> 12) & 0xff;
  nr_plmn_info->cellIdentity.buf[3]= (configuration->cell_identity >> 4) & 0xff;
  nr_plmn_info->cellIdentity.buf[4]= (configuration->cell_identity << 4) & 0xff;
  nr_plmn_info->cellIdentity.size= 5;
  nr_plmn_info->cellIdentity.bits_unused= 4;
  nr_plmn_info->cellReservedForOperatorUse = NR_PLMN_IdentityInfo__cellReservedForOperatorUse_notReserved;

  nr_plmn_info->trackingAreaCode = CALLOC(1,sizeof(NR_TrackingAreaCode_t));
  nr_plmn_info->trackingAreaCode->buf = CALLOC(1,3);
  nr_plmn_info->trackingAreaCode->buf[0] = ( ((uint32_t)configuration->tac) >> 16) & 0xff;
  nr_plmn_info->trackingAreaCode->buf[1] = ( ((uint32_t)configuration->tac) >> 8) & 0xff;
  nr_plmn_info->trackingAreaCode->buf[2] = ( ((uint32_t)configuration->tac) >> 0) & 0xff;
  nr_plmn_info->trackingAreaCode->size = 3;
  nr_plmn_info->trackingAreaCode->bits_unused = 0;

  ASN_SEQUENCE_ADD(&sib1->cellAccessRelatedInfo.plmn_IdentityList.list, nr_plmn_info);

  // connEstFailureControl
  // TODO: add connEstFailureControl

  //si-SchedulingInfo
  /*sib1->si_SchedulingInfo = CALLOC(1,sizeof(struct NR_SI_SchedulingInfo));
  asn_set_empty(&sib1->si_SchedulingInfo->schedulingInfoList.list);
  sib1->si_SchedulingInfo->si_WindowLength = NR_SI_SchedulingInfo__si_WindowLength_s40;
  struct NR_SchedulingInfo *schedulingInfo = CALLOC(1,sizeof(struct NR_SchedulingInfo));
  schedulingInfo->si_BroadcastStatus = NR_SchedulingInfo__si_BroadcastStatus_broadcasting;
  schedulingInfo->si_Periodicity = NR_SchedulingInfo__si_Periodicity_rf8;
  asn_set_empty(&schedulingInfo->sib_MappingInfo.list);

  NR_SIB_TypeInfo_t *sib_type3 = CALLOC(1,sizeof(e_NR_SIB_TypeInfo__type));
  sib_type3->type = NR_SIB_TypeInfo__type_sibType3;
  sib_type3->valueTag = CALLOC(1,sizeof(sib_type3->valueTag));
  ASN_SEQUENCE_ADD(&schedulingInfo->sib_MappingInfo.list,sib_type3);

  NR_SIB_TypeInfo_t *sib_type5 = CALLOC(1,sizeof(e_NR_SIB_TypeInfo__type));
  sib_type5->type = NR_SIB_TypeInfo__type_sibType5;
  sib_type5->valueTag = CALLOC(1,sizeof(sib_type5->valueTag));
  ASN_SEQUENCE_ADD(&schedulingInfo->sib_MappingInfo.list,sib_type5);

  NR_SIB_TypeInfo_t *sib_type4 = CALLOC(1,sizeof(e_NR_SIB_TypeInfo__type));
  sib_type4->type = NR_SIB_TypeInfo__type_sibType4;
  sib_type4->valueTag = CALLOC(1,sizeof(sib_type4->valueTag));
  ASN_SEQUENCE_ADD(&schedulingInfo->sib_MappingInfo.list,sib_type4);

  NR_SIB_TypeInfo_t *sib_type2 = CALLOC(1,sizeof(e_NR_SIB_TypeInfo__type));
  sib_type2->type = NR_SIB_TypeInfo__type_sibType2;
  sib_type2->valueTag = CALLOC(1,sizeof(sib_type2->valueTag));
  ASN_SEQUENCE_ADD(&schedulingInfo->sib_MappingInfo.list,sib_type2);

  ASN_SEQUENCE_ADD(&sib1->si_SchedulingInfo->schedulingInfoList.list,schedulingInfo);*/

  // servingCellConfigCommon
  sib1->servingCellConfigCommon = CALLOC(1,sizeof(struct NR_ServingCellConfigCommonSIB));

  asn_set_empty(&sib1->servingCellConfigCommon->downlinkConfigCommon.frequencyInfoDL.frequencyBandList.list);
  asn_set_empty(&sib1->servingCellConfigCommon->downlinkConfigCommon.frequencyInfoDL.scs_SpecificCarrierList.list);
  sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.genericParameters.locationAndBandwidth = configuration->scc->downlinkConfigCommon->initialDownlinkBWP->genericParameters.locationAndBandwidth;
  sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.genericParameters.subcarrierSpacing = configuration->scc->downlinkConfigCommon->initialDownlinkBWP->genericParameters.subcarrierSpacing;
  sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.genericParameters.cyclicPrefix = configuration->scc->downlinkConfigCommon->initialDownlinkBWP->genericParameters.cyclicPrefix;
  for(int i = 0; i< configuration->scc->downlinkConfigCommon->frequencyInfoDL->frequencyBandList.list.count; i++) {
    struct NR_NR_MultiBandInfo *nrMultiBandInfo = CALLOC(1,sizeof(struct NR_NR_MultiBandInfo));
    nrMultiBandInfo->freqBandIndicatorNR = configuration->scc->downlinkConfigCommon->frequencyInfoDL->frequencyBandList.list.array[i];
    ASN_SEQUENCE_ADD(&sib1->servingCellConfigCommon->downlinkConfigCommon.frequencyInfoDL.frequencyBandList.list,nrMultiBandInfo);
  }
  sib1->servingCellConfigCommon->downlinkConfigCommon.frequencyInfoDL.offsetToPointA = configuration->scc->downlinkConfigCommon->frequencyInfoDL->scs_SpecificCarrierList.list.array[0]->offsetToCarrier;
  for(int i = 0; i< configuration->scc->downlinkConfigCommon->frequencyInfoDL->scs_SpecificCarrierList.list.count; i++) {
    ASN_SEQUENCE_ADD(&sib1->servingCellConfigCommon->downlinkConfigCommon.frequencyInfoDL.scs_SpecificCarrierList.list,configuration->scc->downlinkConfigCommon->frequencyInfoDL->scs_SpecificCarrierList.list.array[i]);
  }

  sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.pdcch_ConfigCommon = configuration->scc->downlinkConfigCommon->initialDownlinkBWP->pdcch_ConfigCommon;
  sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.pdcch_ConfigCommon->choice.setup->commonSearchSpaceList = CALLOC(1,sizeof(struct NR_PDCCH_ConfigCommon__commonSearchSpaceList));
  asn_set_empty(&sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.pdcch_ConfigCommon->choice.setup->commonSearchSpaceList->list);

  NR_SearchSpace_t *ss1 = calloc(1,sizeof(*ss1));
  ss1->searchSpaceId = 1;
  ss1->controlResourceSetId=calloc(1,sizeof(*ss1->controlResourceSetId));
  *ss1->controlResourceSetId=0;
  ss1->monitoringSlotPeriodicityAndOffset = calloc(1,sizeof(*ss1->monitoringSlotPeriodicityAndOffset));
  ss1->monitoringSlotPeriodicityAndOffset->present = NR_SearchSpace__monitoringSlotPeriodicityAndOffset_PR_sl1;
  ss1->monitoringSymbolsWithinSlot = calloc(1,sizeof(*ss1->monitoringSymbolsWithinSlot));
  ss1->monitoringSymbolsWithinSlot->buf = calloc(1,2);
  // should be '1100 0000 0000 00'B (LSB first!), first two symols in slot, adjust if needed
  ss1->monitoringSymbolsWithinSlot->buf[1] = 0;
  ss1->monitoringSymbolsWithinSlot->buf[0] = (1<<7);
  ss1->monitoringSymbolsWithinSlot->size = 2;
  ss1->monitoringSymbolsWithinSlot->bits_unused = 2;
  ss1->nrofCandidates = calloc(1,sizeof(*ss1->nrofCandidates));
  ss1->nrofCandidates->aggregationLevel1 = NR_SearchSpace__nrofCandidates__aggregationLevel1_n0;
  ss1->nrofCandidates->aggregationLevel2 = NR_SearchSpace__nrofCandidates__aggregationLevel2_n0;
  ss1->nrofCandidates->aggregationLevel4 = NR_SearchSpace__nrofCandidates__aggregationLevel4_n1;
  ss1->nrofCandidates->aggregationLevel8 = NR_SearchSpace__nrofCandidates__aggregationLevel8_n0;
  ss1->nrofCandidates->aggregationLevel16 = NR_SearchSpace__nrofCandidates__aggregationLevel16_n0;
  ss1->searchSpaceType = calloc(1,sizeof(*ss1->searchSpaceType));
  ss1->searchSpaceType->present = NR_SearchSpace__searchSpaceType_PR_common;
  ss1->searchSpaceType->choice.common=calloc(1,sizeof(*ss1->searchSpaceType->choice.common));
  ss1->searchSpaceType->choice.common->dci_Format0_0_AndFormat1_0 = calloc(1,sizeof(*ss1->searchSpaceType->choice.common->dci_Format0_0_AndFormat1_0));
  ASN_SEQUENCE_ADD(&sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.pdcch_ConfigCommon->choice.setup->commonSearchSpaceList->list,ss1);

  NR_SearchSpace_t *ss5 = calloc(1,sizeof(*ss5));
  ss5->searchSpaceId = 5;
  ss5->controlResourceSetId=calloc(1,sizeof(*ss5->controlResourceSetId));
  *ss5->controlResourceSetId=0;
  ss5->monitoringSlotPeriodicityAndOffset = calloc(1,sizeof(*ss5->monitoringSlotPeriodicityAndOffset));
  ss5->monitoringSlotPeriodicityAndOffset->present = NR_SearchSpace__monitoringSlotPeriodicityAndOffset_PR_sl5;
  ss5->monitoringSlotPeriodicityAndOffset->choice.sl5 = 0;
  ss5->duration = calloc(1,sizeof(*ss5->duration));
  *ss5->duration = 2;
  ss5->monitoringSymbolsWithinSlot = calloc(1,sizeof(*ss5->monitoringSymbolsWithinSlot));
  ss5->monitoringSymbolsWithinSlot->buf = calloc(1,2);
  // should be '1100 0000 0000 00'B (LSB first!), first two symols in slot, adjust if needed
  ss5->monitoringSymbolsWithinSlot->buf[1] = 0;
  ss5->monitoringSymbolsWithinSlot->buf[0] = (1<<7);
  ss5->monitoringSymbolsWithinSlot->size = 2;
  ss5->monitoringSymbolsWithinSlot->bits_unused = 2;
  ss5->nrofCandidates = calloc(1,sizeof(*ss5->nrofCandidates));
  ss5->nrofCandidates->aggregationLevel1 = NR_SearchSpace__nrofCandidates__aggregationLevel1_n0;
  ss5->nrofCandidates->aggregationLevel2 = NR_SearchSpace__nrofCandidates__aggregationLevel2_n0;
  ss5->nrofCandidates->aggregationLevel4 = NR_SearchSpace__nrofCandidates__aggregationLevel4_n4;
  ss5->nrofCandidates->aggregationLevel8 = NR_SearchSpace__nrofCandidates__aggregationLevel8_n2;
  ss5->nrofCandidates->aggregationLevel16 = NR_SearchSpace__nrofCandidates__aggregationLevel16_n1;
  ss5->searchSpaceType = calloc(1,sizeof(*ss5->searchSpaceType));
  ss5->searchSpaceType->present = NR_SearchSpace__searchSpaceType_PR_common;
  ss5->searchSpaceType->choice.common=calloc(1,sizeof(*ss5->searchSpaceType->choice.common));
  ss5->searchSpaceType->choice.common->dci_Format0_0_AndFormat1_0 = calloc(1,sizeof(*ss5->searchSpaceType->choice.common->dci_Format0_0_AndFormat1_0));
  ASN_SEQUENCE_ADD(&sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.pdcch_ConfigCommon->choice.setup->commonSearchSpaceList->list,ss5);

  NR_SearchSpace_t *ss7 = calloc(1,sizeof(*ss7));
  ss7->searchSpaceId = 7;
  ss7->controlResourceSetId=calloc(1,sizeof(*ss7->controlResourceSetId));
  *ss7->controlResourceSetId=0;
  ss7->monitoringSlotPeriodicityAndOffset = calloc(1,sizeof(*ss7->monitoringSlotPeriodicityAndOffset));
  ss7->monitoringSlotPeriodicityAndOffset->present = NR_SearchSpace__monitoringSlotPeriodicityAndOffset_PR_sl1;
  ss7->monitoringSymbolsWithinSlot = calloc(1,sizeof(*ss7->monitoringSymbolsWithinSlot));
  ss7->monitoringSymbolsWithinSlot->buf = calloc(1,2);
  // should be '1100 0000 0000 00'B (LSB first!), first two symols in slot, adjust if needed
  ss7->monitoringSymbolsWithinSlot->buf[1] = 0;
  ss7->monitoringSymbolsWithinSlot->buf[0] = (1<<7);
  ss7->monitoringSymbolsWithinSlot->size = 2;
  ss7->monitoringSymbolsWithinSlot->bits_unused = 2;
  ss7->nrofCandidates = calloc(1,sizeof(*ss7->nrofCandidates));
  ss7->nrofCandidates->aggregationLevel1 = NR_SearchSpace__nrofCandidates__aggregationLevel1_n0;
  ss7->nrofCandidates->aggregationLevel2 = NR_SearchSpace__nrofCandidates__aggregationLevel2_n0;
  ss7->nrofCandidates->aggregationLevel4 = NR_SearchSpace__nrofCandidates__aggregationLevel4_n4;
  ss7->nrofCandidates->aggregationLevel8 = NR_SearchSpace__nrofCandidates__aggregationLevel8_n2;
  ss7->nrofCandidates->aggregationLevel16 = NR_SearchSpace__nrofCandidates__aggregationLevel16_n1;
  ss7->searchSpaceType = calloc(1,sizeof(*ss7->searchSpaceType));
  ss7->searchSpaceType->present = NR_SearchSpace__searchSpaceType_PR_common;
  ss7->searchSpaceType->choice.common=calloc(1,sizeof(*ss7->searchSpaceType->choice.common));
  ss7->searchSpaceType->choice.common->dci_Format0_0_AndFormat1_0 = calloc(1,sizeof(*ss7->searchSpaceType->choice.common->dci_Format0_0_AndFormat1_0));
  ASN_SEQUENCE_ADD(&sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.pdcch_ConfigCommon->choice.setup->commonSearchSpaceList->list,ss7);

  sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.pdcch_ConfigCommon->choice.setup->searchSpaceSIB1 = calloc(1,sizeof(NR_SearchSpaceId_t));
  *sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.pdcch_ConfigCommon->choice.setup->searchSpaceSIB1 = 0;
  sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.pdcch_ConfigCommon->choice.setup->searchSpaceOtherSystemInformation = calloc(1,sizeof(NR_SearchSpaceId_t));
  *sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.pdcch_ConfigCommon->choice.setup->searchSpaceOtherSystemInformation = 7;
  sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.pdcch_ConfigCommon->choice.setup->pagingSearchSpace = calloc(1,sizeof(NR_SearchSpaceId_t));
  *sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.pdcch_ConfigCommon->choice.setup->pagingSearchSpace = 5;
  sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.pdcch_ConfigCommon->choice.setup->ra_SearchSpace = calloc(1,sizeof(NR_SearchSpaceId_t));
  *sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.pdcch_ConfigCommon->choice.setup->ra_SearchSpace = 1;

  sib1->servingCellConfigCommon->downlinkConfigCommon.initialDownlinkBWP.pdsch_ConfigCommon = configuration->scc->downlinkConfigCommon->initialDownlinkBWP->pdsch_ConfigCommon;
  sib1->servingCellConfigCommon->downlinkConfigCommon.bcch_Config.modificationPeriodCoeff = NR_BCCH_Config__modificationPeriodCoeff_n2;
  sib1->servingCellConfigCommon->downlinkConfigCommon.pcch_Config.defaultPagingCycle = NR_PagingCycle_rf256;
  sib1->servingCellConfigCommon->downlinkConfigCommon.pcch_Config.nAndPagingFrameOffset.present = NR_PCCH_Config__nAndPagingFrameOffset_PR_quarterT;
  sib1->servingCellConfigCommon->downlinkConfigCommon.pcch_Config.nAndPagingFrameOffset.choice.quarterT = 1;
  sib1->servingCellConfigCommon->downlinkConfigCommon.pcch_Config.ns = NR_PCCH_Config__ns_one;

  sib1->servingCellConfigCommon->downlinkConfigCommon.pcch_Config.firstPDCCH_MonitoringOccasionOfPO = calloc(1,sizeof(struct NR_PCCH_Config__firstPDCCH_MonitoringOccasionOfPO));
  sib1->servingCellConfigCommon->downlinkConfigCommon.pcch_Config.firstPDCCH_MonitoringOccasionOfPO->present = NR_PCCH_Config__firstPDCCH_MonitoringOccasionOfPO_PR_sCS120KHZoneT_SCS60KHZhalfT_SCS30KHZquarterT_SCS15KHZoneEighthT;

  sib1->servingCellConfigCommon->downlinkConfigCommon.pcch_Config.firstPDCCH_MonitoringOccasionOfPO->choice.sCS120KHZoneT_SCS60KHZhalfT_SCS30KHZquarterT_SCS15KHZoneEighthT = CALLOC(1,sizeof(struct NR_PCCH_Config__firstPDCCH_MonitoringOccasionOfPO__sCS120KHZoneT_SCS60KHZhalfT_SCS30KHZquarterT_SCS15KHZoneEighthT));
  asn_set_empty(&sib1->servingCellConfigCommon->downlinkConfigCommon.pcch_Config.firstPDCCH_MonitoringOccasionOfPO->choice.sCS120KHZoneT_SCS60KHZhalfT_SCS30KHZquarterT_SCS15KHZoneEighthT->list);

  long *sCS120KHZoneT_SCS60KHZhalfT_SCS30KHZquarterT_SCS15KHZoneEighthT = calloc(1,sizeof(long));
  *sCS120KHZoneT_SCS60KHZhalfT_SCS30KHZquarterT_SCS15KHZoneEighthT = 0;
  ASN_SEQUENCE_ADD(&sib1->servingCellConfigCommon->downlinkConfigCommon.pcch_Config.firstPDCCH_MonitoringOccasionOfPO->choice.sCS120KHZoneT_SCS60KHZhalfT_SCS30KHZquarterT_SCS15KHZoneEighthT->list,sCS120KHZoneT_SCS60KHZhalfT_SCS30KHZquarterT_SCS15KHZoneEighthT);

  sib1->servingCellConfigCommon->uplinkConfigCommon = CALLOC(1,sizeof(struct NR_UplinkConfigCommonSIB));
  asn_set_empty(&sib1->servingCellConfigCommon->uplinkConfigCommon->frequencyInfoUL.scs_SpecificCarrierList.list);
  for(int i = 0; i< configuration->scc->uplinkConfigCommon->frequencyInfoUL->scs_SpecificCarrierList.list.count; i++) {
    ASN_SEQUENCE_ADD(&sib1->servingCellConfigCommon->uplinkConfigCommon->frequencyInfoUL.scs_SpecificCarrierList.list,configuration->scc->uplinkConfigCommon->frequencyInfoUL->scs_SpecificCarrierList.list.array[i]);
  }

  sib1->servingCellConfigCommon->uplinkConfigCommon->frequencyInfoUL.p_Max = CALLOC(1,sizeof(NR_P_Max_t));
  *sib1->servingCellConfigCommon->uplinkConfigCommon->frequencyInfoUL.p_Max = 23;

  sib1->servingCellConfigCommon->uplinkConfigCommon->initialUplinkBWP.genericParameters = configuration->scc->uplinkConfigCommon->initialUplinkBWP->genericParameters;
  sib1->servingCellConfigCommon->uplinkConfigCommon->initialUplinkBWP.rach_ConfigCommon = configuration->scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon;

  sib1->servingCellConfigCommon->uplinkConfigCommon->initialUplinkBWP.pusch_ConfigCommon = configuration->scc->uplinkConfigCommon->initialUplinkBWP->pusch_ConfigCommon;
  sib1->servingCellConfigCommon->uplinkConfigCommon->initialUplinkBWP.pusch_ConfigCommon->choice.setup->groupHoppingEnabledTransformPrecoding = null;

  sib1->servingCellConfigCommon->uplinkConfigCommon->initialUplinkBWP.pucch_ConfigCommon = configuration->scc->uplinkConfigCommon->initialUplinkBWP->pucch_ConfigCommon;

  sib1->servingCellConfigCommon->uplinkConfigCommon->timeAlignmentTimerCommon = NR_TimeAlignmentTimer_infinity;

  sib1->servingCellConfigCommon->n_TimingAdvanceOffset = configuration->scc->n_TimingAdvanceOffset;

  sib1->servingCellConfigCommon->ssb_PositionsInBurst.inOneGroup.buf = calloc(1, sizeof(uint8_t));
  uint8_t bitmap8,temp_bitmap=0;
  switch (configuration->scc->ssb_PositionsInBurst->present) {
    case NR_ServingCellConfigCommon__ssb_PositionsInBurst_PR_shortBitmap:
      sib1->servingCellConfigCommon->ssb_PositionsInBurst.inOneGroup = configuration->scc->ssb_PositionsInBurst->choice.shortBitmap;
      break;
    case NR_ServingCellConfigCommon__ssb_PositionsInBurst_PR_mediumBitmap:
      sib1->servingCellConfigCommon->ssb_PositionsInBurst.inOneGroup = configuration->scc->ssb_PositionsInBurst->choice.mediumBitmap;
      break;
    /*
    groupPresence: This field is present when maximum number of SS/PBCH blocks per half frame equals to 64 as defined in TS 38.213 [13], clause 4.1.
                   The first/leftmost bit corresponds to the SS/PBCH index 0-7, the second bit corresponds to SS/PBCH block 8-15, and so on.
                   Value 0 in the bitmap indicates that the SSBs according to inOneGroup are absent. Value 1 indicates that the SS/PBCH blocks are transmitted in accordance with inOneGroup.
    inOneGroup: When maximum number of SS/PBCH blocks per half frame equals to 64 as defined in TS 38.213 [13], clause 4.1, all 8 bit are valid;
                The first/ leftmost bit corresponds to the first SS/PBCH block index in the group (i.e., to SSB index 0, 8, and so on); the second bit corresponds to the second SS/PBCH block index in the group
                (i.e., to SSB index 1, 9, and so on), and so on. Value 0 in the bitmap indicates that the corresponding SS/PBCH block is not transmitted while value 1 indicates that the corresponding SS/PBCH block is transmitted.
    */
    case NR_ServingCellConfigCommon__ssb_PositionsInBurst_PR_longBitmap:
      sib1->servingCellConfigCommon->ssb_PositionsInBurst.inOneGroup.size = 1;
      sib1->servingCellConfigCommon->ssb_PositionsInBurst.inOneGroup.bits_unused = 0;
      sib1->servingCellConfigCommon->ssb_PositionsInBurst.groupPresence = calloc(1, sizeof(BIT_STRING_t));
      memset(sib1->servingCellConfigCommon->ssb_PositionsInBurst.groupPresence, 0, sizeof(BIT_STRING_t));
      sib1->servingCellConfigCommon->ssb_PositionsInBurst.groupPresence->size = 1;
      sib1->servingCellConfigCommon->ssb_PositionsInBurst.groupPresence->bits_unused = 0;
      sib1->servingCellConfigCommon->ssb_PositionsInBurst.groupPresence->buf = calloc(1, sizeof(uint8_t));
      sib1->servingCellConfigCommon->ssb_PositionsInBurst.groupPresence->buf[0] = 0;
      for (int i=0; i<8; i++){
        bitmap8 = configuration->scc->ssb_PositionsInBurst->choice.longBitmap.buf[i];
        if (bitmap8!=0){
          if(temp_bitmap==0)
            temp_bitmap = bitmap8;
          else
            AssertFatal(temp_bitmap==bitmap8,"For longBitmap the groups of 8 SSBs containing at least 1 transmitted SSB should be all the same\n");

          sib1->servingCellConfigCommon->ssb_PositionsInBurst.inOneGroup.buf[0] = bitmap8;
          sib1->servingCellConfigCommon->ssb_PositionsInBurst.groupPresence->buf[0] |= 1<<(7-i);
        }
      }
      break;
    default:
      AssertFatal(false,"ssb_PositionsInBurst not present\n");
      break;
  }

  sib1->servingCellConfigCommon->ssb_PeriodicityServingCell = *configuration->scc->ssb_periodicityServingCell;
  sib1->servingCellConfigCommon->tdd_UL_DL_ConfigurationCommon = CALLOC(1,sizeof(struct NR_TDD_UL_DL_ConfigCommon));
  sib1->servingCellConfigCommon->tdd_UL_DL_ConfigurationCommon->referenceSubcarrierSpacing = configuration->scc->tdd_UL_DL_ConfigurationCommon->referenceSubcarrierSpacing;
  sib1->servingCellConfigCommon->tdd_UL_DL_ConfigurationCommon->pattern1 = configuration->scc->tdd_UL_DL_ConfigurationCommon->pattern1;
  sib1->servingCellConfigCommon->tdd_UL_DL_ConfigurationCommon->pattern2 = configuration->scc->tdd_UL_DL_ConfigurationCommon->pattern2;
  sib1->servingCellConfigCommon->ss_PBCH_BlockPower = configuration->scc->ss_PBCH_BlockPower;

  // ims-EmergencySupport
  // TODO: add ims-EmergencySupport

  // eCallOverIMS-Support
  // TODO: add eCallOverIMS-Support

  // ue-TimersAndConstants
  sib1->ue_TimersAndConstants = CALLOC(1,sizeof(struct NR_UE_TimersAndConstants));
  sib1->ue_TimersAndConstants->t300 = NR_UE_TimersAndConstants__t300_ms400;
  sib1->ue_TimersAndConstants->t301 = NR_UE_TimersAndConstants__t301_ms400;
  sib1->ue_TimersAndConstants->t310 = NR_UE_TimersAndConstants__t310_ms2000;
  sib1->ue_TimersAndConstants->n310 = NR_UE_TimersAndConstants__n310_n10;
  sib1->ue_TimersAndConstants->t311 = NR_UE_TimersAndConstants__t311_ms3000;
  sib1->ue_TimersAndConstants->n311 = NR_UE_TimersAndConstants__n311_n1;
  sib1->ue_TimersAndConstants->t319 = NR_UE_TimersAndConstants__t319_ms400;

  // uac-BarringInfo
  /*sib1->uac_BarringInfo = CALLOC(1, sizeof(struct NR_SIB1__uac_BarringInfo));
  NR_UAC_BarringInfoSet_t *nr_uac_BarringInfoSet = CALLOC(1, sizeof(NR_UAC_BarringInfoSet_t));
  asn_set_empty(&sib1->uac_BarringInfo->uac_BarringInfoSetList);
  nr_uac_BarringInfoSet->uac_BarringFactor = NR_UAC_BarringInfoSet__uac_BarringFactor_p95;
  nr_uac_BarringInfoSet->uac_BarringTime = NR_UAC_BarringInfoSet__uac_BarringTime_s4;
  nr_uac_BarringInfoSet->uac_BarringForAccessIdentity.buf = CALLOC(1, 1);
  nr_uac_BarringInfoSet->uac_BarringForAccessIdentity.size = 1;
  nr_uac_BarringInfoSet->uac_BarringForAccessIdentity.bits_unused = 1;
  ASN_SEQUENCE_ADD(&sib1->uac_BarringInfo->uac_BarringInfoSetList, nr_uac_BarringInfoSet);*/

  // useFullResumeID
  // TODO: add useFullResumeID

  // lateNonCriticalExtension
  // TODO: add lateNonCriticalExtension

  // nonCriticalExtension
  // TODO: add nonCriticalExtension

  xer_fprint(stdout, &asn_DEF_NR_SIB1, (const void*)sib1_message->message.choice.c1->choice.systemInformationBlockType1);

  if(carrier->SIB1 == NULL) carrier->SIB1=(uint8_t *) malloc16(NR_MAX_SIB_LENGTH/8);
  enc_rval = uper_encode_to_buffer(&asn_DEF_NR_BCCH_DL_SCH_Message,
                                   NULL,
                                   (void *)sib1_message,
                                   carrier->SIB1,
                                   NR_MAX_SIB_LENGTH/8);
  AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %lu)!\n",
               enc_rval.failed_type->name, enc_rval.encoded);

  if (enc_rval.encoded==-1) {
    return(-1);
  }

  return((enc_rval.encoded+7)/8);
}


void  do_RLC_BEARER(uint8_t Mod_id,
                    int CC_id,
                    struct NR_CellGroupConfig__rlc_BearerToAddModList *rlc_BearerToAddModList,
                    rlc_bearer_config_t  *rlc_config) {
  struct NR_RLC_BearerConfig *rlc_bearer;
  rlc_bearer = CALLOC(1,sizeof(struct NR_RLC_BearerConfig));
  rlc_bearer->logicalChannelIdentity = rlc_config->LogicalChannelIdentity[CC_id];
  rlc_bearer->servedRadioBearer = CALLOC(1,sizeof(struct NR_RLC_BearerConfig__servedRadioBearer));
  rlc_bearer->servedRadioBearer->present = rlc_config->servedRadioBearer_present[CC_id];

  if(rlc_bearer->servedRadioBearer->present == NR_RLC_BearerConfig__servedRadioBearer_PR_srb_Identity) {
    rlc_bearer->servedRadioBearer->choice.srb_Identity = rlc_config->srb_Identity[CC_id];
  } else if(rlc_bearer->servedRadioBearer->present == NR_RLC_BearerConfig__servedRadioBearer_PR_drb_Identity) {
    rlc_bearer->servedRadioBearer->choice.drb_Identity = rlc_config->drb_Identity[CC_id];
  }

  rlc_bearer->reestablishRLC = CALLOC(1,sizeof(long));
  *(rlc_bearer->reestablishRLC) = rlc_config->reestablishRLC[CC_id];
  rlc_bearer->rlc_Config = CALLOC(1,sizeof(struct NR_RLC_Config));
  rlc_bearer->rlc_Config->present = rlc_config->rlc_Config_present[CC_id];

  if(rlc_bearer->rlc_Config->present == NR_RLC_Config_PR_am) {
    rlc_bearer->rlc_Config->choice.am = CALLOC(1,sizeof(struct NR_RLC_Config__am));
    rlc_bearer->rlc_Config->choice.am->ul_AM_RLC.sn_FieldLength     = CALLOC(1,sizeof(NR_SN_FieldLengthAM_t));
    *(rlc_bearer->rlc_Config->choice.am->ul_AM_RLC.sn_FieldLength)  = rlc_config->ul_AM_sn_FieldLength[CC_id];
    rlc_bearer->rlc_Config->choice.am->ul_AM_RLC.t_PollRetransmit   = rlc_config->t_PollRetransmit[CC_id];
    rlc_bearer->rlc_Config->choice.am->ul_AM_RLC.pollPDU            = rlc_config->pollPDU[CC_id];
    rlc_bearer->rlc_Config->choice.am->ul_AM_RLC.pollByte           = rlc_config->pollByte[CC_id];
    rlc_bearer->rlc_Config->choice.am->ul_AM_RLC.maxRetxThreshold   = rlc_config->maxRetxThreshold[CC_id];
    rlc_bearer->rlc_Config->choice.am->dl_AM_RLC.sn_FieldLength     = CALLOC(1,sizeof(NR_SN_FieldLengthAM_t));
    *(rlc_bearer->rlc_Config->choice.am->dl_AM_RLC.sn_FieldLength)  = rlc_config->dl_AM_sn_FieldLength[CC_id];
    rlc_bearer->rlc_Config->choice.am->dl_AM_RLC.t_Reassembly       = rlc_config->dl_AM_t_Reassembly[CC_id];
    rlc_bearer->rlc_Config->choice.am->dl_AM_RLC.t_StatusProhibit   = rlc_config->t_StatusProhibit[CC_id];
  } else if(rlc_bearer->rlc_Config->present == NR_RLC_Config_PR_um_Bi_Directional) {
    rlc_bearer->rlc_Config->choice.um_Bi_Directional = CALLOC(1,sizeof(struct NR_RLC_Config__um_Bi_Directional));
    rlc_bearer->rlc_Config->choice.um_Bi_Directional->ul_UM_RLC.sn_FieldLength = CALLOC(1,sizeof(NR_SN_FieldLengthUM_t));
    *(rlc_bearer->rlc_Config->choice.um_Bi_Directional->ul_UM_RLC.sn_FieldLength) = rlc_config->ul_UM_sn_FieldLength[CC_id];
    rlc_bearer->rlc_Config->choice.um_Bi_Directional->dl_UM_RLC.sn_FieldLength = CALLOC(1,sizeof(NR_SN_FieldLengthUM_t));
    *(rlc_bearer->rlc_Config->choice.um_Bi_Directional->dl_UM_RLC.sn_FieldLength) = rlc_config->dl_UM_sn_FieldLength[CC_id];
    rlc_bearer->rlc_Config->choice.um_Bi_Directional->dl_UM_RLC.t_Reassembly   = rlc_config->dl_UM_t_Reassembly[CC_id];
  } else if(rlc_bearer->rlc_Config->present == NR_RLC_Config_PR_um_Uni_Directional_UL) {
    rlc_bearer->rlc_Config->choice.um_Uni_Directional_UL = CALLOC(1,sizeof(struct NR_RLC_Config__um_Uni_Directional_UL));
    rlc_bearer->rlc_Config->choice.um_Uni_Directional_UL->ul_UM_RLC.sn_FieldLength    = CALLOC(1,sizeof(NR_SN_FieldLengthUM_t));
    *(rlc_bearer->rlc_Config->choice.um_Uni_Directional_UL->ul_UM_RLC.sn_FieldLength) = rlc_config->ul_UM_sn_FieldLength[CC_id];
  } else if(rlc_bearer->rlc_Config->present == NR_RLC_Config_PR_um_Uni_Directional_DL) {
    rlc_bearer->rlc_Config->choice.um_Uni_Directional_DL = CALLOC(1,sizeof(struct NR_RLC_Config__um_Uni_Directional_DL));
    rlc_bearer->rlc_Config->choice.um_Uni_Directional_DL->dl_UM_RLC.sn_FieldLength    = CALLOC(1,sizeof(NR_SN_FieldLengthUM_t));
    *(rlc_bearer->rlc_Config->choice.um_Uni_Directional_DL->dl_UM_RLC.sn_FieldLength) = rlc_config->dl_UM_sn_FieldLength[CC_id];
    rlc_bearer->rlc_Config->choice.um_Uni_Directional_DL->dl_UM_RLC.t_Reassembly      = rlc_config->dl_UM_t_Reassembly[CC_id];
  }

  rlc_bearer->mac_LogicalChannelConfig = CALLOC(1,sizeof(struct NR_LogicalChannelConfig));
  rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters = CALLOC(1,sizeof(struct NR_LogicalChannelConfig__ul_SpecificParameters));
  rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters->priority            = rlc_config->priority[CC_id];
  rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters->prioritisedBitRate  = rlc_config->prioritisedBitRate[CC_id];
  rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters->bucketSizeDuration  = rlc_config->bucketSizeDuration[CC_id];
  rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters->allowedServingCells = CALLOC(1,sizeof(struct NR_LogicalChannelConfig__ul_SpecificParameters__allowedServingCells));
  rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters->allowedSCS_List     = CALLOC(1,sizeof(struct NR_LogicalChannelConfig__ul_SpecificParameters__allowedSCS_List));
  NR_ServCellIndex_t *servingcellindex;
  servingcellindex = CALLOC(1,sizeof(NR_ServCellIndex_t));
  *servingcellindex = rlc_config->allowedServingCells[CC_id];
  ASN_SEQUENCE_ADD(&(rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters->allowedServingCells->list),&servingcellindex);
  NR_SubcarrierSpacing_t *subcarrierspacing;
  subcarrierspacing = CALLOC(1,sizeof(NR_SubcarrierSpacing_t));
  *subcarrierspacing = rlc_config->subcarrierspacing[CC_id];
  ASN_SEQUENCE_ADD(&(rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters->allowedSCS_List->list),&subcarrierspacing);
  rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters->maxPUSCH_Duration           = CALLOC(1,sizeof(long));
  rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters->configuredGrantType1Allowed = CALLOC(1,sizeof(long));
  rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters->logicalChannelGroup         = CALLOC(1,sizeof(long));
  rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters->schedulingRequestID         = CALLOC(1,sizeof(NR_SchedulingRequestId_t));
  *(rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters->maxPUSCH_Duration)           = rlc_config->maxPUSCH_Duration[CC_id];
  *(rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters->configuredGrantType1Allowed) = rlc_config->configuredGrantType1Allowed[CC_id];
  *(rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters->logicalChannelGroup)         = rlc_config->logicalChannelGroup[CC_id];
  *(rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters->schedulingRequestID)         = rlc_config->schedulingRequestID[CC_id];
  rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters->logicalChannelSR_Mask               = rlc_config->logicalChannelSR_Mask[CC_id];
  rlc_bearer->mac_LogicalChannelConfig->ul_SpecificParameters->logicalChannelSR_DelayTimerApplied  = rlc_config->logicalChannelSR_DelayTimerApplied[CC_id];
  ASN_SEQUENCE_ADD(&(rlc_BearerToAddModList->list),&rlc_bearer);
}


void do_MAC_CELLGROUP(uint8_t Mod_id,
                      int CC_id,
                      NR_MAC_CellGroupConfig_t *mac_CellGroupConfig,
                      mac_cellgroup_t  *mac_cellgroup_config) {
  mac_CellGroupConfig->drx_Config               = CALLOC(1,sizeof(struct NR_SetupRelease_DRX_Config));
  mac_CellGroupConfig->schedulingRequestConfig  = CALLOC(1,sizeof(struct NR_SchedulingRequestConfig));
  mac_CellGroupConfig->bsr_Config               = CALLOC(1,sizeof(struct NR_BSR_Config));
  mac_CellGroupConfig->tag_Config               = CALLOC(1,sizeof(struct NR_TAG_Config));
  mac_CellGroupConfig->phr_Config               = CALLOC(1,sizeof(struct NR_SetupRelease_PHR_Config));
  mac_CellGroupConfig->drx_Config->present      = mac_cellgroup_config->DRX_Config_PR[CC_id];
  mac_CellGroupConfig->drx_Config->choice.setup = CALLOC(1,sizeof(struct NR_DRX_Config));
  mac_CellGroupConfig->drx_Config->choice.setup->drx_onDurationTimer.present = mac_cellgroup_config->drx_onDurationTimer_PR[CC_id];

  if(mac_CellGroupConfig->drx_Config->choice.setup->drx_onDurationTimer.present == NR_DRX_Config__drx_onDurationTimer_PR_subMilliSeconds) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_onDurationTimer.choice.subMilliSeconds = mac_cellgroup_config->subMilliSeconds[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_onDurationTimer.present == NR_DRX_Config__drx_onDurationTimer_PR_milliSeconds) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_onDurationTimer.choice.milliSeconds    = mac_cellgroup_config->milliSeconds[CC_id];
  }

  mac_CellGroupConfig->drx_Config->choice.setup->drx_InactivityTimer        = mac_cellgroup_config->drx_InactivityTimer[CC_id];
  mac_CellGroupConfig->drx_Config->choice.setup->drx_HARQ_RTT_TimerDL       = mac_cellgroup_config->drx_HARQ_RTT_TimerDL[CC_id];
  mac_CellGroupConfig->drx_Config->choice.setup->drx_HARQ_RTT_TimerUL       = mac_cellgroup_config->drx_HARQ_RTT_TimerUL[CC_id];
  mac_CellGroupConfig->drx_Config->choice.setup->drx_RetransmissionTimerDL  = mac_cellgroup_config->drx_RetransmissionTimerDL[CC_id];
  mac_CellGroupConfig->drx_Config->choice.setup->drx_RetransmissionTimerUL  = mac_cellgroup_config->drx_RetransmissionTimerUL[CC_id];
  mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present = mac_cellgroup_config->drx_LongCycleStartOffset_PR[CC_id];

  if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms10) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms10 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms20) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms20 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms32) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms32 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms40) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms40 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms60) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms60 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms64) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms64 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms70) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms70 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms80) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms80 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms128) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms128 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms160) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms160 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms256) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms256 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms320) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms320 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms512) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms512 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms640) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms640 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms1024) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms1024 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms1280) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms1280 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms2048) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms2048 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms2560) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms2560 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms5120) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms5120 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  } else if(mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.present == NR_DRX_Config__drx_LongCycleStartOffset_PR_ms10240) {
    mac_CellGroupConfig->drx_Config->choice.setup->drx_LongCycleStartOffset.choice.ms10240 = mac_cellgroup_config->drx_LongCycleStartOffset[CC_id];
  }

  mac_CellGroupConfig->drx_Config->choice.setup->shortDRX = CALLOC(1,sizeof(struct NR_DRX_Config__shortDRX));
  mac_CellGroupConfig->drx_Config->choice.setup->shortDRX->drx_ShortCycle       = mac_cellgroup_config->drx_ShortCycle[CC_id];
  mac_CellGroupConfig->drx_Config->choice.setup->shortDRX->drx_ShortCycleTimer  = mac_cellgroup_config->drx_ShortCycleTimer[CC_id];
  mac_CellGroupConfig->drx_Config->choice.setup->drx_SlotOffset                 = mac_cellgroup_config->drx_SlotOffset[CC_id];
  mac_CellGroupConfig->schedulingRequestConfig->schedulingRequestToAddModList = CALLOC(1,sizeof(struct NR_SchedulingRequestConfig__schedulingRequestToAddModList));
  struct NR_SchedulingRequestToAddMod *schedulingrequestlist;
  schedulingrequestlist = CALLOC(1,sizeof(struct NR_SchedulingRequestToAddMod));
  schedulingrequestlist->schedulingRequestId  = mac_cellgroup_config->schedulingRequestId[CC_id];
  schedulingrequestlist->sr_ProhibitTimer = CALLOC(1,sizeof(long));
  *(schedulingrequestlist->sr_ProhibitTimer) = mac_cellgroup_config->sr_ProhibitTimer[CC_id];
  schedulingrequestlist->sr_TransMax      = mac_cellgroup_config->sr_TransMax[CC_id];
  ASN_SEQUENCE_ADD(&(mac_CellGroupConfig->schedulingRequestConfig->schedulingRequestToAddModList->list),&schedulingrequestlist);
  mac_CellGroupConfig->bsr_Config->periodicBSR_Timer              = mac_cellgroup_config->periodicBSR_Timer[CC_id];
  mac_CellGroupConfig->bsr_Config->retxBSR_Timer                  = mac_cellgroup_config->retxBSR_Timer[CC_id];
  mac_CellGroupConfig->bsr_Config->logicalChannelSR_DelayTimer    = CALLOC(1,sizeof(long));
  *(mac_CellGroupConfig->bsr_Config->logicalChannelSR_DelayTimer)    = mac_cellgroup_config->logicalChannelSR_DelayTimer[CC_id];
  mac_CellGroupConfig->tag_Config->tag_ToAddModList = CALLOC(1,sizeof(struct NR_TAG_Config__tag_ToAddModList));
  struct NR_TAG *tag;
  tag = CALLOC(1,sizeof(struct NR_TAG));
  tag->tag_Id             = mac_cellgroup_config->tag_Id[CC_id];
  tag->timeAlignmentTimer = mac_cellgroup_config->timeAlignmentTimer[CC_id];
  ASN_SEQUENCE_ADD(&(mac_CellGroupConfig->tag_Config->tag_ToAddModList->list),&tag);
  mac_CellGroupConfig->phr_Config->present = mac_cellgroup_config->PHR_Config_PR[CC_id];
  mac_CellGroupConfig->phr_Config->choice.setup   = CALLOC(1,sizeof(struct NR_PHR_Config));
  mac_CellGroupConfig->phr_Config->choice.setup->phr_PeriodicTimer         = mac_cellgroup_config->phr_PeriodicTimer[CC_id];
  mac_CellGroupConfig->phr_Config->choice.setup->phr_ProhibitTimer         = mac_cellgroup_config->phr_ProhibitTimer[CC_id];
  mac_CellGroupConfig->phr_Config->choice.setup->phr_Tx_PowerFactorChange  = mac_cellgroup_config->phr_Tx_PowerFactorChange[CC_id];
  mac_CellGroupConfig->phr_Config->choice.setup->multiplePHR               = mac_cellgroup_config->multiplePHR[CC_id];
  mac_CellGroupConfig->phr_Config->choice.setup->dummy                     = mac_cellgroup_config->phr_Type2SpCell[CC_id];
  mac_CellGroupConfig->phr_Config->choice.setup->phr_Type2OtherCell        = mac_cellgroup_config->phr_Type2OtherCell[CC_id];
  mac_CellGroupConfig->phr_Config->choice.setup->phr_ModeOtherCG           = mac_cellgroup_config->phr_ModeOtherCG[CC_id];
  mac_CellGroupConfig->skipUplinkTxDynamic      = mac_cellgroup_config->skipUplinkTxDynamic[CC_id];
}


void  do_PHYSICALCELLGROUP(uint8_t Mod_id,
                           int CC_id,
                           NR_PhysicalCellGroupConfig_t *physicalCellGroupConfig,
                           physicalcellgroup_t *physicalcellgroup_config) {
  physicalCellGroupConfig->harq_ACK_SpatialBundlingPUCCH = CALLOC(1,sizeof(long));
  physicalCellGroupConfig->harq_ACK_SpatialBundlingPUSCH = CALLOC(1,sizeof(long));
  physicalCellGroupConfig->p_NR_FR1                      = CALLOC(1,sizeof(NR_P_Max_t));
  physicalCellGroupConfig->tpc_SRS_RNTI                  = CALLOC(1,sizeof(NR_RNTI_Value_t));
  physicalCellGroupConfig->tpc_PUCCH_RNTI                = CALLOC(1,sizeof(NR_RNTI_Value_t));
  physicalCellGroupConfig->tpc_PUSCH_RNTI                = CALLOC(1,sizeof(NR_RNTI_Value_t));
  physicalCellGroupConfig->sp_CSI_RNTI                   = CALLOC(1,sizeof(NR_RNTI_Value_t));
  *(physicalCellGroupConfig->harq_ACK_SpatialBundlingPUCCH) = physicalcellgroup_config->harq_ACK_SpatialBundlingPUCCH[CC_id];
  *(physicalCellGroupConfig->harq_ACK_SpatialBundlingPUSCH) = physicalcellgroup_config->harq_ACK_SpatialBundlingPUSCH[CC_id];
  *(physicalCellGroupConfig->p_NR_FR1)                      = physicalcellgroup_config->p_NR[CC_id];
  physicalCellGroupConfig->pdsch_HARQ_ACK_Codebook          = physicalcellgroup_config->pdsch_HARQ_ACK_Codebook[CC_id];
  *(physicalCellGroupConfig->tpc_SRS_RNTI)                  = physicalcellgroup_config->tpc_SRS_RNTI[CC_id];
  *(physicalCellGroupConfig->tpc_PUCCH_RNTI)                = physicalcellgroup_config->tpc_PUCCH_RNTI[CC_id];
  *(physicalCellGroupConfig->tpc_PUSCH_RNTI)                = physicalcellgroup_config->tpc_PUSCH_RNTI[CC_id];
  *(physicalCellGroupConfig->sp_CSI_RNTI)                   = physicalcellgroup_config->sp_CSI_RNTI[CC_id];
  physicalCellGroupConfig->cs_RNTI                       = CALLOC(1,sizeof(struct NR_SetupRelease_RNTI_Value));
  physicalCellGroupConfig->cs_RNTI->present              = physicalcellgroup_config->RNTI_Value_PR[CC_id];

  if(physicalCellGroupConfig->cs_RNTI->present == NR_SetupRelease_RNTI_Value_PR_setup) {
    physicalCellGroupConfig->cs_RNTI->choice.setup = physicalcellgroup_config->RNTI_Value[CC_id];
  }
}



void do_SpCellConfig(gNB_RRC_INST *rrc,
                      struct NR_SpCellConfig  *spconfig){
  //gNB_RrcConfigurationReq  *common_configuration;
  //common_configuration = CALLOC(1,sizeof(gNB_RrcConfigurationReq));
  //Fill servingcellconfigcommon config value
  //Fill common config to structure
  //  rrc->configuration = common_configuration;
  spconfig->reconfigurationWithSync = CALLOC(1,sizeof(struct NR_ReconfigurationWithSync));
}

//------------------------------------------------------------------------------
uint8_t do_RRCReject(uint8_t Mod_id,
                  uint8_t *const buffer)
//------------------------------------------------------------------------------
{
    asn_enc_rval_t                                   enc_rval;;
    NR_DL_CCCH_Message_t                             dl_ccch_msg;
    NR_RRCReject_t                                   *rrcReject;
    NR_RejectWaitTime_t                              waitTime = 1;

    memset((void *)&dl_ccch_msg, 0, sizeof(NR_DL_CCCH_Message_t));
    dl_ccch_msg.message.present = NR_DL_CCCH_MessageType_PR_c1;
    dl_ccch_msg.message.choice.c1          = CALLOC(1, sizeof(struct NR_DL_CCCH_MessageType__c1));
    dl_ccch_msg.message.choice.c1->present = NR_RRCReject__criticalExtensions_PR_rrcReject;

    dl_ccch_msg.message.choice.c1->choice.rrcReject = CALLOC(1,sizeof(NR_RRCReject_t));
    rrcReject = dl_ccch_msg.message.choice.c1->choice.rrcReject;

    rrcReject->criticalExtensions.choice.rrcReject           = CALLOC(1, sizeof(struct NR_RRCReject_IEs));
    rrcReject->criticalExtensions.choice.rrcReject->waitTime = CALLOC(1, sizeof(NR_RejectWaitTime_t));

    rrcReject->criticalExtensions.present = NR_RRCReject__criticalExtensions_PR_rrcReject;
    rrcReject->criticalExtensions.choice.rrcReject->waitTime = &waitTime;

    if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
        xer_fprint(stdout, &asn_DEF_NR_DL_CCCH_Message, (void *)&dl_ccch_msg);
    }

    enc_rval = uper_encode_to_buffer(&asn_DEF_NR_DL_CCCH_Message,
                                    NULL,
                                    (void *)&dl_ccch_msg,
                                    buffer,
                                    100);

    if(enc_rval.encoded == -1) {
        LOG_E(NR_RRC, "[gNB AssertFatal]ASN1 message encoding failed (%s, %lu)!\n",
            enc_rval.failed_type->name, enc_rval.encoded);
        return -1;
    }

    LOG_D(NR_RRC,"RRCReject Encoded %zd bits (%zd bytes)\n",
            enc_rval.encoded,(enc_rval.encoded+7)/8);
    return((enc_rval.encoded+7)/8);
}

//------------------------------------------------------------------------------
uint8_t do_RRCSetup(const protocol_ctxt_t        *const ctxt_pP,
                    rrc_gNB_ue_context_t         *const ue_context_pP,
                    int                          CC_id,
                    uint8_t                      *const buffer,
                    const uint8_t                transaction_id,
                    NR_SRB_ToAddModList_t        **SRB_configList)
//------------------------------------------------------------------------------
{
    asn_enc_rval_t                                   enc_rval;;
    NR_DL_CCCH_Message_t                             dl_ccch_msg;
    NR_RRCSetup_t                                    *rrcSetup;
    NR_RRCSetup_IEs_t                                *ie;
    NR_SRB_ToAddMod_t                                *SRB1_config          = NULL;
    NR_PDCP_Config_t                                 *pdcp_Config          = NULL;
    NR_CellGroupConfig_t                             *cellGroupConfig      = NULL;
    NR_RLC_BearerConfig_t                            *rlc_BearerConfig     = NULL;
    NR_RLC_Config_t                                  *rlc_Config           = NULL;
    NR_LogicalChannelConfig_t                        *logicalChannelConfig = NULL;
    NR_MAC_CellGroupConfig_t                         *mac_CellGroupConfig  = NULL;

    char masterCellGroup_buf[1000];
    long *logicalChannelGroup = NULL;

    memset((void *)&dl_ccch_msg, 0, sizeof(NR_DL_CCCH_Message_t));
    dl_ccch_msg.message.present            = NR_DL_CCCH_MessageType_PR_c1;
    dl_ccch_msg.message.choice.c1          = CALLOC(1, sizeof(struct NR_DL_CCCH_MessageType__c1));
    dl_ccch_msg.message.choice.c1->present = NR_DL_CCCH_MessageType__c1_PR_rrcSetup;
    dl_ccch_msg.message.choice.c1->choice.rrcSetup = calloc(1, sizeof(NR_RRCSetup_t));

    rrcSetup = dl_ccch_msg.message.choice.c1->choice.rrcSetup;
    rrcSetup->criticalExtensions.present = NR_RRCSetup__criticalExtensions_PR_rrcSetup;
    rrcSetup->rrc_TransactionIdentifier  = transaction_id;
    rrcSetup->criticalExtensions.choice.rrcSetup = calloc(1, sizeof(NR_RRCSetup_IEs_t));
    ie = rrcSetup->criticalExtensions.choice.rrcSetup;

    /****************************** radioBearerConfig ******************************/
    /* Configure SRB1 */
    if (*SRB_configList) {
        free(*SRB_configList);
    }

    *SRB_configList = calloc(1, sizeof(NR_SRB_ToAddModList_t));
    // SRB1
    /* TODO */
    SRB1_config = calloc(1, sizeof(NR_SRB_ToAddMod_t));
    SRB1_config->srb_Identity = 1;
    // pdcp_Config->t_Reordering
    SRB1_config->pdcp_Config = pdcp_Config;
    ie->radioBearerConfig.srb_ToAddModList = *SRB_configList;
    ASN_SEQUENCE_ADD(&(*SRB_configList)->list, SRB1_config);

    ie->radioBearerConfig.srb3_ToRelease    = NULL;
    ie->radioBearerConfig.drb_ToAddModList  = NULL;
    ie->radioBearerConfig.drb_ToReleaseList = NULL;
    ie->radioBearerConfig.securityConfig    = NULL;

    /****************************** masterCellGroup ******************************/
    /* TODO */
    cellGroupConfig = calloc(1, sizeof(NR_CellGroupConfig_t));
    cellGroupConfig->cellGroupId = 0;

    /* Rlc Bearer Config */
    /* TS38.331 9.2.1	Default SRB configurations */
    cellGroupConfig->rlc_BearerToAddModList                          = calloc(1, sizeof(*cellGroupConfig->rlc_BearerToAddModList));
    rlc_BearerConfig                                                 = calloc(1, sizeof(NR_RLC_BearerConfig_t));
    rlc_BearerConfig->logicalChannelIdentity                         = 1;
    rlc_BearerConfig->servedRadioBearer                              = calloc(1, sizeof(*rlc_BearerConfig->servedRadioBearer));
    rlc_BearerConfig->servedRadioBearer->present                     = NR_RLC_BearerConfig__servedRadioBearer_PR_srb_Identity;
    rlc_BearerConfig->servedRadioBearer->choice.srb_Identity         = 1;
    rlc_BearerConfig->reestablishRLC                                 = NULL;
    rlc_Config = calloc(1, sizeof(NR_RLC_Config_t));
    rlc_Config->present                                              = NR_RLC_Config_PR_am;
    rlc_Config->choice.am                                            = calloc(1, sizeof(*rlc_Config->choice.am));
    rlc_Config->choice.am->dl_AM_RLC.sn_FieldLength                  = calloc(1, sizeof(NR_SN_FieldLengthAM_t));
    *(rlc_Config->choice.am->dl_AM_RLC.sn_FieldLength)               = NR_SN_FieldLengthAM_size12;
    rlc_Config->choice.am->dl_AM_RLC.t_Reassembly                    = NR_T_Reassembly_ms35;
    rlc_Config->choice.am->dl_AM_RLC.t_StatusProhibit                = NR_T_StatusProhibit_ms0;
    rlc_Config->choice.am->ul_AM_RLC.sn_FieldLength                  = calloc(1, sizeof(NR_SN_FieldLengthAM_t));
    *(rlc_Config->choice.am->ul_AM_RLC.sn_FieldLength)               = NR_SN_FieldLengthAM_size12;
    rlc_Config->choice.am->ul_AM_RLC.t_PollRetransmit                = NR_T_PollRetransmit_ms45;
    rlc_Config->choice.am->ul_AM_RLC.pollPDU                         = NR_PollPDU_infinity;
    rlc_Config->choice.am->ul_AM_RLC.pollByte                        = NR_PollByte_infinity;
    rlc_Config->choice.am->ul_AM_RLC.maxRetxThreshold                = NR_UL_AM_RLC__maxRetxThreshold_t8;
    rlc_BearerConfig->rlc_Config                                     = rlc_Config;
    logicalChannelConfig                                             = calloc(1, sizeof(NR_LogicalChannelConfig_t));
    logicalChannelConfig->ul_SpecificParameters                      = calloc(1, sizeof(*logicalChannelConfig->ul_SpecificParameters));
    logicalChannelConfig->ul_SpecificParameters->priority            = 1;
    logicalChannelConfig->ul_SpecificParameters->prioritisedBitRate  = NR_LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_infinity;
    logicalChannelGroup                                              = CALLOC(1, sizeof(long));
    *logicalChannelGroup                                             = 0;
    logicalChannelConfig->ul_SpecificParameters->logicalChannelGroup = logicalChannelGroup;
    rlc_BearerConfig->mac_LogicalChannelConfig                       = logicalChannelConfig;
    ASN_SEQUENCE_ADD(&cellGroupConfig->rlc_BearerToAddModList->list, rlc_BearerConfig);

    cellGroupConfig->rlc_BearerToReleaseList = NULL;
    cellGroupConfig->sCellToAddModList       = NULL;
    cellGroupConfig->sCellToReleaseList      = NULL;

    /* mac CellGroup Config */
    mac_CellGroupConfig                                                     = calloc(1, sizeof(NR_MAC_CellGroupConfig_t));
    mac_CellGroupConfig->bsr_Config                                         = calloc(1, sizeof(*mac_CellGroupConfig->bsr_Config));
    mac_CellGroupConfig->bsr_Config->periodicBSR_Timer                      = NR_BSR_Config__periodicBSR_Timer_sf10;
    mac_CellGroupConfig->bsr_Config->retxBSR_Timer                          = NR_BSR_Config__retxBSR_Timer_sf80;
    mac_CellGroupConfig->phr_Config                                         = calloc(1, sizeof(*mac_CellGroupConfig->phr_Config));
    mac_CellGroupConfig->phr_Config->present                                = NR_SetupRelease_PHR_Config_PR_setup;
    mac_CellGroupConfig->phr_Config->choice.setup                           = calloc(1, sizeof(*mac_CellGroupConfig->phr_Config->choice.setup));
    mac_CellGroupConfig->phr_Config->choice.setup->phr_PeriodicTimer        = NR_PHR_Config__phr_PeriodicTimer_sf10;
    mac_CellGroupConfig->phr_Config->choice.setup->phr_ProhibitTimer        = NR_PHR_Config__phr_ProhibitTimer_sf10;
    mac_CellGroupConfig->phr_Config->choice.setup->phr_Tx_PowerFactorChange = NR_PHR_Config__phr_Tx_PowerFactorChange_dB1;
    cellGroupConfig->mac_CellGroupConfig                                     = mac_CellGroupConfig;

    // cellGroupConfig.physicalCellGroupConfig;

    enc_rval = uper_encode_to_buffer(&asn_DEF_NR_CellGroupConfig,
                                    NULL,
                                    (void *)cellGroupConfig,
                                    masterCellGroup_buf,
                                    100);

    if(enc_rval.encoded == -1) {
        LOG_E(NR_RRC, "ASN1 message CellGroupConfig encoding failed (%s, %lu)!\n",
            enc_rval.failed_type->name, enc_rval.encoded);
        return -1;
    }

    if (OCTET_STRING_fromBuf(&ie->masterCellGroup, masterCellGroup_buf, (enc_rval.encoded+7)/8) == -1) {
        LOG_E(NR_RRC, "fatal: OCTET_STRING_fromBuf failed\n");
        return -1;
    }

    if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
        xer_fprint(stdout, &asn_DEF_NR_DL_CCCH_Message, (void *)&dl_ccch_msg);
    }

    enc_rval = uper_encode_to_buffer(&asn_DEF_NR_DL_CCCH_Message,
                                    NULL,
                                    (void *)&dl_ccch_msg,
                                    buffer,
                                    100);

    if(enc_rval.encoded == -1) {
        LOG_E(NR_RRC, "[gNB AssertFatal]ASN1 message encoding failed (%s, %lu)!\n",
            enc_rval.failed_type->name, enc_rval.encoded);
        return -1;
    }

    LOG_D(NR_RRC,"RRCSetup Encoded %zd bits (%zd bytes)\n",
            enc_rval.encoded,(enc_rval.encoded+7)/8);
    return((enc_rval.encoded+7)/8);
}

uint8_t do_NR_SecurityModeCommand(
  const protocol_ctxt_t *const ctxt_pP,
  uint8_t *const buffer,
  const uint8_t Transaction_id,
  const uint8_t cipheringAlgorithm,
  NR_IntegrityProtAlgorithm_t *integrityProtAlgorithm
)
//------------------------------------------------------------------------------
{
  NR_DL_DCCH_Message_t dl_dcch_msg;
  asn_enc_rval_t enc_rval;
  memset(&dl_dcch_msg,0,sizeof(NR_DL_DCCH_Message_t));
  dl_dcch_msg.message.present           = NR_DL_DCCH_MessageType_PR_c1;
  dl_dcch_msg.message.choice.c1=CALLOC(1,sizeof(struct NR_DL_DCCH_MessageType__c1));
  dl_dcch_msg.message.choice.c1->present = NR_DL_DCCH_MessageType__c1_PR_securityModeCommand;
  dl_dcch_msg.message.choice.c1->choice.securityModeCommand = CALLOC(1, sizeof(struct NR_SecurityModeCommand));
  dl_dcch_msg.message.choice.c1->choice.securityModeCommand->rrc_TransactionIdentifier = Transaction_id;
  dl_dcch_msg.message.choice.c1->choice.securityModeCommand->criticalExtensions.present = NR_SecurityModeCommand__criticalExtensions_PR_securityModeCommand;

  dl_dcch_msg.message.choice.c1->choice.securityModeCommand->criticalExtensions.choice.securityModeCommand =
		  CALLOC(1, sizeof(struct NR_SecurityModeCommand_IEs));
  // the two following information could be based on the mod_id
  dl_dcch_msg.message.choice.c1->choice.securityModeCommand->criticalExtensions.choice.securityModeCommand->securityConfigSMC.securityAlgorithmConfig.cipheringAlgorithm
    = (NR_CipheringAlgorithm_t)cipheringAlgorithm;
  dl_dcch_msg.message.choice.c1->choice.securityModeCommand->criticalExtensions.choice.securityModeCommand->securityConfigSMC.securityAlgorithmConfig.integrityProtAlgorithm
    = integrityProtAlgorithm;

  if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
    xer_fprint(stdout, &asn_DEF_NR_DL_DCCH_Message, (void *)&dl_dcch_msg);
  }

  enc_rval = uper_encode_to_buffer(&asn_DEF_NR_DL_DCCH_Message,
                                   NULL,
                                   (void *)&dl_dcch_msg,
                                   buffer,
                                   100);

  if(enc_rval.encoded == -1) {
    LOG_I(NR_RRC, "[gNB AssertFatal]ASN1 message encoding failed (%s, %lu)!\n",
          enc_rval.failed_type->name, enc_rval.encoded);
    return -1;
  }

  LOG_D(NR_RRC,"[gNB %d] securityModeCommand for UE %x Encoded %zd bits (%zd bytes)\n",
        ctxt_pP->module_id,
        ctxt_pP->rnti,
        enc_rval.encoded,
        (enc_rval.encoded+7)/8);

  if (enc_rval.encoded==-1) {
    LOG_E(NR_RRC,"[gNB %d] ASN1 : securityModeCommand encoding failed for UE %x\n",
          ctxt_pP->module_id,
          ctxt_pP->rnti);
    return(-1);
  }

  //  rrc_ue_process_ueCapabilityEnquiry(0,1000,&dl_dcch_msg.message.choice.c1.choice.ueCapabilityEnquiry,0);
  //  exit(-1);
  return((enc_rval.encoded+7)/8);
}

/*TODO*/
//------------------------------------------------------------------------------
uint8_t do_NR_SA_UECapabilityEnquiry( const protocol_ctxt_t *const ctxt_pP,
                                   uint8_t               *const buffer,
                                   const uint8_t                Transaction_id)
//------------------------------------------------------------------------------
{
  NR_DL_DCCH_Message_t dl_dcch_msg;
  NR_UE_CapabilityRAT_Request_t *ue_capabilityrat_request;

  asn_enc_rval_t enc_rval;
  memset(&dl_dcch_msg,0,sizeof(NR_DL_DCCH_Message_t));
  dl_dcch_msg.message.present           = NR_DL_DCCH_MessageType_PR_c1;
  dl_dcch_msg.message.choice.c1 = CALLOC(1,sizeof(struct NR_DL_DCCH_MessageType__c1));
  dl_dcch_msg.message.choice.c1->present = NR_DL_DCCH_MessageType__c1_PR_ueCapabilityEnquiry;
  dl_dcch_msg.message.choice.c1->choice.ueCapabilityEnquiry = CALLOC(1,sizeof(struct NR_UECapabilityEnquiry));
  dl_dcch_msg.message.choice.c1->choice.ueCapabilityEnquiry->rrc_TransactionIdentifier = Transaction_id;
  dl_dcch_msg.message.choice.c1->choice.ueCapabilityEnquiry->criticalExtensions.present = NR_UECapabilityEnquiry__criticalExtensions_PR_ueCapabilityEnquiry;
  dl_dcch_msg.message.choice.c1->choice.ueCapabilityEnquiry->criticalExtensions.choice.ueCapabilityEnquiry = CALLOC(1,sizeof(struct NR_UECapabilityEnquiry_IEs));
  ue_capabilityrat_request =  CALLOC(1,sizeof(NR_UE_CapabilityRAT_Request_t));
  memset(ue_capabilityrat_request,0,sizeof(NR_UE_CapabilityRAT_Request_t));
  ue_capabilityrat_request->rat_Type = NR_RAT_Type_nr;

  ASN_SEQUENCE_ADD(&dl_dcch_msg.message.choice.c1->choice.ueCapabilityEnquiry->criticalExtensions.choice.ueCapabilityEnquiry->ue_CapabilityRAT_RequestList.list,
                   ue_capabilityrat_request);


  if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
    xer_fprint(stdout, &asn_DEF_NR_DL_DCCH_Message, (void *)&dl_dcch_msg);
  }

  enc_rval = uper_encode_to_buffer(&asn_DEF_NR_DL_DCCH_Message,
                                   NULL,
                                   (void *)&dl_dcch_msg,
                                   buffer,
                                   100);

  if(enc_rval.encoded == -1) {
    LOG_I(NR_RRC, "[gNB AssertFatal]ASN1 message encoding failed (%s, %lu)!\n",
          enc_rval.failed_type->name, enc_rval.encoded);
    return -1;
  }

  LOG_D(NR_RRC,"[gNB %d] NR UECapabilityRequest for UE %x Encoded %zd bits (%zd bytes)\n",
        ctxt_pP->module_id,
        ctxt_pP->rnti,
        enc_rval.encoded,
        (enc_rval.encoded+7)/8);

  if (enc_rval.encoded==-1) {
    LOG_E(NR_RRC,"[gNB %d] ASN1 : NR UECapabilityRequest encoding failed for UE %x\n",
          ctxt_pP->module_id,
          ctxt_pP->rnti);
    return(-1);
  }

  return((enc_rval.encoded+7)/8);
}


uint8_t do_NR_RRCRelease(uint8_t                            *buffer,
                         uint8_t                             Transaction_id) {
  asn_enc_rval_t enc_rval;
  NR_DL_DCCH_Message_t dl_dcch_msg;
  NR_RRCRelease_t *rrcConnectionRelease;
  memset(&dl_dcch_msg,0,sizeof(NR_DL_DCCH_Message_t));
  dl_dcch_msg.message.present           = NR_DL_DCCH_MessageType_PR_c1;
  dl_dcch_msg.message.choice.c1=CALLOC(1,sizeof(struct NR_DL_DCCH_MessageType__c1));
  dl_dcch_msg.message.choice.c1->present = NR_DL_DCCH_MessageType__c1_PR_rrcRelease;
  dl_dcch_msg.message.choice.c1->choice.rrcRelease = CALLOC(1, sizeof(NR_RRCRelease_t));
  rrcConnectionRelease = dl_dcch_msg.message.choice.c1->choice.rrcRelease;
  // RRCConnectionRelease
  rrcConnectionRelease->rrc_TransactionIdentifier = Transaction_id;
  rrcConnectionRelease->criticalExtensions.present = NR_RRCRelease__criticalExtensions_PR_rrcRelease;
  rrcConnectionRelease->criticalExtensions.choice.rrcRelease = CALLOC(1, sizeof(NR_RRCRelease_IEs_t));
  rrcConnectionRelease->criticalExtensions.choice.rrcRelease->deprioritisationReq =
      CALLOC(1, sizeof(struct NR_RRCRelease_IEs__deprioritisationReq));
  rrcConnectionRelease->criticalExtensions.choice.rrcRelease->deprioritisationReq->deprioritisationType =
      NR_RRCRelease_IEs__deprioritisationReq__deprioritisationType_nr;
  rrcConnectionRelease->criticalExtensions.choice.rrcRelease->deprioritisationReq->deprioritisationTimer =
      NR_RRCRelease_IEs__deprioritisationReq__deprioritisationTimer_min10;

  enc_rval = uper_encode_to_buffer(&asn_DEF_NR_DL_DCCH_Message,
                                   NULL,
                                   (void *)&dl_dcch_msg,
                                   buffer,
                                   RRC_BUF_SIZE);
  if(enc_rval.encoded == -1) {
    LOG_I(NR_RRC, "[gNB AssertFatal]ASN1 message encoding failed (%s, %lu)!\n",
        enc_rval.failed_type->name, enc_rval.encoded);
    return -1;
  }
  return((enc_rval.encoded+7)/8);
}

//------------------------------------------------------------------------------
uint16_t do_RRCReconfiguration(
    const protocol_ctxt_t        *const ctxt_pP,
    uint8_t                      *buffer,
    uint8_t                       Transaction_id,
    NR_SRB_ToAddModList_t        *SRB_configList,
    NR_DRB_ToAddModList_t        *DRB_configList,
    NR_DRB_ToReleaseList_t       *DRB_releaseList,
    NR_SecurityConfig_t          *security_config,
    NR_SDAP_Config_t             *sdap_config,
    NR_MeasConfig_t              *meas_config,
    struct NR_RRCReconfiguration_v1530_IEs__dedicatedNAS_MessageList
                                 *dedicatedNAS_MessageList,
    NR_MAC_CellGroupConfig_t     *mac_CellGroupConfig)
//------------------------------------------------------------------------------
{
    NR_DL_DCCH_Message_t                             dl_dcch_msg;
    asn_enc_rval_t                                   enc_rval;
    NR_RRCReconfiguration_IEs_t                      *ie;

    memset(&dl_dcch_msg, 0, sizeof(NR_DL_DCCH_Message_t));
    dl_dcch_msg.message.present            = NR_DL_DCCH_MessageType_PR_c1;
    dl_dcch_msg.message.choice.c1          = CALLOC(1, sizeof(struct NR_DL_DCCH_MessageType__c1));
    dl_dcch_msg.message.choice.c1->present = NR_DL_DCCH_MessageType__c1_PR_rrcReconfiguration;

    dl_dcch_msg.message.choice.c1->choice.rrcReconfiguration = calloc(1, sizeof(NR_RRCReconfiguration_t));
    dl_dcch_msg.message.choice.c1->choice.rrcReconfiguration->rrc_TransactionIdentifier = Transaction_id;
    dl_dcch_msg.message.choice.c1->choice.rrcReconfiguration->criticalExtensions.present = NR_RRCReconfiguration__criticalExtensions_PR_rrcReconfiguration;

    /******************** Radio Bearer Config ********************/
    /* Configure Security */
    // security_config    =  CALLOC(1, sizeof(NR_SecurityConfig_t));
    // security_config->securityAlgorithmConfig = CALLOC(1, sizeof(*ie->radioBearerConfig->securityConfig->securityAlgorithmConfig));
    // security_config->securityAlgorithmConfig->cipheringAlgorithm     = NR_CipheringAlgorithm_nea0;
    // security_config->securityAlgorithmConfig->integrityProtAlgorithm = NULL;
    // security_config->keyToUse = CALLOC(1, sizeof(*ie->radioBearerConfig->securityConfig->keyToUse));
    // *security_config->keyToUse = NR_SecurityConfig__keyToUse_master;

    ie = calloc(1, sizeof(NR_RRCReconfiguration_IEs_t));
    ie->radioBearerConfig = calloc(1, sizeof(NR_RadioBearerConfig_t));
    ie->radioBearerConfig->srb_ToAddModList  = SRB_configList;
    ie->radioBearerConfig->drb_ToAddModList  = DRB_configList;
    ie->radioBearerConfig->securityConfig    = security_config;
    ie->radioBearerConfig->srb3_ToRelease    = NULL;
    ie->radioBearerConfig->drb_ToReleaseList = DRB_releaseList;

    /******************** Secondary Cell Group ********************/
    // rrc_gNB_carrier_data_t *carrier = &(gnb_rrc_inst->carrier);
    // fill_default_secondaryCellGroup( carrier->servingcellconfigcommon,
    //                                  ue_context_pP->ue_context.secondaryCellGroup,
    //                                  1,
    //                                  1,
    //                                  carrier->pdsch_AntennaPorts,
    //                                  carrier->initial_csi_index[gnb_rrc_inst->Nb_ue]);

    /******************** Meas Config ********************/
    // measConfig
    ie->measConfig = meas_config;
    // lateNonCriticalExtension
    ie->lateNonCriticalExtension = NULL;
    // nonCriticalExtension
    ie->nonCriticalExtension = calloc(1, sizeof(NR_RRCReconfiguration_v1530_IEs_t));
    ie->nonCriticalExtension->dedicatedNAS_MessageList = dedicatedNAS_MessageList;

    dl_dcch_msg.message.choice.c1->choice.rrcReconfiguration->criticalExtensions.choice.rrcReconfiguration = ie;

    if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
        xer_fprint(stdout, &asn_DEF_NR_DL_DCCH_Message, (void *)&dl_dcch_msg);
    }

    enc_rval = uper_encode_to_buffer(&asn_DEF_NR_DL_DCCH_Message,
                                    NULL,
                                    (void *)&dl_dcch_msg,
                                    buffer,
                                    100);

    if(enc_rval.encoded == -1) {
        LOG_I(NR_RRC, "[gNB AssertFatal]ASN1 message encoding failed (%s, %lu)!\n",
            enc_rval.failed_type->name, enc_rval.encoded);
        return -1;
    }

    LOG_D(NR_RRC,"[gNB %d] RRCReconfiguration for UE %x Encoded %zd bits (%zd bytes)\n",
            ctxt_pP->module_id,
            ctxt_pP->rnti,
            enc_rval.encoded,
            (enc_rval.encoded+7)/8);

    if (enc_rval.encoded == -1) {
        LOG_E(NR_RRC,"[gNB %d] ASN1 : RRCReconfiguration encoding failed for UE %x\n",
            ctxt_pP->module_id,
            ctxt_pP->rnti);
        return(-1);
    }

    return((enc_rval.encoded+7)/8);
}


uint8_t do_RRCSetupRequest(uint8_t Mod_id, uint8_t *buffer,uint8_t *rv) {
  asn_enc_rval_t enc_rval;
  uint8_t buf[5],buf2=0;
  NR_UL_CCCH_Message_t ul_ccch_msg;
  NR_RRCSetupRequest_t *rrcSetupRequest;
  memset((void *)&ul_ccch_msg,0,sizeof(NR_UL_CCCH_Message_t));
  ul_ccch_msg.message.present           = NR_UL_CCCH_MessageType_PR_c1;
  ul_ccch_msg.message.choice.c1          = CALLOC(1, sizeof(struct NR_UL_CCCH_MessageType__c1));
  ul_ccch_msg.message.choice.c1->present = NR_UL_CCCH_MessageType__c1_PR_rrcSetupRequest;
  ul_ccch_msg.message.choice.c1->choice.rrcSetupRequest = CALLOC(1, sizeof(NR_RRCSetupRequest_t));
  rrcSetupRequest          = ul_ccch_msg.message.choice.c1->choice.rrcSetupRequest;


  if (1) {
    rrcSetupRequest->rrcSetupRequest.ue_Identity.present = NR_InitialUE_Identity_PR_randomValue;
    rrcSetupRequest->rrcSetupRequest.ue_Identity.choice.randomValue.size = 5;
    rrcSetupRequest->rrcSetupRequest.ue_Identity.choice.randomValue.bits_unused = 1;
    rrcSetupRequest->rrcSetupRequest.ue_Identity.choice.randomValue.buf = buf;
    rrcSetupRequest->rrcSetupRequest.ue_Identity.choice.randomValue.buf[0] = rv[0];
    rrcSetupRequest->rrcSetupRequest.ue_Identity.choice.randomValue.buf[1] = rv[1];
    rrcSetupRequest->rrcSetupRequest.ue_Identity.choice.randomValue.buf[2] = rv[2];
    rrcSetupRequest->rrcSetupRequest.ue_Identity.choice.randomValue.buf[3] = rv[3];
    rrcSetupRequest->rrcSetupRequest.ue_Identity.choice.randomValue.buf[4] = rv[4]&0xfe;
  } else {
    rrcSetupRequest->rrcSetupRequest.ue_Identity.present = NR_InitialUE_Identity_PR_ng_5G_S_TMSI_Part1;
    rrcSetupRequest->rrcSetupRequest.ue_Identity.choice.ng_5G_S_TMSI_Part1.size = 1;
    rrcSetupRequest->rrcSetupRequest.ue_Identity.choice.ng_5G_S_TMSI_Part1.bits_unused = 0;
    rrcSetupRequest->rrcSetupRequest.ue_Identity.choice.ng_5G_S_TMSI_Part1.buf = buf;
    rrcSetupRequest->rrcSetupRequest.ue_Identity.choice.ng_5G_S_TMSI_Part1.buf[0] = 0x12;
  }

  rrcSetupRequest->rrcSetupRequest.establishmentCause = NR_EstablishmentCause_mo_Signalling; //EstablishmentCause_mo_Data;
  rrcSetupRequest->rrcSetupRequest.spare.buf = &buf2;
  rrcSetupRequest->rrcSetupRequest.spare.size=1;
  rrcSetupRequest->rrcSetupRequest.spare.bits_unused = 7;

  if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
    xer_fprint(stdout, &asn_DEF_NR_UL_CCCH_Message, (void *)&ul_ccch_msg);
  }

  enc_rval = uper_encode_to_buffer(&asn_DEF_NR_UL_CCCH_Message,
                                   NULL,
                                   (void *)&ul_ccch_msg,
                                   buffer,
                                   100);
  AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %lu)!\n", enc_rval.failed_type->name, enc_rval.encoded);
  LOG_D(NR_RRC,"[UE] RRCSetupRequest Encoded %zd bits (%zd bytes)\n", enc_rval.encoded, (enc_rval.encoded+7)/8);
  return((enc_rval.encoded+7)/8);
}
//------------------------------------------------------------------------------
uint8_t
do_NR_RRCReconfigurationComplete(
  const protocol_ctxt_t *const ctxt_pP,
  uint8_t *buffer,
  const uint8_t Transaction_id
)
//------------------------------------------------------------------------------
{
  asn_enc_rval_t enc_rval;
  NR_UL_DCCH_Message_t ul_dcch_msg;
  NR_RRCReconfigurationComplete_t *rrcReconfigurationComplete;
  memset((void *)&ul_dcch_msg,0,sizeof(NR_UL_DCCH_Message_t));
  ul_dcch_msg.message.present                     = NR_UL_DCCH_MessageType_PR_c1;
  ul_dcch_msg.message.choice.c1                   = CALLOC(1, sizeof(struct NR_UL_DCCH_MessageType__c1));
  ul_dcch_msg.message.choice.c1->present           = NR_UL_DCCH_MessageType__c1_PR_rrcReconfigurationComplete;
  ul_dcch_msg.message.choice.c1->choice.rrcReconfigurationComplete = CALLOC(1, sizeof(NR_RRCReconfigurationComplete_t));
  rrcReconfigurationComplete            = ul_dcch_msg.message.choice.c1->choice.rrcReconfigurationComplete;
  rrcReconfigurationComplete->rrc_TransactionIdentifier = Transaction_id;
  rrcReconfigurationComplete->criticalExtensions.choice.rrcReconfigurationComplete = CALLOC(1, sizeof(NR_RRCReconfigurationComplete_IEs_t));
  rrcReconfigurationComplete->criticalExtensions.present =
		  NR_RRCReconfigurationComplete__criticalExtensions_PR_rrcReconfigurationComplete;
  rrcReconfigurationComplete->criticalExtensions.choice.rrcReconfigurationComplete->nonCriticalExtension = NULL;
  rrcReconfigurationComplete->criticalExtensions.choice.rrcReconfigurationComplete->lateNonCriticalExtension = NULL;
  if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
    xer_fprint(stdout, &asn_DEF_NR_UL_DCCH_Message, (void *)&ul_dcch_msg);
  }

  enc_rval = uper_encode_to_buffer(&asn_DEF_NR_UL_DCCH_Message,
                                   NULL,
                                   (void *)&ul_dcch_msg,
                                   buffer,
                                   100);
  AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %lu)!\n",
               enc_rval.failed_type->name, enc_rval.encoded);
  LOG_I(NR_RRC,"rrcReconfigurationComplete Encoded %zd bits (%zd bytes)\n",enc_rval.encoded,(enc_rval.encoded+7)/8);
  return((enc_rval.encoded+7)/8);
}

uint8_t do_RRCSetupComplete(uint8_t Mod_id, uint8_t *buffer, const uint8_t Transaction_id, uint8_t sel_plmn_id, const int dedicatedInfoNASLength, const char *dedicatedInfoNAS){
  asn_enc_rval_t enc_rval;
  
  NR_UL_DCCH_Message_t  ul_dcch_msg;
  NR_RRCSetupComplete_t *RrcSetupComplete;
  memset((void *)&ul_dcch_msg,0,sizeof(NR_UL_DCCH_Message_t));

  uint8_t buf[6];

  ul_dcch_msg.message.present = NR_UL_DCCH_MessageType_PR_c1;
  ul_dcch_msg.message.choice.c1 = CALLOC(1,sizeof(struct NR_UL_DCCH_MessageType__c1));
  ul_dcch_msg.message.choice.c1->present = NR_UL_DCCH_MessageType__c1_PR_rrcSetupComplete;
  ul_dcch_msg.message.choice.c1->choice.rrcSetupComplete = CALLOC(1, sizeof(NR_RRCSetupComplete_t));
  RrcSetupComplete                       = ul_dcch_msg.message.choice.c1->choice.rrcSetupComplete;
  RrcSetupComplete->rrc_TransactionIdentifier    = Transaction_id;
  RrcSetupComplete->criticalExtensions.present   = NR_RRCSetupComplete__criticalExtensions_PR_rrcSetupComplete;
  RrcSetupComplete->criticalExtensions.choice.rrcSetupComplete = CALLOC(1, sizeof(NR_RRCSetupComplete_IEs_t));
  // RrcSetupComplete->criticalExtensions.choice.rrcSetupComplete->nonCriticalExtension = CALLOC(1,
  //   sizeof(*RrcSetupComplete->criticalExtensions.choice.rrcSetupComplete->nonCriticalExtension));
  RrcSetupComplete->criticalExtensions.choice.rrcSetupComplete->selectedPLMN_Identity = sel_plmn_id;
  RrcSetupComplete->criticalExtensions.choice.rrcSetupComplete->registeredAMF = NULL;

  RrcSetupComplete->criticalExtensions.choice.rrcSetupComplete->ng_5G_S_TMSI_Value = CALLOC(1, sizeof(struct NR_RRCSetupComplete_IEs__ng_5G_S_TMSI_Value));
  RrcSetupComplete->criticalExtensions.choice.rrcSetupComplete->ng_5G_S_TMSI_Value->present = NR_RRCSetupComplete_IEs__ng_5G_S_TMSI_Value_PR_ng_5G_S_TMSI;
  RrcSetupComplete->criticalExtensions.choice.rrcSetupComplete->ng_5G_S_TMSI_Value->choice.ng_5G_S_TMSI.size = 6;
  RrcSetupComplete->criticalExtensions.choice.rrcSetupComplete->ng_5G_S_TMSI_Value->choice.ng_5G_S_TMSI.buf = buf;
  RrcSetupComplete->criticalExtensions.choice.rrcSetupComplete->ng_5G_S_TMSI_Value->choice.ng_5G_S_TMSI.buf[0] = 0x12;
  RrcSetupComplete->criticalExtensions.choice.rrcSetupComplete->ng_5G_S_TMSI_Value->choice.ng_5G_S_TMSI.buf[1] = 0x34;
  RrcSetupComplete->criticalExtensions.choice.rrcSetupComplete->ng_5G_S_TMSI_Value->choice.ng_5G_S_TMSI.buf[2] = 0x56;
  RrcSetupComplete->criticalExtensions.choice.rrcSetupComplete->ng_5G_S_TMSI_Value->choice.ng_5G_S_TMSI.buf[3] = 0x78;
  RrcSetupComplete->criticalExtensions.choice.rrcSetupComplete->ng_5G_S_TMSI_Value->choice.ng_5G_S_TMSI.buf[4] = 0x9A;
  RrcSetupComplete->criticalExtensions.choice.rrcSetupComplete->ng_5G_S_TMSI_Value->choice.ng_5G_S_TMSI.buf[5] = 0xBC;

 memset(&RrcSetupComplete->criticalExtensions.choice.rrcSetupComplete->dedicatedNAS_Message,0,sizeof(OCTET_STRING_t));
 OCTET_STRING_fromBuf(&RrcSetupComplete->criticalExtensions.choice.rrcSetupComplete->dedicatedNAS_Message,dedicatedInfoNAS,dedicatedInfoNASLength);
if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
  xer_fprint(stdout, &asn_DEF_NR_UL_DCCH_Message, (void *)&ul_dcch_msg);
}

enc_rval = uper_encode_to_buffer(&asn_DEF_NR_UL_DCCH_Message,
                                 NULL,
                                 (void *)&ul_dcch_msg,
                                 buffer,
                                 100);
AssertFatal(enc_rval.encoded > 0,"ASN1 message encoding failed (%s, %lu)!\n",
    enc_rval.failed_type->name,enc_rval.encoded);
LOG_D(NR_RRC,"RRCSetupComplete Encoded %zd bits (%zd bytes)\n",enc_rval.encoded,(enc_rval.encoded+7)/8);

return((enc_rval.encoded+7)/8);
}

//------------------------------------------------------------------------------
uint8_t 
do_NR_DLInformationTransfer(
    uint8_t Mod_id,
    uint8_t **buffer,
    uint8_t transaction_id,
    uint32_t pdu_length,
    uint8_t *pdu_buffer
)
//------------------------------------------------------------------------------
{
    ssize_t encoded;
    NR_DL_DCCH_Message_t   dl_dcch_msg;
    memset(&dl_dcch_msg, 0, sizeof(NR_DL_DCCH_Message_t));
    dl_dcch_msg.message.present            = NR_DL_DCCH_MessageType_PR_c1;
    dl_dcch_msg.message.choice.c1          = CALLOC(1, sizeof(struct NR_DL_DCCH_MessageType__c1));
    dl_dcch_msg.message.choice.c1->present = NR_DL_DCCH_MessageType__c1_PR_dlInformationTransfer;

    dl_dcch_msg.message.choice.c1->choice.dlInformationTransfer = CALLOC(1, sizeof(NR_DLInformationTransfer_t));
    dl_dcch_msg.message.choice.c1->choice.dlInformationTransfer->rrc_TransactionIdentifier = transaction_id;
    dl_dcch_msg.message.choice.c1->choice.dlInformationTransfer->criticalExtensions.present =
        NR_DLInformationTransfer__criticalExtensions_PR_dlInformationTransfer;

    dl_dcch_msg.message.choice.c1->choice.dlInformationTransfer->
        criticalExtensions.choice.dlInformationTransfer = CALLOC(1, sizeof(NR_DLInformationTransfer_IEs_t));
    dl_dcch_msg.message.choice.c1->choice.dlInformationTransfer->
        criticalExtensions.choice.dlInformationTransfer->dedicatedNAS_Message = CALLOC(1, sizeof(NR_DedicatedNAS_Message_t));
    dl_dcch_msg.message.choice.c1->choice.dlInformationTransfer->
        criticalExtensions.choice.dlInformationTransfer->dedicatedNAS_Message->buf = pdu_buffer;
    dl_dcch_msg.message.choice.c1->choice.dlInformationTransfer->
        criticalExtensions.choice.dlInformationTransfer->dedicatedNAS_Message->size = pdu_length;

    encoded = uper_encode_to_new_buffer (&asn_DEF_NR_DL_DCCH_Message, NULL, (void *) &dl_dcch_msg, (void **)buffer);
    AssertFatal(encoded > 0,"ASN1 message encoding failed (%s, %ld)!\n",
                "DLInformationTransfer", encoded);
    LOG_D(NR_RRC,"DLInformationTransfer Encoded %zd bytes\n", encoded);
    return encoded;
}

uint8_t do_NR_ULInformationTransfer(uint8_t **buffer, uint32_t pdu_length, uint8_t *pdu_buffer) {
    ssize_t encoded;
    NR_UL_DCCH_Message_t ul_dcch_msg;
    memset(&ul_dcch_msg, 0, sizeof(NR_UL_DCCH_Message_t));
    ul_dcch_msg.message.present           = NR_UL_DCCH_MessageType_PR_c1;
    ul_dcch_msg.message.choice.c1          = CALLOC(1,sizeof(struct NR_UL_DCCH_MessageType__c1));
    ul_dcch_msg.message.choice.c1->present = NR_UL_DCCH_MessageType__c1_PR_ulInformationTransfer;
    ul_dcch_msg.message.choice.c1->choice.ulInformationTransfer = CALLOC(1,sizeof(struct NR_ULInformationTransfer));
    ul_dcch_msg.message.choice.c1->choice.ulInformationTransfer->criticalExtensions.present = NR_ULInformationTransfer__criticalExtensions_PR_ulInformationTransfer;
    ul_dcch_msg.message.choice.c1->choice.ulInformationTransfer->criticalExtensions.choice.ulInformationTransfer = CALLOC(1,sizeof(struct NR_ULInformationTransfer_IEs));
    struct NR_ULInformationTransfer_IEs *ulInformationTransfer = ul_dcch_msg.message.choice.c1->choice.ulInformationTransfer->criticalExtensions.choice.ulInformationTransfer;
    ulInformationTransfer->dedicatedNAS_Message = CALLOC(1,sizeof(NR_DedicatedNAS_Message_t));
    ulInformationTransfer->dedicatedNAS_Message->buf = pdu_buffer;
    ulInformationTransfer->dedicatedNAS_Message->size = pdu_length;
    ulInformationTransfer->lateNonCriticalExtension = NULL;
    encoded = uper_encode_to_new_buffer (&asn_DEF_NR_UL_DCCH_Message, NULL, (void *) &ul_dcch_msg, (void **) buffer);
    AssertFatal(encoded > 0,"ASN1 message encoding failed (%s, %ld)!\n",
                "ULInformationTransfer",encoded);
    LOG_D(NR_RRC,"ULInformationTransfer Encoded %zd bytes\n",encoded);

    return encoded;
}

uint8_t do_RRCReestablishmentRequest(uint8_t Mod_id, uint8_t *buffer, uint16_t c_rnti) {
  asn_enc_rval_t enc_rval;
  NR_UL_CCCH_Message_t ul_ccch_msg;
  NR_RRCReestablishmentRequest_t *rrcReestablishmentRequest;
  uint8_t buf[2];

  memset((void *)&ul_ccch_msg,0,sizeof(NR_UL_CCCH_Message_t));
  ul_ccch_msg.message.present            = NR_UL_CCCH_MessageType_PR_c1;
  ul_ccch_msg.message.choice.c1          = CALLOC(1, sizeof(struct NR_UL_CCCH_MessageType__c1));
  ul_ccch_msg.message.choice.c1->present = NR_UL_CCCH_MessageType__c1_PR_rrcReestablishmentRequest;
  ul_ccch_msg.message.choice.c1->choice.rrcReestablishmentRequest = CALLOC(1, sizeof(NR_RRCReestablishmentRequest_t));

  rrcReestablishmentRequest = ul_ccch_msg.message.choice.c1->choice.rrcReestablishmentRequest;
  // test
  rrcReestablishmentRequest->rrcReestablishmentRequest.reestablishmentCause = NR_ReestablishmentCause_reconfigurationFailure;
  rrcReestablishmentRequest->rrcReestablishmentRequest.ue_Identity.c_RNTI = c_rnti;
  rrcReestablishmentRequest->rrcReestablishmentRequest.ue_Identity.physCellId = 0;
  rrcReestablishmentRequest->rrcReestablishmentRequest.ue_Identity.shortMAC_I.buf = buf;
  rrcReestablishmentRequest->rrcReestablishmentRequest.ue_Identity.shortMAC_I.buf[0] = 0x08;
  rrcReestablishmentRequest->rrcReestablishmentRequest.ue_Identity.shortMAC_I.buf[1] = 0x32;
  rrcReestablishmentRequest->rrcReestablishmentRequest.ue_Identity.shortMAC_I.size = 2;


  if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
    xer_fprint(stdout, &asn_DEF_NR_UL_CCCH_Message, (void *)&ul_ccch_msg);
  }

  enc_rval = uper_encode_to_buffer(&asn_DEF_NR_UL_CCCH_Message,
                                   NULL,
                                   (void *)&ul_ccch_msg,
                                   buffer,
                                   100);
  AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %lu)!\n", enc_rval.failed_type->name, enc_rval.encoded);
  LOG_D(NR_RRC,"[UE] RRCReestablishmentRequest Encoded %zd bits (%zd bytes)\n", enc_rval.encoded, (enc_rval.encoded+7)/8);
  return((enc_rval.encoded+7)/8);
}

//------------------------------------------------------------------------------
uint8_t
do_RRCReestablishment(
const protocol_ctxt_t     *const ctxt_pP,
rrc_gNB_ue_context_t      *const ue_context_pP,
int                              CC_id,
uint8_t                   *const buffer,
//const uint8_t                    transmission_mode,
const uint8_t                    Transaction_id,
NR_SRB_ToAddModList_t               **SRB_configList
) {
    asn_enc_rval_t enc_rval;
    //long *logicalchannelgroup = NULL;
    struct NR_SRB_ToAddMod *SRB1_config = NULL;
    struct NR_SRB_ToAddMod *SRB2_config = NULL;
    //gNB_RRC_INST *nrrrc               = RC.nrrrc[ctxt_pP->module_id];
    NR_DL_DCCH_Message_t dl_dcch_msg;
    NR_RRCReestablishment_t *rrcReestablishment = NULL;
    int i = 0;
    ue_context_pP->ue_context.reestablishment_xid = Transaction_id;
    NR_SRB_ToAddModList_t **SRB_configList2 = NULL;
    SRB_configList2 = &ue_context_pP->ue_context.SRB_configList2[Transaction_id];

    if (*SRB_configList2) {
      free(*SRB_configList2);
    }

    *SRB_configList2 = CALLOC(1, sizeof(NR_SRB_ToAddModList_t));
    memset((void *)&dl_dcch_msg, 0, sizeof(NR_DL_DCCH_Message_t));
    dl_dcch_msg.message.present           = NR_DL_DCCH_MessageType_PR_c1;
    dl_dcch_msg.message.choice.c1 = calloc(1,sizeof(struct NR_DL_DCCH_MessageType__c1));
    dl_dcch_msg.message.choice.c1->present = NR_DL_DCCH_MessageType__c1_PR_rrcReestablishment;
    dl_dcch_msg.message.choice.c1->choice.rrcReestablishment = CALLOC(1,sizeof(NR_RRCReestablishment_t));
    rrcReestablishment = dl_dcch_msg.message.choice.c1->choice.rrcReestablishment;

    // get old configuration of SRB2
    if (*SRB_configList != NULL) {
      for (i = 0; (i < (*SRB_configList)->list.count) && (i < 3); i++) {
        LOG_D(NR_RRC, "(*SRB_configList)->list.array[%d]->srb_Identity=%ld\n",
              i, (*SRB_configList)->list.array[i]->srb_Identity);
    
        if ((*SRB_configList)->list.array[i]->srb_Identity == 2 ) {
          SRB2_config = (*SRB_configList)->list.array[i];
        } else if ((*SRB_configList)->list.array[i]->srb_Identity == 1 ) {
          SRB1_config = (*SRB_configList)->list.array[i];
        }
      }
    }

    if (SRB1_config == NULL) {
      // default SRB1 configuration
      LOG_W(NR_RRC,"SRB1 configuration does not exist in SRB configuration list, use default\n");
      /// SRB1
      SRB1_config = CALLOC(1, sizeof(*SRB1_config));
      SRB1_config->srb_Identity = 1;
    }

    if (SRB2_config == NULL) {
      LOG_W(NR_RRC,"SRB2 configuration does not exist in SRB configuration list\n");
    } else {
      ASN_SEQUENCE_ADD(&(*SRB_configList2)->list, SRB2_config);
    }

    if (*SRB_configList) {
      free(*SRB_configList);
    }

    *SRB_configList = CALLOC(1, sizeof(LTE_SRB_ToAddModList_t));
    ASN_SEQUENCE_ADD(&(*SRB_configList)->list,SRB1_config);

    rrcReestablishment->rrc_TransactionIdentifier = Transaction_id;
    rrcReestablishment->criticalExtensions.present = NR_RRCReestablishment__criticalExtensions_PR_rrcReestablishment;
    rrcReestablishment->criticalExtensions.choice.rrcReestablishment = CALLOC(1,sizeof(NR_RRCReestablishment_IEs_t));

    uint8_t KgNB_star[32] = { 0 };
    /** TODO
    uint16_t pci = nrrrc->carrier[CC_id].physCellId;
    uint32_t earfcn_dl = (uint32_t)freq_to_arfcn10(RC.mac[ctxt_pP->module_id]->common_channels[CC_id].eutra_band,
                         nrrrc->carrier[CC_id].dl_CarrierFreq);
    bool     is_rel8_only = true;
    
    if (earfcn_dl > 65535) {
      is_rel8_only = false;
    }
    LOG_D(NR_RRC, "pci=%d, eutra_band=%d, downlink_frequency=%d, earfcn_dl=%u, is_rel8_only=%s\n",
          pci,
          RC.mac[ctxt_pP->module_id]->common_channels[CC_id].eutra_band,
          nrrrc->carrier[CC_id].dl_CarrierFreq,
          earfcn_dl,
          is_rel8_only == true ? "true": "false");
    */
    
    if (ue_context_pP->ue_context.nh_ncc >= 0) {
      //TODO derive_keNB_star(ue_context_pP->ue_context.nh, pci, earfcn_dl, is_rel8_only, KgNB_star);
      rrcReestablishment->criticalExtensions.choice.rrcReestablishment->nextHopChainingCount = ue_context_pP->ue_context.nh_ncc;
    } else { // first HO
      //TODO derive_keNB_star (ue_context_pP->ue_context.kgnb, pci, earfcn_dl, is_rel8_only, KgNB_star);
      // LG: really 1
      rrcReestablishment->criticalExtensions.choice.rrcReestablishment->nextHopChainingCount = 0;
    }
    // copy KgNB_star to ue_context_pP->ue_context.kgnb
    memcpy (ue_context_pP->ue_context.kgnb, KgNB_star, 32);
    ue_context_pP->ue_context.kgnb_ncc = 0;
    rrcReestablishment->criticalExtensions.choice.rrcReestablishment->lateNonCriticalExtension = NULL;
    rrcReestablishment->criticalExtensions.choice.rrcReestablishment->nonCriticalExtension = NULL;

    if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
      xer_fprint(stdout, &asn_DEF_NR_DL_DCCH_Message, (void *)&dl_dcch_msg);
    }

    enc_rval = uper_encode_to_buffer(&asn_DEF_NR_DL_DCCH_Message,
                                     NULL,
                                     (void *)&dl_dcch_msg,
                                     buffer,
                                     100);

    if(enc_rval.encoded == -1) {
      LOG_E(NR_RRC, "[gNB AssertFatal]ASN1 message encoding failed (%s, %lu)!\n",
            enc_rval.failed_type->name, enc_rval.encoded);
      return -1;
    }
    
    LOG_D(NR_RRC,"RRCReestablishment Encoded %u bits (%u bytes)\n",
          (uint32_t)enc_rval.encoded, (uint32_t)(enc_rval.encoded+7)/8);
    return((enc_rval.encoded+7)/8);

}

uint8_t 
do_RRCReestablishmentComplete(uint8_t *buffer, int64_t rrc_TransactionIdentifier) {
  asn_enc_rval_t enc_rval;
  NR_UL_DCCH_Message_t ul_dcch_msg;
  NR_RRCReestablishmentComplete_t *rrcReestablishmentComplete;

  memset((void *)&ul_dcch_msg,0,sizeof(NR_UL_DCCH_Message_t));
  ul_dcch_msg.message.present            = NR_UL_DCCH_MessageType_PR_c1;
  ul_dcch_msg.message.choice.c1          = CALLOC(1, sizeof(struct NR_UL_DCCH_MessageType__c1));
  ul_dcch_msg.message.choice.c1->present = NR_UL_DCCH_MessageType__c1_PR_rrcReestablishmentComplete;
  ul_dcch_msg.message.choice.c1->choice.rrcReestablishmentComplete = CALLOC(1, sizeof(NR_RRCReestablishmentComplete_t));

  rrcReestablishmentComplete = ul_dcch_msg.message.choice.c1->choice.rrcReestablishmentComplete;
  rrcReestablishmentComplete->rrc_TransactionIdentifier = rrc_TransactionIdentifier;
  rrcReestablishmentComplete->criticalExtensions.present = NR_RRCReestablishmentComplete__criticalExtensions_PR_rrcReestablishmentComplete;
  rrcReestablishmentComplete->criticalExtensions.choice.rrcReestablishmentComplete = CALLOC(1, sizeof(NR_RRCReestablishmentComplete_IEs_t));
  rrcReestablishmentComplete->criticalExtensions.choice.rrcReestablishmentComplete->lateNonCriticalExtension = NULL;
  rrcReestablishmentComplete->criticalExtensions.choice.rrcReestablishmentComplete->nonCriticalExtension = NULL;

  if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
    xer_fprint(stdout, &asn_DEF_NR_UL_CCCH_Message, (void *)&ul_dcch_msg);
  }

  enc_rval = uper_encode_to_buffer(&asn_DEF_NR_UL_DCCH_Message,
                                   NULL,
                                   (void *)&ul_dcch_msg,
                                   buffer,
                                   100);
  AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %lu)!\n", enc_rval.failed_type->name, enc_rval.encoded);
  LOG_D(NR_RRC,"[UE] RRCReestablishmentComplete Encoded %zd bits (%zd bytes)\n", enc_rval.encoded, (enc_rval.encoded+7)/8);
  return((enc_rval.encoded+7)/8);
}

