/*! \file nfapi/open-nFAPI/nfapi/public_inc/nfapi_nr_interface_scfv2.h
 * \brief Create FAPI(P5) data structure and revise FAPI(P7) data structure
 * \author NCTU OpinConnect Terng-Yin Hsu, Sendren Xu, WEI-YING LIN, Hong-Ming Huang
 * \email  a22490010@gmail.com
 * \date   31-10-2021
 * \version 2.0
 * \note
 * \warning
 */


#include "stddef.h"
#include "nfapi_interface.h"
#include "nfapi_nr_interface.h"
//  2021.05
// SCF222_5G-FAPI_PHY_SPI_Specificayion.pdf
// Section 3.2 Messages Types
typedef enum {
  NFAPI_NR_PHY_MSG_TYPE_PARAM_REQUEST=  0x00,
  NFAPI_NR_PHY_MSG_TYPE_PARAM_RESPONSE= 0x01,
  NFAPI_NR_PHY_MSG_TYPE_CONFIG_REQUEST= 0x02,
  NFAPI_NR_PHY_MSG_TYPE_CONFIG_RESPONSE=0X03,
  NFAPI_NR_PHY_MSG_TYPE_START_REQUEST=  0X04,
  NFAPI_NR_PHY_MSG_TYPE_STOP_REQUEST=   0X05,
  NFAPI_NR_PHY_MSG_TYPE_STOP_INDICATION=0X06,
  NFAPI_NR_PHY_MSG_TYPE_ERROR_INDICATION=0X07,
  NFAPI_NR_PHY_MSG_TYPE_START_RESPONSE=0x108,
  //RESERVED 0X08 ~ 0X7F
  NFAPI_NR_PHY_MSG_TYPE_DL_TTI_REQUEST= 0X80,
  NFAPI_NR_PHY_MSG_TYPE_UL_TTI_REQUEST= 0X81,
  NFAPI_NR_PHY_MSG_TYPE_SLOT_INDICATION=0X82,
  NFAPI_NR_PHY_MSG_TYPE_UL_DCI_REQUEST= 0X83,
  NFAPI_NR_PHY_MSG_TYPE_TX_DATA_REQUEST=0X84, 
  NFAPI_NR_PHY_MSG_TYPE_RX_DATA_INDICATION=0X85,
  NFAPI_NR_PHY_MSG_TYPE_CRC_INDICATION= 0X86,
  NFAPI_NR_PHY_MSG_TYPE_UCI_INDICATION= 0X87,
  NFAPI_NR_PHY_MSG_TYPE_SRS_INDICATION= 0X88,
  NFAPI_NR_PHY_MSG_TYPE_RACH_INDICATION= 0X89,
  //RESERVED 0X8a ~ 0xff
  NFAPI_NR_PHY_MSG_TYPE_UL_NODE_SYNC = 0x180,
	NFAPI_NR_PHY_MSG_TYPE_DL_NODE_SYNC = 0x181,
	NFAPI_NR_PHY_MSG_TYPE_TIMING_INFO = 0x182
} nfapi_nr_phy_msg_type_e;

/* Section 3.3 Configuration Messages */
/* PARAM.response */
#define  NFAPI_NR_PARAM_TLV_RELEASE_CAPABILITY_TAG 0x0001
#define  NFAPI_NR_PARAM_TLV_PHY_STATE_TAG         0x0002
#define  NFAPI_NR_PARAM_TLV_SKIP_BLANK_DL_CONFIG_TAG 0x0003
#define  NFAPI_NR_PARAM_TLV_SKIP_BLANK_UL_CONFIG_TAG 0x0004
#define  NFAPI_NR_PARAM_TLV_NUM_CONFIG_TLVS_TO_REPORT_TAG 0x0005
#define  NFAPI_NR_PARAM_TLV_CYCLIC_PREFIX_TAG 0x0006
#define  NFAPI_NR_PARAM_TLV_SUPPORTED_SUBCARRIER_SPACINGS_DL_TAG 0x0007
#define  NFAPI_NR_PARAM_TLV_SUPPORTED_BANDWIDTH_DL_TAG 0x0008
#define  NFAPI_NR_PARAM_TLV_SUPPORTED_SUBCARRIER_SPACINGS_UL_TAG 0x0009
#define  NFAPI_NR_PARAM_TLV_SUPPORTED_BANDWIDTH_UL_TAG 0x000A
#define  NFAPI_NR_PARAM_TLV_CCE_MAPPING_TYPE_TAG 0x000B
#define  NFAPI_NR_PARAM_TLV_CORESET_OUTSIDE_FIRST_3_OFDM_SYMS_OF_SLOT_TAG 0x000C
#define  NFAPI_NR_PARAM_TLV_PRECODER_GRANULARITY_CORESET_TAG 0x000D
#define  NFAPI_NR_PARAM_TLV_PDCCH_MU_MIMO_TAG 0x000E
#define  NFAPI_NR_PARAM_TLV_PDCCH_PRECODER_CYCLING_TAG 0x000F
#define  NFAPI_NR_PARAM_TLV_MAX_PDCCHS_PER_SLOT_TAG 0x0010
#define  NFAPI_NR_PARAM_TLV_PUCCH_FORMATS_TAG 0x0011
#define  NFAPI_NR_PARAM_TLV_MAX_PUCCHS_PER_SLOT_TAG 0x0012
#define  NFAPI_NR_PARAM_TLV_PDSCH_MAPPING_TYPE_TAG 0x0013
#define  NFAPI_NR_PARAM_TLV_PDSCH_ALLOCATION_TYPES_TAG 0x0014
#define  NFAPI_NR_PARAM_TLV_PDSCH_VRB_TO_PRB_MAPPING_TAG 0x0015
#define  NFAPI_NR_PARAM_TLV_PDSCH_CBG_TAG 0x0016
#define  NFAPI_NR_PARAM_TLV_PDSCH_DMRS_CONFIG_TYPES_TAG 0x0017
#define  NFAPI_NR_PARAM_TLV_PDSCH_DMRS_MAX_LENGTH_TAG 0x0018
#define  NFAPI_NR_PARAM_TLV_PDSCH_DMRS_ADDITIONAL_POS_TAG 0x0019
#define  NFAPI_NR_PARAM_TLV_MAX_PDSCHS_TBS_PER_SLOT_TAG 0x001A
#define  NFAPI_NR_PARAM_TLV_MAX_NUMBER_MIMO_LAYERS_PDSCH_TAG 0x001B
#define  NFAPI_NR_PARAM_TLV_SUPPORTED_MAX_MODULATION_ORDER_DL_TAG 0x001C
#define  NFAPI_NR_PARAM_TLV_MAX_MU_MIMO_USERS_DL_TAG 0x001D
#define  NFAPI_NR_PARAM_TLV_PDSCH_DATA_IN_DMRS_SYMBOLS_TAG 0x001E
#define  NFAPI_NR_PARAM_TLV_PREMPTION_SUPPORT_TAG 0x001F
#define  NFAPI_NR_PARAM_TLV_PDSCH_NON_SLOT_SUPPORT_TAG 0x0020
#define  NFAPI_NR_PARAM_TLV_UCI_MUX_ULSCH_IN_PUSCH_TAG 0x0021
#define  NFAPI_NR_PARAM_TLV_UCI_ONLY_PUSCH_TAG 0x0022
#define  NFAPI_NR_PARAM_TLV_PUSCH_FREQUENCY_HOPPING_TAG 0x0023
#define  NFAPI_NR_PARAM_TLV_PUSCH_DMRS_CONFIG_TYPES_TAG 0x0024
#define  NFAPI_NR_PARAM_TLV_PUSCH_DMRS_MAX_LEN_TAG 0x0025
#define  NFAPI_NR_PARAM_TLV_PUSCH_DMRS_ADDITIONAL_POS_TAG 0x0026
#define  NFAPI_NR_PARAM_TLV_PUSCH_CBG_TAG 0x0027
#define  NFAPI_NR_PARAM_TLV_PUSCH_MAPPING_TYPE_TAG 0x0028
#define  NFAPI_NR_PARAM_TLV_PUSCH_ALLOCATION_TYPES_TAG 0x0029
#define  NFAPI_NR_PARAM_TLV_PUSCH_VRB_TO_PRB_MAPPING_TAG 0x002A
#define  NFAPI_NR_PARAM_TLV_PUSCH_MAX_PTRS_PORTS_TAG 0x002B
#define  NFAPI_NR_PARAM_TLV_MAX_PDUSCHS_TBS_PER_SLOT_TAG 0x002C
#define  NFAPI_NR_PARAM_TLV_MAX_NUMBER_MIMO_LAYERS_NON_CB_PUSCH_TAG 0x002D
#define  NFAPI_NR_PARAM_TLV_SUPPORTED_MODULATION_ORDER_UL_TAG 0x002E
#define  NFAPI_NR_PARAM_TLV_MAX_MU_MIMO_USERS_UL_TAG 0x002F
#define  NFAPI_NR_PARAM_TLV_DFTS_OFDM_SUPPORT_TAG 0x0030
#define  NFAPI_NR_PARAM_TLV_PUSCH_AGGREGATION_FACTOR_TAG 0x0031
#define  NFAPI_NR_PARAM_TLV_PRACH_LONG_FORMATS_TAG 0x0032
#define  NFAPI_NR_PARAM_TLV_PRACH_SHORT_FORMATS_TAG 0x0033
#define  NFAPI_NR_PARAM_TLV_PRACH_RESTRICTED_SETS_TAG 0x0034
#define  NFAPI_NR_PARAM_TLV_MAX_PRACH_FD_OCCASIONS_IN_A_SLOT_TAG 0x0035
#define  NFAPI_NR_PARAM_TLV_RSSI_MEASUREMENT_SUPPORT_TAG 0x0036
// FAPI V3
//#define NFAPI_NR_PARAM_TLV__TAG
#define NFAPI_NR_PARAM_TLV_POWER_PROFILES_SUPPORTED_TAG 0x0038
#define NFAPI_NR_PARAM_TLV_MAX_NUM_PDUS_INDL_TTI_TAG 0x0039
#define NFAPI_NR_PARAM_TLV_MAX_NUM_PDUS_INUL_TTI_TAG 0x003A
#define NFAPI_NR_PARAM_TLV_MAX_NUM_PDUS_INUL_DCI_TAG 0x003B
#define NFAPI_NR_PARAM_TLV_SS_PBCH_MULTIPLE_CARRIERS_IN_AB_AND_TAG 0x003C
#define NFAPI_NR_PARAM_TLV_MULTIPLE_CELLS_SS_PBCH_IN_A_CARRIER_TAG 0x003D
#define NFAPI_NR_PARAM_TLV_PUCCH_GROUP_AND_SEQUENCE_HOPPING_TAG 0x003E
#define NFAPI_NR_PARAM_TLV_MAX_NUM_UL_BWP_IDS_TAG 0x003F
#define NFAPI_NR_PARAM_TLV_PUCCH_AGGREGATION_TAG 0x0040
#define NFAPI_NR_PARAM_TLV_SSB_RATE_MATCH_TAG 0x0041
#define NFAPI_NR_PARAM_TLV_SUPPORTED_RATE_MATCH_PATTERN_TYPE_TAG 0x0042
#define NFAPI_NR_PARAM_TLV_PDCCH_RATE_MATCH_TAG 0x0043
#define NFAPI_NR_PARAM_TLV_NUM_OF_RATE_MATCH_PATTERN_LTE_CRS_PER_SLOT_TAG 0x0044
#define NFAPI_NR_PARAM_TLV_NUM_OF_RATE_MATCH_PATTERN_LTE_CRS_IN_PHY_TAG 0x0045
#define NFAPI_NR_PARAM_TLV_CSI_RS_RATE_MATCH_TAG 0x0046
#define NFAPI_NR_PARAM_TLV_PDSCH_TRANS_PYPE_SUPPORT_TAG 0x0047
#define NFAPI_NR_PARAM_TLV_PDSCH_MAC_PDU_BIT_ALIGNMENT_TAG 0x0048
#define NFAPI_NR_PARAM_TLV_MAX_NUMBER_MIMO_LAYERS_CB_PUSCH_TAG 0x0049
#define NFAPI_NR_PARAM_TLV_PUSCH_LBRM_SUPPORT_TAG 0x004A
#define NFAPI_NR_PARAM_TLV_PUSCH_TRANS_TYPE_SUPPORT_TAG 0x004B
#define NFAPI_NR_PARAM_TLV_PUSCH_MAC_PDU_BIT_ALIGNMENT_TAG 0x004C
#define NFAPI_NR_PARAM_TLV_MAX_NUM_PRACH_CONFIGURATIONS_TAG 0x004D
#define NFAPI_NR_PARAM_TLV_PRACH_MULTIPLE_CARRIERS_IN_AB_AND_TAG 0x004E
#define NFAPI_NR_PARAM_TLV_MAX_NUM_UCI_MAPS_TAG 0x004F
#define NFAPI_NR_PARAM_TLV_NUM_CAPABILITIES_TAG 0x0050
// may fix!
//table 3-22
#define NFAPI_NR_PARAM_TLV_PHY_PROFILE_SUPPORT_PARAMETERS_TAG 0x0051
#define NFAPI_NR_PARAM_TLV_TIME_MANAGEMENT_TAG 0x0052
#define NFAPI_NR_PARAM_TLV_PROTOCOL_VERSION_TAG 0x0053
#define NFAPI_NR_PARAM_TLV_MORE_THAN_ONE_INDICATION_PER_SLOT_TAG 0x0055
#define NFAPI_NR_PARAM_TLV_MORE_THAN_ONE_REQUEST_PER_SLOT_TAG 0x0056
//table 3-23
#define NFAPI_NR_PARAM_TLV_PAIRINGS_OF_PHY_AND_DFE_PROFILES_TAG 0x0054
//table 3-25 Rel-16 mTRP parameters
#define NFAPI_NR_PARAM_TLV_MTRP_PARAMETERS_TAG 0x0057


/* 3.3.1.1 PARAM.request */
typedef struct{
    nfapi_nr_p5_message_header_t header;
    uint8_t protocolVersion;
}nfapi_nr_param_request_scf_t;


/* 3.3.1.2 PARAM.response */
//Table 3-12 Cell and PHY parameters (incomplete)
typedef struct{
  
    nfapi_uint16_tlv_t release_capability; //TAG 0x0001
    nfapi_uint16_tlv_t phy_state;
    nfapi_uint8_tlv_t  skip_blank_dl_config;
    nfapi_uint8_tlv_t  skip_blank_ul_config;
    nfapi_uint16_tlv_t num_config_tlvs_to_report;
    nfapi_uint8_tlv_t* config_tlvs_to_report_list;
    //fix me!!
    nfapi_uint8_tlv_t power;

    nfapi_uint16_tlv_t maxNumPDUsInDL_TTI;
    nfapi_uint16_tlv_t maxNumPDUsInUL_TTI;
    nfapi_uint16_tlv_t maxNumPDUsInUL_DCI;
} nfapi_nr_cell_param_t;

//Table 3-13 Carrier parameters
typedef struct {
  nfapi_uint8_tlv_t  cyclic_prefix;//TAG 0x0006
  nfapi_uint8_tlv_t  supported_subcarrier_spacings_dl;
  nfapi_uint16_tlv_t supported_bandwidth_dl;
  nfapi_uint8_tlv_t  supported_subcarrier_spacings_ul;
  nfapi_uint16_tlv_t supported_bandwidth_ul;
  nfapi_uint8_tlv_t ssPbchMultipleCarriersInABand;
  nfapi_uint8_tlv_t multipleCellsSsPbchInACarrier;
} nfapi_nr_carrier_param_t;

//Table 3-14 PDCCH parameters
typedef struct {
  nfapi_uint8_tlv_t  cce_mapping_type;
  nfapi_uint8_tlv_t  coreset_outside_first_3_of_ofdm_syms_of_slot;
  nfapi_uint8_tlv_t  precoder_granularity_coreset;
  nfapi_uint8_tlv_t  pdcch_mu_mimo;
  nfapi_uint8_tlv_t  pdcch_precoder_cycling;
  nfapi_uint8_tlv_t  max_pdcch_per_slot;
} nfapi_nr_pdcch_param_t;

//Table 3-15 PUCCH parameters
typedef struct {
    nfapi_uint8_tlv_t pucch_formats;//0011
    nfapi_uint8_tlv_t max_pucchs_per_slot;
    nfapi_uint8_tlv_t pucchGroupAndSequenceHopping;//003E
    nfapi_uint8_tlv_t maxNumUlBwpIds;//003F
} nfapi_nr_pucch_param_t;

//Table 3-16 PDSCH parameters
typedef struct {
    nfapi_uint8_tlv_t pdsch_mapping_type;//0013
    nfapi_uint8_tlv_t pdsch_allocation_types;
    nfapi_uint8_tlv_t pdsch_vrb_to_prb_mapping;
    nfapi_uint8_tlv_t pdsch_cbg;
    nfapi_uint8_tlv_t pdsch_dmrs_config_types;
    nfapi_uint8_tlv_t pdsch_dmrs_max_length;
    nfapi_uint8_tlv_t pdsch_dmrs_additional_pos;
    nfapi_uint8_tlv_t max_pdsch_tbs_per_slot;
    nfapi_uint8_tlv_t max_number_mimo_layers_pdsch;
    nfapi_uint8_tlv_t supported_max_modulation_order_dl;
    nfapi_uint8_tlv_t max_mu_mimo_users_dl;
    nfapi_uint8_tlv_t pdsch_data_in_dmrs_symbols;
    nfapi_uint8_tlv_t premption_support;
    nfapi_uint8_tlv_t pdsch_non_slot_support;
    nfapi_uint8_tlv_t ssbRateMatch;//0041
    nfapi_uint16_tlv_t supportedRateMatchPatternType;//0042
    nfapi_uint8_tlv_t pdcchRateMatch;//0043
    nfapi_uint8_tlv_t numOfRateMatchPatternLTECrsPerSlot;//0044
    nfapi_uint8_tlv_t numOfRateMatchPatternLTECrsInPhy;//0045
    nfapi_uint8_tlv_t csiRsRateMatch;//0046
    nfapi_uint8_tlv_t pdschTransTypeSupport;//0047
    nfapi_uint8_tlv_t pdschMacPduBitAlignment;//0048
} nfapi_nr_pdsch_param_t;
//Table 3-17 PUSCH parameters
typedef struct {
  nfapi_uint8_tlv_t uci_mux_ulsch_in_pusch;
  nfapi_uint8_tlv_t uci_only_pusch;
  nfapi_uint8_tlv_t pusch_frequency_hopping;
  nfapi_uint8_tlv_t pusch_dmrs_config_types;
  nfapi_uint8_tlv_t pusch_dmrs_max_len;
  nfapi_uint8_tlv_t pusch_dmrs_additional_pos;
  nfapi_uint8_tlv_t pusch_cbg;
  nfapi_uint8_tlv_t pusch_mapping_type;
  nfapi_uint8_tlv_t pusch_allocation_types;
  nfapi_uint8_tlv_t pusch_vrb_to_prb_mapping;
  nfapi_uint8_tlv_t pusch_max_ptrs_ports;
  nfapi_uint8_tlv_t max_pduschs_tbs_per_slot;
  nfapi_uint8_tlv_t max_number_mimo_layers_non_cb_pusch;
  nfapi_uint8_tlv_t maxNumberMimoLayersCbPusch;//0049
  nfapi_uint8_tlv_t supported_modulation_order_ul;
  nfapi_uint8_tlv_t max_mu_mimo_users_ul;
  nfapi_uint8_tlv_t dfts_ofdm_support;
  nfapi_uint8_tlv_t pusch_aggregation_factor;
  nfapi_uint8_tlv_t puschLbrmSupport;//004A
  nfapi_uint8_tlv_t puschTransTypeSupport;//004B
  nfapi_uint8_tlv_t puschMacPduBitAlignment;//004C
} nfapi_nr_pusch_param_t;

//Table 3-18 PRACH parameters
typedef struct {
    nfapi_uint8_tlv_t prach_long_formats;
    nfapi_uint8_tlv_t prach_short_formats;
    nfapi_uint8_tlv_t prach_restricted_sets;
    nfapi_uint8_tlv_t max_prach_fd_occasions_in_a_slot;
    nfapi_uint16_tlv_t maxNumPrachConfigurations;
    nfapi_uint8_tlv_t prachMultipleCarriersInABand;
} nfapi_nr_prach_param_t;

//Table 3-19 Measurement parameters
typedef struct {
    nfapi_uint8_tlv_t rssi_measurement_support;
} nfapi_nr_measurement_param_t;

//Table 3-20 UCI parameters
typedef struct{
    nfapi_uint16_tlv_t maxNumUciMaps;
}nfapi_nr_uci_param_t;

//table 3-21 Capability validity scope
typedef struct{
    nfapi_uint16_tlv_t  CapabilityTag;
    nfapi_uint8_tlv_t   ValidityScope;
}nfapi_nr_num_capabilities_t;
typedef struct {
    nfapi_uint16_tlv_t numCapabilities;
    nfapi_nr_num_capabilities_t* numCapabilities_list;
}nfapi_nr_capability_validity_scope_t;

//Table 3-22 PHY Support
typedef struct{
    nfapi_uint8_tlv_t numDlPortRanges;
    nfapi_uint16_tlv_t dlPortRangeStart;//[numDlPortRanges]
    nfapi_uint16_tlv_t dlPortRangeLen;//[numDlPortRanges]
    nfapi_uint8_tlv_t numUlPortRanges;
    nfapi_uint16_tlv_t ulPortRangeStart;//[numUlPortRanges]
    nfapi_uint16_tlv_t ulPortRangeLen;//[numUlPortRanges]
}nfapi_nr_max_num_phys_t;
typedef struct{
    nfapi_uint8_tlv_t maxNumPhys;
    nfapi_nr_max_num_phys_t maxnumphyslist;
}nfapi_nr_num_phy_profiles_t;
typedef struct{
    nfapi_uint16_tlv_t numPhyProfiles;
    nfapi_uint16_tlv_t numDlBbPorts;
    nfapi_uint16_tlv_t numUlBbPorts;
    nfapi_nr_num_phy_profiles_t numphyprofile;
    nfapi_uint8_tlv_t Time_Management;
    nfapi_uint8_tlv_t phyFapiProtocolVersion;
    nfapi_uint8_tlv_t phyFapiNegotiatedProtocolVersion;
    nfapi_uint8_tlv_t moreThanOneIndicationPerSlot[6];
    nfapi_uint8_tlv_t moreThanOneRequestPerSlot[4];
} nfapi_nr_phy_profile_support_param_t;
//Table 3-23 PHY / DFE Profile Validity Map
typedef struct{
    nfapi_uint16_tlv_t numPhyProfiles;
    nfapi_uint16_tlv_t numDfeProfiles;
    nfapi_uint8_tlv_t profileValidityMap;
}nfapi_nr_phy_dfe_profile_validity_map_t;
//Table 3-24 Delay management parameters
typedef struct{
    nfapi_uint32_tlv_t DL_TTI_Timingoffset;//0106
    nfapi_uint32_tlv_t UL_TTI_Timingoffset;//0107
    nfapi_uint32_tlv_t UL_DCI_Timingoffset;//0108
    nfapi_uint32_tlv_t Tx_Data_Timingoffset;//0109
    nfapi_uint16_tlv_t Timing_window;//011E
    nfapi_uint8_tlv_t Timing_info_period;//0120
}nfapi_nr_delay_management_param_t;
//Table 3-25 Rel-16 mTRP parameters
typedef struct{
    nfapi_uint32_tlv_t mTRP_Support; 
}nfapi_nr_rel16_mtrp_param_t;
// Table 3-10 PARAM.response TLVs
typedef struct {
    nfapi_nr_p5_message_header_t header;
    uint8_t error_code;
    uint8_t num_tlv;

    nfapi_nr_cell_param_t         cell_param;
    nfapi_nr_carrier_param_t      carrier_param;
    nfapi_nr_pdcch_param_t        pdcch_param;
    nfapi_nr_pucch_param_t        pucch_param;
    nfapi_nr_pdsch_param_t        pdsch_param;
    nfapi_nr_pusch_param_t        pusch_param;
    nfapi_nr_prach_param_t        prach_param;
    nfapi_nr_measurement_param_t  measurement_param;
    nfapi_nr_capability_validity_scope_t capa_vali_scope;
    nfapi_nr_phy_profile_support_param_t phy_profiles_param;
    nfapi_nr_phy_dfe_profile_validity_map_t phydfeprofilevaildity;
    nfapi_nr_delay_management_param_t delaymanageparam;
    nfapi_nr_rel16_mtrp_param_t rel16mtrpparam;
} nfapi_nr_param_response_scf_t;

/* 3.3.2 CONFIG */

//Table 3-30 PHY configuration
#define NFAPI_NR_CONFIG_PHY_PROFILE_ID_TAG 0x102A
#define NFAPI_NR_CONFIG_INDICATION_INSTANCES_PER_SLOT_TAG 0x102B
#define NFAPI_NR_CONFIG_REQUEST_INSTANCES_PER_SLOT_TAG 0x102C
//Table 3-31 Carrier configuration
#define NFAPI_NR_CONFIG_CARRIER_CONFIGURATION_TAG 0x102D
//Table 3-32 Cell configuration
#define NFAPI_NR_CONFIG_PHY_CELL_ID_TAG 0x100C
#define NFAPI_NR_CONFIG_FRAME_DUPLEX_TYPE_TAG 0x100D
#define NFAPI_NR_CONFIG_PDSCH_TRANS_TYPE_VALIDITY_TAG 0x102E
#define NFAPI_NR_CONFIG_PUSCH_TRANS_TYPE_VALIDITY_TAG 0x102F
//Table 3-33 SSB power and PBCH configuration
#define NFAPI_NR_CONFIG_SS_PBCH_POWER_TAG 0x100E
#define NFAPI_NR_CONFIG_SS_PBCH_BLOCK_POWER_SCALING_TAG 0x1030
#define NFAPI_NR_CONFIG_BCH_PAYLOAD_TAG 0x100F
#define NFAPI_NR_CONFIG_PRACH_CONFIGURATION_TAG 0x1031
#define NFAPI_NR_CONFIG_MULTI_PRACH_CONFIGURATION_TABLE_TAG 0x1032
#define NFAPI_NR_CONFIG_SSB_RESOURCE_CONFIGURATION_TABLE_TAG 0x1033
#define NFAPI_NR_CONFIG_MULTI_SSB_RESOURCE_CONFIGURATION_TABLE_TAG 0x1034
#define NFAPI_NR_CONFIG_TDD_TABLE_TAG 0x1035
#define NFAPI_NR_CONFIG_RSSI_MEASUREMENT_TAG 0x1028
#define NFAPI_NR_CONFIG_UCI_CONFIGURATION_TAG 0x1036
#define NFAPI_NR_CONFIG_PRB_SYMBOL_RATE_MATCH_PATTERNS_BITMAP_TAG 0x0137
#define NFAPI_NR_CONFIG_LTE_CRS_RATE_MATCH_PATTERNS_TAG 0x0138
#define NFAPI_NR_CONFIG_PUCCH_SEMI_STATIC_CONFIGURATION_TAG 0x1039
#define NFAPI_NR_CONFIG_PDSCH_SEMI_STATIC_CONFIGURATION_TAG 0x103A
#define NFAPI_NR_CONFIG_TIMING_WINDOW_TAG 0x011E
#define NFAPI_NR_CONFIG_TIMING_INFO_MODE_TAG 0x011F
#define NFAPI_NR_CONFIG_TIMING_INFO_PERIOD_TAG 0x0120
#define NFAPI_NR_CONFIG_NUM_TX_PORTS_TRP1_TAG 0x103B
#define NFAPI_NR_CONFIG_NUM_RX_PORTS_TRP1_TAG 0x103C

// 3.3.2.1 CONFIG.request
//table 3-30 phy config
typedef struct{
    nfapi_uint16_tlv_t phyProfileId;
    nfapi_uint8_tlv_t indicationInstancesPerSlot[6];
    nfapi_uint8_tlv_t requestInstancesPerSlot[4];
}nfapi_nr_phy_config_t;

//table 3-31 carrier config
typedef struct  {
  nfapi_uint16_tlv_t dl_bandwidth;//Carrier bandwidth for DL in MHz [38.104, sec 5.3.2] Values: 5, 10, 15, 20, 25, 30, 40,50, 60, 70, 80,90,100,200,400
  nfapi_uint32_tlv_t dl_frequency; //Absolute frequency of DL point A in KHz [38.104, sec5.2 and 38.211 sec 4.4.4.2] Value: 450000 -> 52600000
  nfapi_uint16_tlv_t dl_k0[5];//𝑘_{0}^{𝜇} for each of the numerologies [38.211, sec 5.3.1] Value: 0 ->23699
  nfapi_uint16_tlv_t dl_grid_size[5];//Grid size 𝑁_{𝑔𝑟𝑖𝑑}^{𝑠𝑖𝑧𝑒,𝜇} for each of the numerologies [38.211, sec 4.4.2] Value: 0->275 0 = this numerology not used
  nfapi_uint16_tlv_t num_tx_ant;//Number of Tx antennas
  nfapi_uint16_tlv_t uplink_bandwidth;//Carrier bandwidth for UL in MHz. [38.104, sec 5.3.2] Values: 5, 10, 15, 20, 25, 30, 40,50, 60, 70, 80,90,100,200,400
  nfapi_uint32_tlv_t uplink_frequency;//Absolute frequency of UL point A in KHz [38.104, sec5.2 and 38.211 sec 4.4.4.2] Value: 450000 -> 52600000
  nfapi_uint16_tlv_t ul_k0[5];//𝑘0 𝜇 for each of the numerologies [38.211, sec 5.3.1] Value: : 0 ->23699
  nfapi_uint16_tlv_t ul_grid_size[5];//Grid size 𝑁𝑔𝑟𝑖𝑑 𝑠𝑖𝑧𝑒,𝜇 for each of the numerologies [38.211, sec 4.4.2]. Value: 0->275 0 = this numerology not used
  nfapi_uint16_tlv_t num_rx_ant;
  nfapi_uint8_tlv_t  frequency_shift_7p5khz;
  nfapi_uint8_tlv_t PowerProfile;
  nfapi_uint8_tlv_t PowerOffsetRsIndex;
} nfapi_nr_carrier_config_t; 

//table 3-32 cell configuration
typedef struct {
  nfapi_uint16_tlv_t phy_cell_id;//Physical Cell ID, 𝑁_{𝐼𝐷}^{𝑐𝑒𝑙𝑙} [38.211, sec 7.4.2.1] Value: 0 ->1007
  nfapi_uint8_tlv_t frame_duplex_type;//Frame duplex type Value: 0 = FDD 1 = TDD
  nfapi_uint8_tlv_t pdschTransTypeValidity;
  nfapi_uint8_tlv_t puschTransTypeValidity;
} nfapi_nr_cell_config_t;

//table 3-33 ssb power and pbch configuration
typedef struct {
  nfapi_int32_tlv_t ss_pbch_power;//SSB Block Power Value: TBD (-60..50 dBm)
  nfapi_uint16_tlv_t ssPbchBlockPowerScaling;
  nfapi_uint8_tlv_t  bch_payload;
} nfapi_nr_ssb_config_t;

//table 3-34 PRACH configuration
typedef struct {
  nfapi_uint16_tlv_t prach_root_sequence_index;//Starting logical root sequence index, 𝑖, equivalent to higher layer parameter prach-RootSequenceIndex [38.211, sec 6.3.3.1] Value: 0 -> 837
  nfapi_uint8_tlv_t  num_root_sequences;//Number of root sequences for a particular FD occasion that are required to generate the necessary number of preambles
  nfapi_uint16_tlv_t k1;//Frequency offset (from UL bandwidth part) for each FD. [38.211, sec 6.3.3.2] Value: from 0 to 272
  nfapi_uint8_tlv_t  prach_zero_corr_conf;//PRACH Zero CorrelationZone Config which is used to dervive 𝑁𝑐𝑠 [38.211, sec 6.3.3.1] Value: from 0 to 15
  nfapi_uint16_tlv_t  num_unused_root_sequences;//Number of unused sequences available for noise estimation per FD occasion. At least one unused root sequence is required per FD occasion.
  nfapi_uint16_tlv_t* unused_root_sequences_list;//Unused root sequence or sequences per FD occasion. Required for noise estimation.
} nfapi_nr_num_prach_fd_occasions_t;
typedef struct {
    nfapi_uint16_tlv_t prachResConfigIndex;
    nfapi_uint8_tlv_t prach_sequence_length;//RACH sequence length. Only short sequence length is supported for FR2. [38.211, sec 6.3.3.1] Value: 0 = Long sequence 1 = Short sequence
    nfapi_uint8_tlv_t prach_sub_c_spacing;//Subcarrier spacing of PRACH. [38.211 sec 4.2] Value:0->4
    nfapi_uint8_tlv_t ulBwpPuschScs;
    nfapi_uint8_tlv_t restricted_set_config;//PRACH restricted set config Value: 0: unrestricted 1: restricted set type A 2: restricted set type B
    nfapi_uint8_tlv_t num_prach_fd_occasions;//Corresponds to the parameter 𝑀 in [38.211, sec 6.3.3.2] which equals the higher layer parameter msg1FDM Value: 1,2,4,8
    nfapi_uint8_tlv_t prach_ConfigurationIndex;//PRACH configuration index. Value:0->255
    nfapi_nr_num_prach_fd_occasions_t* num_prach_fd_occasions_list;
    nfapi_uint8_tlv_t ssb_per_rach;//SSB-per-RACH-occasion Value: 0: 1/8 1:1/4, 2:1/2 3:1 4:2 5:4, 6:8 7:16
} nfapi_nr_prach_config_t;

//table 3-35  Multi-PRACH configuration table
typedef struct{
    nfapi_uint16_tlv_t numPrachConfigurations;
    nfapi_nr_prach_config_t numPrachConfigurationsTLVs;
} nfapi_nr_multi_prach_config_table_t ;

//table 3-36 SSB resource configuration table
typedef struct{
    nfapi_uint32_tlv_t SsbMask;
}nfapi_nr_ssb_mask_list_t;
typedef struct{
    nfapi_uint8_tlv_t BeamId;
}nfapi_nr_ssb_beam_id_list_t;
typedef struct{
    nfapi_uint16_tlv_t ssbConfigIndex;
    nfapi_uint16_tlv_t ssbOffsetPointA;
    nfapi_uint8_tlv_t betaPssProfileNR;
    nfapi_uint16_tlv_t betaPssProfileSSS;
    nfapi_uint8_tlv_t ssbPeriod;
    nfapi_uint8_tlv_t ssbSubcarrierOffset;
    nfapi_uint8_tlv_t Case;
    nfapi_uint8_tlv_t subcarrierSpacing;
    nfapi_uint8_tlv_t subCarrierSpacingCommon;
    nfapi_nr_ssb_mask_list_t ssb_mask_list[2];
    nfapi_nr_ssb_beam_id_list_t ssb_id_list[64];
    nfapi_uint8_tlv_t lMax;
    nfapi_uint8_tlv_t rmsiPresence;
}nfapi_nr_ssb_resource_config_t;

//table 3-37 Multi-SSB resource configuration table
typedef struct{
    nfapi_uint8_tlv_t numSsbConfigurations;
    nfapi_nr_ssb_resource_config_t ssbConfigurations_TLVs;
}nfapi_nr_multi_ssb_resource_config_table_t;

//table 3-38 TDD table
typedef struct {
  nfapi_uint8_tlv_t slot_config;//For each symbol in each slot a uint8_t value is provided indicating: 0: DL slot 1: UL slot 2: Guard slot
} nfapi_nr_max_num_of_symbol_per_slot_t;
typedef struct {
  nfapi_nr_max_num_of_symbol_per_slot_t* max_num_of_symbol_per_slot_list;
} nfapi_nr_max_tdd_periodicity_t;
typedef struct {
  nfapi_uint8_tlv_t tdd_period;//DL UL Transmission Periodicity. Value:0: ms0p5 1: ms0p625 2: ms1 3: ms1p25 4: ms2 5: ms2p5 6: ms5 7: ms10 8: ms3 9: ms4
  nfapi_nr_max_tdd_periodicity_t* max_tdd_periodicity_list;
} nfapi_nr_tdd_table_t;

//table 3-39 Measurement configuration
typedef struct {
  nfapi_uint8_tlv_t rssi_measurement;//RSSI measurement unit. See Table 3-16 for RSSI definition. Value: 0: Do not report RSSI 1: dBm 2: dBFS
} nfapi_nr_measurement_config_t;

//table 3-40 UCI configuration
typedef struct{
    nfapi_uint8_tlv_t numPart1Params;
    nfapi_uint8_tlv_t* sizesPart1Params;// [numPart1Params]
    nfapi_uint16_tlv_t* map;//[2^(Σ(sizesPart1Params))]
}nfapi_nr_num_uci_to_maps_t;
typedef struct{
    nfapi_uint16_tlv_t numUci2Maps;
    nfapi_nr_num_uci_to_maps_t numucitomaps_list;
} nfapi_nr_uci_config_t;

//table 3-41 PRB-symbol rate match patterns bitmap (non-CORESET) configuration
typedef struct{
    nfapi_uint8_tlv_t prbSymbRateMatchPatternID;
    nfapi_uint8_tlv_t freqDomainRB[35];
    nfapi_uint8_tlv_t oneOrTwoSlots;
    nfapi_uint32_tlv_t symbolsInRB;
    nfapi_uint8_tlv_t timeDomainPeriodicity;
    nfapi_uint8_tlv_t timeDomainPattern[5];
    nfapi_uint8_tlv_t subcarrierSpacing;
}nfapi_nr_prb_symb_rate_match_patterns;
typedef struct{
    nfapi_uint8_tlv_t numberOfPrb_SymbRateMatch_Patterns;
    nfapi_nr_prb_symb_rate_match_patterns  PrbSymbRateMatchPattern;
}nfapi_nr_prb_symbol_rate_match_pattern_bitmap_config_t;

//Table 3-42 LTE-CRS rate match patterns configuration
typedef struct{
    nfapi_uint8_tlv_t radioframeAllocationPeriod;
    nfapi_uint8_tlv_t radioframeAllocationOffset;
    nfapi_uint8_tlv_t lteFrameStructureType;
    nfapi_uint8_tlv_t subframeAllocLength;
    nfapi_uint8_tlv_t subframeAllocationBitmap;
}nfapi_nr_size_mbsfn_subframe_config_list_t;
typedef struct{
    nfapi_uint8_tlv_t crsRateMatchPatternID;
    nfapi_uint16_tlv_t carrierFreqDL;
    nfapi_uint8_tlv_t carrierBandwidthDL;
    nfapi_uint8_tlv_t nrofCrsPorts;
    nfapi_uint8_tlv_t vShift;
    nfapi_uint8_tlv_t sizeMbsfnSubframeConfigList;
    nfapi_nr_size_mbsfn_subframe_config_list_t sizeMbsfnSubframeConfig_list;
}nfapi_nr_each_LteCRS_RateMatchPattern_t;
typedef struct{
    nfapi_uint8_tlv_t numberOfLteCRS_RateMatchPatterns;
    nfapi_nr_each_LteCRS_RateMatchPattern_t LteCRS_RateMatchPattern_list;
} nfapi_nr_lte_crs_rate_match_patterns_config_t;

//table 3-43 PUCCH semi-static configuration
typedef struct{
    nfapi_uint8_tlv_t pucchGroupHopping;
    nfapi_uint16_tlv_t nIdPucchHopping;
}nfapi_nr_ul_bwp_id_t;
typedef struct{
    nfapi_uint8_tlv_t numUlBwpIds;
    nfapi_nr_ul_bwp_id_t UlBwpId_list;
}nfapi_nr_pucch_semi_static_config_t;
//table 3-44 PDSCH semi-static configuration
typedef struct {
    nfapi_uint8_tlv_t pdschCbgScheme;
}nfapi_nr_pdsch_semi_static_config_t;

//table 3-45 delay management configuration
typedef struct {
    nfapi_uint16_tlv_t timing_window;
    nfapi_uint8_tlv_t timing_info_mode;
    nfapi_uint8_tlv_t timing_info_period;
}nfapi_nr_delay_management_config_t;

//Table 3-46 Rel-16 mTRP configuration
typedef struct{
    nfapi_uint8_tlv_t numTxPortsTRP1;
    nfapi_uint8_tlv_t numRxPortsTRP1;
}nfapi_nr_rel16_mtrp_config_t;

/* 3.3.7 Storing Precoding and Beamforming Tables */
//table 3-51 Digital beam table (DBT) PDU
typedef struct {
	uint16_t dig_beam_weight_Re;
    uint16_t dig_beam_weight_Im;
} nfapi_nr_txru_t;
typedef struct {
	uint16_t beam_idx;    
    nfapi_nr_txru_t*  txru_list;
} nfapi_nr_dig_beam_t;
typedef struct {
	uint16_t num_dig_beams; 
    uint16_t num_txrus;
    nfapi_nr_dig_beam_t* dig_beam_list;
} nfapi_nr_dbt_pdu_t;

//table 3-52 Precoding matrix (PM) PDU
typedef struct {
	int16_t precoder_weight_Re;
    int16_t precoder_weight_Im;
} nfapi_nr_num_ant_ports_t;
typedef struct {
	nfapi_nr_num_ant_ports_t* num_ant_ports_list;
} nfapi_nr_num_layers_t;
typedef struct {
    uint16_t pm_idx;       
    uint16_t numLayers;
    uint16_t numAntPorts;
    nfapi_nr_num_layers_t* num_layers_list;   
} nfapi_nr_pm_pdu_t;

//Table 3-29 (total)
typedef struct {
    nfapi_nr_p5_message_header_t header;
  
    uint8_t                       num_tlv;

    nfapi_nr_phy_config_t         phy_config;//30
    nfapi_nr_carrier_config_t     carrier_config;//31
    nfapi_nr_cell_config_t        cell_config;//32
    nfapi_nr_ssb_config_t         ssb_config;//33
    nfapi_nr_prach_config_t       prach_config;//34
    //may fix !
    nfapi_nr_multi_prach_config_table_t multi_prach_config;//35
    nfapi_nr_ssb_resource_config_t ssb_resource_config;//36
    nfapi_nr_multi_ssb_resource_config_table_t multissbresourceconfig;//37
    nfapi_nr_tdd_table_t          tdd_table;//38
    nfapi_nr_measurement_config_t measurement_config;//39
    nfapi_nr_uci_config_t uci_config;//40
    nfapi_nr_prb_symbol_rate_match_pattern_bitmap_config_t prbsymbol_config;//41
    nfapi_nr_lte_crs_rate_match_patterns_config_t ltecrs_config;//42
    //may fix !
    nfapi_nr_pucch_semi_static_config_t pucch_semi_static_config;//43
    nfapi_nr_pdsch_semi_static_config_t pdsch_semi_static_config;//44
    nfapi_nr_delay_management_config_t  delay_management_config;//45
    nfapi_nr_rel16_mtrp_config_t Rel16_mTRP_config;//46
    nfapi_nr_dbt_pdu_t digital_beam_dbt_pdu;//50
    nfapi_nr_pm_pdu_t precoding_matrix_PDU;//51
} nfapi_nr_config_request_scf_t;
// 3.3.2.2 CONFIG.response

typedef struct {
  nfapi_nr_p5_message_header_t header;
  uint8_t error_code;
} nfapi_nr_config_response_scf_t;


// 3.3.2.3 CONFIG errors
typedef enum {    
  NFAPI_NR_CONFIG_MSG_OK = 0,
  NFAPI_NR_CONFIG_MSG_INVALID_CONFIG  
} nfapi_nr_config_errors_e;


/* 3.3.4 START */
// 3.3.4.1 START.request
typedef struct{
  nfapi_nr_p5_message_header_t header;
}nfapi_nr_start_request_scf_t;

// 3.3.4.2 START.errors
typedef enum {
    NFAPI_NR_START_MSG_INVALID_STATE
} nfapi_nr_start_errors_e;
// 3.3.4.3 START.reponse
typedef struct {
  nfapi_nr_p5_message_header_t header;
  nfapi_nr_start_errors_e error_code;
} nfapi_nr_start_response_scf_t;

/* 3.3.5 STOP */
// 3.3.5.1 STOP.request
typedef struct {
	nfapi_nr_p5_message_header_t header;
} nfapi_nr_stop_request_t;
// 3.3.5.2 STOP.indication
typedef struct {
	nfapi_nr_p5_message_header_t header;
} nfapi_nr_stop_indication_t;
// 3.3.5.3 STOP.errors
typedef enum {
    NFAPI_NR_STOP_MSG_INVALID_STATE
} nfapi_nr_stop_errors_e;

/* 3.3.6 PHY Notifications */
// 3.3.6.2 Error Codes
typedef enum {
    NFAPI_NR_PHY_API_MSG_OK                  =0x0,
	NFAPI_NR_PHY_API_MSG_INVALID_STATE       =0x1,
    NFAPI_NR_PHY_API_MSG_INVALID_CONFIG      =0x2,
    NFAPI_NR_PHY_API_SFN_OUT_OF_SYNC         =0X3,
    NFAPI_NR_PHY_API_MSG_SLOR_ERR            =0X4,
    NFAPI_NR_PHY_API_MSG_BCH_MISSING         =0X5,
    NFAPI_NR_PHY_API_MSG_INVALID_SFN         =0X6,
    NFAPI_NR_PHY_API_MSG_UL_DCI_ERR          =0X7,
    NFAPI_NR_PHY_API_MSG_TX_ERR              =0X8,
    NFAPI_NR_PHY_API_MSG_INVALID_PHY_ID      =0X9,
    NFAPI_NR_PHY_API_MSG_UNINSTANTIATED_PHY  =0XA,
    NFAPI_NR_PHY_API_MSG_INVALID_DFE_Profile =0XB,
    NFAPI_NR_PHY_API_PHY_Profile_Selection   =0XC
} nfapi_nr_phy_notifications_errors_e;

// 3.3.6.1 ERROR.indication
typedef struct {
    uint16_t sfn; 
    uint16_t slot;
    // nfapi_nr_phy_msg_type_e msg_id;
    nfapi_nr_phy_notifications_errors_e error_code;
    uint16_t expected_sfn;
    uint16_t expected_slot;
} nfapi_nr_phy_notifications_error_indicate_t;



/* Section 3.4 Slot messages */
#define NFAPI_NR_SLOT_INDICATION_PERIOD_NUMEROLOGY_0 1000 //us
#define NFAPI_NR_SLOT_INDICATION_PERIOD_NUMEROLOGY_1 500 //us
#define NFAPI_NR_SLOT_INDICATION_PERIOD_NUMEROLOGY_2 250 //us
#define NFAPI_NR_SLOT_INDICATION_PERIOD_NUMEROLOGY_3 125 //us
#define NFAPI_NR_SLOT_INDICATION_PERIOD_NUMEROLOGY_4 62 //us

typedef struct {
  uint16_t tag;
  uint16_t length;
  union { 
    uint32_t *ptr;
    uint32_t direct[16384];
  } value;
} nfapi_nr_tx_data_request_tlv_t;

typedef struct{
  uint16_t beamIdx;//Index of the digital beam weight vector pre-stored at cell configuration. The vector maps this input port to output TXRUs. Value: 0->65535
}nfapi_nr_dig_BF_Interface_t;

//  3.4.1 Slot indiction 
//table 3-53
typedef struct {
    nfapi_nr_p7_message_header_t header;
	uint16_t sfn; 
  uint16_t slot;
} nfapi_nr_slot_indication_scf_t;


//  3.4.2 DL_TTI.request
/// 3.4.2.5 Tx Precoding and Beamforming PDU  and 3.4.2.6 Rel-16 mTRP Tx Precoding and Beamforming PDU
//// 3.4.2.5
typedef struct {
  uint16_t beamIdx;
}nfapi_nr_dig_bf_interface_t;

typedef struct {
  uint16_t PMidx;
  nfapi_nr_dig_bf_interface_t* dig_bf_interface_list;

}nfapi_nr_tx_precoding_and_beamforming_number_of_prgs_t;

typedef struct {
  uint8_t TRP_scheme;
  uint16_t numPRGs;
  uint16_t prgSize;
  uint8_t  digBFInterfaces;
  nfapi_nr_tx_precoding_and_beamforming_number_of_prgs_t* prgs_list;
} nfapi_nr_tx_precoding_and_beamforming_t;

//// 3.4.2.6

//   <<<< Not yet joined >>>>
#define DCI_PAYLOAD_BYTE_LEN 10
/// 3.4.2.1 PDCCH PDU
typedef struct {
  uint16_t RNTI;
  uint16_t ScramblingId;
  uint16_t ScramblingRNTI;
  uint8_t CceIndex;
  uint8_t AggregationLevel;
  // Beamforming info 
  nfapi_nr_tx_precoding_and_beamforming_t precodingAndBeamforming;
  // Tx Power info
  uint8_t beta_PDCCH_1_0;
  uint8_t powerControlOffsetSS;

  uint16_t PayloadSizeBits;
  uint8_t Payload[DCI_PAYLOAD_BYTE_LEN];
  //FAPI V3
  uint16_t  dciIndex;
  uint8_t  collocatedAl16Candidate;
  int16_t pdcchDmrsPowerOffsetProfileSSS;
  int16_t pdcchDataPowerOffsetProfileSSS;
} nfapi_nr_dl_dci_pdu_t;
#define MAX_DCI_CORESET 10
typedef struct {
  //  BWP
  uint16_t CoresetBWPSize;
  uint16_t CoresetBWPStart;
  uint8_t SubcarrierSpacing;
  uint8_t CyclicPrefix;
  //  Coreset
  uint8_t StartSymbolIndex;
  uint8_t DurationSymbols; 
  uint8_t FreqDomainResource[6];
  uint8_t CceRegMappingType;
  uint8_t RegBundleSize;
  uint8_t InterleaverSize; 
  uint8_t CoreSetType;
  uint16_t ShiftIndex;
  uint8_t precoderGranularity;
  uint16_t numDlDci;
  nfapi_nr_dl_dci_pdu_t dci_pdu[MAX_DCI_CORESET];
  //FPAI V3
  //Coreset parameter
  uint16_t  pdcchPduIndex;
  uint16_t  nIdPdcchDmrs;
}  nfapi_nr_dl_tti_pdcch_pdu;

/// 3.4.2.2 PDSCH PDU
//Table 3-60  Set of PDSCH PDU parameters added in FAPIv3
typedef struct {
  //BWP information
  uint8_t pdschTransType;
  uint16_t  coresetStartPoint;
  uint16_t  initialDlBwpSize;
  //Codeword information
  uint8_t ldpcBaseGraph;
  uint32_t  tbSizeLbrmBytes;
  uint8_t tbCrcRequired;
  //Rate Matching references
  uint16_t ssbPdusForRateMatching[2];
  uint16_t  ssbConfigForRateMatching;
  uint8_t prbSymRmPatternBitmapSizeByReference;
  uint8_t prbSymRateMatchPatternBitmapByReference;
  uint8_t numPrbSymRmPatternsByValue;
  ///For numPrbSymRmPatternsByValue
  uint8_t freqDomainRB[35];
  uint16_t  symbolsInRB;

  uint8_t numCoresetRmPatterns;
  ///For numCoresetRmPatterns
  uint8_t freqDomainResources[6];
  uint16_t  symbolsPattern;

  uint16_t  pdcchPduIndex;
  uint16_t  dciIndex;
  uint8_t lteCrsRateMatchPatternBitmapSize; 
  uint8_t lteCrsRateMatchPattern;
  uint8_t numCsiRsForRateMatching;
  uint16_t  csiRsForRateMatching;
  //Tx Power info
  int16_t pdschDmrsPowerOffsetProfileSSS;
  int16_t pdschDataPowerOffsetProfileSSS;
  //CBG
  uint8_t maxNumCbgPerTb;
  uint8_t cbgTxInformation;

}nfapi_nr_pdsch_maintenance_parameters_fapiv3;

// table 3-61 PDSCH-PTRS parameter added in FAPIv3
typedef struct {
  //Tx Power info
  int16_t pdschPtrsPowerOffsetProfileSSS;
}nfapi_nr_pdsch_ptrs_maintenance_parameters_fapiv3;

// table 3-62 Rel-16 parameters added in FAPIv3
typedef struct {
  uint8_t repetitionScheme;
  // PTRS [TS38.214, sec 5.1.6.3]
  uint8_t PTRSPortIndex;
  uint8_t PTRSTimeDensity;
  uint8_t PTRSFreqDensity;
  uint8_t PTRSReOffset;
  uint8_t nEpreRatioOfPDSCHToPTRS;
  nfapi_nr_pdsch_ptrs_maintenance_parameters_fapiv3 pdschPtrsMaintenanceParameters;
}nfapi_nr_pdsch_rel16_parameters_fapiv3;

// table 3-58
typedef struct {
  uint16_t pduBitmap;
  uint16_t RNTI;
  uint16_t pduIndex;
  // BWP  [TS38.213 sec 12]
  uint16_t BWPSize;
  uint16_t BWPStart;
  uint8_t SubcarrierSpacing;
  uint8_t CyclicPrefix;
  // Codeword information
  uint8_t NrOfCodewords;
  uint16_t targetCodeRate[2]; 
  uint8_t qamModOrder[2];
  uint8_t mcsIndex[2];
  uint8_t mcsTable[2];   
  uint8_t rvIndex[2];
  uint32_t TBSize[2];

  uint16_t nIdPdsch;
  uint8_t nrOfLayers;
  uint8_t transmissionScheme;
  uint8_t refPoint;
  // DMRS  [TS38.211 sec 7.4.1.1]
  uint16_t dlDmrsSymbPos;  
  uint8_t dmrsConfigType;
  uint16_t dlDmrsScramblingId;
  uint8_t SCID;
  uint8_t numDmrsCdmGrpsNoData;
  uint16_t dmrsPorts;

  // Pdsch Allocation in frequency domain [TS38.214, sec 5.1.2.2]
  uint8_t resourceAlloc;
  uint8_t rbBitmap[36];
  uint16_t rbStart;
  uint16_t rbSize;
  uint8_t VRBtoPRBMapping;

  // Resource Allocation in time domain [TS38.214, sec 5.1.2.1]
  uint8_t StartSymbolIndex;
  uint8_t NrOfSymbols;

  // PTRS [TS38.214, sec 5.1.6.3] (PTRS Parameters (existing in FAPIv2))
  uint8_t PTRSPortIndex ;
  uint8_t PTRSTimeDensity;
  uint8_t PTRSFreqDensity;
  uint8_t PTRSReOffset;
  uint8_t nEpreRatioOfPDSCHToPTRS;
  // Beamforming
  nfapi_nr_tx_precoding_and_beamforming_t precodingAndBeamforming;

  //Tx power info
  uint8_t powerControlOffsetProfileNR;
  uint8_t powerControlOffsetSSProfileNR;
  
//  cbgReTxCtrl: CBG fields for Segmentation in L2. This structure is only included if and only if
//  (pduBitmap[1] = 1 and L1 is configured to support L2 CBG segmentation)
  
  uint8_t IsLastCbPresent;
  uint8_t isInlineTbCrc;
  uint32_t dlTbCrcCW[2];

  nfapi_nr_pdsch_maintenance_parameters_fapiv3 pdsch_maintenance_param;
  nfapi_nr_pdsch_ptrs_maintenance_parameters_fapiv3 pdsch_ptrs_maintenance_param;
  nfapi_nr_pdsch_rel16_parameters_fapiv3 pdsch_rel16_param;

}nfapi_nr_dl_tti_pdsch_pdu;

///  3.4.2.3 CSI-RS PDU
// table 3-63
typedef struct
{
  uint16_t BWPSize;// Not sure need it
  uint16_t BWPStart;// Not sure need it
  uint8_t  SubcarrierSpacing;
  uint8_t  CyclicPrefix;
  uint16_t StartRB;
  uint16_t NrOfRBs;
  uint8_t  CSIType;//Value: 0:TRS 1:CSI-RS NZP 2:CSI-RS ZP
  uint8_t  Row;//Row entry into the CSI Resource location table. [TS38.211, sec 7.4.1.5.3 and table 7.4.1.5.3-1] Value: 1-18
  uint16_t FreqDomain;//Value: Up to the 12 LSBs, actual size is determined by the Row parameter
  uint8_t  SymbL0;//The time domain location l0 and firstOFDMSymbolInTimeDomain Value: 0->13
  uint8_t  SymbL1;//
  uint8_t  CDMType;
  uint8_t  FreqDensity;//The density field, p and comb offset (for dot5).0: dot5 (even RB), 1: dot5 (odd RB), 2: one, 3: three
  uint16_t ScrambId;//ScramblingID of the CSI-RS [TS38.214, sec 5.2.2.3.1] Value: 0->1023
  //Tx power info
  uint8_t  powerControlOffsetProfileNR;//Ratio of PDSCH EPRE to NZP CSI-RSEPRE Value :0->23 representing -8 to 15 dB in 1dB steps
  uint8_t  powerControlOffsetSSProfileNR;//Ratio of SSB/PBCH block EPRE to NZP CSI-RS EPRES 0: -3dB, 1: 0dB, 2: 3dB, 3: 6dB
  // Beamforming
  nfapi_nr_tx_precoding_and_beamforming_t precodingAndBeamforming;

  //FAPI V3
  //Basic
  uint16_t  csiRsPduIndex;
  //Tx Power info
  int16_t csiRsPowerOffsetProfileSSS;
} nfapi_nr_dl_tti_csi_rs_pdu;

//  3.4.2.4 SSB PDU
//  table 3-66
typedef struct {
  uint32_t bchPayload;
} nfapi_nr_mac_generated_mib_pdu_t;

//  table-67
typedef struct {
  uint8_t  DmrsTypeAPosition;
  uint8_t  PdcchConfigSib1;
  uint8_t  CellBarred;
  uint8_t  IntraFreqReselection;
} nfapi_nr_phy_generated_mib_pdu_t;

// table 3-66 and 3-77
typedef struct {
  nfapi_nr_mac_generated_mib_pdu_t* mac_generated_mib_pdu;
  nfapi_nr_phy_generated_mib_pdu_t* phy_generated_mib_pdu;
} nfapi_nr_bch_payload_t;

// table 3-65
typedef struct {
  uint16_t PhysCellId;
  uint8_t  betaPssProfileNR;
  uint8_t  ssbBlockIndex;
  uint8_t  ssbSubcarrierOffset;
  uint16_t SsbOffsetPointA;
  uint8_t  bchPayloadFlag;
  uint32_t bchPayload; // See table 3-66 and 3-67
  nfapi_nr_tx_precoding_and_beamforming_t precodingAndBeamforming;

  // FAPI V3 
  //Basic Parameters
  uint8_t ssbPduIndex;
  uint8_t Case;
  uint8_t SubcarrierSpacing;
  uint8_t lMax;
  //Tx Power info
  int16_t ssPbchBlockPowerScaling;
  int16_t betaPSSProfileSSS;

} nfapi_nr_dl_tti_ssb_pdu;

// Consolidate
typedef struct {
  uint16_t PDUType;
  uint32_t PDUSize;

  union {
  nfapi_nr_dl_tti_pdcch_pdu      pdcch_pdu;
  nfapi_nr_dl_tti_pdsch_pdu      pdsch_pdu;
  nfapi_nr_dl_tti_csi_rs_pdu     csi_rs_pdu;
  nfapi_nr_dl_tti_ssb_pdu        ssb_pdu;
  };
} nfapi_nr_dl_tti_request_pdu_t;

#define NFAPI_NR_MAX_DL_TTI_PDUS 10
typedef struct {
    nfapi_nr_p7_message_header_t header;
    uint16_t SFN;
    uint16_t Slot;
    uint16_t nPDUs;
    uint8_t nDlTypes;
    uint16_t nPDUsOfEachType;
    uint8_t nGroup;
    nfapi_nr_dl_tti_request_pdu_t dl_tti_pdu_list[NFAPI_NR_MAX_DL_TTI_PDUS];
} nfapi_nr_dl_tti_request_t;


// 3.4.3 UL_TTI.request
typedef struct{
  nfapi_nr_dig_BF_Interface_t* dig_bf_interface_list;
} nfapi_nr_ul_beamforming_number_of_PRGs_t;
// table 3-85
typedef struct{
  uint8_t TRP_scheme;
  uint16_t numPRGs;
  uint16_t prgSize;
  uint8_t  digBFInterface;
  nfapi_nr_ul_beamforming_number_of_PRGs_t* prgs_list;
} nfapi_nr_ul_beamforming_t;

/// 3.4.3.1  PRACH PDU
//  table 3-74
typedef struct{
  uint32_t  handle;
  uint8_t prachConfigScope;
  uint16_t  prachResConfigIndex;
  uint8_t numFdRa;
  uint8_t startPreambleIndex;
  uint8_t numPreambleIndices;
}nfapi_nr_prach_maintenance_parameters_fapiv3;
//  table 3-73
typedef struct{
  uint16_t physCellID;
  uint8_t  numPrachOcas;
  uint8_t  prachFormat;
  uint8_t  indexFdRa;
  uint8_t  prachStartSymbol;
  uint16_t numCs;
  nfapi_nr_ul_beamforming_t beamforming;
  // FAPI V3
  nfapi_nr_prach_maintenance_parameters_fapiv3 prachmaintenanceParam;
} nfapi_nr_prach_pdu_t;

/// 3.4.3.2 PUSCH PDU
//  table 3-76
typedef struct{
  //BWP [3GPP TS 38.213 [4], sec 12]
  uint8_t puschTransType;
  uint16_t deltaBwp0StartFromActiveBwp;
  uint16_t  initialUlBwpSize;
  //DMRS [3GPP TS 38.211 [2], sec 6.4.1.1]
  uint8_t groupOrSequenceHopping;
 //Frequency Domain Allocation [3GPP TS 38.214 [5], sec 6.1.2.2] and Hopping [3GPP TS38.214 [5], sec 6.3]
  uint16_t puschSecondHopPRB;
  uint8_t ldpcBaseGraph;
  uint32_t  tbSizeLbrmBytes;
}nfapi_nr_pusch_maintenance_parameters_fapiv3;

//  table 3-77
typedef struct{
  uint16_t  priority;
  uint8_t numPart1Params;
  uint16_t  paramOffsets;
  uint8_t paramSizes;
  uint16_t  part2SizeMapIndex;
}nfapi_nr_pusch_uci_each_part2;
typedef struct{
  uint16_t  numPart2s;
  nfapi_nr_pusch_uci_each_part2 part2vlaue;
}nfapi_nr_pusch_uci_part1_to_part2;

#define NFAPI_MAX_NUM_CB 9
//  table 3-78 for pusch data
typedef struct {
  uint8_t  rvIndex;
  uint8_t  harqProcessID;
  uint8_t  newData;
  uint32_t TBSize;
  uint16_t numCb;
  uint8_t cbPresentAndPosition[(NFAPI_MAX_NUM_CB+7) / 8];
} nfapi_nr_pusch_data_t;

//  table 3-79 for pusch uci
typedef struct{
  uint16_t harqAckBitLength;
  uint16_t csiPart1BitLength;
  uint16_t flagCsiPart2;//flagCsiPart2
  uint8_t  AlphaScaling;
  uint8_t  betaOffsetHarqAck;
  uint8_t  betaOffsetCsi1;
  uint8_t  betaOffsetCsi2;
} nfapi_nr_pusch_uci_t;

//table 3-80 for pusch ptrs
typedef struct {
  uint16_t PTRSPortIndex;//PT-RS antenna ports [TS38.214, sec6.2.3.1 and 38.212, section 7.3.1.1.2] Bitmap occupying the 12 LSBs with: bit 0: antenna port 0 bit 11: antenna port 11 and for each bit 0: PTRS port not used 1: PTRS port used
  uint8_t  PTRSDmrsPort;//DMRS port corresponding to PTRS.
  uint8_t  PTRSReOffset;//PT-RS resource element offset value taken from 0~11
} nfapi_nr_ptrs_ports_t;
typedef struct{
  uint8_t  numPtrsPorts;
  nfapi_nr_ptrs_ports_t* ptrs_ports_list;
  uint8_t  PTRSTimeDensity;
  uint8_t  PTRSFreqDensity;
  uint8_t  ulPTRSPower;
}nfapi_nr_pusch_ptrs_t;

//  table 3-81 for dfts Ofdm
typedef struct{
  uint8_t  lowPaprGroupNumber;//Group number for Low PAPR sequence generation.
  uint16_t lowPaprSequenceNumber;//[TS38.211, sec 5.2.2] For DFT-S-OFDM.
  uint8_t  ulPtrsSampleDensity;//Number of PTRS groups [But I suppose this sentence is misplaced, so as the next one. --Chenyu]
  uint8_t  ulPtrsTimeDensityTransformPrecoding;//Number of samples per PTRS group

} nfapi_nr_dfts_ofdm_t;

//  table 3-75
typedef struct
{
  uint16_t pduBitmap;//Bitmap indicating presence of optional PDUs (see above)
  uint16_t RNTI;
  uint32_t Handle;//An opaque handling returned in the RxData.indication and/or UCI.indication message
  //BWP
  uint16_t BWPSize;
  uint16_t BWPStart;
  uint8_t  SubcarrierSpacing;
  uint8_t  CyclicPrefix;
  //pusch information always include
  uint16_t targetCodeRate;
  uint8_t  qamModOrder;
  uint8_t  mcsIndex;
  uint8_t  mcsTable;
  uint8_t  TransformPrecoding;
  uint16_t nIdPusch;
  uint8_t  nrOfLayers;
  //DMRS
  uint16_t  ulDmrsSymbPos;
  uint8_t  dmrsConfigType;
  uint16_t puschDmrsScramblingId;
  uint16_t puschDmrsIdentity;
  uint8_t  nSCID;
  uint8_t  numDmrsCdmGrpsNoData;
  uint16_t dmrsPorts;//DMRS ports. [TS38.212 7.3.1.1.2] provides description between DCI 0-1 content and DMRS ports. Bitmap occupying the 11 LSBs with: bit 0: antenna port 1000 bit 11: antenna port 1011 and for each bit 0: DMRS port not used 1: DMRS port used
  //Pusch Allocation in frequency domain [TS38.214, sec 6.1.2.2]
  uint8_t  resourceAlloc;
  uint8_t  rbBitmap[36];//
  uint16_t rbStart;
  uint16_t rbSize;
  uint8_t  VRBtoPRBMapping;
  uint8_t  IntraSlotFrequencyHopping; 
  uint16_t txDirectCurrentLocation;//The uplink Tx Direct Current location for the carrier. Only values in the value range of this field between 0 and 3299, which indicate the subcarrier index within the carrier corresponding 1o the numerology of the corresponding uplink BWP and value 3300, which indicates "Outside the carrier" and value 3301, which indicates "Undetermined position within the carrier" are used. [TS38.331, UplinkTxDirectCurrentBWP IE]
  uint8_t  uplinkFrequencyShift7p5khz;
  //Resource Allocation in time domain
  uint8_t  StartSymbolIndex;
  uint8_t  NrOfSymbols;
  //Optional Data only included if indicated in pduBitmap
  nfapi_nr_pusch_data_t puschData; // see table 3-78
  nfapi_nr_pusch_uci_t  puschUci; // see table 3-79
  nfapi_nr_pusch_ptrs_t puschPtrs;  // see table 3-80
  nfapi_nr_dfts_ofdm_t dftsOfdm;  //see table 3-81
  //beamforming
  nfapi_nr_ul_beamforming_t beamforming;

  //FAPI V3
  nfapi_nr_pusch_maintenance_parameters_fapiv3 puschmaintenance; // see table 3-76
  nfapi_nr_pusch_uci_part1_to_part2 puschucioptional; // see table 3-77
} nfapi_nr_pusch_pdu_t;

/// 3.4.3.3 PUCCH PDU
//  table 3-83  In this specification version, L2 always includes this TLV to extend PUCCH PDU
typedef struct{
  uint8_t maxCodeRate;
  uint8_t ulBwpId;
}nfapi_nr_pucch_basic_extension;

// table 3-82
typedef struct {
  uint16_t RNTI;
  uint32_t Handle;
  //BWP
  uint16_t BWPSize;
  uint16_t BWPStart;
  uint8_t  SubcarrierSpacing;
  uint8_t  CyclicPrefix;

  uint8_t  FormatType;
  uint8_t  multiSlotTxIndicator;
  uint8_t  pi2Bpsk;
  //pucch allocation in freq domain
  uint16_t prbStart;
  uint16_t prbSize;
  //pucch allocation in tome domain
  uint8_t  StartSymbolIndex;
  uint8_t  NrOfSymbols;
  //hopping info
  uint8_t  intraSlotFrequencyHopping;
  uint16_t secondHopPRB;
  uint8_t  pucchGroupHopping;
  uint8_t  obsolete8bit;
  uint16_t nIdPucchHopping;
  uint16_t InitialCyclicShift;

  uint16_t nIdPucchScrambling;
  uint8_t  TimeDomainOccIdx;
  uint8_t  PreDftOccIdx;
  uint8_t  PreDftOccLen;
  //DMRS
  uint8_t  AddDmrsFlag;
  uint16_t DmrsScramblingId;
  uint8_t  DMRScyclicshift;

  uint8_t  SRFlag;
  uint8_t  BitLenHarq;
  uint16_t csiPart1BitLength;

  nfapi_nr_ul_beamforming_t beamforming;

  //FAPI V3 Extension TLVs
  nfapi_nr_pucch_basic_extension pucchBasicExtension; // see table 3-83
  nfapi_nr_pusch_uci_part1_to_part2 puschucioptional; // see table 3-77

} nfapi_nr_pucch_pdu_t;

/// 3.4.3.4 SRS PDU
// table 3-84
typedef struct
{
  uint16_t RNTI;
  uint32_t Handle;
  //BWP
  uint16_t BWPSize;
  uint16_t BWPStart;
  uint8_t  SubcarrierSpacing;
  uint8_t  CyclicPrefix;

  uint8_t  numAntPorts;
  uint8_t  numSymbols;
  uint8_t  numRepetitions;
  uint8_t  timeStartPosition;//Starting position in the time domain l0; Note: the MAC undertakes the translation from startPosition to 𝑙0
  uint8_t  configIndex;
  uint16_t sequenceId;
  uint8_t  bandwidthIndex;
  uint8_t  combSize;
  uint8_t  combOffset;//Transmission comb offset 𝑘 ̄ TC [TS38.211, Sec 6.4.1.4.3] Value: 0 → 1 (combSize = 0) Value: 0 → 3 (combSize = 1)
  uint8_t  cyclicShift;
  uint8_t  frequencyPosition;
  uint8_t  frequencyShift;
  uint8_t  frequencyHopping;
  uint8_t  groupOrSequenceHopping;//Group or sequence hopping configuration (RRC parameter groupOrSequenceHopping in SRS-Resource
  uint8_t  resourceType;//Type of SRS resource allocation
  uint16_t Tsrs;//SRS-Periodicity in slots [TS38.211 Sec 6.4.1.4.4] Value: 1,2,3,4,5,8,10,16,20,32,40,64,80,160,320,640,1280,2560
  uint16_t Toffset;//Slot offset value [TS38.211, Sec 6.4.1.4.3] Value:0->2559

  nfapi_nr_ul_beamforming_t beamforming;

} nfapi_nr_srs_pdu_t;

//  Consolidate
typedef struct {
  uint16_t PDUType;//0: PRACH PDU, 1: PUSCH PDU, 2: PUCCH PDU, 3: SRS PDU
  uint16_t PDUSize;//Value: 0 -> 65535
  union {
    nfapi_nr_prach_pdu_t prach_pdu;
    nfapi_nr_pusch_pdu_t pusch_pdu;
    nfapi_nr_pucch_pdu_t pucch_pdu;
    nfapi_nr_srs_pdu_t srs_pdu;
  };
} nfapi_nr_ul_tti_request_number_of_pdus_t;

typedef struct {
  uint8_t  PduIdx;
} nfapi_nr_ul_tti_request_number_of_ue_t;

typedef struct {
  uint8_t  nUe;
  nfapi_nr_ul_tti_request_number_of_ue_t ue_list;
} nfapi_nr_ul_tti_request_number_of_groups_t;

// table 3-72
typedef struct {
    nfapi_nr_p7_message_header_t header;
    uint16_t SFN; 
    uint16_t Slot;
    uint16_t nPDUs;
    uint8_t nUlTypes;
    uint16_t  nPDUsOfEachType;
    uint8_t nGroup;
    // uint8_t  rach_present;//Indicates if a RACH PDU will be included in this message. 0: no RACH in this slot 1: RACH in this slot
    // uint8_t  n_ulsch;//Number of ULSCH PDUs that are included in this message.
    // uint8_t  n_ulcch;//Number of ULCCH PDUs
    // uint8_t n_group;//Number of UE Groups included in this message. Value 0 -> 8
    nfapi_nr_ul_tti_request_number_of_pdus_t pdus_list;
    nfapi_nr_ul_tti_request_number_of_groups_t groups_list;
} nfapi_nr_ul_tti_request_t;



// 3.4.4 UL_DCI.request

typedef struct {
  uint16_t PDUType;
  uint16_t PDUSize;
  nfapi_nr_dl_tti_pdcch_pdu pdcch_pdu;
} nfapi_nr_ul_dci_request_pdus_t;
#define NFAPI_NR_MAX_UL_DCI_PDUS 10 //
typedef struct {
    nfapi_nr_p7_message_header_t header;
    uint16_t SFN;
    uint16_t Slot;
    uint8_t  numPdus;
    uint8_t nDlTypes;
    uint16_t nPDUsOfEachType;
    nfapi_nr_ul_dci_request_pdus_t ul_dci_pdu_list[NFAPI_NR_MAX_UL_DCI_PDUS];
} nfapi_nr_ul_dci_request_t;

// 3.4.5 SLOT errors 
// table 3-87
typedef enum {
  NFAPI_NR_SLOT_DL_TTI_MSG_INVALID_STATE,
  NFAPI_NR_SLOT_DL_TTI_OUT_OF_SYNC,
  NFAPI_NR_SLOT_DL_TTI_MSG_BCH_MISSING,
  NFAPI_NR_SLOT_DL_TTI_MSG_SLOT_ERR
} nfapi_nr_slot_errors_ul_tti_e;

// table 3-88
typedef enum {
  NFAPI_NR_SLOT_UL_TTI_MSG_INVALID_STATE,
  NFAPI_NR_SLOT_UL_TTI_MSG_SLOT_ERR
} nfapi_nr_slot_errors_dl_tti_e;

// table 3-89
typedef enum {
  NFAPI_NR_SLOT_UL_DCI_MSG_INVALID_STATE,
  NFAPI_NR_SLOT_UL_DCI_MSG_INVALID_SFN,
  NFAPI_NR_SLOT_UL_DCI_MSG_UL_DCI_ERR
} nfapi_nr_slot_errors_ul_dci_e;

// 3.4.6 Tx_Data.request
//
#define NFAPI_NR_MAX_TX_REQUEST_TLV 2
typedef struct {
  uint32_t PDU_length;
  uint16_t PDU_index;
  uint8_t CW_index;
  uint32_t num_TLV;
  nfapi_nr_tx_data_request_tlv_t TLVs[NFAPI_NR_MAX_TX_REQUEST_TLV]; 
} nfapi_nr_pdu_t;

// table 3-90
#define NFAPI_NR_MAX_TX_REQUEST_PDUS 16
typedef struct {
  nfapi_nr_p7_message_header_t header;
  uint16_t SFN;
  uint16_t Slot;
  uint16_t Number_of_PDUs;
  nfapi_nr_pdu_t pdu_list[NFAPI_NR_MAX_TX_REQUEST_PDUS];
} nfapi_nr_tx_data_request_t;

//  3.4.6.1 Downlink Data Errors
typedef enum {
	NFAPI_NR_DL_DATA_MSG_INVALID_STATE,
  NFAPI_NR_DL_DATA_MSG_INVALID_SFN,
  NFAPI_NR_DL_DATA_MSG_TX_ERR
} nfapi_nr_dl_data_errors_e;



// UL_indication
// 3.4.7 Rx_Data.indication
typedef struct {
  uint32_t Handle;
  uint16_t RNTI;
  uint8_t  HarqID;
  uint16_t PDU_Length;
  uint8_t  UL_CQI;
  uint16_t Timing_advance;//Timing advance 𝑇𝐴 measured for the UE [TS 38.213, Section 4.2] NTA_new = NTA_old + (TA − 31) ⋅ 16 ⋅ 64⁄2μ Value: 0 → 63 0xffff should be set if this field is invalid
  uint16_t RSSI;
  uint8_t *PDU; //MAC PDU
} nfapi_nr_rx_data_pdu_t;
//  table 3-93
typedef struct {
  nfapi_nr_p7_message_header_t header;
  uint16_t SFN;
  uint16_t Slot;
  uint16_t Number_of_PDUs;
  nfapi_nr_rx_data_pdu_t *pdu_list; 
} nfapi_nr_rx_data_indication_t;

//  3.4.8 CRC.indication
typedef struct {
  uint32_t Handle;
  uint16_t RNTI;
  uint8_t  HarqID;
  uint8_t  TbCrcStatus;
  uint16_t NumCb;//If CBG is not used this parameter can be set to zero. Otherwise the number of CBs in the TB. Value: 0->65535
  //! fixme
  uint8_t* CbCrcStatus;//cb_crc_status[ceil(NumCb/8)];
  uint8_t  UL_CQI;
  uint16_t Timing_advance;
  uint16_t RSSI;
} nfapi_nr_crc_t;
//  table 3-94
typedef struct {
  nfapi_nr_p7_message_header_t header;
  uint16_t SFN;
  uint16_t Slot;
  uint16_t NumCRCs;
  nfapi_nr_crc_t* crc_list;
} nfapi_nr_crc_indication_t;

//  3.4.9 UCI.indication
//  3.4.9.1 for PUSCH PDU
//  table 3-102
typedef struct {
  uint8_t  HarqCrc;
  uint16_t HarqBitLen;
  //! fixme
  uint8_t*  HarqPayload;//harq_payload[ceil(harq_bit_len)];
} nfapi_nr_harq_pdu_2_3_4_t;
//table 3-103
typedef struct {
  uint8_t  CsiPart1Crc;
  uint16_t CsiPart1BitLen;
  //! fixme
  uint8_t*  CsiPart1Payload;//uint8_t[ceil(csiPart1BitLen/8)]
} nfapi_nr_csi_part1_pdu_t;

//table 3-104
typedef struct
{
  uint8_t  CsiPart2Crc;
  uint16_t CsiPart2BitLen;
  //! fixme
  uint8_t*  CsiPart2Payload;//uint8_t[ceil(csiPart2BitLen/8)]
} nfapi_nr_csi_part2_pdu_t;

//  table 3-96
typedef struct
{
  uint8_t  pduBitmap;
  uint32_t Handle;
  uint16_t RNTI;
  uint8_t  UL_CQI;
  uint16_t Timing_advance;
  uint16_t RSSI;
  nfapi_nr_harq_pdu_2_3_4_t harq;// table 3-102
  nfapi_nr_csi_part1_pdu_t csi_part1;
  nfapi_nr_csi_part2_pdu_t csi_part2;

}nfapi_nr_uci_pusch_pdu_t;

/// 3.4.9.2 for PUCCH PDU Format 0/1
//  table 3-99
typedef struct {
  uint8_t SRindication;
  uint8_t SRconfidenceLevel;

} nfapi_nr_sr_pdu_0_1_t;
typedef struct{
  uint8_t  harq_value;
} nfapi_nr_harq_t;
//  table 3-100
typedef struct {
  uint8_t NumHarq;
  uint8_t HarqconfidenceLevel;
  nfapi_nr_harq_t* harq_list;

} nfapi_nr_harq_pdu_0_1_t;
//  table 3-97
typedef struct
{
  uint8_t  pduBitmap;
  uint32_t Handle;
  uint16_t RNTI;
  uint8_t  PucchFormat;
  uint8_t  UL_CQI;
  uint16_t Timing_advance;
  uint16_t RSSI;
  nfapi_nr_sr_pdu_0_1_t *sr;
  nfapi_nr_harq_pdu_0_1_t *harq;
}nfapi_nr_uci_pucch_pdu_format_0_1_t;

/// 3.4.9.3 for PUCCH PDU Format 2/3/4
// table 3-101
typedef struct {
  uint16_t SrBitLen;
  //! fixme
  uint8_t* SrPayload;//sr_payload[ceil(sr_bit_len/8)];
} nfapi_nr_sr_pdu_2_3_4_t;
//  table 3-98
typedef struct
{
  uint8_t  pduBitmap;
  uint32_t Handle;
  uint16_t RNTI;
  uint8_t  PucchFormat;//PUCCH format Value: 0 -> 2 0: PUCCH Format2 1: PUCCH Format3 2: PUCCH Format4
  uint8_t  UL_CQI;
  uint16_t Timing_advance;
  uint16_t RSSI;
  nfapi_nr_sr_pdu_2_3_4_t* sr;//  table 3-101
  nfapi_nr_harq_pdu_2_3_4_t* harq;// table 3-102
  nfapi_nr_csi_part1_pdu_t* csi_part1;//  table 3-103
  nfapi_nr_csi_part2_pdu_t* csi_part2;//  table 3-104

}nfapi_nr_uci_pucch_pdu_format_2_3_4_t;

typedef struct {
  uint16_t PDUType;  
  uint16_t PDUSize;
  union
  {
    nfapi_nr_uci_pusch_pdu_t pusch_pdu;
    nfapi_nr_uci_pucch_pdu_format_0_1_t pucch_pdu_format_0_1;
    nfapi_nr_uci_pucch_pdu_format_2_3_4_t pucch_pdu_format_2_3_4;
  };
} nfapi_nr_uci_t;
//  table 3-95
typedef struct {
  nfapi_nr_p7_message_header_t header;
  uint16_t SFN;
  uint16_t Slot;
  uint16_t NumUCIs;
  nfapi_nr_uci_t *uci_list;
} nfapi_nr_uci_indication_t;

//  3.4.10 SRS.indication

typedef struct{
  uint8_t  rbSNR;
}nfapi_nr_srs_indication_reported_symbol_resource_block_t;
typedef struct{
  uint16_t numRBs;
  nfapi_nr_srs_indication_reported_symbol_resource_block_t* rb_list;
}nfapi_nr_srs_indication_reported_symbol_t;
typedef struct{
  uint32_t Handle;
  uint16_t RNTI;
  uint16_t Timing_advance;
  uint8_t  numSymbols;
  uint8_t  wideBandSNR;
  uint8_t  numReportedSymbols;
  nfapi_nr_srs_indication_reported_symbol_t* reported_symbol_list;

}nfapi_nr_srs_indication_pdu_t;
// table 3-105
typedef struct {
  nfapi_nr_p7_message_header_t header;
  uint16_t SFN;
  uint16_t Slot;
  uint8_t Number_of_PDUs;
  nfapi_nr_srs_indication_pdu_t* pdu_list;
} nfapi_nr_srs_indication_t;


//  3.4.11  RACH.indication

typedef struct {
  uint8_t  preambleIndex;
  uint16_t timingadvance;
  uint32_t preamblePwr;
  uint8_t preambleSnr;
} nfapi_nr_prach_indication_preamble_t;

typedef struct {
  uint16_t handle;
  uint8_t  SymbolIndex;
  uint8_t  SlotIndex;
  uint8_t  raIndex;
  uint16_t  avgRssi;
  uint8_t  avgSnr;
  uint8_t  numPreambles;
  nfapi_nr_prach_indication_preamble_t* preamble_list;
}nfapi_nr_prach_indication_pdu_t;
//  table 3-106
typedef struct {
  nfapi_nr_p7_message_header_t header;
  uint16_t SFN;
  uint16_t Slot;
  uint8_t Number_of_PDUs;
  nfapi_nr_prach_indication_pdu_t* pdu_list;
} nfapi_nr_rach_indication_t;




