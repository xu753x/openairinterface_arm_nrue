
/*! \file openairinterface5g/openair1/PHY/BF/bf_struct.h
 * \brief merge ISIP beamforming and QR decomposer
 * \author NCTU OpinConnect Terng-Yin Hsu, Sendren Xu, WEI-YING LIN, Min-Hsun Wu
 * \email  a22490010@gmail.com
 * \date   25-9-2021
 * \version 1.0
 * \note
 * \warning
 */

#ifndef __BF_STRUCT_H
#define __BF_STRUCT_H


/* to set for UE capabilities */
#define MAX_NR_OF_SRS_RESOURCE_SET         (1)
#define MAX_NR_OF_SRS_RESOURCES_PER_SET    (1)

#define NR_MAX_SLOTS_PER_FRAME                   (160)                    /* number of slots per frame */


typedef enum {
  nr_FR1 = 0,
  nr_FR2
} nr_frequency_range_e;

typedef enum {
  port1           = 1,
  port2           = 2,
  port4           = 4
} nrof_Srs_Ports_t;

typedef enum {
  neitherHopping  = 0,
  groupHopping    = 1,
  sequenceHopping = 2
} groupOrSequenceHopping_t;

typedef enum {
  srs_sl1    = 0,
  srs_sl2    = 1,
  srs_sl4    = 2,
  srs_sl5    = 3,
  srs_sl8    = 4,
  srs_sl10   = 5,
  srs_sl16   = 6,
  srs_sl20   = 7,
  srs_sl32   = 8,
  srs_sl40   = 9,
  srs_sl64   = 10,
  srs_sl80   = 11,
  srs_sl160  = 12,
  srs_sl320  = 13,
  srs_sl640  = 14,
  srs_sl1280 = 15,
  srs_sl2560 = 16
} SRS_Periodicity_t;

typedef enum {
  aperiodic       = 0,
  semi_persistent = 1,
  periodic        = 2
} resourceType_t;

typedef enum {
  beamManagement     = 0,
  codebook           = 1,
  nonCodebook        = 2,
  antennaSwitching   = 3,
} SRS_resourceSet_usage_t;

typedef enum {
  ssb_Index          = 0,
  csi_RS_Index       = 1,
} pathlossReferenceRS_t;


typedef enum {
  sameAsFci2         = 0,
  separateClosedLoop = 1
} srs_PowerControlAdjustmentStates_t;






/// PRACH-ConfigInfo from 36.331 RRC spec
typedef struct {
  /// Parameter: prach-ConfigurationIndex, see TS 36.211 (5.7.1). \vr{[0..63]}
  uint8_t prach_ConfigIndex;
  /// Parameter: High-speed-flag, see TS 36.211 (5.7.2). \vr{[0..1]} 1 corresponds to Restricted set and 0 to Unrestricted set.
  uint8_t highSpeedFlag;
  /// Parameter: \f$N_\text{CS}\f$, see TS 36.211 (5.7.2). \vr{[0..15]}\n Refer to table 5.7.2-2 for preamble format 0..3 and to table 5.7.2-3 for preamble format 4.
  uint8_t zeroCorrelationZoneConfig;
  /// Parameter: prach-FrequencyOffset, see TS 36.211 (5.7.1). \vr{[0..94]}\n For TDD the value range is dependent on the value of \ref prach_ConfigIndex.
  uint8_t prach_FreqOffset;
} NR_PRACH_CONFIG_INFO;

/// PRACH-ConfigSIB or PRACH-Config
typedef struct {
  /// Parameter: RACH_ROOT_SEQUENCE, see TS 36.211 (5.7.1). \vr{[0..837]}
  uint16_t rootSequenceIndex;
  /// prach_Config_enabled=1 means enabled. \vr{[0..1]}
  uint8_t prach_Config_enabled;
  /// PRACH Configuration Information
  NR_PRACH_CONFIG_INFO prach_ConfigInfo;
} NR_PRACH_CONFIG_COMMON;

/// SRS_Resource of SRS_Config information element from 38.331 RRC specifications
typedef struct {
  /// \brief srs resource identity.
  uint8_t srs_ResourceId;
  /// \brief number of srs ports.
  nrof_Srs_Ports_t nrof_SrsPorts;
  /// \brief index of prts port index.
  uint8_t ptrs_PortIndex;
  /// \brief srs transmission comb see parameter SRS-TransmissionComb 38.214 section &6.2.1.
  uint8_t transmissionComb;
  /// \brief comb offset.
  uint8_t combOffset;
  /// \brief cyclic shift configuration - see parameter SRS-CyclicShiftConfig 38.214 &6.2.1.
  uint8_t cyclicShift;
  /// \brief OFDM symbol location of the SRS resource within a slot.
  // Corresponds to L1 parameter 'SRS-ResourceMapping' (see 38.214, section 6.2.1 and 38.211, section 6.4.1.4).
  // startPosition (SRSSymbolStartPosition = 0..5; "0" refers to the last symbol, "1" refers to the second last symbol.
  uint8_t resourceMapping_startPosition;
  /// \brief number of OFDM symbols (N = 1, 2 or 4 per SRS resource).
  uint8_t resourceMapping_nrofSymbols;
  /// \brief RepetitionFactor (r = 1, 2 or 4).
  uint8_t resourceMapping_repetitionFactor;
  /// \brief Parameter(s) defining frequency domain position and configurable shift to align SRS allocation to 4 PRB grid.
  // Corresponds to L1 parameter 'SRS-FreqDomainPosition' (see 38.214, section 6.2.1)
  uint8_t  freqDomainPosition;    // INTEGER (0..67),
  uint16_t freqDomainShift;       // INTEGER (0..268),
  /// \brief Includes  parameters capturing SRS frequency hopping
  // Corresponds to L1 parameter 'SRS-FreqHopping' (see 38.214, section 6.2.1)
  uint8_t freqHopping_c_SRS;      // INTEGER (0..63),
  uint8_t freqHopping_b_SRS;      // INTEGER (0..3),
  uint8_t freqHopping_b_hop;      // INTEGER (0..3)
  // Parameter(s) for configuring group or sequence hopping
  // Corresponds to L1 parameter 'SRS-GroupSequenceHopping' see 38.211
  groupOrSequenceHopping_t groupOrSequenceHopping;
  /// \brief Time domain behavior of SRS resource configuration.
  // Corresponds to L1 parameter 'SRS-ResourceConfigType' (see 38.214, section 6.2.1).
  // For codebook based uplink transmission, the network configures SRS resources in the same resource set with the same
  // time domain behavior on periodic, aperiodic and semi-persistent SRS
  SRS_Periodicity_t SRS_Periodicity;
  uint16_t          SRS_Offset;
  /// \brief Sequence ID used to initialize psedo random group and sequence hopping.
  // Corresponds to L1 parameter 'SRS-SequenceId' (see 38.214, section 6.2.1).
  uint16_t sequenceId;            // BIT STRING (SIZE (10))
  /// \brief Configuration of the spatial relation between a reference RS and the target SRS. Reference RS can be SSB/CSI-RS/SRS.
  // Corresponds to L1 parameter 'SRS-SpatialRelationInfo' (see 38.214, section 6.2.1)
  uint8_t spatialRelationInfo_ssb_Index;      // SSB-Index,
  uint8_t spatialRelationInfo_csi_RS_Index;   // NZP-CSI-RS-ResourceId,
  uint8_t spatialRelationInfo_srs_Id;         // SRS-ResourceId
} SRS_Resource_t;

// SRS_Config information element from 38.331 RRC specifications from 38.331 RRC specifications.
typedef struct {
  /// \brief The ID of this resource set. It is unique in the context of the BWP in which the parent SRS-Config is defined.
  uint8_t srs_ResourceSetId;
  /// \brief number of resources in the resource set
  uint8_t number_srs_Resource;
  /// \brief The IDs of the SRS-Resources used in this SRS-ResourceSet.
  /// in fact this is an array of pointers to resource structures
  SRS_Resource_t *p_srs_ResourceList[MAX_NR_OF_SRS_RESOURCES_PER_SET];
  resourceType_t resourceType;
  /// \brief The DCI "code point" upon which the UE shall transmit SRS according to this SRS resource set configuration.
  // Corresponds to L1 parameter 'AperiodicSRS-ResourceTrigger' (see 38.214, section 6.1.1.2)
  uint8_t aperiodicSRS_ResourceTrigger;		 // INTEGER (0..maxNrofSRS-TriggerStates-1) : [0:3]
  /// \brief ID of CSI-RS resource associated with this SRS resource set. (see 38.214, section 6.1.1.2).
  uint8_t NZP_CSI_RS_ResourceId;
  /// \brief An offset in number of slots between the triggering DCI and the actual transmission of this SRS-ResourceSet.
  // If the field is absent the UE applies no offset (value 0)
  uint8_t aperiodic_slotOffset; // INTEGER (1..8)
  /// \brief Indicates if the SRS resource set is used for beam management vs. used for either codebook based or non-codebook based transmission.
  // Corresponds to L1 parameter 'SRS-SetUse' (see 38.214, section 6.2.1)
  // FFS_CHECK: Isn't codebook/noncodebook already known from the ulTxConfig in the SRS-Config? If so, isn't the only distinction
  // in the set between BeamManagement, AtennaSwitching and "Other”? Or what happens if SRS-Config=Codebook but a Set=NonCodebook?
  SRS_resourceSet_usage_t usage;
  /// \brief alpha value for SRS power control. Corresponds to L1 parameter 'alpha-srs' (see 38.213, section 7.3).
  // When the field is absent the UE applies the value 1
  uint8_t alpha;
  /// \brief P0 value for SRS power control. The value is in dBm. Only even values (step size 2) are allowed.
  // Corresponds to L1 parameter 'p0-srs' (see 38.213, section 7.3)
  int8_t p0;          // INTEGER (-202..24)
  /// \brief A reference signal (e.g. a CSI-RS config or a SSblock) to be used for SRS path loss estimation.
  /// Corresponds to L1 parameter 'srs-pathlossReference-rs-config' (see 38.213, section 7.3)
  pathlossReferenceRS_t pathlossReferenceRS_t;
  uint8_t path_loss_SSB_Index;
  uint8_t path_loss_NZP_CSI_RS_ResourceId;
  /// \brief Indicates whether hsrs,c(i) = fc(i,1) or hsrs,c(i) = fc(i,2) (if twoPUSCH-PC-AdjustmentStates are configured)
  /// or separate close loop is configured for SRS. This parameter is applicable only for Uls on which UE also transmits PUSCH.
  /// If absent or release, the UE applies the value sameAs-Fci1
  /// Corresponds to L1 parameter 'srs-pcadjustment-state-config' (see 38.213, section 7.3)
  srs_PowerControlAdjustmentStates_t srs_PowerControlAdjustmentStates;
} SRS_ResourceSet_t;

typedef struct {
  uint8_t active_srs_Resource_Set;  /* ue implementation specific */
  uint8_t number_srs_Resource_Set;  /* ue implementation specific */
  SRS_ResourceSet_t *p_SRS_ResourceSetList[MAX_NR_OF_SRS_RESOURCE_SET]; /* ue implementation specific */
} SRS_NR;

typedef enum{
  nr_ssb_type_A = 0,
  nr_ssb_type_B,
  nr_ssb_type_C,
  nr_ssb_type_D,
  nr_ssb_type_E
} nr_ssb_type_e;

typedef struct decoder_node_t_s {
  struct decoder_node_t_s *left;
  struct decoder_node_t_s *right;
  int level;
  int leaf;
  int Nv;
  int first_leaf_index;
  int all_frozen;
  int16_t *alpha;
  int16_t *beta;
} decoder_node_t;


typedef struct decoder_tree_t_s {
  decoder_node_t *root;
  int num_nodes;
} decoder_tree_t;


struct nrPolar_params {
  //messageType: 0=PBCH, 1=DCI, -1=UCI
  int idx; //idx = (messageType * messageLength * aggregation_prime);
  struct nrPolar_params *nextPtr;

  uint8_t n_max;
  uint8_t i_il;
  uint8_t i_seg;
  uint8_t n_pc;
  uint8_t n_pc_wm;
  uint8_t i_bil;
  uint16_t payloadBits;
  uint16_t encoderLength;
  uint8_t crcParityBits;
  uint8_t crcCorrectionBits;
  uint16_t K;
  uint16_t N;
  uint8_t n;
  uint32_t crcBit;

  uint16_t *interleaving_pattern;
  uint16_t *deinterleaving_pattern;
  uint16_t *rate_matching_pattern;
  const uint16_t *Q_0_Nminus1;
  int16_t *Q_I_N;
  int16_t *Q_F_N;
  int16_t *Q_PC_N;
  uint8_t *information_bit_pattern;
  uint16_t *channel_interleaver_pattern;
  //uint32_t crc_polynomial;

  uint8_t **crc_generator_matrix; //G_P
  uint8_t **G_N;
  uint64_t **G_N_tab;
  int groupsize;
  int *rm_tab;
  uint64_t cprime_tab0[32][256];
  uint64_t cprime_tab1[32][256];
  uint64_t B_tab0[32][256];
  uint64_t B_tab1[32][256];
  uint8_t **extended_crc_generator_matrix;
  //lowercase: bits, Uppercase: Bits stored in bytes
  //polar_encoder vectors
  uint8_t *nr_polar_crc;
  uint8_t *nr_polar_aPrime;
  uint8_t *nr_polar_APrime;
  uint8_t *nr_polar_D;
  uint8_t *nr_polar_E;

  //Polar Coding vectors
  uint8_t *nr_polar_A;
  uint8_t *nr_polar_CPrime;
  uint8_t *nr_polar_B;
  uint8_t *nr_polar_U;

  decoder_tree_t tree;
} __attribute__ ((__packed__));
typedef struct nrPolar_params t_nrPolar_params;

typedef struct {
	uint16_t tag;
	uint16_t length;
} nfapi_tl_t;

// Generic strucutre for single tlv value.
typedef struct {
	nfapi_tl_t tl;
	uint16_t value;
} nfapi_uint16_tlv_t;

typedef struct {
	nfapi_tl_t tl;
	uint64_t value;
} nfapi_uint64_tlv_t;


typedef struct {
  nfapi_uint16_tlv_t  dl_carrier_bandwidth;
  nfapi_uint16_tlv_t  ul_carrier_bandwidth;
  nfapi_uint16_tlv_t  dl_bwp_subcarrierspacing;
  nfapi_uint16_tlv_t  ul_bwp_subcarrierspacing;
  nfapi_uint16_tlv_t  dl_locationandbandwidth;
  nfapi_uint16_tlv_t  ul_locationandbandwidth;
  nfapi_uint16_tlv_t  dl_absolutefrequencypointA;
  nfapi_uint16_tlv_t  ul_absolutefrequencypointA;
  nfapi_uint16_tlv_t  dl_offsettocarrier;
  nfapi_uint16_tlv_t  ul_offsettocarrier;
  nfapi_uint16_tlv_t  dl_subcarrierspacing;
  nfapi_uint16_tlv_t  ul_subcarrierspacing;
  nfapi_uint16_tlv_t  dl_specificcarrier_k0;
  nfapi_uint16_tlv_t  ul_specificcarrier_k0;
  nfapi_uint16_tlv_t  NIA_subcarrierspacing;
} nfapi_nr_rf_config_t;

typedef struct {
  nfapi_uint16_tlv_t  physical_cell_id;
  nfapi_uint16_tlv_t  half_frame_index;
  nfapi_uint16_tlv_t  ssb_subcarrier_offset;
  nfapi_uint16_tlv_t  ssb_sib1_position_in_burst; // in sib1
  nfapi_uint64_tlv_t  ssb_scg_position_in_burst;  // in servingcellconfigcommon 
  nfapi_uint16_tlv_t  ssb_periodicity;
  nfapi_uint16_tlv_t  ss_pbch_block_power;
  nfapi_uint16_tlv_t  n_ssb_crb;
  nfapi_uint16_tlv_t  rmsi_pdcch_config;
} nfapi_nr_sch_config_t;

typedef enum {TDD=1,FDD=0} lte_frame_type_t;


typedef enum {EXTENDED=1,NORMAL=0} lte_prefix_type_t;

typedef enum {
  ms0p5    = 500,                 /* duration is given in microsecond */
  ms0p625  = 625,
  ms1      = 1000,
  ms1p25   = 1250,
  ms2      = 2000,
  ms2p5    = 2500,
  ms5      = 5000,
  ms10     = 10000,
} dl_UL_TransmissionPeriodicity_t;


typedef struct {
  /// Reference SCS used to determine the time domain boundaries in the UL-DL pattern which must be common across all subcarrier specific
  /// virtual carriers, i.e., independent of the actual subcarrier spacing using for data transmission.
  /// Only the values 15 or 30 kHz  (<6GHz), 60 or 120 kHz (>6GHz) are applicable.
  /// Corresponds to L1 parameter 'reference-SCS' (see 38.211, section FFS_Section)
  /// \ subcarrier spacing
  uint8_t referenceSubcarrierSpacing;
  /// \ Periodicity of the DL-UL pattern. Corresponds to L1 parameter 'DL-UL-transmission-periodicity' (see 38.211, section FFS_Section)
  dl_UL_TransmissionPeriodicity_t dl_UL_TransmissionPeriodicity;
  /// \ Number of consecutive full DL slots at the beginning of each DL-UL pattern.
  /// Corresponds to L1 parameter 'number-of-DL-slots' (see 38.211, Table 4.3.2-1)
  uint8_t nrofDownlinkSlots;
  /// \ Number of consecutive DL symbols in the beginning of the slot following the last full DL slot (as derived from nrofDownlinkSlots).
  /// If the field is absent or released, there is no partial-downlink slot.
  /// Corresponds to L1 parameter 'number-of-DL-symbols-common' (see 38.211, section FFS_Section).
  uint8_t nrofDownlinkSymbols;
  /// \ Number of consecutive full UL slots at the end of each DL-UL pattern.
  /// Corresponds to L1 parameter 'number-of-UL-slots' (see 38.211, Table 4.3.2-1)
  uint8_t nrofUplinkSlots;
  /// \ Number of consecutive UL symbols in the end of the slot preceding the first full UL slot (as derived from nrofUplinkSlots).
  /// If the field is absent or released, there is no partial-uplink slot.
  /// Corresponds to L1 parameter 'number-of-UL-symbols-common' (see 38.211, section FFS_Section)
  uint8_t nrofUplinkSymbols;
  /// \ for setting a sequence
  struct TDD_UL_DL_configCommon_t *p_next_TDD_UL_DL_configCommon_t;
} TDD_UL_DL_configCommon_t;

typedef struct {
  /// \ Identifies a slot within a dl-UL-TransmissionPeriodicity (given in tdd-UL-DL-configurationCommon)
  uint16_t slotIndex;
  /// \ The direction (downlink or uplink) for the symbols in this slot. "allDownlink" indicates that all symbols in this slot are used
  /// for downlink; "allUplink" indicates that all symbols in this slot are used for uplink; "explicit" indicates explicitly how many symbols
  /// in the beginning and end of this slot are allocated to downlink and uplink, respectively.
  /// Number of consecutive DL symbols in the beginning of the slot identified by slotIndex.
  /// If the field is absent the UE assumes that there are no leading DL symbols.
  /// Corresponds to L1 parameter 'number-of-DL-symbols-dedicated' (see 38.211, section FFS_Section)
  uint16_t nrofDownlinkSymbols;
  /// Number of consecutive UL symbols in the end of the slot identified by slotIndex.
  /// If the field is absent the UE assumes that there are no trailing UL symbols.
  /// Corresponds to L1 parameter 'number-of-UL-symbols-dedicated' (see 38.211, section FFS_Section)
  uint16_t nrofUplinkSymbols;
  /// \ for setting a sequence
  struct TDD_UL_DL_SlotConfig_t *p_next_TDD_UL_DL_SlotConfig;
} TDD_UL_DL_SlotConfig_t;


typedef struct NR_DL_FRAME_PARMS {
  /// frequency range
  nr_frequency_range_e freq_range;
  /// Placeholder to replace overlapping fields below
  nfapi_nr_rf_config_t rf_config;
  /// Placeholder to replace SSB overlapping fields below
  nfapi_nr_sch_config_t sch_config;
  /// Number of resource blocks (RB) in DL
  int N_RB_DL;
  /// Number of resource blocks (RB) in UL
  int N_RB_UL;
  ///  total Number of Resource Block Groups: this is ceil(N_PRB/P)
  uint8_t N_RBG;
  /// Total Number of Resource Block Groups SubSets: this is P
  uint8_t N_RBGS;
  /// EUTRA Band
  uint16_t eutra_band;
  /// DL carrier frequency
  uint32_t dl_CarrierFreq;
  /// UL carrier frequency
  uint32_t ul_CarrierFreq;
  /// TX attenuation
  uint32_t att_tx;
  /// RX attenuation
  uint32_t att_rx;
  ///  total Number of Resource Block Groups: this is ceil(N_PRB/P)
  /// Frame type (0 FDD, 1 TDD)
  lte_frame_type_t frame_type;
  /// TDD subframe assignment (0-7) (default = 3) (254=RX only, 255=TX only)
  uint8_t tdd_config;
  /// TDD S-subframe configuration (0-9)
  /// Cell ID
  uint16_t Nid_cell;
  uint32_t subcarrier_spacing;
  /// 3/4 sampling
  uint8_t threequarter_fs;
  /// Size of FFT
  uint16_t ofdm_symbol_size;
  /// Number of prefix samples in all but first symbol of slot
  uint16_t nb_prefix_samples;
  /// Number of prefix samples in first symbol of slot
  uint16_t nb_prefix_samples0;
  /// Carrier offset in FFT buffer for first RE in PRB0
  uint16_t first_carrier_offset;
  /// Number of OFDM/SC-FDMA symbols in one slot
  uint16_t symbols_per_slot;
  /// Number of slots per subframe
  uint16_t slots_per_subframe;
    /// Number of slots per frame
  uint16_t slots_per_frame;
  /// Number of samples in a subframe
  uint32_t samples_per_subframe;
  /// Number of samples in a slot
  uint32_t samples_per_slot;
  /// Number of OFDM/SC-FDMA symbols in one subframe (to be modified to account for potential different in UL/DL)
  uint16_t symbols_per_tti;
  /// Number of samples in a radio frame
  uint32_t samples_per_frame;
  /// Number of samples in a subframe without CP
  uint32_t samples_per_subframe_wCP;
  /// Number of samples in a slot without CP
  uint32_t samples_per_slot_wCP;
  /// Number of samples in a radio frame without CP
  uint32_t samples_per_frame_wCP;
  /// Number of samples in a tti (same as subrame in LTE, slot in NR)
  uint32_t samples_per_tti;
  /// NR numerology index [0..5] as specified in 38.211 Section 4 (mu). 0=15khZ SCS, 1=30khZ, 2=60kHz, etc
  uint8_t numerology_index;
  /// NR number of ttis per subframe deduced from numerology (cf 38.211): 1, 2, 4, 8(not supported),16(not supported),32(not supported)
  uint8_t ttis_per_subframe;
  /// NR number of slots per tti . Assumption only 2 Slot per TTI is supported (Slot Config 1 in 38.211)
  uint8_t slots_per_tti;
//#endif
  /// Number of Physical transmit antennas in node
  uint8_t nb_antennas_tx;
  /// Number of Receive antennas in node
  uint8_t nb_antennas_rx;
  /// Number of common transmit antenna ports in eNodeB (1 or 2)
  uint8_t nb_antenna_ports_eNB;
  /// PRACH_CONFIG
  NR_PRACH_CONFIG_COMMON prach_config_common;
  /// Cyclic Prefix for DL (0=Normal CP, 1=Extended CP)
  lte_prefix_type_t Ncp;
  /// shift of pilot position in one RB
  uint8_t nushift;
  /// SRS configuration from TS 38.331 RRC
  SRS_NR srs_nr;

  /// for NR TDD management
  TDD_UL_DL_configCommon_t  *p_tdd_UL_DL_Configuration;

  TDD_UL_DL_configCommon_t  *p_tdd_UL_DL_ConfigurationCommon2;

  TDD_UL_DL_SlotConfig_t    *p_TDD_UL_DL_ConfigDedicated;

  /// TDD configuration
  uint16_t tdd_uplink_nr[2*NR_MAX_SLOTS_PER_FRAME]; /* this is a bitmap of symbol of each slot given for 2 frames */

  //SSB related params
  /// Start in Subcarrier index of the SSB block
  uint16_t ssb_start_subcarrier;
  /// SSB type
  nr_ssb_type_e ssb_type;
  /// Max number of SSB in frame
  uint8_t Lmax;
  /// SS block pattern (max 64 ssb, each bit is on/off ssb)
  uint64_t L_ssb;
  /// Total number of SSB transmitted
  uint8_t N_ssb;
  /// PBCH polar encoder params
  t_nrPolar_params pbch_polar_params;

} NR_DL_FRAME_PARMS;






#endif