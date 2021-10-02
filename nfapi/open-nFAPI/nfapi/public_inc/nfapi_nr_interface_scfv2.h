/*! \file nfapi/open-nFAPI/nfapi/public_inc/nfapi_nr_interface_scfv2.h
 * \brief Create FAPI(P7) data structure
 * \author NCTU OpinConnect Terng-Yin Hsu, Sendren Xu, WEI-YING LIN, Hong-Ming Huang
 * \email  a22490010@gmail.com
 * \date   2-10-2021
 * \version 1.0
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

// Section 3.3 Configuration Messages
/// 3.3.1 PARAM



// Section 3.4 Slot messages

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


typedef struct {
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

typedef struct {
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
  uint16_t SFN;
  uint16_t Slot;
  uint8_t Number_of_PDUs;
  nfapi_nr_prach_indication_pdu_t* pdu_list;
} nfapi_nr_rach_indication_t;




