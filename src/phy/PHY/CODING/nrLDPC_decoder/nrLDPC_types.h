



#ifndef __NR_LDPC_TYPES__H__
#define __NR_LDPC_TYPES__H__

#include "PHY/TOOLS/time_meas.h"
#include "nrLDPCdecoder_defs.h"
// ==============================================================================
// TYPES

/**
   Structure containing the pointers to the LUTs.
 */
typedef struct nrLDPC_lut {
    const uint32_t* startAddrCnGroups; /**< Start addresses for CN groups in CN processing buffer */
    const uint8_t*  numCnInCnGroups; /**< Number of CNs in every CN group */
    const uint8_t*  numBnInBnGroups; /**< Number of CNs in every BN group */
    const uint32_t* startAddrBnGroups; /**< Start addresses for BN groups in BN processing buffer  */
    const uint16_t* startAddrBnGroupsLlr; /**< Start addresses for BN groups in LLR processing buffer  */
    const uint16_t** circShift[NR_LDPC_NUM_CN_GROUPS_BG1]; /**< LUT for circular shift values for all CN groups and Zs */
    const uint32_t** startAddrBnProcBuf[NR_LDPC_NUM_CN_GROUPS_BG1]; /**< LUT of start addresses of CN groups in BN proc buffer */
    const uint8_t**  bnPosBnProcBuf[NR_LDPC_NUM_CN_GROUPS_BG1]; /**< LUT of BN positions in BG for CN groups */
    const uint16_t* llr2llrProcBufAddr; /**< LUT for transferring input LLRs to LLR processing buffer */
    const uint8_t*  llr2llrProcBufBnPos; /**< LUT BN position in BG */
    const uint8_t** posBnInCnProcBuf[NR_LDPC_NUM_CN_GROUPS_BG1]; /**< LUT for llr2cnProcBuf */
} t_nrLDPC_lut;

/**
   Enum with possible LDPC output formats.
 */
typedef enum nrLDPC_outMode {
    nrLDPC_outMode_BIT, /**< 32 bits per uint32_t output */
    nrLDPC_outMode_BITINT8, /**< 1 bit per int8_t output */
    nrLDPC_outMode_LLRINT8 /**< Single LLR value per int8_t output */
} e_nrLDPC_outMode;

/**
   Structure containing LDPC decoder parameters.
 */
typedef struct nrLDPC_dec_params {
    uint8_t BG; /**< Base graph */
    uint16_t Z; /**< Lifting size */
    uint8_t R; /**< Decoding rate: Format 15,13,... for code rates 1/5, 1/3,... */
    uint8_t numMaxIter; /**< Maximum number of iterations */
    e_nrLDPC_outMode outMode; /**< Output format */
} t_nrLDPC_dec_params;

/**
   Structure containing LDPC decoder processing time statistics.
 */
typedef struct nrLDPC_time_stats {
    time_stats_t llr2llrProcBuf; /**< Statistics for function llr2llrProcBuf */
    time_stats_t llr2CnProcBuf; /**< Statistics for function llr2CnProcBuf */
    time_stats_t cnProc; /**< Statistics for function cnProc */
    time_stats_t cnProcPc; /**< Statistics for function cnProcPc */
    time_stats_t bnProcPc; /**< Statistics for function bnProcPc */
    time_stats_t bnProc; /**< Statistics for function bnProc */
    time_stats_t cn2bnProcBuf; /**< Statistics for function cn2bnProcBuf */
    time_stats_t bn2cnProcBuf; /**< Statistics for function bn2cnProcBuf */
    time_stats_t llrRes2llrOut; /**< Statistics for function llrRes2llrOut */
    time_stats_t llr2bit; /**< Statistics for function llr2bit */
    time_stats_t total; /**< Statistics for total processing time */
} t_nrLDPC_time_stats;

/**
   Structure containing the processing buffers
 */
typedef struct nrLDPC_procBuf {
    int8_t* cnProcBuf; /**< CN processing buffer */
    int8_t* cnProcBufRes; /**< Buffer for CN processing results */
    int8_t* bnProcBuf; /**< BN processing buffer */
    int8_t* bnProcBufRes; /**< Buffer for BN processing results */
    int8_t* llrRes; /**< Buffer for LLR results */
    int8_t* llrProcBuf; /**< LLR processing buffer */
} t_nrLDPC_procBuf;



#endif
