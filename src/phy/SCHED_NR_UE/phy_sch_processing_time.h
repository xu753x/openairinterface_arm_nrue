



/* TS 38.214 Table 5.3-1: PDSCH processing time for PDSCH processing capability 1
//   corresponding to the PDSCH decoding time N_1 [symbols]
// where mu corresponds to the one of {mu_PDCCH, mu_PDSCH, mu_UL}
//   resulting with the largest T_proc_1
// where mu_PDCCH is the SCS of PDCCH scheduling PDSCH
//   mu_PDSCH is the SCS of the scheduled PDSCH
//   mu_UL is the SCS of the UL channel with which the HARQ-ACK is to be transmitted
// column A is N_1 corresponding to dmrs-AdditionalPosition pos0 in DMRS-DownlinkConfig
//   in both dmrs-DownlinkForPDSCH-MappingTypeA and dmrs-DownlinkForPDSCH-MappingTypeB
// column B is N_1 corresponding to corresponds to dmrs-AdditionalPosition !0
//   in DMRS-DownlinkConfig in both dmrs-DownlinkForPDSCH-MappingTypeA,
//   dmrs-DownlinkForPDSCH-MappingTypeB or if the higher layer param is not configured
//   when PDSCH DM-RS position l1 for the additional DM-RS is l1 = 1,2
// column C is N_1 corresponding to corresponds to dmrs-AdditionalPosition !0
//   in DMRS-DownlinkConfig in both dmrs-DownlinkForPDSCH-MappingTypeA,
//   dmrs-DownlinkForPDSCH-MappingTypeB or if the higher layer param is not configured
//   when PDSCH DM-RS position l1 for the additional DM-RS is != 1,2

*/
int8_t pdsch_N_1_capability_1[4][4] = {
/* mu      A            B            C   */
{  0,      8,           14,          13  },
{  1,      10,          13,          13  },
{  2,      17,          20,          20  },
{  3,      20,          24,          24  },
};

/* TS 38.214 Table 5.3-2: PDSCH processing time for PDSCH processing capability 2
//   corresponding to the PDSCH decoding time N_1 [symbols]
// where mu corresponds to the one of {mu_PDCCH, mu_PDSCH, mu_UL}
//   resulting with the largest T_proc_1
// where mu_PDCCH is the SCS of PDCCH scheduling PDSCH
//   mu_PDSCH is the SCS of the scheduled PDSCH
//   mu_UL is the SCS of the UL channel with which the HARQ-ACK is to be transmitted
// column A is N_1 corresponding to dmrs-AdditionalPosition pos0 in DMRS-DownlinkConfig in both
//   dmrs-DownlinkForPDSCH-MappingTypeA and dmrs-DownlinkForPDSCH-MappingTypeB
// mu == 2 is for FR1 only
*/
float pdsch_N_1_capability_2[3][2] = {
/* mu      A */   
{  0,      3   },
{  1,      4.5 },
{  2,      9   },
};

/* TS 38.214 Table 6.4-1: PUSCH preparation time for PUSCH timing capability 1
//   corresponding to the PUSCH preparation time N_2 [symbols]
// where mu corresponds to the one of {mu_DL, mu_UL}
//   resulting with the largest T_proc_2
// where mu_DL is the SCS with which the PDCCH
//   carrying the DCI scheduling the PUSCH was transmitted
//   mu_UL is the SCS of the UL channel with which PUSCH to be transmitted
*/
int8_t pusch_N_2_timing_capability_1[4][2] = {
/* mu      N_2   */
{  0,      10 },
{  1,      12 },
{  2,      23 },
{  3,      36 },
};

/* TS 38.214 Table 6.4-2: PUSCH preparation time for PUSCH timing capability 2
//   corresponding to the PUSCH preparation time N_2 [symbols]
// where mu corresponds to the one of {mu_DL, mu_UL}
//   resulting with the largest T_proc_2
// where mu_DL is the SCS with which the PDCCH
//   carrying the DCI scheduling the PUSCH was transmitted
//   mu_UL is the SCS of the UL channel with which PUSCH to be transmitted
// mu == 2 is for FR1 only
*/
float pusch_N_2_timing_capability_2[3][2] = {
/* mu      N_2   */
{  0,      5   },
{  1,      5.5 },
{  2,      11  },
};