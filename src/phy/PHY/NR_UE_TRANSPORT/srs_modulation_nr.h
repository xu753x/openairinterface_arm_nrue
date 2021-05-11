



#ifndef SRS_MODULATION_NR_H
#define SRS_MODULATION_NR_H

#ifdef DEFINE_VARIABLES_SRS_MODULATION_NR_H
#define EXTERN
#define INIT_VARIABLES_SRS_MODULATION_NR_H
#else
#define EXTERN extern
#undef INIT_VARIABLES_SRS_MODULATION_NR_H
#endif


/************** DEFINE ********************************************/

#define C_SRS_NUMBER       (64)
#define B_SRS_NUMBER       (4)

/************** VARIABLES *****************************************/

/* TS 38.211 Table 6.4.1.4.3-1: SRS bandwidth configuration */
EXTERN const unsigned short srs_bandwidth_config[C_SRS_NUMBER][B_SRS_NUMBER][2]
#ifdef INIT_VARIABLES_SRS_MODULATION_NR_H
= {
/*           B_SRS = 0    B_SRS = 1   B_SRS = 2    B_SRS = 3     */
/* C SRS   m_srs0  N_0  m_srs1 N_1  m_srs2   N_2 m_srs3   N_3    */
/* 0  */   { {   4,  1 },{   4,  1 },{   4,  1  },{   4,  1  } },
/* 1  */   { {   8,  1 },{   4,  2 },{   4,  1  },{   4,  1  } },
/* 2  */   { {  12,  1 },{   4,  3 },{   4,  1  },{   4,  1  } },
/* 3  */   { {  16,  1 },{   4,  4 },{   4,  1  },{   4,  1  } },
/* 4  */   { {  16,  1 },{   8,  2 },{   4,  2  },{   4,  1  } },
/* 5  */   { {  20,  1 },{   4,  5 },{   4,  1  },{   4,  1  } },
/* 6  */   { {  24,  1 },{   4,  6 },{   4,  1  },{   4,  1  } },
/* 7  */   { {  24,  1 },{  12,  2 },{   4,  3  },{   4,  1  } },
/* 8  */   { {  28,  1 },{   4,  7 },{   4,  1  },{   4,  1  } },
/* 9  */   { {  32,  1 },{  16,  2 },{   8,  2  },{   4,  2  } },
/* 10 */   { {  36,  1 },{  12,  3 },{   4,  3  },{   4,  1  } },
/* 11 */   { {  40,  1 },{  20,  2 },{   4,  5  },{   4,  1  } },
/* 12 */   { {  48,  1 },{  16,  3 },{   8,  2  },{   4,  2  } },
/* 13 */   { {  48,  1 },{  24,  2 },{  12,  2  },{   4,  3  } },
/* 14 */   { {  52,  1 },{   4, 13 },{   4,  1  },{   4,  1  } },
/* 15 */   { {  56,  1 },{  28,  2 },{   4,  7  },{   4,  1  } },
/* 16 */   { {  60,  1 },{  20,  3 },{   4,  5  },{   4,  1  } },
/* 17 */   { {  64,  1 },{  32,  2 },{  16,  2  },{   4,  4  } },
/* 18 */   { {  72,  1 },{  24,  3 },{  12,  2  },{   4,  3  } },
/* 19 */   { {  72,  1 },{  36,  2 },{  12,  3  },{   4,  3  } },
/* 20 */   { {  76,  1 },{   4, 19 },{   4,  1  },{   4,  1  } },
/* 21 */   { {  80,  1 },{  40,  2 },{  20,  2  },{   4,  5  } },
/* 22 */   { {  88,  1 },{  44,  2 },{   4, 11  },{   4,  1  } },
/* 23 */   { {  96,  1 },{  32,  3 },{  16,  2  },{   4,  4  } },
/* 24 */   { {  96,  1 },{  48,  2 },{  24,  2  },{   4,  6  } },
/* 25 */   { { 104,  1 },{  52,  2 },{   4, 13  },{   4,  1  } },
/* 26 */   { { 112,  1 },{  56,  2 },{  28,  2  },{   4,  7  } },
/* 27 */   { { 120,  1 },{  60,  2 },{  20,  3  },{   4,  5  } },
/* 28 */   { { 120,  1 },{  40,  3 },{   8,  5  },{   4,  2  } },
/* 29 */   { { 120,  1 },{  24,  5 },{  12,  2  },{   4,  3  } },
/* 30 */   { { 128,  1 },{  64,  2 },{  32,  2  },{   4,  8  } },
/* 31 */   { { 128,  1 },{  64,  2 },{  16,  4  },{   4,  4  } },
/* 32 */   { { 128,  1 },{  16,  8 },{   8,  2  },{   4,  2  } },
/* 33 */   { { 132,  1 },{  44,  3 },{   4, 11  },{   4,  1  } },
/* 34 */   { { 136,  1 },{  68,  2 },{   4, 17  },{   4,  1  } },
/* 35 */   { { 144,  1 },{  72,  2 },{  36,  2  },{   4,  9  } },
/* 36 */   { { 144,  1 },{  48,  3 },{  24,  2  },{  12,  2  } },
/* 37 */   { { 144,  1 },{  48,  3 },{  16,  3  },{   4,  4  } },
/* 38 */   { { 144,  1 },{  16,  9 },{   8,  2  },{   4,  2  } },
/* 39 */   { { 152,  1 },{  76,  2 },{   4, 19  },{   4,  1  } },
/* 40 */   { { 160,  1 },{  80,  2 },{  40,  2  },{   4, 10  } },
/* 41 */   { { 160,  1 },{  80,  2 },{  20,  4  },{   4,  5  } },
/* 42 */   { { 160,  1 },{  32,  5 },{  16,  2  },{   4,  4  } },
/* 43 */   { { 168,  1 },{  84,  2 },{  28,  3  },{   4,  7  } },
/* 44 */   { { 176,  1 },{  88,  2 },{  44,  2  },{   4, 11  } },
/* 45 */   { { 184,  1 },{  92,  2 },{   4, 23  },{   4,  1  } },
/* 46 */   { { 192,  1 },{  96,  2 },{  48,  2  },{   4, 12  } },
/* 47 */   { { 192,  1 },{  96,  2 },{  24,  4  },{   4,  6  } },
/* 48 */   { { 192,  1 },{  64,  3 },{  16,  4  },{   4,  4  } },
/* 49 */   { { 192,  1 },{  24,  8 },{   8,  3  },{   4,  2  } },
/* 50 */   { { 208,  1 },{ 104,  2 },{  52,  2  },{   4, 13  } },
/* 51 */   { { 216,  1 },{ 108,  2 },{  36,  3  },{   4,  9  } },
/* 52 */   { { 224,  1 },{ 112,  2 },{  56,  2  },{   4, 14  } },
/* 53 */   { { 240,  1 },{ 120,  2 },{  60,  2  },{   4, 15  } },
/* 54 */   { { 240,  1 },{  80,  3 },{  20,  4  },{   4,  5  } },
/* 55 */   { { 240,  1 },{  48,  5 },{  16,  3  },{   8,  2  } },
/* 56 */   { { 240,  1 },{  24, 10 },{  12,  2  },{   4,  3  } },
/* 57 */   { { 256,  1 },{ 128,  2 },{  64,  2  },{   4, 16  } },
/* 58 */   { { 256,  1 },{ 128,  2 },{  32,  4  },{   4,  8  } },
/* 59 */   { { 256,  1 },{  16, 16 },{   8,  2  },{   4,  2  } },
/* 60 */   { { 264,  1 },{ 132,  2 },{  44,  3  },{   4, 11  } },
/* 61 */   { { 272,  1 },{ 136,  2 },{  68,  2  },{   4, 17  } },
/* 62 */   { { 272,  1 },{  68,  4 },{   4, 17  },{   4,  1  } },
/* 63 */   { { 272,  1 },{  16, 17 },{   8,  2  },{   4,  2  } },
}
#endif
;

#define SRS_PERIODICITY                 (17)

EXTERN const uint16_t srs_periodicity[SRS_PERIODICITY]
#ifdef INIT_VARIABLES_SRS_MODULATION_NR_H
= { 1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64, 80, 160, 320, 640, 1280, 2560}
#endif
;

/*************** FUNCTIONS *****************************************/

/** \brief This function generates the sounding reference symbol (SRS) for the uplink according to 38.211 6.4.1.4 Sounding reference signal
    @param frame_parms NR DL Frame parameters
    @param txdataF pointer to the frequency domain TX signal
    @param amp amplitude of generated signal
    @param proc pointer to the transmit parameters
    @returns 0 on success -1 on error with message */

int32_t generate_srs_nr(SRS_ResourceSet_t *p_srs_resource_set,
                        NR_DL_FRAME_PARMS *frame_parms,
                        int32_t *txptr,
                        int16_t amp,
                        UE_nr_rxtx_proc_t *proc);

/** \brief This function checks for periodic srs if srs should be transmitted in this slot
 *  @param p_SRS_Resource pointer to active resource
    @param frame_parms NR DL Frame parameters
    @param txdataF pointer to the frequency domain TX signal
    @returns 0 if srs should be transmitted -1 on error with message */

int is_srs_period_nr(SRS_Resource_t *p_SRS_Resource,
                     NR_DL_FRAME_PARMS *frame_parms,
                     int frame_tx, int slot_tx);
/** \brief This function processes srs configuration
 *  @param ue context
    @param rxtx context
    @param current eNB identifier
    @returns 0 if srs is transmitted -1 otherwise */

int ue_srs_procedure_nr(PHY_VARS_NR_UE *ue, UE_nr_rxtx_proc_t *proc, uint8_t eNB_id);

#undef EXTERN
#undef INIT_VARIABLES_SRS_MODULATION_NR_H

#endif /* SRS_MODULATION_NR_H */
