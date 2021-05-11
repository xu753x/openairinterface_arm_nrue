



#ifndef __NR_POLAR_DCI_DEFS__H__
#define __NR_POLAR_DCI_DEFS__H__

#define NR_POLAR_DCI_MESSAGE_TYPE 1 //int8_t
#define NR_POLAR_DCI_CRC_PARITY_BITS 24
#define NR_POLAR_DCI_CRC_ERROR_CORRECTION_BITS 3

//Sec. 7.3.3: Channel Coding
#define NR_POLAR_DCI_N_MAX 9   //uint8_t
#define NR_POLAR_DCI_I_IL 1    //uint8_t
#define NR_POLAR_DCI_I_SEG 0   //uint8_t
#define NR_POLAR_DCI_N_PC 0    //uint8_t
#define NR_POLAR_DCI_N_PC_WM 0 //uint8_t

//Sec. 7.3.4: Rate Matching
#define NR_POLAR_DCI_I_BIL 0 //uint8_t

#endif
