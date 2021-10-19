
/*! \file openairinterface5g/openair1/PHY/BF/cu_function.h
 * \brief merge ISIP beamforming and MVDR algorithm
 * \author NCTU OpinConnect Terng-Yin Hsu, Sendren Xu, WEI-YING LIN, Min-Hsun Wu
 * \email  a22490010@gmail.com
 * \date   19-10-2021
 * \version 1.2
 * \note
 * \warning
 */

#ifndef __CU_FUNCTION_H
#define __CU_FUNCTION_H

#ifdef __cplusplus
extern "C" {  
#endif

    int global_antenna;
    int global_QR_iteration;
    int global_total_round;
    int global_angle;
    int global_type;
    int global_SNR;
    int global_multi_input;
    int global_RA;

    void qr_test(double *matA ,int rowA, int colA);
    void MVDR_DOA_1D_CPU(int M, int snr, int qr_iter, int multi_input, float *result);
   
    

#ifdef __cplusplus  
} // extern "C"  
#endif

#endif