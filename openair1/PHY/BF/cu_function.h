
/*! \file openairinterface5g/openair1/PHY/BF/cu_function.h
 * \brief merge ISIP beamforming and QR decomposer
 * \author NCTU OpinConnect Terng-Yin Hsu, Sendren Xu, WEI-YING LIN, Min-Hsun Wu
 * \email  a22490010@gmail.com
 * \date   25-9-2021
 * \version 1.0
 * \note
 * \warning
 */

#ifndef __CU_FUNCTION_H
#define __CU_FUNCTION_H

#ifdef __cplusplus
extern "C" {  
#endif

    int global_music_antenna;
    int global_music_QR_iteration;
    int global_music_total_round;
    int global_music_angle;
    int global_music_type;
    int global_music_SNR;
    int global_music_multi_input;
    int global_RA;

    void qr_test(double *matA ,int rowA, int colA);

#ifdef __cplusplus  
} // extern "C"  
#endif

#endif