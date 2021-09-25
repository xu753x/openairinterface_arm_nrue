
/*! \file openairinterface5g/openair1/PHY/BF/bf.h
 * \brief merge ISIP beamforming and QR decomposer
 * \author NCTU OpinConnect Terng-Yin Hsu, Sendren Xu, WEI-YING LIN, Min-Hsun Wu
 * \email  a22490010@gmail.com
 * \date   25-9-2021
 * \version 1.0
 * \note
 * \warning
 */

#ifndef __BF_H
#define __BF_H

#define NB_ANTENNA_PORTS_GNB  8                                         // total number of gNB antenna ports
int counter = 0;

int nr_beam_precoding(int32_t **txdataF,
	                  int32_t **txdataF_BF,
                      NR_DL_FRAME_PARMS *frame_parms,
	                  int32_t ***beam_weights,
                      int slot,
                      int symbol,
                      int aa,
                      int nb_antenna_ports);


int multadd_cpx_vector(int16_t *x1,
                    int16_t *x2,
                    int16_t *y,
                    uint8_t zero_flag,
                    uint32_t N,
                    int output_shift);

#endif