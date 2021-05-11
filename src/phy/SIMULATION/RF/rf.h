


void rf_rx(double **r_re,
           double **r_im,
           double **r_re_i1,
           double **r_im_i1,
           double I0_dB,
           unsigned int nb_rx_antennas,
           unsigned int length,
           double s_time,
           double f_off,
           double drift,
           double noise_figure,
           double rx_gain_dB,
           int IP3_dBm,
           double *initial_phase,
           double pn_cutoff,
           double pn_amp_dBc,
           double IQ_imb_dB,
           double IQ_phase);

void rf_rx_simple(double *r_re[2],
                  double *r_im[2],
                  unsigned int nb_rx_antennas,
                  unsigned int length,
                  double s_time,
                  double rx_gain_dB);


void adc(double *r_re[2],
         double *r_im[2],
         unsigned int input_offset,
         unsigned int output_offset,
         int32_t **output,
         unsigned int nb_rx_antennas,
         unsigned int length,
         unsigned char B);

void dac(double *s_re[2],
         double *s_im[2],
         int32_t **input,
         unsigned int input_offset,
         unsigned int nb_tx_antennas,
         unsigned int length,
         double amp_dBm,
         unsigned char B,
         unsigned int meas_length,
         unsigned int meas_offset);

double dac_fixed_gain(double *s_re[2],
                      double *s_im[2],
                      int32_t **input,
                      unsigned int input_offset,
                      unsigned int nb_tx_antennas,
                      unsigned int length,
                      unsigned int input_offset_meas,
                      unsigned int length_meas,
                      unsigned char B,
                      double gain,
		      uint8_t do_amp_compute,
		      double *amp1,
                      int NB_RE);
