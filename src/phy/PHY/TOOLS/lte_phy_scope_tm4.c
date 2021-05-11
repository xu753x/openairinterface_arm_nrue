


#include <stdlib.h>
#include "lte_phy_scope.h"
#define TPUT_WINDOW_LENGTH 100
int otg_enabled;
int use_sic_receiver=0;
FL_COLOR rx_antenna_colors[4] = {FL_RED,FL_BLUE,FL_GREEN,FL_YELLOW};
float tput_time_enb[NUMBER_OF_UE_MAX][TPUT_WINDOW_LENGTH] = {{0}};
float tput_enb[NUMBER_OF_UE_MAX][TPUT_WINDOW_LENGTH] = {{0}};
float tput_time_ue[NUMBER_OF_UE_MAX][TPUT_WINDOW_LENGTH] = {{0}};
float tput_ue[NUMBER_OF_UE_MAX][TPUT_WINDOW_LENGTH] = {{0}};
float tput_ue_max[NUMBER_OF_UE_MAX] = {0};

static void ia_receiver_on_off( FL_OBJECT *button, long arg) {
  if (fl_get_button(button)) {
    fl_set_object_label(button, "IA Receiver ON");
    //    PHY_vars_UE_g[0][0]->use_ia_receiver = 1;
    fl_set_object_color(button, FL_GREEN, FL_GREEN);
  } else {
    fl_set_object_label(button, "IA Receiver OFF");
    //    PHY_vars_UE_g[0][0]->use_ia_receiver = 0;
    fl_set_object_color(button, FL_RED, FL_RED);
  }
}

static void dl_traffic_on_off( FL_OBJECT *button, long arg) {
  if (fl_get_button(button)) {
    fl_set_object_label(button, "DL Traffic ON");
    otg_enabled = 1;
    fl_set_object_color(button, FL_GREEN, FL_GREEN);
  } else {
    fl_set_object_label(button, "DL Traffic OFF");
    otg_enabled = 0;
    fl_set_object_color(button, FL_RED, FL_RED);
  }
}

static void sic_receiver_on_off( FL_OBJECT *button, long arg) {
  if (fl_get_button(button)) {
    fl_set_object_label(button, "SIC Receiver ON");
    use_sic_receiver = 1;
    fl_set_object_color(button, FL_GREEN, FL_GREEN);
  } else {
    fl_set_object_label(button, "SIC Receiver OFF");
    use_sic_receiver = 0;
    fl_set_object_color(button, FL_RED, FL_RED);
  }
}

FD_lte_phy_scope_enb *create_lte_phy_scope_enb( void ) {
  FL_OBJECT *obj;
  FD_lte_phy_scope_enb *fdui = fl_malloc( sizeof *fdui );
  // Define form
  fdui->lte_phy_scope_enb = fl_bgn_form( FL_NO_BOX, 800, 800 );
  // This the whole UI box
  obj = fl_add_box( FL_BORDER_BOX, 0, 0, 800, 800, "" );
  fl_set_object_color( obj, FL_BLACK, FL_BLACK );
  // Received signal
  fdui->rxsig_t = fl_add_xyplot( FL_NORMAL_XYPLOT, 20, 20, 370, 100, "Received Signal (Time-Domain, dB)" );
  fl_set_object_boxtype( fdui->rxsig_t, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->rxsig_t, FL_BLACK, FL_RED );
  fl_set_object_lcolor( fdui->rxsig_t, FL_WHITE ); // Label color
  fl_set_xyplot_ybounds(fdui->rxsig_t,10,70);
  // Time-domain channel response
  fdui->chest_t = fl_add_xyplot( FL_NORMAL_XYPLOT, 410, 20, 370, 100, "Channel Impulse Response (samples, abs)" );
  fl_set_object_boxtype( fdui->chest_t, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->chest_t, FL_BLACK, FL_RED );
  fl_set_object_lcolor( fdui->chest_t, FL_WHITE ); // Label color
  // Frequency-domain channel response
  fdui->chest_f = fl_add_xyplot( FL_IMPULSE_XYPLOT, 20, 140, 760, 100, "Channel Frequency  Response (RE, dB)" );
  fl_set_object_boxtype( fdui->chest_f, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->chest_f, FL_BLACK, FL_RED );
  fl_set_object_lcolor( fdui->chest_f, FL_WHITE ); // Label color
  fl_set_xyplot_ybounds( fdui->chest_f,30,70);
  // LLR of PUSCH
  fdui->pusch_llr = fl_add_xyplot( FL_POINTS_XYPLOT, 20, 260, 500, 200, "PUSCH Log-Likelihood Ratios (LLR, mag)" );
  fl_set_object_boxtype( fdui->pusch_llr, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->pusch_llr, FL_BLACK, FL_YELLOW );
  fl_set_object_lcolor( fdui->pusch_llr, FL_WHITE ); // Label color
  fl_set_xyplot_symbolsize( fdui->pusch_llr,2);
  // I/Q PUSCH comp
  fdui->pusch_comp = fl_add_xyplot( FL_POINTS_XYPLOT, 540, 260, 240, 200, "PUSCH I/Q of MF Output" );
  fl_set_object_boxtype( fdui->pusch_comp, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->pusch_comp, FL_BLACK, FL_YELLOW );
  fl_set_object_lcolor( fdui->pusch_comp, FL_WHITE ); // Label color
  fl_set_xyplot_symbolsize( fdui->pusch_comp,2);
  fl_set_xyplot_xgrid( fdui->pusch_llr,FL_GRID_MAJOR);
  // I/Q PUCCH comp (format 1)
  fdui->pucch_comp1 = fl_add_xyplot( FL_POINTS_XYPLOT, 540, 480, 240, 100, "PUCCH1 Energy (SR)" );
  fl_set_object_boxtype( fdui->pucch_comp1, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->pucch_comp1, FL_BLACK, FL_YELLOW );
  fl_set_object_lcolor( fdui->pucch_comp1, FL_WHITE ); // Label color
  fl_set_xyplot_symbolsize( fdui->pucch_comp1,2);
  //  fl_set_xyplot_xgrid( fdui->pusch_llr,FL_GRID_MAJOR);
  // I/Q PUCCH comp (fromat 1a/b)
  fdui->pucch_comp = fl_add_xyplot( FL_POINTS_XYPLOT, 540, 600, 240, 100, "PUCCH I/Q of MF Output" );
  fl_set_object_boxtype( fdui->pucch_comp, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->pucch_comp, FL_BLACK, FL_YELLOW );
  fl_set_object_lcolor( fdui->pucch_comp, FL_WHITE ); // Label color
  fl_set_xyplot_symbolsize( fdui->pucch_comp,2);
  //  fl_set_xyplot_xgrid( fdui->pusch_llr,FL_GRID_MAJOR);
  // Throughput on PUSCH
  fdui->pusch_tput = fl_add_xyplot( FL_NORMAL_XYPLOT, 20, 480, 500, 100, "PUSCH Throughput [frame]/[kbit/s]" );
  fl_set_object_boxtype( fdui->pusch_tput, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->pusch_tput, FL_BLACK, FL_WHITE );
  fl_set_object_lcolor( fdui->pusch_tput, FL_WHITE ); // Label color
  // Generic eNB Button
  fdui->button_0 = fl_add_button( FL_PUSH_BUTTON, 20, 600, 240, 40, "" );
  fl_set_object_lalign(fdui->button_0, FL_ALIGN_CENTER );
  fl_set_button(fdui->button_0,0);
  otg_enabled = 0;
  fl_set_object_label(fdui->button_0, "DL Traffic OFF");
  fl_set_object_color(fdui->button_0, FL_RED, FL_RED);
  fl_set_object_callback(fdui->button_0, dl_traffic_on_off, 0 );
  fl_end_form( );
  fdui->lte_phy_scope_enb->fdui = fdui;
  return fdui;
}

void phy_scope_eNB(FD_lte_phy_scope_enb *form,
                   PHY_VARS_eNB *phy_vars_enb,
                   int UE_id) {
  int eNB_id = 0;
  int i,i2,arx,atx,ind,k;
  LTE_DL_FRAME_PARMS *frame_parms = &phy_vars_enb->frame_parms;
  int nsymb_ce = 12*frame_parms->N_RB_UL*frame_parms->symbols_per_tti;
  uint8_t nb_antennas_rx = frame_parms->nb_antennas_rx;
  uint8_t nb_antennas_tx = 1; // frame_parms->nb_antennas_tx; // in LTE Rel. 8 and 9 only a single transmit antenna is assumed at the UE
  int16_t **rxsig_t;
  int16_t **chest_t;
  int16_t **chest_f;
  int16_t *pusch_llr;
  int16_t *pusch_comp;
  int32_t *pucch1_comp;
  int32_t *pucch1_thres;
  int32_t *pucch1ab_comp;
  float Re,Im,ymax;
  float *llr, *bit;
  float I[nsymb_ce*2], Q[nsymb_ce*2];
  float I_pucch[10240],Q_pucch[10240],A_pucch[10240],B_pucch[10240],C_pucch[10240];
  float rxsig_t_dB[nb_antennas_rx][FRAME_LENGTH_COMPLEX_SAMPLES];
  float chest_t_abs[nb_antennas_rx][frame_parms->ofdm_symbol_size];
  float *chest_f_abs;
  float time[FRAME_LENGTH_COMPLEX_SAMPLES];
  float time2[2048];
  float freq[nsymb_ce*nb_antennas_rx*nb_antennas_tx];
  int frame = phy_vars_enb->proc.proc_rxtx[0].frame_tx;
  uint32_t total_dlsch_bitrate = phy_vars_enb->total_dlsch_bitrate;
  int coded_bits_per_codeword = 0;
  uint8_t harq_pid; // in TDD config 3 it is sf-2, i.e., can be 0,1,2
  int mcs = 0;

  // choose max MCS to compute coded_bits_per_codeword
  if (phy_vars_enb->ulsch[UE_id]!=NULL) {
    for (harq_pid=0; harq_pid<3; harq_pid++) {
      mcs = cmax(phy_vars_enb->ulsch[UE_id]->harq_processes[harq_pid]->mcs,mcs);
    }
  }

  coded_bits_per_codeword = frame_parms->N_RB_UL*12*get_Qm(mcs)*frame_parms->symbols_per_tti;
  chest_f_abs = (float *) calloc(nsymb_ce*nb_antennas_rx*nb_antennas_tx,sizeof(float));
  llr = (float *) calloc(coded_bits_per_codeword,sizeof(float)); // init to zero
  bit = malloc(coded_bits_per_codeword*sizeof(float));
  rxsig_t = (int16_t **) phy_vars_enb->common_vars.rxdata[eNB_id];
  chest_t = (int16_t **) phy_vars_enb->pusch_vars[UE_id]->drs_ch_estimates_time[eNB_id];
  chest_f = (int16_t **) phy_vars_enb->pusch_vars[UE_id]->drs_ch_estimates[eNB_id];
  pusch_llr = (int16_t *) phy_vars_enb->pusch_vars[UE_id]->llr;
  pusch_comp = (int16_t *) phy_vars_enb->pusch_vars[UE_id]->rxdataF_comp[eNB_id][0];
  pucch1_comp = (int32_t *) phy_vars_enb->pucch1_stats[UE_id];
  pucch1_thres = (int32_t *) phy_vars_enb->pucch1_stats_thres[UE_id];
  pucch1ab_comp = (int32_t *) phy_vars_enb->pucch1ab_stats[UE_id];

  // Received signal in time domain of receive antenna 0
  if (rxsig_t != NULL) {
    if (rxsig_t[0] != NULL) {
      for (i=0; i<FRAME_LENGTH_COMPLEX_SAMPLES; i++) {
        rxsig_t_dB[0][i] = 10*log10(1.0+(float) ((rxsig_t[0][2*i])*(rxsig_t[0][2*i])+(rxsig_t[0][2*i+1])*(rxsig_t[0][2*i+1])));
        time[i] = (float) i;
      }

      fl_set_xyplot_data(form->rxsig_t,time,rxsig_t_dB[0],FRAME_LENGTH_COMPLEX_SAMPLES,"","","");
    }

    for (arx=1; arx<nb_antennas_rx; arx++) {
      if (rxsig_t[arx] != NULL) {
        for (i=0; i<FRAME_LENGTH_COMPLEX_SAMPLES; i++) {
          rxsig_t_dB[arx][i] = 10*log10(1.0+(float) ((rxsig_t[arx][2*i])*(rxsig_t[arx][2*i])+(rxsig_t[arx][2*i+1])*(rxsig_t[arx][2*i+1])));
        }

        fl_add_xyplot_overlay(form->rxsig_t,arx,time,rxsig_t_dB[arx],FRAME_LENGTH_COMPLEX_SAMPLES,rx_antenna_colors[arx]);
      }
    }
  }

  // Channel Impulse Response
  if (chest_t != NULL) {
    ymax = 0;

    if (chest_t[0] !=NULL) {
      for (i=0; i<(frame_parms->ofdm_symbol_size); i++) {
        i2 = (i+(frame_parms->ofdm_symbol_size>>1))%frame_parms->ofdm_symbol_size;
        time2[i] = (float)(i-(frame_parms->ofdm_symbol_size>>1));
        chest_t_abs[0][i] = 10*log10((float) (1+chest_t[0][2*i2]*chest_t[0][2*i2]+chest_t[0][2*i2+1]*chest_t[0][2*i2+1]));

        if (chest_t_abs[0][i] > ymax)
          ymax = chest_t_abs[0][i];
      }

      fl_set_xyplot_data(form->chest_t,time2,chest_t_abs[0],(frame_parms->ofdm_symbol_size),"","","");
    }

    for (arx=1; arx<nb_antennas_rx; arx++) {
      if (chest_t[arx] !=NULL) {
        for (i=0; i<(frame_parms->ofdm_symbol_size>>3); i++) {
          chest_t_abs[arx][i] = 10*log10((float) (1+chest_t[arx][2*i]*chest_t[arx][2*i]+chest_t[arx][2*i+1]*chest_t[arx][2*i+1]));

          if (chest_t_abs[arx][i] > ymax)
            ymax = chest_t_abs[arx][i];
        }

        fl_add_xyplot_overlay(form->chest_t,arx,time,chest_t_abs[arx],(frame_parms->ofdm_symbol_size>>3),rx_antenna_colors[arx]);
        fl_set_xyplot_overlay_type(form->chest_t,arx,FL_DASHED_XYPLOT);
      }
    }

    // Avoid flickering effect
    //        fl_get_xyplot_ybounds(form->chest_t,&ymin,&ymax);
    fl_set_xyplot_ybounds(form->chest_t,0,ymax);
  }

  // Channel Frequency Response
  if (chest_f != NULL) {
    ind = 0;

    for (atx=0; atx<nb_antennas_tx; atx++) {
      for (arx=0; arx<nb_antennas_rx; arx++) {
        if (chest_f[(atx<<1)+arx] != NULL) {
          for (k=0; k<nsymb_ce; k++) {
            freq[ind] = (float)ind;
            Re = (float)(chest_f[(atx<<1)+arx][(2*k)]);
            Im = (float)(chest_f[(atx<<1)+arx][(2*k)+1]);
            chest_f_abs[ind] = (short)10*log10(1.0+((double)Re*Re + (double)Im*Im));
            ind++;
          }
        }
      }
    }

    // tx antenna 0
    fl_set_xyplot_xbounds(form->chest_f,0,nb_antennas_rx*nb_antennas_tx*nsymb_ce);
    fl_set_xyplot_xtics(form->chest_f,nb_antennas_rx*nb_antennas_tx*frame_parms->symbols_per_tti,3);
    fl_set_xyplot_xgrid(form->chest_f,FL_GRID_MAJOR);
    fl_set_xyplot_data(form->chest_f,freq,chest_f_abs,nsymb_ce,"","","");

    for (arx=1; arx<nb_antennas_rx; arx++) {
      fl_add_xyplot_overlay(form->chest_f,1,&freq[arx*nsymb_ce],&chest_f_abs[arx*nsymb_ce],nsymb_ce,rx_antenna_colors[arx]);
    }

    // other tx antennas
    if (nb_antennas_tx > 1) {
      if (nb_antennas_rx > 1) {
        for (atx=1; atx<nb_antennas_tx; atx++) {
          for (arx=0; arx<nb_antennas_rx; arx++) {
            fl_add_xyplot_overlay(form->chest_f,(atx<<1)+arx,&freq[((atx<<1)+arx)*nsymb_ce],&chest_f_abs[((atx<<1)+arx)*nsymb_ce],nsymb_ce,rx_antenna_colors[arx]);
          }
        }
      } else { // 1 rx antenna
        atx=1;
        arx=0;
        fl_add_xyplot_overlay(form->chest_f,atx,&freq[atx*nsymb_ce],&chest_f_abs[atx*nsymb_ce],nsymb_ce,rx_antenna_colors[arx]);
      }
    }
  }

  // PUSCH LLRs
  if (pusch_llr != NULL) {
    for (i=0; i<coded_bits_per_codeword; i++) {
      llr[i] = (float) pusch_llr[i];
      bit[i] = (float) i;
    }

    fl_set_xyplot_data(form->pusch_llr,bit,llr,coded_bits_per_codeword,"","","");
  }

  // PUSCH I/Q of MF Output
  if (pusch_comp!=NULL) {
    ind=0;

    for (k=0; k<frame_parms->symbols_per_tti; k++) {
      for (i=0; i<12*frame_parms->N_RB_UL; i++) {
        I[ind] = pusch_comp[(2*frame_parms->N_RB_UL*12*k)+2*i];
        Q[ind] = pusch_comp[(2*frame_parms->N_RB_UL*12*k)+2*i+1];
        ind++;
      }
    }

    fl_set_xyplot_data(form->pusch_comp,I,Q,ind,"","","");
  }

  // PUSCH I/Q of MF Output
  if (pucch1ab_comp!=NULL) {
    for (ind=0; ind<10240; ind++) {
      I_pucch[ind] = (float)pucch1ab_comp[2*(ind)];
      Q_pucch[ind] = (float)pucch1ab_comp[2*(ind)+1];
      A_pucch[ind] = 10*log10(pucch1_comp[ind]);
      B_pucch[ind] = ind;
      C_pucch[ind] = (float)pucch1_thres[ind];
    }

    fl_set_xyplot_data(form->pucch_comp,I_pucch,Q_pucch,10240,"","","");
    fl_set_xyplot_data(form->pucch_comp1,B_pucch,A_pucch,1024,"","","");
    fl_add_xyplot_overlay(form->pucch_comp1,1,B_pucch,C_pucch,1024,FL_RED);
    fl_set_xyplot_ybounds(form->pucch_comp,-5000,5000);
    fl_set_xyplot_xbounds(form->pucch_comp,-5000,5000);
    fl_set_xyplot_ybounds(form->pucch_comp1,0,80);
  }

  // PUSCH Throughput
  memmove( tput_time_enb[UE_id], &tput_time_enb[UE_id][1], (TPUT_WINDOW_LENGTH-1)*sizeof(float) );
  memmove( tput_enb[UE_id], &tput_enb[UE_id][1], (TPUT_WINDOW_LENGTH-1)*sizeof(float) );
  tput_time_enb[UE_id][TPUT_WINDOW_LENGTH-1]  = (float) frame;
  tput_enb[UE_id][TPUT_WINDOW_LENGTH-1] = ((float) total_dlsch_bitrate)/1000.0;
  fl_set_xyplot_data(form->pusch_tput,tput_time_enb[UE_id],tput_enb[UE_id],TPUT_WINDOW_LENGTH,"","","");
  //    fl_get_xyplot_ybounds(form->pusch_tput,&ymin,&ymax);
  //    fl_set_xyplot_ybounds(form->pusch_tput,0,ymax);
  fl_check_forms();
  free(llr);
  free(bit);
  free(chest_f_abs);
}

FD_lte_phy_scope_ue *create_lte_phy_scope_ue( void ) {
  FL_OBJECT *obj;
  FD_lte_phy_scope_ue *fdui = fl_malloc( sizeof *fdui );
  // Define form
  fdui->lte_phy_scope_ue = fl_bgn_form( FL_NO_BOX, 800, 1000 );
  // This the whole UI box
  obj = fl_add_box( FL_BORDER_BOX, 0, 0, 800, 1000, "" );
  fl_set_object_color( obj, FL_BLACK, FL_BLACK );
  // Received signal
  fdui->rxsig_t = fl_add_xyplot( FL_NORMAL_XYPLOT, 20, 20, 370, 100, "Received Signal (Time-Domain, dB)" );
  fl_set_object_boxtype( fdui->rxsig_t, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->rxsig_t, FL_BLACK, FL_RED );
  fl_set_object_lcolor( fdui->rxsig_t, FL_WHITE ); // Label color
  fl_set_xyplot_ybounds(fdui->rxsig_t,30,70);
  // Time-domain channel response
  fdui->chest_t = fl_add_xyplot( FL_NORMAL_XYPLOT, 410, 20, 370, 100, "Channel Impulse Response (samples, abs)" );
  fl_set_object_boxtype( fdui->chest_t, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->chest_t, FL_BLACK, FL_RED );
  fl_set_object_lcolor( fdui->chest_t, FL_WHITE ); // Label color
  // Frequency-domain channel response
  fdui->chest_f = fl_add_xyplot( FL_IMPULSE_XYPLOT, 20, 140, 760, 100, "Channel Frequency Response (RE, dB)" );
  fl_set_object_boxtype( fdui->chest_f, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->chest_f, FL_BLACK, FL_RED );
  fl_set_object_lcolor( fdui->chest_f, FL_WHITE ); // Label color
  fl_set_xyplot_ybounds( fdui->chest_f,30,70);
  /*
  // LLR of PBCH
  fdui->pbch_llr = fl_add_xyplot( FL_POINTS_XYPLOT, 20, 260, 500, 100, "PBCH Log-Likelihood Ratios (LLR, mag)" );
  fl_set_object_boxtype( fdui->pbch_llr, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->pbch_llr, FL_BLACK, FL_GREEN );
  fl_set_object_lcolor( fdui->pbch_llr, FL_WHITE ); // Label color
  fl_set_xyplot_symbolsize( fdui->pbch_llr,2);
  fl_set_xyplot_xgrid( fdui->pbch_llr,FL_GRID_MAJOR);
  fl_set_xyplot_xbounds( fdui->pbch_llr,0,1920);
  // I/Q PBCH comp
  fdui->pbch_comp = fl_add_xyplot( FL_POINTS_XYPLOT, 540, 260, 240, 100, "PBCH I/Q of MF Output" );
  fl_set_object_boxtype( fdui->pbch_comp, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->pbch_comp, FL_BLACK, FL_GREEN );
  fl_set_object_lcolor( fdui->pbch_comp, FL_WHITE ); // Label color
  fl_set_xyplot_symbolsize( fdui->pbch_comp,2);
  fl_set_xyplot_xbounds( fdui->pbch_comp,-100,100);
  fl_set_xyplot_ybounds( fdui->pbch_comp,-100,100);
  // LLR of PDCCH
  fdui->pdcch_llr = fl_add_xyplot( FL_POINTS_XYPLOT, 20, 380, 500, 100, "PDCCH Log-Likelihood Ratios (LLR, mag)" );
  fl_set_object_boxtype( fdui->pdcch_llr, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->pdcch_llr, FL_BLACK, FL_CYAN );
  fl_set_object_lcolor( fdui->pdcch_llr, FL_WHITE ); // Label color
  fl_set_xyplot_symbolsize( fdui->pdcch_llr,2);
  // I/Q PDCCH comp
  fdui->pdcch_comp = fl_add_xyplot( FL_POINTS_XYPLOT, 540, 380, 240, 100, "PDCCH I/Q of MF Output" );
  fl_set_object_boxtype( fdui->pdcch_comp, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->pdcch_comp, FL_BLACK, FL_CYAN );
  fl_set_object_lcolor( fdui->pdcch_comp, FL_WHITE ); // Label color
  fl_set_xyplot_symbolsize( fdui->pdcch_comp,2);
  fl_set_xyplot_xgrid( fdui->pdcch_llr,FL_GRID_MAJOR);
  */
  int offset=240;
  // LLR of PDSCH
  fdui->pdsch_llr = fl_add_xyplot( FL_POINTS_XYPLOT, 20, 500-offset, 500, 200, "PDSCH Log-Likelihood Ratios (LLR, mag)" );
  fl_set_object_boxtype( fdui->pdsch_llr, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->pdsch_llr, FL_BLACK, FL_YELLOW );
  fl_set_object_lcolor( fdui->pdsch_llr, FL_WHITE ); // Label color
  fl_set_xyplot_symbolsize( fdui->pdsch_llr,2);
  fl_set_xyplot_xgrid( fdui->pdsch_llr,FL_GRID_MAJOR);
  // I/Q PDSCH comp
  fdui->pdsch_comp = fl_add_xyplot( FL_POINTS_XYPLOT, 540, 500-offset, 240, 200, "PDSCH I/Q of MF Output" );
  fl_set_object_boxtype( fdui->pdsch_comp, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->pdsch_comp, FL_BLACK, FL_YELLOW );
  fl_set_object_lcolor( fdui->pdsch_comp, FL_WHITE ); // Label color
  fl_set_xyplot_symbolsize( fdui->pdsch_comp,2);
  // LLR of PDSCH
  fdui->pdsch_llr1 = fl_add_xyplot( FL_POINTS_XYPLOT, 20, 720-offset, 500, 200, "PDSCH Log-Likelihood Ratios (LLR, mag)" );
  fl_set_object_boxtype( fdui->pdsch_llr1, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->pdsch_llr1, FL_BLACK, FL_YELLOW );
  fl_set_object_lcolor( fdui->pdsch_llr1, FL_WHITE ); // Label color
  fl_set_xyplot_symbolsize( fdui->pdsch_llr1,2);
  fl_set_xyplot_xgrid( fdui->pdsch_llr1,FL_GRID_MAJOR);
  // I/Q PDSCH comp
  fdui->pdsch_comp1 = fl_add_xyplot( FL_POINTS_XYPLOT, 540, 720-offset, 240, 200, "PDSCH I/Q of MF Output" );
  fl_set_object_boxtype( fdui->pdsch_comp1, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->pdsch_comp1, FL_BLACK, FL_YELLOW );
  fl_set_object_lcolor( fdui->pdsch_comp1, FL_WHITE ); // Label color
  fl_set_xyplot_symbolsize( fdui->pdsch_comp1,2);
  /*
  // Throughput on PDSCH
  fdui->pdsch_tput = fl_add_xyplot( FL_NORMAL_XYPLOT, 20, 720, 500, 100, "PDSCH Throughput [frame]/[kbit/s]" );
  fl_set_object_boxtype( fdui->pdsch_tput, FL_EMBOSSED_BOX );
  fl_set_object_color( fdui->pdsch_tput, FL_BLACK, FL_WHITE );
  fl_set_object_lcolor( fdui->pdsch_tput, FL_WHITE ); // Label color
  */
  // Generic UE Button
  fdui->button_0 = fl_add_button( FL_PUSH_BUTTON, 540, 720, 240, 40, "" );
  fl_set_object_lalign(fdui->button_0, FL_ALIGN_CENTER );
  //use_sic_receiver = 0;
  fl_set_button(fdui->button_0,0);
  fl_set_object_label(fdui->button_0, "SIC Receiver OFF");
  fl_set_object_color(fdui->button_0, FL_RED, FL_RED);
  fl_set_object_callback(fdui->button_0, sic_receiver_on_off, 0 );
  fl_hide_object(fdui->button_0);
  fl_end_form( );
  fdui->lte_phy_scope_ue->fdui = fdui;
  return fdui;
}

void phy_scope_UE(FD_lte_phy_scope_ue *form,
                  PHY_VARS_UE *phy_vars_ue,
                  int eNB_id,
                  int UE_id,
                  uint8_t subframe) {
  int i,arx,atx,ind,k;
  LTE_DL_FRAME_PARMS *frame_parms = &phy_vars_ue->frame_parms;
  int nsymb_ce = frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti;
  uint8_t nb_antennas_rx = frame_parms->nb_antennas_rx;
  uint8_t nb_antennas_tx = frame_parms->nb_antenna_ports_eNB;
  int16_t **rxsig_t;
  int16_t **chest_t;
  int16_t **chest_f;
  int16_t *pdsch_llr,*pdsch_llr1;
  int16_t *pdsch_comp,*pdsch_comp1;
  int16_t *pdsch_mag0,*pdsch_mag1,*pdsch_magb0,*pdsch_magb1;
  int8_t *pdcch_llr;
  int16_t *pdcch_comp;
  int8_t *pbch_llr;
  int16_t *pbch_comp;
  float Re,Im,ymax=1;
  int num_pdcch_symbols=3;
  float *llr0, *bit0, *llr1, *bit1, *chest_f_abs, llr_pbch[1920], bit_pbch[1920], *llr_pdcch, *bit_pdcch;
  float *I, *Q;
  float rxsig_t_dB[nb_antennas_rx][FRAME_LENGTH_COMPLEX_SAMPLES];
  float **chest_t_abs;
  float time[FRAME_LENGTH_COMPLEX_SAMPLES];
  float freq[nsymb_ce*nb_antennas_rx*nb_antennas_tx];
  int frame = phy_vars_ue->proc.proc_rxtx[0].frame_rx;
  uint32_t total_dlsch_bitrate = phy_vars_ue->bitrate[eNB_id];
  int coded_bits_per_codeword0=0,coded_bits_per_codeword1=1;
  int mod0,mod1;
  int mcs0 = 0;
  int mcs1=0;
  unsigned char harq_pid = 0;
  int beamforming_mode = phy_vars_ue->transmission_mode[eNB_id]>6 ? phy_vars_ue->transmission_mode[eNB_id] : 0;

  if (phy_vars_ue->dlsch[phy_vars_ue->current_thread_id[subframe]][eNB_id][0]!=NULL) {
    harq_pid = phy_vars_ue->dlsch[phy_vars_ue->current_thread_id[subframe]][eNB_id][0]->current_harq_pid;

    if (harq_pid>=8)
      return;

    mcs0 = phy_vars_ue->dlsch[phy_vars_ue->current_thread_id[subframe]][eNB_id][0]->harq_processes[harq_pid]->mcs;
    // Button 0
    /*
          if(!phy_vars_ue->dlsch_ue[eNB_id][0]->harq_processes[harq_pid]->dl_power_off) {
              // we are in TM5
              fl_show_object(form->button_0);
          }
    */
  }

  fl_show_object(form->button_0);

  if (phy_vars_ue->dlsch[phy_vars_ue->current_thread_id[subframe]][eNB_id][1]!=NULL) {
    harq_pid = phy_vars_ue->dlsch[phy_vars_ue->current_thread_id[subframe]][eNB_id][1]->current_harq_pid;

    if (harq_pid>=8)
      return;

    mcs1 = phy_vars_ue->dlsch[phy_vars_ue->current_thread_id[subframe]][eNB_id][1]->harq_processes[harq_pid]->mcs;
  }

  if (phy_vars_ue->pdcch_vars[phy_vars_ue->current_thread_id[subframe]][eNB_id]!=NULL) {
    num_pdcch_symbols = phy_vars_ue->pdcch_vars[phy_vars_ue->current_thread_id[subframe]][eNB_id]->num_pdcch_symbols;
  }

  //    coded_bits_per_codeword = frame_parms->N_RB_DL*12*get_Qm(mcs)*(frame_parms->symbols_per_tti);
  if (phy_vars_ue->dlsch[phy_vars_ue->current_thread_id[subframe]][eNB_id][0]!=NULL) {
    mod0 = get_Qm(mcs0);
    coded_bits_per_codeword0 = get_G(frame_parms,
                                     phy_vars_ue->dlsch[phy_vars_ue->current_thread_id[subframe]][eNB_id][0]->harq_processes[harq_pid]->nb_rb,
                                     phy_vars_ue->dlsch[phy_vars_ue->current_thread_id[subframe]][eNB_id][0]->harq_processes[harq_pid]->rb_alloc_even,
                                     get_Qm(mcs0),
                                     phy_vars_ue->dlsch[phy_vars_ue->current_thread_id[subframe]][eNB_id][0]->harq_processes[harq_pid]->Nl,
                                     num_pdcch_symbols,
                                     frame,
                                     subframe,
                                     beamforming_mode);
  } else {
    coded_bits_per_codeword0 = 0; //frame_parms->N_RB_DL*12*get_Qm(mcs)*(frame_parms->symbols_per_tti);
    mod0=0;
  }

  if (phy_vars_ue->dlsch[phy_vars_ue->current_thread_id[subframe]][eNB_id][1]!=NULL) {
    mod1 = get_Qm(mcs1);
    coded_bits_per_codeword1 = get_G(frame_parms,
                                     phy_vars_ue->dlsch[phy_vars_ue->current_thread_id[subframe]][eNB_id][1]->harq_processes[harq_pid]->nb_rb,
                                     phy_vars_ue->dlsch[phy_vars_ue->current_thread_id[subframe]][eNB_id][1]->harq_processes[harq_pid]->rb_alloc_even,
                                     get_Qm(mcs1),
                                     phy_vars_ue->dlsch[phy_vars_ue->current_thread_id[subframe]][eNB_id][1]->harq_processes[harq_pid]->Nl,
                                     num_pdcch_symbols,
                                     frame,
                                     subframe,
                                     beamforming_mode);
  } else {
    coded_bits_per_codeword1 = 0; //frame_parms->N_RB_DL*12*get_Qm(mcs)*(frame_parms->symbols_per_tti);
    mod1=0;
  }

  I = (float *) calloc(nsymb_ce*2,sizeof(float));
  Q = (float *) calloc(nsymb_ce*2,sizeof(float));
  chest_t_abs = (float **) malloc(nb_antennas_rx*sizeof(float *));

  for (arx=0; arx<nb_antennas_rx; arx++) {
    chest_t_abs[arx] = (float *) calloc(frame_parms->ofdm_symbol_size,sizeof(float));
  }

  chest_f_abs = (float *) calloc(nsymb_ce*nb_antennas_rx*nb_antennas_tx,sizeof(float));
  //llr0 = (float*) calloc(coded_bits_per_codeword0,sizeof(float)); // Cppcheck returns "invalidFunctionArg" error.
  llr0 = (float *) malloc(coded_bits_per_codeword0*sizeof(float));
  memset((void *)llr0, 0,coded_bits_per_codeword0*sizeof(float)); // init to zero
  bit0 = malloc(coded_bits_per_codeword0*sizeof(float));
  //llr1 = (float*) calloc(coded_bits_per_codeword1,sizeof(float)); // Cppcheck returns "invalidFunctionArg" error.
  llr1 = (float *) malloc(coded_bits_per_codeword1*sizeof(float));
  memset((void *)llr1, 0,coded_bits_per_codeword1*sizeof(float)); // init to zero
  bit1 = malloc(coded_bits_per_codeword1*sizeof(float));
  llr_pdcch = (float *) calloc(12*frame_parms->N_RB_DL*num_pdcch_symbols*2,sizeof(float)); // init to zero
  bit_pdcch = (float *) calloc(12*frame_parms->N_RB_DL*num_pdcch_symbols*2,sizeof(float));
  rxsig_t = (int16_t **) phy_vars_ue->common_vars.rxdata;
  chest_t = (int16_t **) phy_vars_ue->common_vars.common_vars_rx_data_per_thread[phy_vars_ue->current_thread_id[subframe]].dl_ch_estimates_time[eNB_id];
  chest_f = (int16_t **) phy_vars_ue->common_vars.common_vars_rx_data_per_thread[phy_vars_ue->current_thread_id[subframe]].dl_ch_estimates[eNB_id];
  pbch_llr = (int8_t *) phy_vars_ue->pbch_vars[eNB_id]->llr;
  pbch_comp = (int16_t *) phy_vars_ue->pbch_vars[eNB_id]->rxdataF_comp[0];
  pdcch_llr = (int8_t *) phy_vars_ue->pdcch_vars[phy_vars_ue->current_thread_id[subframe]][eNB_id]->llr;
  pdcch_comp = (int16_t *) phy_vars_ue->pdcch_vars[phy_vars_ue->current_thread_id[subframe]][eNB_id]->rxdataF_comp[0];
  pdsch_llr = (int16_t *) phy_vars_ue->pdsch_vars[phy_vars_ue->current_thread_id[subframe]][eNB_id]->llr[0]; // stream 0
  pdsch_llr1 = (int16_t *) phy_vars_ue->pdsch_vars[phy_vars_ue->current_thread_id[subframe]][eNB_id]->llr[1]; // stream 1
  pdsch_comp = (int16_t *) phy_vars_ue->pdsch_vars[phy_vars_ue->current_thread_id[subframe]][eNB_id]->rxdataF_comp0[0];
  //pdsch_comp = (int16_t*) phy_vars_ue->lte_ue_pdsch_vars[phy_vars_ue->current_thread_id[subframe]][eNB_id]->rxdataF_ext[0];
  //pdsch_comp1 = (int16_t*) phy_vars_ue->lte_ue_pdsch_vars[phy_vars_ue->current_thread_id[subframe]][eNB_id]->rxdataF_ext[1];
  pdsch_comp1 = (int16_t *) (phy_vars_ue->pdsch_vars[phy_vars_ue->current_thread_id[subframe]][eNB_id]->rxdataF_comp1[0][0])[0];
  //pdsch_comp1 = (int16_t*) (phy_vars_ue->lte_ue_pdsch_vars[eNB_id]->dl_ch_rho_ext[0][0])[0];
  pdsch_mag0 = (int16_t *) phy_vars_ue->pdsch_vars[phy_vars_ue->current_thread_id[subframe]][eNB_id]->dl_ch_mag0[0];
  pdsch_mag1 = (int16_t *) (phy_vars_ue->pdsch_vars[phy_vars_ue->current_thread_id[subframe]][eNB_id]->dl_ch_mag1[0][0])[0];
  pdsch_magb0 = (int16_t *) phy_vars_ue->pdsch_vars[phy_vars_ue->current_thread_id[subframe]][eNB_id]->dl_ch_magb0[0];
  pdsch_magb1 = (int16_t *) (phy_vars_ue->pdsch_vars[phy_vars_ue->current_thread_id[subframe]][eNB_id]->dl_ch_magb1[0][0])[0];
  fl_freeze_form(form->lte_phy_scope_ue);

  // Received signal in time domain of receive antenna 0
  if (rxsig_t != NULL) {
    if (rxsig_t[0] != NULL) {
      for (i=0; i<FRAME_LENGTH_COMPLEX_SAMPLES; i++) {
        rxsig_t_dB[0][i] = 10*log10(1.0+(float) ((rxsig_t[0][2*i])*(rxsig_t[0][2*i])+(rxsig_t[0][2*i+1])*(rxsig_t[0][2*i+1])));
        time[i] = (float) i;
      }

      fl_set_xyplot_data(form->rxsig_t,time,rxsig_t_dB[0],FRAME_LENGTH_COMPLEX_SAMPLES,"","","");
    }

    for (arx=1; arx<nb_antennas_rx; arx++) {
      if (rxsig_t[arx] != NULL) {
        for (i=0; i<FRAME_LENGTH_COMPLEX_SAMPLES; i++) {
          rxsig_t_dB[arx][i] = 10*log10(1.0+(float) ((rxsig_t[arx][2*i])*(rxsig_t[arx][2*i])+(rxsig_t[arx][2*i+1])*(rxsig_t[arx][2*i+1])));
        }

        fl_add_xyplot_overlay(form->rxsig_t,arx,time,rxsig_t_dB[arx],FRAME_LENGTH_COMPLEX_SAMPLES,rx_antenna_colors[arx]);
      }
    }
  }

  // Channel Impulse Response (still repeated format)
  if (chest_t != NULL) {
    ymax = 0;

    if (chest_t[0] !=NULL) {
      for (i=0; i<(frame_parms->ofdm_symbol_size>>3); i++) {
        chest_t_abs[0][i] = (float) (chest_t[0][4*i]*chest_t[0][4*i]+chest_t[0][4*i+1]*chest_t[0][4*i+1]);

        if (chest_t_abs[0][i] > ymax)
          ymax = chest_t_abs[0][i];
      }

      fl_set_xyplot_data(form->chest_t,time,chest_t_abs[0],(frame_parms->ofdm_symbol_size>>3),"","","");
    }

    for (arx=1; arx<nb_antennas_rx; arx++) {
      if (chest_t[arx] !=NULL) {
        for (i=0; i<(frame_parms->ofdm_symbol_size>>3); i++) {
          chest_t_abs[arx][i] = (float) (chest_t[arx][4*i]*chest_t[arx][4*i]+chest_t[arx][4*i+1]*chest_t[arx][4*i+1]);

          if (chest_t_abs[arx][i] > ymax)
            ymax = chest_t_abs[arx][i];
        }

        fl_add_xyplot_overlay(form->chest_t,arx,time,chest_t_abs[arx],(frame_parms->ofdm_symbol_size>>3),rx_antenna_colors[arx]);
        fl_set_xyplot_overlay_type(form->chest_t,arx,FL_DASHED_XYPLOT);
      }
    }

    // Avoid flickering effect
    //        fl_get_xyplot_ybounds(form->chest_t,&ymin,&ymax);
    fl_set_xyplot_ybounds(form->chest_t,0,ymax);
  }

  // Channel Frequency Response (includes 5 complex sample for filter)
  if (chest_f != NULL) {
    ind = 0;

    for (atx=0; atx<nb_antennas_tx; atx++) {
      for (arx=0; arx<nb_antennas_rx; arx++) {
        if (chest_f[(atx<<1)+arx] != NULL) {
          for (k=0; k<nsymb_ce; k++) {
            freq[ind] = (float)ind;
            Re = (float)(chest_f[(atx<<1)+arx][(2*k)]);
            Im = (float)(chest_f[(atx<<1)+arx][(2*k)+1]);
            chest_f_abs[ind] = (short)10*log10(1.0+((double)Re*Re + (double)Im*Im));
            ind++;
          }
        }
      }
    }

    // tx antenna 0
    fl_set_xyplot_xbounds(form->chest_f,0,nb_antennas_rx*nb_antennas_tx*nsymb_ce);
    //fl_set_xyplot_xtics(form->chest_f,nb_antennas_rx*nb_antennas_tx*frame_parms->symbols_per_tti,2);
    //        fl_set_xyplot_xtics(form->chest_f,nb_antennas_rx*nb_antennas_tx*2,2);
    fl_set_xyplot_xgrid(form->chest_f,FL_GRID_MAJOR);
    fl_set_xyplot_data(form->chest_f,freq,chest_f_abs,nsymb_ce,"","","");

    for (arx=1; arx<nb_antennas_rx; arx++) {
      fl_add_xyplot_overlay(form->chest_f,1,&freq[arx*nsymb_ce],&chest_f_abs[arx*nsymb_ce],nsymb_ce,rx_antenna_colors[arx]);
    }

    // other tx antennas
    if (nb_antennas_tx > 1) {
      if (nb_antennas_rx > 1) {
        for (atx=1; atx<nb_antennas_tx; atx++) {
          for (arx=0; arx<nb_antennas_rx; arx++) {
            fl_add_xyplot_overlay(form->chest_f,(atx<<1)+arx,&freq[((atx<<1)+arx)*nsymb_ce],&chest_f_abs[((atx<<1)+arx)*nsymb_ce],nsymb_ce,rx_antenna_colors[arx]);
          }
        }
      } else { // 1 rx antenna
        atx=1;
        arx=0;
        fl_add_xyplot_overlay(form->chest_f,atx,&freq[atx*nsymb_ce],&chest_f_abs[atx*nsymb_ce],nsymb_ce,rx_antenna_colors[arx]);
      }
    }
  }

  /*
     // PBCH LLRs
     if (pbch_llr != NULL) {
         for (i=0; i<1920;i++) {
             llr_pbch[i] = (float) pbch_llr[i];
             bit_pbch[i] = (float) i;
         }
         fl_set_xyplot_data(form->pbch_llr,bit_pbch,llr_pbch,1920,"","","");
     }
     // PBCH I/Q of MF Output
     if (pbch_comp!=NULL) {
         for (i=0; i<72*2; i++) {
             I[i] = pbch_comp[2*i];
             Q[i] = pbch_comp[2*i+1];
         }
         fl_set_xyplot_data(form->pbch_comp,I,Q,72*2,"","","");
     }
     // PDCCH LLRs
     if (pdcch_llr != NULL) {
         for (i=0; i<12*frame_parms->N_RB_DL*2*num_pdcch_symbols;i++) {
             llr_pdcch[i] = (float) pdcch_llr[i];
             bit_pdcch[i] = (float) i;
         }
         fl_set_xyplot_xbounds(form->pdcch_llr,0,12*frame_parms->N_RB_DL*2*3);
         fl_set_xyplot_data(form->pdcch_llr,bit_pdcch,llr_pdcch,12*frame_parms->N_RB_DL*2*num_pdcch_symbols,"","","");
     }
     // PDCCH I/Q of MF Output
     if (pdcch_comp!=NULL) {
         for (i=0; i<12*frame_parms->N_RB_DL*num_pdcch_symbols; i++) {
             I[i] = pdcch_comp[2*i];
             Q[i] = pdcch_comp[2*i+1];
         }
         fl_set_xyplot_data(form->pdcch_comp,I,Q,12*frame_parms->N_RB_DL*num_pdcch_symbols,"","","");
     }
     */
  // PDSCH LLRs CW0
  if (pdsch_llr != NULL) {
    for (i=0; i<coded_bits_per_codeword0; i++) {
      llr0[i] = (float) pdsch_llr[i];
      bit0[i] = (float) i;
    }

    fl_set_xyplot_xbounds(form->pdsch_llr,0,coded_bits_per_codeword0);
    fl_set_xyplot_data(form->pdsch_llr,bit0,llr0,coded_bits_per_codeword0,"","","");
  }

  // PDSCH I/Q of MF Output
  if (pdsch_comp!=NULL) {
    ind=0;

    for (k=0; k<frame_parms->symbols_per_tti; k++) {
      for (i=0; i<12*frame_parms->N_RB_DL/2; i++) {
        I[ind] = pdsch_comp[(2*frame_parms->N_RB_DL*12*k)+4*i];
        Q[ind] = pdsch_comp[(2*frame_parms->N_RB_DL*12*k)+4*i+1];
        ind++;
      }
    }

    fl_set_xyplot_data(form->pdsch_comp,I,Q,ind,"","","");
  }

  if (pdsch_mag0 != NULL) {
    ind=0;

    for (k=0; k<frame_parms->symbols_per_tti; k++) {
      for (i=0; i<12*frame_parms->N_RB_DL/2; i++) {
        I[ind] = pdsch_mag0[(2*frame_parms->N_RB_DL*12*k)+4*i]*cos(i*2*M_PI/(12*frame_parms->N_RB_DL/2));
        Q[ind] = pdsch_mag0[(2*frame_parms->N_RB_DL*12*k)+4*i+1]*sin(i*2*M_PI/(12*frame_parms->N_RB_DL/2));
        ind++;
      }
    }

    fl_add_xyplot_overlay(form->pdsch_comp,1,I,Q,ind,FL_GREEN);
  }

  if (pdsch_magb0 != NULL) {
    ind=0;

    for (k=0; k<frame_parms->symbols_per_tti; k++) {
      for (i=0; i<12*frame_parms->N_RB_DL/2; i++) {
        I[ind] = pdsch_magb0[(2*frame_parms->N_RB_DL*12*k)+4*i]*cos(i*2*M_PI/(12*frame_parms->N_RB_DL/2));
        Q[ind] = pdsch_magb0[(2*frame_parms->N_RB_DL*12*k)+4*i+1]*sin(i*2*M_PI/(12*frame_parms->N_RB_DL/2));
        ind++;
      }
    }

    fl_add_xyplot_overlay(form->pdsch_comp,2,I,Q,ind,FL_RED);
  }

  if ((pdsch_mag0 != NULL) && (pdsch_magb0 != NULL)) {
    ind=0;

    for (k=0; k<frame_parms->symbols_per_tti; k++) {
      for (i=0; i<12*frame_parms->N_RB_DL/2; i++) {
        I[ind] =
          (pdsch_mag0[(2*frame_parms->N_RB_DL*12*k)+4*i]+
           pdsch_magb0[(2*frame_parms->N_RB_DL*12*k)+4*i])*
          cos(i*2*M_PI/(12*frame_parms->N_RB_DL/2));
        Q[ind] =
          (pdsch_mag0[(2*frame_parms->N_RB_DL*12*k)+4*i+1]+
           pdsch_magb0[(2*frame_parms->N_RB_DL*12*k)+4*i+1])*
          sin(i*2*M_PI/(12*frame_parms->N_RB_DL/2));
        ind++;
      }
    }

    fl_add_xyplot_overlay(form->pdsch_comp,3,I,Q,ind,FL_BLUE);
  }

  // PDSCH LLRs CW1
  if (pdsch_llr1 != NULL) {
    for (i=0; i<coded_bits_per_codeword1; i++) {
      llr1[i] = (float) pdsch_llr1[i];
      bit1[i] = (float) i;
    }

    fl_set_xyplot_xbounds(form->pdsch_llr1,0,coded_bits_per_codeword1);
    fl_set_xyplot_data(form->pdsch_llr1,bit1,llr1,coded_bits_per_codeword1,"","","");
  }

  // PDSCH I/Q of MF Output
  if (pdsch_comp1!=NULL) {
    ind=0;

    for (k=0; k<frame_parms->symbols_per_tti; k++) {
      for (i=0; i<12*frame_parms->N_RB_DL/2; i++) {
        I[ind] = pdsch_comp1[(2*frame_parms->N_RB_DL*12*k)+4*i];
        Q[ind] = pdsch_comp1[(2*frame_parms->N_RB_DL*12*k)+4*i+1];
        ind++;
      }
    }

    fl_set_xyplot_data(form->pdsch_comp1,I,Q,ind,"","","");
  }

  if (pdsch_mag1 != NULL) {
    ind=0;

    for (k=0; k<frame_parms->symbols_per_tti; k++) {
      for (i=0; i<12*frame_parms->N_RB_DL/2; i++) {
        I[ind] = pdsch_mag1[(2*frame_parms->N_RB_DL*12*k)+4*i]*cos(i*2*M_PI/(12*frame_parms->N_RB_DL/2));
        Q[ind] = pdsch_mag1[(2*frame_parms->N_RB_DL*12*k)+4*i+1]*sin(i*2*M_PI/(12*frame_parms->N_RB_DL/2));
        ind++;
      }
    }

    fl_add_xyplot_overlay(form->pdsch_comp1,1,I,Q,ind,FL_GREEN);
  }

  if (pdsch_magb1 != NULL) {
    ind=0;

    for (k=0; k<frame_parms->symbols_per_tti; k++) {
      for (i=0; i<12*frame_parms->N_RB_DL/2; i++) {
        I[ind] = pdsch_magb1[(2*frame_parms->N_RB_DL*12*k)+4*i]*cos(i*2*M_PI/(12*frame_parms->N_RB_DL/2));
        Q[ind] = pdsch_magb1[(2*frame_parms->N_RB_DL*12*k)+4*i+1]*sin(i*2*M_PI/(12*frame_parms->N_RB_DL/2));
        ind++;
      }
    }

    fl_add_xyplot_overlay(form->pdsch_comp1,2,I,Q,ind,FL_RED);
  }

  if ((pdsch_mag1 != NULL) && (pdsch_magb1 != NULL)) {
    ind=0;

    for (k=0; k<frame_parms->symbols_per_tti; k++) {
      for (i=0; i<12*frame_parms->N_RB_DL/2; i++) {
        I[ind] =
          (pdsch_mag1[(2*frame_parms->N_RB_DL*12*k)+4*i]+
           pdsch_magb1[(2*frame_parms->N_RB_DL*12*k)+4*i])*
          cos(i*2*M_PI/(12*frame_parms->N_RB_DL/2));
        Q[ind] =
          (pdsch_mag1[(2*frame_parms->N_RB_DL*12*k)+4*i+1]+
           pdsch_magb1[(2*frame_parms->N_RB_DL*12*k)+4*i+1])*
          sin(i*2*M_PI/(12*frame_parms->N_RB_DL/2));
        ind++;
      }
    }

    fl_add_xyplot_overlay(form->pdsch_comp1,3,I,Q,ind,FL_BLUE);
  }

  /*
  // PDSCH Throughput
  memcpy((void*)tput_time_ue[UE_id],(void*)&tput_time_ue[UE_id][1],(TPUT_WINDOW_LENGTH-1)*sizeof(float));
  memcpy((void*)tput_ue[UE_id],(void*)&tput_ue[UE_id][1],(TPUT_WINDOW_LENGTH-1)*sizeof(float));
  tput_time_ue[UE_id][TPUT_WINDOW_LENGTH-1]  = (float) frame;
  tput_ue[UE_id][TPUT_WINDOW_LENGTH-1] = ((float) total_dlsch_bitrate)/1000.0;
  if (tput_ue[UE_id][TPUT_WINDOW_LENGTH-1] > tput_ue_max[UE_id]) {
      tput_ue_max[UE_id] = tput_ue[UE_id][TPUT_WINDOW_LENGTH-1];
  }
  fl_set_xyplot_data(form->pdsch_tput,tput_time_ue[UE_id],tput_ue[UE_id],TPUT_WINDOW_LENGTH,"","","");
  fl_set_xyplot_ybounds(form->pdsch_tput,0,tput_ue_max[UE_id]);
  */
  fl_unfreeze_form(form->lte_phy_scope_ue);
  fl_check_forms();
  free(I);
  free(Q);
  free(chest_f_abs);
  free(llr0);
  free(bit0);
  free(llr1);
  free(bit1);
  free(bit_pdcch);
  free(llr_pdcch);
  //This is done to avoid plotting old data when TB0 is disabled, and TB1 is mapped onto CW0
  /*if (phy_vars_ue->transmission_mode[eNB_id]==3 && phy_vars_ue->transmission_mode[eNB_id]==4){
    for (int i = 0; i<8; ++i)
      for (int j = 0; j < 7*2*frame_parms->N_RB_DL*12+4; ++j )
        phy_vars_ue->pdsch_vars[subframe&0x1][eNB_id]->rxdataF_comp1[0][0][i][j]=0;

    for (int m=0; m<coded_bits_per_codeword1; ++m)
        phy_vars_ue->pdsch_vars[subframe&0x1][eNB_id]->llr[0][m]=0;
    }*/
}
