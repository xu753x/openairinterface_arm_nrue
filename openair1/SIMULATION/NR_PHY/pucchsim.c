/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include "common/config/config_userapi.h"
#include "common/utils/LOG/log.h"
#include "common/ran_context.h" 
#include "PHY/types.h"
#include "PHY/defs_nr_common.h"
#include "PHY/defs_nr_UE.h"
#include "PHY/defs_gNB.h"
#include "PHY/NR_REFSIG/refsig_defs_ue.h"
#include "PHY/MODULATION/modulation_eNB.h"
#include "PHY/MODULATION/modulation_UE.h"
#include "PHY/NR_ESTIMATION/nr_ul_estimation.h"
#include "PHY/INIT/phy_init.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "PHY/NR_UE_TRANSPORT/pucch_nr.h"
#include "SCHED_NR/sched_nr.h"
#include "openair1/SIMULATION/TOOLS/sim.h"
#include "openair1/SIMULATION/RF/rf.h"
#include "openair1/SIMULATION/NR_PHY/nr_unitary_defs.h"
#include "openair1/SIMULATION/NR_PHY/nr_dummy_functions.c"

PHY_VARS_gNB *gNB;
PHY_VARS_NR_UE *UE;
RAN_CONTEXT_t RC;
openair0_config_t openair0_cfg[MAX_CARDS];
int32_t uplink_frequency_offset[MAX_NUM_CCs][4];

double cpuf;
uint8_t nfapi_mode = 0;
uint16_t NB_UE_INST = 1;
uint8_t const nr_rv_round_map[4] = {0, 2, 1, 3};
uint8_t const nr_rv_round_map_ue[4] = {0, 2, 1, 3};

// needed for some functions
PHY_VARS_NR_UE * PHY_vars_UE_g[1][1]={{NULL}};

void init_downlink_harq_status(NR_DL_UE_HARQ_t *dl_harq) {}

int main(int argc, char **argv)
{
  char c;
  int i,aa=0;//,l;
  double sigma2, sigma2_dB=10,SNR,snr0=-2.0,snr1=2.0;
  double cfo=0;
  uint8_t snr1set=0;
  int **txdataF,**rxdataF;
  double **s_re,**s_im,**r_re,**r_im;
  //int sync_pos, sync_pos_slot;
  //FILE *rx_frame_file;
  FILE *output_fd = NULL;
  //uint8_t write_output_file=0;
  //int result;
  //int freq_offset;
  //int subframe_offset;
  //char fname[40], vname[40];
  int trial,n_trials=100,n_errors=0,ack_nack_errors=0,sr_errors=0;
  uint8_t transmission_mode = 1,n_tx=1,n_rx=1;
  uint16_t Nid_cell=0;
  uint64_t SSB_positions=0x01;
  channel_desc_t *UE2gNB;
  int format=0;
  //uint8_t extended_prefix_flag=0;
  FILE *input_fd=NULL;
  //uint8_t nacktoack_flag=0;
  int16_t amp=0x7FFF;
  int nr_slot_tx=0;
  uint64_t actual_payload=0,payload_received;
  int nr_bit=1; // maximum value possible is 2
  uint8_t m0=0;// higher layer paramater initial cyclic shift
  uint8_t nrofSymbols=1; //number of OFDM symbols can be 1-2 for format 1
  uint8_t startingSymbolIndex=0; // resource allocated see 9.2.1, 38.213 for more info.should be actually present in the resource set provided
  uint16_t startingPRB=0,startingPRB_intraSlotHopping=0; //PRB number not sure see 9.2.1, 38.213 for more info. Should be actually present in the resource set provided
  uint16_t nrofPRB=2;
  uint8_t timeDomainOCC=0;
  SCM_t channel_model=AWGN;//Rayleigh1_anticorr;

  double DS_TDL = .03;
  
  int N_RB_DL=273,mu=1;
  float target_error_rate=0.001;
  int frame_length_complex_samples;
  //int frame_length_complex_samples_no_prefix;
  NR_DL_FRAME_PARMS *frame_parms;
  //unsigned char frame_type = 0;
  int loglvl=OAILOG_WARNING;
  int sr_flag = 0;

  cpuf = get_cpu_freq_GHz();

  if ( load_configmodule(argc,argv,CONFIG_ENABLECMDLINEONLY) == 0) {
    exit_fun("[NR_PUCCHSIM] Error, configuration module init failed\n");
  }

  randominit(0);
  logInit();

  while ((c = getopt (argc, argv, "f:hA:f:g:i:I:P:B:b:T:m:n:r:o:s:S:x:y:z:N:F:GR:IL:q:c")) != -1) {
    switch (c) {
    case 'f':
      //write_output_file=1;
      output_fd = fopen(optarg,"w");

      if (output_fd==NULL) {
        printf("Error opening %s\n",optarg);
        exit(-1);
      }
      break;



    case 'g':
      switch((char)*optarg) {
      case 'A':
        channel_model=SCM_A;
        break;

      case 'B':
        channel_model=SCM_B;
        break;

      case 'C':
        channel_model=SCM_C;
        break;

      case 'D':
        channel_model=SCM_D;
        break;

      case 'E':
        channel_model=EPA;
        break;

      case 'F':
        channel_model=EVA;
        break;

      case 'G':
        channel_model=ETU;
        break;
      case 'H':
        channel_model = TDL_C;
	DS_TDL = .030; // 30 ns
	break;
  
      case 'I':
	channel_model = TDL_C;
	DS_TDL = .3;  // 300ns
        break;
     
      case 'J':
	channel_model=TDL_D;
	DS_TDL = .03;
	break;

      default:
        printf("Unsupported channel model!\n");
        exit(-1);
      }
      break;

    case 'n':
      n_trials = atoi(optarg);
      break;

    case 'o':
      cfo = atof(optarg);
      printf("Setting CFO to %f Hz\n",cfo);
      break;

    case 's':
      snr0 = atof(optarg);
      printf("Setting SNR0 to %f\n",snr0);
      break;

    case 'S':
      snr1 = atof(optarg);
      snr1set=1;
      printf("Setting SNR1 to %f\n",snr1);
      break;

      /*
    case 't':
      Td= atof(optarg);
      break;

    case 'p':
      extended_prefix_flag=1;
      break;

    case 'd':
      frame_type = 1;
      break;

    case 'r':
      ricean_factor = pow(10,-.1*atof(optarg));
      if (ricean_factor>1) {
        printf("Ricean factor must be between 0 and 1\n");
        exit(-1);
      }
      break;
      */
    case 'x':
      transmission_mode=atoi(optarg);

      if ((transmission_mode!=1) &&
          (transmission_mode!=2) &&
          (transmission_mode!=6)) {
        printf("Unsupported transmission mode %d\n",transmission_mode);
        exit(-1);
      }
      break;

    case 'y':
      n_tx=atoi(optarg);

      if ((n_tx==0) || (n_tx>2)) {
        printf("Unsupported number of tx antennas %d\n",n_tx);
        exit(-1);
      }
      break;

    case 'z':
      n_rx=atoi(optarg);

      if ((n_rx==0) || (n_rx>2)) {
        printf("Unsupported number of rx antennas %d\n",n_rx);
        exit(-1);
      }
      break;

    case 'N':
      Nid_cell = atoi(optarg);
      break;

    case 'R':
      N_RB_DL = atoi(optarg);
      break;

    case 'F':
      input_fd = fopen(optarg,"r");

      if (input_fd==NULL) {
        printf("Problem with filename %s\n",optarg);
        exit(-1);
      }
      break;

    case 'L':
      loglvl = atoi(optarg);
      break;
    case 'i':
      nrofSymbols=(uint8_t)atoi(optarg);
      break;
    case 'I':
      startingSymbolIndex=(uint8_t)atoi(optarg);
      break;
    case 'r':
      startingPRB=atoi(optarg);
      break;
    case 'q':
      nrofPRB=atoi(optarg);
      break;
    case 'P':
      format=atoi(optarg);
      break;
    case 'm':
      m0=atoi(optarg);
      break;
    case 'b':
      nr_bit=atoi(optarg);
      break;
    case 'c':
      sr_flag=1;
      break;
    case 'B':
      actual_payload=atoi(optarg);
      break;
    case 'T':
      //nacktoack_flag=(uint8_t)atoi(optarg);
      target_error_rate=0.001;
      break;
    default:
    case 'h':
      printf("%s -h(elp) -p(extended_prefix) -N cell_id -f output_filename -F input_filename -g channel_model -n n_frames -t Delayspread -s snr0 -S snr1 -x transmission_mode -y TXant -z RXant -i Intefrence0 -j Interference1 -A interpolation_file -C(alibration offset dB) -N CellId\n", argv[0]);
      printf("-h This message\n");
      printf("-p Use extended prefix mode\n");
      printf("-d Use TDD\n");
      printf("-n Number of frames to simulate\n");
      printf("-s Starting SNR, runs from SNR0 to SNR0 + 5 dB.  If n_frames is 1 then just SNR is simulated\n");
      printf("-S Ending SNR, runs from SNR0 to SNR1\n");
      printf("-t Delay spread for multipath channel\n");
      printf("-g [A,B,C,D,E,F,G] Use 3GPP SCM (A,B,C,D) or 36-101 (E-EPA,F-EVA,G-ETU) models (ignores delay spread and Ricean factor)\n");
      printf("-x Transmission mode (1,2,6 for the moment)\n");
      printf("-y Number of TX antennas used in eNB\n");
      printf("-z Number of RX antennas used in UE\n");
      printf("-i Relative strength of first intefering eNB (in dB) - cell_id mod 3 = 1\n");
      printf("-j Relative strength of second intefering eNB (in dB) - cell_id mod 3 = 2\n");
      printf("-o Carrier frequency offset in Hz\n");
      printf("-N Nid_cell\n");
      printf("-R N_RB_DL\n");
      printf("-O oversampling factor (1,2,4,8,16)\n");
      printf("-A Interpolation_filname Run with Abstraction to generate Scatter plot using interpolation polynomial in file\n");
      //printf("-C Generate Calibration information for Abstraction (effective SNR adjustment to remove Pe bias w.r.t. AWGN)\n");
      printf("-f Output filename (.txt format) for Pe/SNR results\n");
      printf("-F Input filename (.txt format) for RX conformance testing\n");
      printf("-i Enter number of ofdm symbols for pucch\n");
      printf("-I Starting symbol index for pucch\n");
      printf("-r PUCCH starting PRB\n");
      printf("-q PUCCH number of PRB\n");
      printf("-P Enter the format of PUCCH\n");
      printf("-b number of HARQ bits (1-2)\n");
      printf("-B payload to be transmitted on PUCCH\n");
      printf("-m initial cyclic shift m0\n");
      printf("-T to check nacktoack miss for format 1");
      exit (-1);
      break;
    }
  }

  set_glog(loglvl);

  if (snr1set==0) snr1 = snr0+10;

  printf("Initializing gNodeB for mu %d, N_RB_DL %d\n",mu,N_RB_DL);

  if((format!=0) && (format!=1) && (format!=2)){
    printf("PUCCH format %d not supported\n",format);
    exit(0); 
  }

  AssertFatal(((format < 2)&&(nr_bit<3)&&(actual_payload<4)) ||
	      ((format == 2)&&(nr_bit>2)&&(nr_bit<65)),"illegal combination format %d, nr_bit %d\n",
	      format,nr_bit);

  actual_payload &= ((1<<nr_bit)-1);

  printf("Transmitted payload is %ld\n",actual_payload);

  RC.gNB = (PHY_VARS_gNB**) malloc(sizeof(PHY_VARS_gNB *));
  RC.gNB[0] = malloc(sizeof(PHY_VARS_gNB));
  gNB = RC.gNB[0];
  memset((void*)gNB,0,sizeof(*gNB));
  frame_parms = &gNB->frame_parms; //to be initialized I suppose (maybe not necessary for PBCH)
  frame_parms->nb_antennas_tx = n_tx;
  frame_parms->nb_antennas_rx = n_rx;
  frame_parms->N_RB_DL = N_RB_DL;
  frame_parms->N_RB_UL = N_RB_DL;
  frame_parms->Nid_cell = Nid_cell;

  nfapi_nr_config_request_scf_t *cfg = &gNB->gNB_config;
  cfg->carrier_config.num_tx_ant.value = n_tx;
  cfg->carrier_config.num_rx_ant.value = n_rx;
  nr_phy_config_request_sim(gNB,N_RB_DL,N_RB_DL,mu,Nid_cell,SSB_positions);
  phy_init_nr_gNB(gNB,0,0);

  double fs,bw,scs,eps;

  if (mu == 1 && N_RB_DL == 217) { 
    fs = 122.88e6;
    bw = 80e6;
    scs = 30000;
  }					       
  else if (mu == 1 && N_RB_DL == 245) {
    fs = 122.88e6;
    bw = 90e6;
    scs = 30000;
  }
  else if (mu == 1 && N_RB_DL == 273) {
    fs = 122.88e6;
    bw = 100e6;
    scs = 30000;
  }
  else if (mu == 1 && N_RB_DL == 106) { 
    fs = 61.44e6;
    bw = 40e6;
    scs = 30000;
  }
  else AssertFatal(1==0,"Unsupported numerology for mu %d, N_RB %d\n",mu, N_RB_DL);

  // cfo with respect to sub-carrier spacing
  eps = cfo/scs;

  // computation of integer and fractional FO to compare with estimation results
  int IFO;
  if(eps!=0.0){
	printf("Introducing a CFO of %lf relative to SCS of %d kHz\n",eps,(int)(scs/1000));
	if (eps>0)	
  	  IFO=(int)(eps+0.5);
	else
	  IFO=(int)(eps-0.5);
	printf("FFO = %lf; IFO = %d\n",eps-IFO,IFO);
  }

  UE2gNB = new_channel_desc_scm(n_tx, n_rx, channel_model, fs, bw, DS_TDL,0, 0, 0);

  if (UE2gNB==NULL) {
    printf("Problem generating channel model. Exiting.\n");
    exit(-1);
  }

  frame_length_complex_samples = frame_parms->samples_per_subframe*NR_NUMBER_OF_SUBFRAMES_PER_FRAME;
  //frame_length_complex_samples_no_prefix = frame_parms->samples_per_subframe_wCP;

  s_re = malloc(2*sizeof(double*));
  s_im = malloc(2*sizeof(double*));
  r_re = malloc(2*sizeof(double*));
  r_im = malloc(2*sizeof(double*));
  txdataF = malloc(2*sizeof(int*));
  rxdataF = malloc(2*sizeof(int*));
  gNB->common_vars.rxdataF=rxdataF;
  memcpy((void*)&gNB->frame_parms,(void*)frame_parms,sizeof(frame_parms));
  for (i=0; i<2; i++) {

    s_re[i] = malloc(frame_length_complex_samples*sizeof(double));
    bzero(s_re[i],frame_length_complex_samples*sizeof(double));
    s_im[i] = malloc(frame_length_complex_samples*sizeof(double));
    bzero(s_im[i],frame_length_complex_samples*sizeof(double));

    r_re[i] = malloc(frame_length_complex_samples*sizeof(double));
    bzero(r_re[i],frame_length_complex_samples*sizeof(double));
    r_im[i] = malloc(frame_length_complex_samples*sizeof(double));
    bzero(r_im[i],frame_length_complex_samples*sizeof(double));

    printf("Allocating %d samples for txdataF/rxdataF\n",14*frame_parms->ofdm_symbol_size);
    txdataF[i] = memalign(32,14*frame_parms->ofdm_symbol_size*sizeof(int));
    bzero(txdataF[i],14*frame_parms->ofdm_symbol_size*sizeof(int));
    rxdataF[i] = memalign(32,14*frame_parms->ofdm_symbol_size*sizeof(int));
    bzero(rxdataF[i],14*frame_parms->ofdm_symbol_size*sizeof(int));
  }


  //configure UE
  UE = malloc(sizeof(PHY_VARS_NR_UE));
  memcpy(&UE->frame_parms,frame_parms,sizeof(NR_DL_FRAME_PARMS));
  UE->pucch_config_common_nr->hoppingId = Nid_cell;
  //phy_init_nr_top(UE); //called from init_nr_ue_signal
                      
  UE->perfect_ce = 0;

  if(eps!=0.0)
	UE->UE_fo_compensation = 1; // if a frequency offset is set then perform fo estimation and compensation

  if (init_nr_ue_signal(UE, 1, 0) != 0)
  {
    printf("Error at UE NR initialisation\n");
    exit(-1);
  }
  uint8_t mcs=0;
  startingPRB_intraSlotHopping=N_RB_DL-1;
  pucch_GroupHopping_t PUCCH_GroupHopping=UE->pucch_config_common_nr->pucch_GroupHopping;
  uint32_t hopping_id=UE->pucch_config_common_nr->hoppingId;
  uint32_t dmrs_scrambling_id = 0, data_scrambling_id=0;
  //t_nrPolar_params *currentPtr;

  int shift = 0;

  if(format==0){
    if (sr_flag)
      shift = 1<<nr_bit;
    if (nr_bit ==0) 
      mcs=table1_mcs[0];
    else if(nr_bit==1)
      mcs=table1_mcs[actual_payload+shift];
    else if(nr_bit==2)
      mcs=table2_mcs[actual_payload+shift];
    else AssertFatal(1==0,"Either nr_bit %d or sr_flag %d must be non-zero\n", nr_bit, sr_flag);
  }
  else if (format == 2 && nr_bit > 11) gNB->uci_polarParams = nr_polar_params(2, nr_bit, nrofPRB, 1, NULL);
  
  for(SNR=snr0;SNR<=snr1;SNR=SNR+1){
    ack_nack_errors=0;
    sr_errors=0;
    n_errors = 0;
    for (trial=0; trial<n_trials; trial++) {
      bzero(txdataF[aa],frame_parms->ofdm_symbol_size*sizeof(int));
      if(format==0){
        nr_generate_pucch0(UE,txdataF,frame_parms,PUCCH_GroupHopping,hopping_id,amp,nr_slot_tx,m0,mcs,nrofSymbols,startingSymbolIndex,startingPRB);
      }
      else if (format == 1){
        nr_generate_pucch1(UE,txdataF,frame_parms,UE->pucch_config_dedicated,actual_payload,amp,nr_slot_tx,m0,nrofSymbols,startingSymbolIndex,startingPRB,startingPRB_intraSlotHopping,0,nr_bit);
      }
      else {
	nr_generate_pucch2(UE,0x1234,dmrs_scrambling_id,data_scrambling_id,txdataF,frame_parms,UE->pucch_config_dedicated,actual_payload,amp,nr_slot_tx,nrofSymbols,startingSymbolIndex,nrofPRB,startingPRB,nr_bit);
      }
      
      int txlev = signal_energy(&txdataF[aa][startingSymbolIndex*frame_parms->ofdm_symbol_size],
				frame_parms->ofdm_symbol_size);
      //      printf("txlev %d (%d dB), offset %d\n",txlev,dB_fixed(txlev),startingSymbolIndex*frame_parms->ofdm_symbol_size);
	    
      // note : this scaling
      int nb_re = (format == 0 || format == 1)? 12 : 12*nrofPRB;
      sigma2_dB = 10*log10((double)txlev*UE->frame_parms.ofdm_symbol_size/nb_re)-SNR;
      sigma2 = pow(10,sigma2_dB/10);
      
      for(i=startingSymbolIndex*frame_parms->ofdm_symbol_size; i<(startingSymbolIndex+1)*frame_parms->ofdm_symbol_size; i++) {
        ((int16_t*)rxdataF[aa])[i<<1] = (int16_t)(100.0*((double)(((int16_t *)txdataF[aa])[(i<<1)]) + sqrt(sigma2/2)*gaussdouble(0.0,1.0))/sqrt((double)txlev));
        ((int16_t*)rxdataF[aa])[1+(i<<1)]=(int16_t)(100.0*((double)(((int16_t *)txdataF[aa])[(i<<1)+1])+ sqrt(sigma2/2)*gaussdouble(0.0,1.0))/sqrt((double)txlev));
      }
      int rxlev = signal_energy(&rxdataF[aa][startingSymbolIndex*frame_parms->ofdm_symbol_size],
				frame_parms->ofdm_symbol_size);

      // noise measurement
      gNB->ulmask_symb = startingSymbolIndex;
      for (int rb=0; rb<nrofPRB; rb++) {
        int rb2 = rb+startingPRB;
        gNB->rb_mask_ul[rb2>>5] |= (1<<(rb2&31));
      }
      gNB_I0_measurements(gNB);

      if (n_trials==1) printf("rxlev %d (%d dB), sigma2 %f dB, SNR %f, TX %f\n",rxlev,dB_fixed(rxlev),sigma2_dB,SNR,10*log10((double)txlev*UE->frame_parms.ofdm_symbol_size/12));
      if(format==0){
        nfapi_nr_uci_pucch_pdu_format_0_1_t uci_pdu;
        nfapi_nr_pucch_pdu_t pucch_pdu;
        pucch_pdu.subcarrier_spacing    = 1;
        pucch_pdu.group_hop_flag        = PUCCH_GroupHopping&1;
        pucch_pdu.sequence_hop_flag     = (PUCCH_GroupHopping>>1)&1;
        pucch_pdu.bit_len_harq          = nr_bit;
        pucch_pdu.bit_len_csi_part1     = 0;
        pucch_pdu.bit_len_csi_part2     = 0;
        pucch_pdu.sr_flag               = sr_flag;
        pucch_pdu.nr_of_symbols         = nrofSymbols;
        pucch_pdu.hopping_id            = hopping_id;
        pucch_pdu.initial_cyclic_shift  = 0;
        pucch_pdu.start_symbol_index    = startingSymbolIndex;
        pucch_pdu.prb_start             = startingPRB;
        nr_decode_pucch0(gNB,nr_slot_tx,&uci_pdu,&pucch_pdu);
        if(sr_flag==1){
          if (uci_pdu.sr->sr_indication == 0 || uci_pdu.sr->sr_confidence_level == 1)
            sr_errors+=1;
        }
        if(nr_bit>0){
          if(nr_bit==1)
            ack_nack_errors+=(actual_payload^uci_pdu.harq->harq_list[0].harq_value);
          else
            ack_nack_errors+=(((actual_payload&1)^uci_pdu.harq->harq_list[0].harq_value)+((actual_payload>>1)^uci_pdu.harq->harq_list[1].harq_value));
	  free(uci_pdu.harq->harq_list);
        }
      }
      else if (format==1) {
        nr_decode_pucch1(rxdataF,PUCCH_GroupHopping,hopping_id,
                         &(payload_received),frame_parms,amp,nr_slot_tx,
                         m0,nrofSymbols,startingSymbolIndex,startingPRB,
                         startingPRB_intraSlotHopping,timeDomainOCC,nr_bit);
        if(nr_bit==1)
          ack_nack_errors+=((actual_payload^payload_received)&1);
        else
          ack_nack_errors+=((actual_payload^payload_received)&1) + (((actual_payload^payload_received)&2)>>1);
      }
      else if (format==2) {
        nfapi_nr_uci_pucch_pdu_format_2_3_4_t uci_pdu;
        nfapi_nr_pucch_pdu_t pucch_pdu;
        pucch_pdu.rnti = 0x1234;
        pucch_pdu.subcarrier_spacing    = 1;
        pucch_pdu.group_hop_flag        = PUCCH_GroupHopping&1;
        pucch_pdu.sequence_hop_flag     = (PUCCH_GroupHopping>>1)&1;
        pucch_pdu.bit_len_csi_part1     = nr_bit;
        pucch_pdu.bit_len_harq          = 0;
        pucch_pdu.bit_len_csi_part2     = 0;
        pucch_pdu.sr_flag               = 0;
        pucch_pdu.nr_of_symbols         = nrofSymbols;
        pucch_pdu.hopping_id            = hopping_id;
        pucch_pdu.initial_cyclic_shift  = 0;
        pucch_pdu.start_symbol_index    = startingSymbolIndex;
        pucch_pdu.prb_size              = nrofPRB;
        pucch_pdu.prb_start             = startingPRB;
        pucch_pdu.dmrs_scrambling_id    = dmrs_scrambling_id;
        pucch_pdu.data_scrambling_id    = data_scrambling_id;
        nr_decode_pucch2(gNB,nr_slot_tx,&uci_pdu,&pucch_pdu);
        int csi_part1_bytes=pucch_pdu.bit_len_csi_part1>>3;
        if ((pucch_pdu.bit_len_csi_part1&7) > 0) csi_part1_bytes++;
        for (int i=0;i<csi_part1_bytes;i++) {
          if (uci_pdu.csi_part1.csi_part1_payload[i] != ((uint8_t*)&actual_payload)[i]) {
            ack_nack_errors++;
            break;
          }
        }
        free(uci_pdu.csi_part1.csi_part1_payload);

      }
      n_errors=((actual_payload^payload_received)&1)+(((actual_payload^payload_received)&2)>>1)+(((actual_payload^payload_received)&4)>>2)+n_errors;
    }
    if (sr_flag == 1)
      printf("SR: SNR=%f, n_trials=%d, n_bit_errors=%d\n",SNR,n_trials,sr_errors);
    if(nr_bit > 0)
      printf("ACK/NACK: SNR=%f, n_trials=%d, n_bit_errors=%d\n",SNR,n_trials,ack_nack_errors);
    if((float)(ack_nack_errors+sr_errors)/(float)(n_trials)<=target_error_rate){
      printf("PUCCH test OK\n");
      break;
    }
  }

  for (i=0; i<2; i++) {
    free(s_re[i]);
    free(s_im[i]);
    free(r_re[i]);
    free(r_im[i]);
    free(txdataF[i]);
    free(rxdataF[i]);
  }
  free(s_re);
  free(s_im);
  free(r_re);
  free(r_im);
  free(txdataF);
  free(rxdataF);

  if (output_fd) fclose(output_fd);
  if (input_fd)  fclose(input_fd);

  return(n_errors);
}
