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

#include <fcntl.h>
#include <math.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "common/ran_context.h"
#include "common/config/config_userapi.h"
#include "common/utils/LOG/log.h"
#include "LAYER2/NR_MAC_gNB/nr_mac_gNB.h"
#include "LAYER2/NR_MAC_UE/mac_defs.h"
#include "LAYER2/NR_MAC_UE/mac_extern.h"
#include "PHY/defs_gNB.h"
#include "PHY/defs_nr_common.h"
#include "PHY/defs_nr_UE.h"
#include "PHY/phy_vars_nr_ue.h"
#include "PHY/types.h"
#include "PHY/INIT/phy_init.h"
#include "PHY/MODULATION/modulation_eNB.h"
#include "PHY/MODULATION/nr_modulation.h"
#include "PHY/MODULATION/modulation_UE.h"
#include "PHY/NR_REFSIG/refsig_defs_ue.h"
#include "PHY/NR_TRANSPORT/nr_dlsch.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "SCHED_NR/fapi_nr_l1.h"
#include "SCHED_NR/sched_nr.h"
#include "SCHED_NR_UE/defs.h"
#include "SCHED_NR_UE/fapi_nr_ue_l1.h"
#include "NR_PHY_INTERFACE/NR_IF_Module.h"
#include "NR_UE_PHY_INTERFACE/NR_IF_Module.h"

#include "LAYER2/NR_MAC_UE/mac_proto.h"
//#include "LAYER2/NR_MAC_gNB/mac_proto.h"
//#include "openair2/LAYER2/NR_MAC_UE/mac_proto.h"
#include "LAYER2/NR_MAC_gNB/mac_proto.h"
#include "NR_asn_constant.h"
#include "RRC/NR/MESSAGES/asn1_msg.h"
#include "openair1/SIMULATION/RF/rf.h"
#include "openair1/SIMULATION/TOOLS/sim.h"
#include "openair1/SIMULATION/NR_PHY/nr_unitary_defs.h"
//#include "openair1/SIMULATION/NR_PHY/nr_dummy_functions.c"

#include "NR_RRCReconfiguration.h"
#define inMicroS(a) (((double)(a))/(cpu_freq_GHz*1000.0))
#include "SIMULATION/LTE_PHY/common_sim.h"

#include <openair2/LAYER2/MAC/mac_vars.h>
#include <openair2/RRC/LTE/rrc_vars.h>

LCHAN_DESC DCCH_LCHAN_DESC,DTCH_DL_LCHAN_DESC,DTCH_UL_LCHAN_DESC;
rlc_info_t Rlc_info_um,Rlc_info_am_config;

PHY_VARS_gNB *gNB;
PHY_VARS_NR_UE *UE;
RAN_CONTEXT_t RC;
int32_t uplink_frequency_offset[MAX_NUM_CCs][4];

double cpuf;

int sf_ahead=4 ;
int sl_ahead=0;
uint8_t nfapi_mode = 0;
uint64_t downlink_frequency[MAX_NUM_CCs][4];

// dummy functions
int dummy_nr_ue_ul_indication(nr_uplink_indication_t *ul_info)              { return(0);  }

int8_t nr_mac_rrc_data_ind_ue(const module_id_t     module_id,
			      const int             CC_id,
			      const uint8_t         gNB_index,
			      const int8_t          channel,
			      const uint8_t*        pduP,
			      const sdu_size_t      pdu_len)
{
  return 0;
}

void
rrc_data_ind(
  const protocol_ctxt_t *const ctxt_pP,
  const rb_id_t                Srb_id,
  const sdu_size_t             sdu_sizeP,
  const uint8_t   *const       buffer_pP
)
{
}

int
gtpv1u_create_s1u_tunnel(
  const instance_t                              instanceP,
  const gtpv1u_enb_create_tunnel_req_t *const  create_tunnel_req_pP,
  gtpv1u_enb_create_tunnel_resp_t *const create_tunnel_resp_pP
) {
  return 0;
}

int
rrc_gNB_process_GTPV1U_CREATE_TUNNEL_RESP(
  const protocol_ctxt_t *const ctxt_pP,
  const gtpv1u_enb_create_tunnel_resp_t *const create_tunnel_resp_pP,
  uint8_t                         *inde_list
) {
  return 0;
}

void config_common(int Mod_idP,
                   int pdsch_AntennaPorts, 
		   NR_ServingCellConfigCommon_t *scc
		   );

int generate_dlsch_header(unsigned char *mac_header,
                          unsigned char num_sdus,
                          unsigned short *sdu_lengths,
                          unsigned char *sdu_lcids,
                          unsigned char drx_cmd,
                          unsigned short timing_advance_cmd,
                          unsigned char *ue_cont_res_id,
                          unsigned char short_padding,
                          unsigned short post_padding){return 0;}

// Dummy function to avoid linking error at compilation of nr-dlsim
int is_x2ap_enabled(void)
{
  return 0;
}

// needed for some functions
openair0_config_t openair0_cfg[MAX_CARDS];


/* specific dlsim DL preprocessor: uses rbStart/rbSize/mcs from command line of
   dlsim, does not search for CCE/PUCCH occasion but simply sets to 0 */
int g_mcsIndex = -1, g_rbStart = -1, g_rbSize = -1;
void nr_dlsim_preprocessor(module_id_t module_id,
                           frame_t frame,
                           sub_frame_t slot,
                           int num_slots_per_tdd) {
  NR_UE_info_t *UE_info = &RC.nrmac[module_id]->UE_info;
  AssertFatal(UE_info->num_UEs == 1, "can have only a single UE\n");
  NR_UE_sched_ctrl_t *sched_ctrl = &UE_info->UE_sched_ctrl[0];

  /* manually set free CCE to 0 */
  const int target_ss = NR_SearchSpace__searchSpaceType_PR_ue_Specific;
  sched_ctrl->search_space = get_searchspace(sched_ctrl->active_bwp, target_ss);
  uint8_t nr_of_candidates;
  find_aggregation_candidates(&sched_ctrl->aggregation_level,
                              &nr_of_candidates,
                              sched_ctrl->search_space);
  sched_ctrl->coreset = get_coreset(
      sched_ctrl->active_bwp, sched_ctrl->search_space, 1 /* dedicated */);
  sched_ctrl->cce_index = 0;

  /* set "any" value for PUCCH (simulator evaluates PDSCH only) */
  sched_ctrl->pucch_sched_idx = 0;
  sched_ctrl->pucch_occ_idx = 0;

  sched_ctrl->rbStart = g_rbStart;
  sched_ctrl->rbSize = g_rbSize;
  sched_ctrl->mcs = g_mcsIndex;
  sched_ctrl->time_domain_allocation = 2;
  sched_ctrl->mcsTableIdx = 0;
  AssertFatal(sched_ctrl->rbStart >= 0, "invalid rbStart %d\n", sched_ctrl->rbStart);
  AssertFatal(sched_ctrl->rbSize > 0, "invalid rbSize %d\n", sched_ctrl->rbSize);
  AssertFatal(sched_ctrl->mcs >= 0, "invalid sched_ctrl->mcs %d\n", sched_ctrl->mcs);
  sched_ctrl->numDmrsCdmGrpsNoData = 1;
}


int main(int argc, char **argv)
{
  char c;
  int i,aa;//,l;
  double sigma2, sigma2_dB=10, SNR, snr0=-2.0, snr1=2.0;
  uint8_t snr1set=0;
  float roundStats[50];
  float effRate;
  //float psnr;
  float eff_tp_check = 0.7;
  uint8_t snrRun;
  uint32_t TBS;
  int **txdata;
  double **s_re,**s_im,**r_re,**r_im;
  //double iqim = 0.0;
  //unsigned char pbch_pdu[6];
  //  int sync_pos, sync_pos_slot;
  //  FILE *rx_frame_file;
  FILE *output_fd = NULL;
  //uint8_t write_output_file=0;
  //int result;
  //int freq_offset;
  //  int subframe_offset;
  //  char fname[40], vname[40];
  int trial, n_trials = 1, n_errors = 0, n_false_positive = 0;
  //int n_errors2, n_alamouti;
  uint8_t transmission_mode = 1,n_tx=1,n_rx=1;
  uint8_t round;
  uint8_t num_rounds = 4;

  channel_desc_t *gNB2UE;
  //uint32_t nsymb,tx_lev,tx_lev1 = 0,tx_lev2 = 0;
  //uint8_t extended_prefix_flag=0;
  //int8_t interf1=-21,interf2=-21;

  FILE *input_fd=NULL,*pbch_file_fd=NULL;
  //char input_val_str[50],input_val_str2[50];

  //uint8_t frame_mod4,num_pdcch_symbols = 0;

  SCM_t channel_model=AWGN;//Rayleigh1_anticorr;

  NB_UE_INST = 1;
  //double pbch_sinr;
  //int pbch_tx_ant;
  int N_RB_DL=106,mu=1;

  //unsigned char frame_type = 0;

  int frame=1,slot=1;
  int frame_length_complex_samples;
  //int frame_length_complex_samples_no_prefix;
  NR_DL_FRAME_PARMS *frame_parms;
  UE_nr_rxtx_proc_t UE_proc;
  NR_Sched_Rsp_t Sched_INFO;
  gNB_MAC_INST *gNB_mac;
  NR_UE_MAC_INST_t *UE_mac;
  int cyclic_prefix_type = NFAPI_CP_NORMAL;
  int run_initial_sync=0;
  int pusch_tgt_snrx10 = 200;
  int pucch_tgt_snrx10 = 200;
  int loglvl=OAILOG_INFO;

  //float target_error_rate = 0.01;
  int css_flag=0;

  cpuf = get_cpu_freq_GHz();

  if ( load_configmodule(argc,argv,CONFIG_ENABLECMDLINEONLY) == 0) {
    exit_fun("[NR_DLSIM] Error, configuration module init failed\n");
  }

  randominit(0);

  int print_perf             = 0;

  FILE *scg_fd=NULL;
  
  while ((c = getopt (argc, argv, "f:hA:pf:g:i:j:n:s:S:t:x:y:z:M:N:F:GR:dPIL:Ea:b:e:m:w")) != -1) {
    switch (c) {
    case 'f':
      scg_fd = fopen(optarg,"r");

      if (scg_fd==NULL) {
        printf("Error opening %s\n",optarg);
        exit(-1);
      }
      break;

    /*case 'd':
      frame_type = 1;
      break;*/

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

      default:
        printf("Unsupported channel model!\n");
        exit(-1);
      }

      break;

    /*case 'i':
      interf1=atoi(optarg);
      break;

    case 'j':
      interf2=atoi(optarg);
      break;*/

    case 'n':
      n_trials = atoi(optarg);
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
      */
    /*case 'p':
      extended_prefix_flag=1;
      break;*/

      /*
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

    case 'P':
      print_perf=1;
      opp_enabled=1;
      break;
      
    case 'I':
      run_initial_sync=1;
      //target_error_rate=0.1;
      slot = 0;
      break;

    case 'L':
      loglvl = atoi(optarg);
      break;


    case 'E':
	css_flag=1;
	break;


    case 'a':
      g_rbStart = atoi(optarg);
      break;

    case 'b':
      g_rbSize = atoi(optarg);
      break;

    case 'e':
      g_mcsIndex = atoi(optarg);
      break;

    case 'm':
      mu = atoi(optarg);
      break;

    case 't':
      eff_tp_check = (float)atoi(optarg)/100;
      break;

    case 'w':
      output_fd = fopen("txdata.dat", "w+");
      break;

    default:
    case 'h':
      printf("%s -h(elp) -p(extended_prefix) -N cell_id -f output_filename -F input_filename -g channel_model -n n_frames -t Delayspread -s snr0 -S snr1 -x transmission_mode -y TXant -z RXant -i Intefrence0 -j Interference1 -A interpolation_file -C(alibration offset dB) -N CellId\n",
             argv[0]);
      printf("-h This message\n");
      //printf("-p Use extended prefix mode\n");
      //printf("-d Use TDD\n");
      printf("-n Number of frames to simulate\n");
      printf("-s Starting SNR, runs from SNR0 to SNR0 + 5 dB.  If n_frames is 1 then just SNR is simulated\n");
      printf("-S Ending SNR, runs from SNR0 to SNR1\n");
      printf("-t Delay spread for multipath channel\n");
      printf("-g [A,B,C,D,E,F,G] Use 3GPP SCM (A,B,C,D) or 36-101 (E-EPA,F-EVA,G-ETU) models (ignores delay spread and Ricean factor)\n");
      printf("-y Number of TX antennas used in gNB\n");
      printf("-z Number of RX antennas used in UE\n");
      //printf("-i Relative strength of first intefering gNB (in dB) - cell_id mod 3 = 1\n");
      //printf("-j Relative strength of second intefering gNB (in dB) - cell_id mod 3 = 2\n");
      printf("-R N_RB_DL\n");
      printf("-O oversampling factor (1,2,4,8,16)\n");
      printf("-A Interpolation_filname Run with Abstraction to generate Scatter plot using interpolation polynomial in file\n");
      //printf("-C Generate Calibration information for Abstraction (effective SNR adjustment to remove Pe bias w.r.t. AWGN)\n");
      printf("-f raw file containing RRC configuration (generated by gNB)\n");
      printf("-F Input filename (.txt format) for RX conformance testing\n");
      printf("-E used CSS scheduler\n");
      printf("-o CORESET offset\n");
      printf("-a Start PRB for PDSCH\n");
      printf("-b Number of PRB for PDSCH\n");
      printf("-c Start symbol for PDSCH (fixed for now)\n");
      printf("-j Number of symbols for PDSCH (fixed for now)\n");
      printf("-e MSC index\n");
      printf("-t Acceptable effective throughput (in percentage)\n");
      printf("-P Print DLSCH performances\n");
      printf("-w Write txdata to binary file (one frame)\n");
      exit (-1);
      break;
    }
  }
  
  logInit();
  set_glog(loglvl);
  T_stdout = 1;

  get_softmodem_params()->phy_test = 1;
  
  if (snr1set==0)
    snr1 = snr0+10;


  RC.gNB = (PHY_VARS_gNB**) malloc(sizeof(PHY_VARS_gNB *));
  RC.gNB[0] = (PHY_VARS_gNB*) malloc(sizeof(PHY_VARS_gNB ));
  memset(RC.gNB[0],0,sizeof(PHY_VARS_gNB));

  gNB = RC.gNB[0];
  frame_parms = &gNB->frame_parms; //to be initialized I suppose (maybe not necessary for PBCH)
  frame_parms->nb_antennas_tx = n_tx;
  frame_parms->nb_antennas_rx = n_rx;
  frame_parms->N_RB_DL = N_RB_DL;
  frame_parms->N_RB_UL = N_RB_DL;

  RC.nb_nr_macrlc_inst = 1;
  RC.nb_nr_mac_CC = (int*)malloc(RC.nb_nr_macrlc_inst*sizeof(int));
  for (i = 0; i < RC.nb_nr_macrlc_inst; i++)
    RC.nb_nr_mac_CC[i] = 1;
  mac_top_init_gNB();
  gNB_mac = RC.nrmac[0];
  gNB_mac->pre_processor_dl = nr_dlsim_preprocessor;
  gNB_RRC_INST rrc;
  memset((void*)&rrc,0,sizeof(rrc));

  /*
  // read in SCGroupConfig
  AssertFatal(scg_fd != NULL,"no reconfig.raw file\n");
  char buffer[1024];
  int msg_len=fread(buffer,1,1024,scg_fd);
  NR_RRCReconfiguration_t *NR_RRCReconfiguration;

  printf("Decoding NR_RRCReconfiguration (%d bytes)\n",msg_len);
  asn_dec_rval_t dec_rval = uper_decode_complete( NULL,
						  &asn_DEF_NR_RRCReconfiguration,
						  (void **)&NR_RRCReconfiguration,
						  (uint8_t *)buffer,
						  msg_len); 
  
  if ((dec_rval.code != RC_OK) && (dec_rval.consumed == 0)) {
    AssertFatal(1==0,"NR_RRCReConfiguration decode error\n");
    // free the memory
    SEQUENCE_free( &asn_DEF_NR_RRCReconfiguration, NR_RRCReconfiguration, 1 );
    exit(-1);
  }      
  fclose(scg_fd);

  AssertFatal(NR_RRCReconfiguration->criticalExtensions.present == NR_RRCReconfiguration__criticalExtensions_PR_rrcReconfiguration,"wrong NR_RRCReconfiguration->criticalExstions.present type\n");

  NR_RRCReconfiguration_IEs_t *reconfig_ies = NR_RRCReconfiguration->criticalExtensions.choice.rrcReconfiguration;
  NR_CellGroupConfig_t *secondaryCellGroup;
  dec_rval = uper_decode_complete( NULL,
				   &asn_DEF_NR_CellGroupConfig,
				   (void **)&secondaryCellGroup,
				   (uint8_t *)reconfig_ies->secondaryCellGroup->buf,
				   reconfig_ies->secondaryCellGroup->size); 
  
  if ((dec_rval.code != RC_OK) && (dec_rval.consumed == 0)) {
    AssertFatal(1==0,"NR_CellGroupConfig decode error\n");
    // free the memory
    SEQUENCE_free( &asn_DEF_NR_CellGroupConfig, secondaryCellGroup, 1 );
    exit(-1);
  }      
  
  NR_ServingCellConfigCommon_t *scc = secondaryCellGroup->spCellConfig->reconfigurationWithSync->spCellConfigCommon;
  */


  rrc.carrier.servingcellconfigcommon = calloc(1,sizeof(*rrc.carrier.servingcellconfigcommon));

  NR_ServingCellConfigCommon_t *scc = rrc.carrier.servingcellconfigcommon;
  NR_CellGroupConfig_t *secondaryCellGroup=calloc(1,sizeof(*secondaryCellGroup));
  prepare_scc(rrc.carrier.servingcellconfigcommon);
  uint64_t ssb_bitmap;
  fill_scc(rrc.carrier.servingcellconfigcommon,&ssb_bitmap,N_RB_DL,N_RB_DL,mu,mu);

  fix_scc(scc,ssb_bitmap);

  fill_default_secondaryCellGroup(scc,
				  secondaryCellGroup,
				  0,
				  1,
				  n_tx,
				  0);

  xer_fprint(stdout, &asn_DEF_NR_CellGroupConfig, (const void*)secondaryCellGroup);

  AssertFatal((gNB->if_inst         = NR_IF_Module_init(0))!=NULL,"Cannot register interface");
  gNB->if_inst->NR_PHY_config_req      = nr_phy_config_request;
  // common configuration
  rrc_mac_config_req_gNB(0,0,1,pusch_tgt_snrx10,pucch_tgt_snrx10,scc,0,0,NULL);
  // UE dedicated configuration
  rrc_mac_config_req_gNB(0,0,1,pusch_tgt_snrx10,pucch_tgt_snrx10,NULL,1,secondaryCellGroup->spCellConfig->reconfigurationWithSync->newUE_Identity,secondaryCellGroup);
  phy_init_nr_gNB(gNB,0,0);
  N_RB_DL = gNB->frame_parms.N_RB_DL;
  // stub to configure frame_parms
  //  nr_phy_config_request_sim(gNB,N_RB_DL,N_RB_DL,mu,Nid_cell,SSB_positions);
  // call MAC to configure common parameters

  /* rrc_mac_config_req_gNB() has created one user, so set the scheduling
   * parameters from command line in global variables that will be picked up by
   * scheduling preprocessor */
  if (g_mcsIndex < 0) g_mcsIndex = 9;
  if (g_rbStart < 0) g_rbStart=0;
  if (g_rbSize < 0) g_rbSize = N_RB_DL - g_rbStart;

  double fs,bw;

  if (mu == 1 && N_RB_DL == 217) { 
    fs = 122.88e6;
    bw = 80e6;
  }					       
  else if (mu == 1 && N_RB_DL == 245) {
    fs = 122.88e6;
    bw = 90e6;
  }
  else if (mu == 1 && N_RB_DL == 273) {
    fs = 122.88e6;
    bw = 100e6;
  }
  else if (mu == 1 && N_RB_DL == 106) { 
    fs = 61.44e6;
    bw = 40e6;
  }
  else if (mu == 3 && N_RB_DL == 66) {
    fs = 122.88e6;
    bw = 100e6;
  }
  else if (mu == 3 && N_RB_DL == 32) {
    fs = 61.44e6;
    bw = 50e6;
  }
  else AssertFatal(1==0,"Unsupported numerology for mu %d, N_RB %d\n",mu, N_RB_DL);

  gNB2UE = new_channel_desc_scm(n_tx,
                                n_rx,
                                channel_model,
 				fs,
				bw,
				30e-9,
                                0,
                                0,
                                0);

  if (gNB2UE==NULL) {
    printf("Problem generating channel model. Exiting.\n");
    exit(-1);
  }

  frame_length_complex_samples = frame_parms->samples_per_subframe*NR_NUMBER_OF_SUBFRAMES_PER_FRAME;
  //frame_length_complex_samples_no_prefix = frame_parms->samples_per_subframe_wCP*NR_NUMBER_OF_SUBFRAMES_PER_FRAME;

  s_re = malloc(2*sizeof(double*));
  s_im = malloc(2*sizeof(double*));
  r_re = malloc(2*sizeof(double*));
  r_im = malloc(2*sizeof(double*));
  txdata = malloc(2*sizeof(int*));

  for (i=0; i<2; i++) {

    s_re[i] = malloc(frame_length_complex_samples*sizeof(double));
    bzero(s_re[i],frame_length_complex_samples*sizeof(double));
    s_im[i] = malloc(frame_length_complex_samples*sizeof(double));
    bzero(s_im[i],frame_length_complex_samples*sizeof(double));

    r_re[i] = malloc(frame_length_complex_samples*sizeof(double));
    bzero(r_re[i],frame_length_complex_samples*sizeof(double));
    r_im[i] = malloc(frame_length_complex_samples*sizeof(double));
    bzero(r_im[i],frame_length_complex_samples*sizeof(double));

    printf("Allocating %d samples for txdata\n",frame_length_complex_samples);
    txdata[i] = malloc(frame_length_complex_samples*sizeof(int));
    bzero(txdata[i],frame_length_complex_samples*sizeof(int));
  
  }

  if (pbch_file_fd!=NULL) {
    load_pbch_desc(pbch_file_fd);
  }


  //configure UE
  UE = malloc(sizeof(PHY_VARS_NR_UE));
  memset((void*)UE,0,sizeof(PHY_VARS_NR_UE));
  PHY_vars_UE_g = malloc(sizeof(PHY_VARS_NR_UE**));
  PHY_vars_UE_g[0] = malloc(sizeof(PHY_VARS_NR_UE*));
  PHY_vars_UE_g[0][0] = UE;
  memcpy(&UE->frame_parms,frame_parms,sizeof(NR_DL_FRAME_PARMS));

  if (run_initial_sync==1)  UE->is_synchronized = 0;
  else                      {UE->is_synchronized = 1; UE->UE_mode[0]=PUSCH;}
                      
  UE->perfect_ce = 0;

  if (init_nr_ue_signal(UE, 1, 0) != 0)
  {
    printf("Error at UE NR initialisation\n");
    exit(-1);
  }

  init_nr_ue_transport(UE,0);

  nr_gold_pbch(UE);
  nr_gold_pdcch(UE,0,2);

  nr_l2_init_ue(NULL);
  UE_mac = get_mac_inst(0);
  
  UE->if_inst = nr_ue_if_module_init(0);
  UE->if_inst->scheduled_response = nr_ue_scheduled_response;
  UE->if_inst->phy_config_request = nr_ue_phy_config_request;
  UE->if_inst->dl_indication = nr_ue_dl_indication;
  UE->if_inst->ul_indication = dummy_nr_ue_ul_indication;
  

  UE_mac->if_module = nr_ue_if_module_init(0);
  
  unsigned int available_bits=0;
  unsigned char *estimated_output_bit;
  unsigned char *test_input_bit;
  unsigned int errors_bit    = 0;
  uint32_t errors_scrambling = 0;


  test_input_bit       = (unsigned char *) malloc16(sizeof(unsigned char) * 16 * 68 * 384);
  estimated_output_bit = (unsigned char *) malloc16(sizeof(unsigned char) * 16 * 68 * 384);
  
  // generate signal
  AssertFatal(input_fd==NULL,"Not ready for input signal file\n");
  gNB->pbch_configured = 1;
  gNB->ssb_pdu.ssb_pdu_rel15.bchPayload=0x001234;

  //Configure UE
  rrc.carrier.MIB = (uint8_t*) malloc(4);
  rrc.carrier.sizeof_MIB = do_MIB_NR(&rrc,0);

  nr_rrc_mac_config_req_ue(0,0,0,rrc.carrier.mib.message.choice.mib,secondaryCellGroup);


  nr_dcireq_t dcireq;
  nr_scheduled_response_t scheduled_response;
  memset((void*)&dcireq,0,sizeof(dcireq));
  memset((void*)&scheduled_response,0,sizeof(scheduled_response));
  dcireq.module_id = 0;
  dcireq.gNB_index = 0;
  dcireq.cc_id     = 0;
  
  scheduled_response.dl_config = &dcireq.dl_config_req;
  scheduled_response.ul_config = &dcireq.ul_config_req;
  scheduled_response.tx_request = NULL;
  scheduled_response.module_id = 0;
  scheduled_response.CC_id     = 0;
  scheduled_response.frame = frame;
  scheduled_response.slot  = slot;
  scheduled_response.thread_id = UE_proc.thread_id;

  nr_ue_phy_config_request(&UE_mac->phy_config);
  NR_UE_info_t *UE_info = &RC.nrmac[0]->UE_info;
  //NR_COMMON_channels_t *cc = RC.nrmac[0]->common_channels;
  snrRun = 0;

  for (SNR = snr0; SNR < snr1; SNR += .2) {

    varArray_t *table_tx=initVarArray(1000,sizeof(double));
    reset_meas(&gNB->phy_proc_tx); // total gNB tx
    reset_meas(&gNB->dlsch_scrambling_stats);
    reset_meas(&gNB->dlsch_interleaving_stats);
    reset_meas(&gNB->dlsch_rate_matching_stats);
    reset_meas(&gNB->dlsch_segmentation_stats);
    reset_meas(&gNB->dlsch_modulation_stats);
    reset_meas(&gNB->dlsch_encoding_stats);
    reset_meas(&gNB->tinput);
    reset_meas(&gNB->tprep);
    reset_meas(&gNB->tparity);
    reset_meas(&gNB->toutput);  

    clear_pdsch_stats(gNB);

    n_errors = 0;
    effRate = 0;
    //n_errors2 = 0;
    //n_alamouti = 0;
    errors_scrambling=0;
    n_false_positive = 0;
    if (n_trials== 1) num_rounds = 1;

    for (trial = 0; trial < n_trials; trial++) {

      errors_bit = 0;
      //multipath channel
      //multipath_channel(gNB2UE,s_re,s_im,r_re,r_im,frame_length_complex_samples,0);

      UE->rx_offset=0;
      UE_proc.thread_id  = 0;
      UE_proc.frame_rx   = frame;
      UE_proc.nr_slot_rx = slot;
      
      dcireq.frame     = frame;
      dcireq.slot      = slot;

      NR_UE_DLSCH_t *dlsch0 = UE->dlsch[UE_proc.thread_id][0][0];

      int harq_pid = slot;
      NR_DL_UE_HARQ_t *UE_harq_process = dlsch0->harq_processes[harq_pid];

      NR_gNB_DLSCH_t *gNB_dlsch = gNB->dlsch[0][0];
      nfapi_nr_dl_tti_pdsch_pdu_rel15_t *rel15 = &gNB_dlsch->harq_processes[slot]->pdsch_pdu.pdsch_pdu_rel15;
      
      UE_harq_process->harq_ack.ack = 0;
      round = 0;
      UE_harq_process->round = round;
      UE_harq_process->first_tx = 1;
        
      while ((round<num_rounds) && (UE_harq_process->harq_ack.ack==0)) {
        memset(RC.nrmac[0]->cce_list[1][0],0,MAX_NUM_CCE*sizeof(int));
        memset(RC.nrmac[0]->cce_list[1][1],0,MAX_NUM_CCE*sizeof(int));
        clear_nr_nfapi_information(RC.nrmac[0], 0, frame, slot);

        UE_info->UE_sched_ctrl[0].harq_processes[harq_pid].ndi = !(trial&1);


        UE_info->UE_sched_ctrl[0].harq_processes[harq_pid].round = round;
        UE_info->UE_sched_ctrl[0].current_harq_pid = harq_pid;
        gNB->dlsch[0][0]->harq_processes[harq_pid]->round = round;
        for (int i=0; i<MAX_NUM_CORESET; i++)
          gNB_mac->UE_info.num_pdcch_cand[0][i] = 0;
      
        if (css_flag == 0) {
          const uint8_t slots_per_frame[5] = {10, 20, 40, 80, 160};
          const NR_TDD_UL_DL_Pattern_t *tdd_pattern = &scc->tdd_UL_DL_ConfigurationCommon->pattern1;
          const int num_slots_per_tdd = slots_per_frame[*scc->ssbSubcarrierSpacing] >> (7 - tdd_pattern->dl_UL_TransmissionPeriodicity);
          nr_schedule_ue_spec(0, frame, slot, num_slots_per_tdd);
        } else {
          nr_schedule_css_dlsch_phytest(0,frame,slot);
        }
        Sched_INFO.module_id = 0;
        Sched_INFO.CC_id     = 0;
        Sched_INFO.frame     = frame;
        Sched_INFO.slot      = slot;
        Sched_INFO.DL_req    = &gNB_mac->DL_req[0];
        Sched_INFO.UL_tti_req    = &gNB_mac->UL_tti_req[0];
        Sched_INFO.UL_dci_req  = NULL;
        Sched_INFO.TX_req    = &gNB_mac->TX_req[0];
        nr_schedule_response(&Sched_INFO);
        
        if (run_initial_sync)
          nr_common_signal_procedures(gNB,frame,slot);
        else
          phy_procedures_gNB_TX(gNB,frame,slot,0);
            
        int txdataF_offset = (slot%2) * frame_parms->samples_per_slot_wCP;
        
        if (n_trials==1) {
          LOG_M("txsigF0.m","txsF0", &gNB->common_vars.txdataF[0][txdataF_offset],frame_parms->samples_per_slot_wCP,1,1);
          if (gNB->frame_parms.nb_antennas_tx>1)
          LOG_M("txsigF1.m","txsF1", &gNB->common_vars.txdataF[1][txdataF_offset],frame_parms->samples_per_slot_wCP,1,1);
        }
        int tx_offset = frame_parms->get_samples_slot_timestamp(slot,frame_parms,0);
        if (n_trials==1) printf("tx_offset %d, txdataF_offset %d \n", tx_offset,txdataF_offset);
        
        //TODO: loop over slots
        for (aa=0; aa<gNB->frame_parms.nb_antennas_tx; aa++) {
    
          if (cyclic_prefix_type == 1) {
            PHY_ofdm_mod(&gNB->common_vars.txdataF[aa][txdataF_offset],
                         &txdata[aa][tx_offset],
                         frame_parms->ofdm_symbol_size,
                         12,
                         frame_parms->nb_prefix_samples,
                         CYCLIC_PREFIX);
          } else {
            nr_normal_prefix_mod(&gNB->common_vars.txdataF[aa][txdataF_offset],
                                 &txdata[aa][tx_offset],
                                 14,
                                 frame_parms);
          }
        }
       
        if (n_trials==1) {
          LOG_M("txsig0.m","txs0", &txdata[0][tx_offset],frame_parms->get_samples_slot_timestamp(slot,frame_parms,0),1,1);
          if (gNB->frame_parms.nb_antennas_tx>1)
            LOG_M("txsig1.m","txs1", &txdata[1][tx_offset],frame_parms->get_samples_slot_timestamp(slot,frame_parms,0),1,1);
        }
        if (output_fd) {
          printf("writing txdata to binary file\n");
          fwrite(txdata[0],sizeof(int32_t),frame_length_complex_samples,output_fd);
        }

        int txlev = signal_energy(&txdata[0][frame_parms->get_samples_slot_timestamp(slot,frame_parms,0)+5*frame_parms->ofdm_symbol_size + 4*frame_parms->nb_prefix_samples + frame_parms->nb_prefix_samples0], frame_parms->ofdm_symbol_size + frame_parms->nb_prefix_samples);
        
        //  if (n_trials==1) printf("txlev %d (%f)\n",txlev,10*log10((double)txlev));
        
        for (i=(frame_parms->get_samples_slot_timestamp(slot,frame_parms,0)); 
             i<(frame_parms->get_samples_slot_timestamp(slot+1,frame_parms,0)); 
             i++) {
    
          for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
            r_re[aa][i] = ((double)(((short *)txdata[aa]))[(i<<1)]);
            r_im[aa][i] = ((double)(((short *)txdata[aa]))[(i<<1)+1]);
          }
        }
        
        //AWGN
        sigma2_dB = 10 * log10((double)txlev * ((double)UE->frame_parms.ofdm_symbol_size/(12*rel15->rbSize))) - SNR;
        sigma2    = pow(10, sigma2_dB/10);
        if (n_trials==1) printf("sigma2 %f (%f dB), txlev %f (factor %f)\n",sigma2,sigma2_dB,10*log10((double)txlev),(double)(double)UE->frame_parms.ofdm_symbol_size/(12*rel15->rbSize));
        
        for (i=frame_parms->get_samples_slot_timestamp(slot,frame_parms,0); 
             i<frame_parms->get_samples_slot_timestamp(slot+1,frame_parms,0);
             i++) {

          for (aa=0; aa<frame_parms->nb_antennas_rx; aa++) {
            ((short*) UE->common_vars.rxdata[aa])[2*i]   = (short) ((r_re[aa][i] + sqrt(sigma2/2)*gaussdouble(0.0,1.0)));
            ((short*) UE->common_vars.rxdata[aa])[2*i+1] = (short) ((r_im[aa][i] + sqrt(sigma2/2)*gaussdouble(0.0,1.0)));
          }
        }
        
        nr_ue_dcireq(&dcireq); //to be replaced with function pointer later
        nr_ue_scheduled_response(&scheduled_response);
        
        phy_procedures_nrUE_RX(UE,
                               &UE_proc,
                               0,
                               normal_txrx);
        
        //printf("dlsim round %d ends\n",round);
        round++;
      } // round
      
      //----------------------------------------------------------
      //---------------------- count errors ----------------------
      //----------------------------------------------------------

      if (UE->dlsch[UE_proc.thread_id][0][0]->last_iteration_cnt >=
        UE->dlsch[UE_proc.thread_id][0][0]->max_ldpc_iterations+1)
        n_errors++;

      NR_UE_PDSCH **pdsch_vars = UE->pdsch_vars[UE_proc.thread_id];
      int16_t *UE_llr = pdsch_vars[0]->llr[0];

      TBS                  = UE_harq_process->TBS;//rel15->TBSize[0];
      uint16_t length_dmrs = 1;
      uint16_t nb_rb       = rel15->rbSize;
      uint8_t  nb_re_dmrs  = rel15->dmrsConfigType == NFAPI_NR_DMRS_TYPE1 ? 6 : 4;
      uint8_t  mod_order   = rel15->qamModOrder[0];
      uint8_t  nb_symb_sch = rel15->NrOfSymbols;

      available_bits = nr_get_G(nb_rb, nb_symb_sch, nb_re_dmrs, length_dmrs, mod_order, rel15->nrOfLayers);
      
      for (i = 0; i < available_bits; i++) {
	
	if(((gNB_dlsch->harq_processes[harq_pid]->f[i] == 0) && (UE_llr[i] <= 0)) || 
	   ((gNB_dlsch->harq_processes[harq_pid]->f[i] == 1) && (UE_llr[i] >= 0)))
	  {
	    if(errors_scrambling == 0) {
	      LOG_D(PHY,"\n");
	      LOG_D(PHY,"First bit in error in unscrambling = %d\n",i);
	    }
	    errors_scrambling++;
	  }
	
      }
      for (i = 0; i < TBS; i++) {
	
	estimated_output_bit[i] = (UE_harq_process->b[i/8] & (1 << (i & 7))) >> (i & 7);
	test_input_bit[i]       = (gNB_dlsch->harq_processes[harq_pid]->b[i / 8] & (1 << (i & 7))) >> (i & 7); // Further correct for multiple segments
	
	if (estimated_output_bit[i] != test_input_bit[i]) {
	  if(errors_bit == 0)
	    LOG_D(PHY,"First bit in error in decoding = %d (errors scrambling %d)\n",i,errors_scrambling);
	  errors_bit++;
	}
	
      }
      
      ////////////////////////////////////////////////////////////
      
      if (errors_scrambling > 0) {
	if (n_trials == 1)
	  printf("errors_scrambling = %u/%u (trial %d)\n", errors_scrambling, available_bits,trial);
      }
      
      if (errors_bit > 0) {
	n_false_positive++;
	if (n_trials == 1)
	  printf("errors_bit = %u (trial %d)\n", errors_bit, trial);
      }
      roundStats[snrRun]+=((float)round); 
      if (UE_harq_process->harq_ack.ack==1) effRate += ((float)TBS)/round;
    } // noise trials

    roundStats[snrRun]/=((float)n_trials);
    effRate /= n_trials;
    printf("*****************************************\n");
    printf("SNR %f, (false positive %f)\n", SNR,
           (float) n_errors / (float) n_trials);
    printf("*****************************************\n");
    printf("\n");
    dump_pdsch_stats(gNB);
    printf("SNR %f : n_errors (negative CRC) = %d/%d, Avg round %.2f, Channel BER %e, Eff Rate %.4f bits/slot, Eff Throughput %.2f, TBS %d bits/slot\n", SNR, n_errors, n_trials,roundStats[snrRun],(double)errors_scrambling/available_bits/n_trials,effRate,effRate/TBS*100,TBS);
    printf("\n");

    if (print_perf==1) {
      printf("\ngNB TX function statistics (per %d us slot, NPRB %d, mcs %d, TBS %d, Kr %d (Zc %d))\n",
	     1000>>*scc->ssbSubcarrierSpacing, g_rbSize, g_mcsIndex,
	     gNB->dlsch[0][0]->harq_processes[0]->pdsch_pdu.pdsch_pdu_rel15.TBSize[0]<<3,
	     gNB->dlsch[0][0]->harq_processes[0]->K,
	     gNB->dlsch[0][0]->harq_processes[0]->K/((gNB->dlsch[0][0]->harq_processes[0]->pdsch_pdu.pdsch_pdu_rel15.TBSize[0]<<3)>3824?22:10));
      printDistribution(&gNB->phy_proc_tx,table_tx,"PHY proc tx");
      printStatIndent2(&gNB->dlsch_encoding_stats,"DLSCH encoding time");
      printStatIndent3(&gNB->dlsch_segmentation_stats,"DLSCH segmentation time");
      printStatIndent3(&gNB->tinput,"DLSCH LDPC input processing time");
      printStatIndent3(&gNB->tprep,"DLSCH LDPC input preparation time");
      printStatIndent3(&gNB->tparity,"DLSCH LDPC parity generation time");
      printStatIndent3(&gNB->toutput,"DLSCH LDPC output generation time");
      printStatIndent3(&gNB->dlsch_rate_matching_stats,"DLSCH Rate Mataching time");
      printStatIndent3(&gNB->dlsch_interleaving_stats,  "DLSCH Interleaving time");
      printStatIndent2(&gNB->dlsch_modulation_stats,"DLSCH modulation time");
      printStatIndent2(&gNB->dlsch_scrambling_stats,  "DLSCH scrambling time");


      printf("\nUE RX function statistics (per %d us slot)\n",1000>>*scc->ssbSubcarrierSpacing);
      /*
      printDistribution(&phy_proc_rx_tot, table_rx,"Total PHY proc rx");
      printStatIndent(&ue_front_end_tot,"Front end processing");
      printStatIndent(&dlsch_llr_tot,"rx_pdsch processing");
      printStatIndent2(&pdsch_procedures_tot,"pdsch processing");
      printStatIndent2(&dlsch_procedures_tot,"dlsch processing");
      printStatIndent2(&UE->crnti_procedures_stats,"C-RNTI processing");
      printStatIndent(&UE->ofdm_demod_stats,"ofdm demodulation");
      printStatIndent(&UE->dlsch_channel_estimation_stats,"DLSCH channel estimation time");
      printStatIndent(&UE->dlsch_freq_offset_estimation_stats,"DLSCH frequency offset estimation time");
      printStatIndent(&dlsch_decoding_tot, "DLSCH Decoding time ");
      printStatIndent(&UE->dlsch_unscrambling_stats,"DLSCH unscrambling time");
      printStatIndent(&UE->dlsch_rate_unmatching_stats,"DLSCH Rate Unmatching");
      printf("|__ DLSCH Turbo Decoding(%d bits), avg iterations: %.1f       %.2f us (%d cycles, %d trials)\n",
	     UE->dlsch[UE_proc.thread_id][0][0]->harq_processes[0]->Cminus ?
	     UE->dlsch[UE_proc.thread_id][0][0]->harq_processes[0]->Kminus :
	     UE->dlsch[UE_proc.thread_id][0][0]->harq_processes[0]->Kplus,
	     UE->dlsch_tc_intl1_stats.trials/(double)UE->dlsch_tc_init_stats.trials,
	     (double)UE->dlsch_turbo_decoding_stats.diff/UE->dlsch_turbo_decoding_stats.trials*timeBase,
	     (int)((double)UE->dlsch_turbo_decoding_stats.diff/UE->dlsch_turbo_decoding_stats.trials),
	     UE->dlsch_turbo_decoding_stats.trials);
      printStatIndent2(&UE->dlsch_tc_init_stats,"init");
      printStatIndent2(&UE->dlsch_tc_alpha_stats,"alpha");
      printStatIndent2(&UE->dlsch_tc_beta_stats,"beta");
      printStatIndent2(&UE->dlsch_tc_gamma_stats,"gamma");
      printStatIndent2(&UE->dlsch_tc_ext_stats,"ext");
      printStatIndent2(&UE->dlsch_tc_intl1_stats,"turbo internal interleaver");
      printStatIndent2(&UE->dlsch_tc_intl2_stats,"intl2+HardDecode+CRC");
      */
    }

    if (n_trials == 1) {
      
      LOG_M("rxsig0.m","rxs0", UE->common_vars.rxdata[0], frame_length_complex_samples, 1, 1);
      if (UE->frame_parms.nb_antennas_rx>1)
	LOG_M("rxsig1.m","rxs1", UE->common_vars.rxdata[1], frame_length_complex_samples, 1, 1);
      LOG_M("chestF0.m","chF0",UE->pdsch_vars[0][0]->dl_ch_estimates_ext,N_RB_DL*12*14,1,1);
      write_output("rxF_comp.m","rxFc",&UE->pdsch_vars[0][0]->rxdataF_comp0[0][0],N_RB_DL*12*14,1,1);
      LOG_M("rxF_llr.m","rxFllr",UE->pdsch_vars[UE_proc.thread_id][0]->llr[0],available_bits,1,0);
      break;
    }

    //if ((float)n_errors/(float)n_trials <= target_error_rate) {
    if (effRate > (eff_tp_check*TBS)) {
      printf("PDSCH test OK\n");
      break;
    }

    snrRun++;
  } // NSR

  /*if (n_trials>1) {
    printf("HARQ stats:\nSNR\tRounds\n");
    psnr = snr0;
    for (uint8_t i=0; i<snrRun; i++) {
      printf("%.1f\t%.2f\n",psnr,roundStats[i]);
      psnr+=0.2;
    }
  }*/

  for (i = 0; i < 2; i++) {
    free(s_re[i]);
    free(s_im[i]);
    free(r_re[i]);
    free(r_im[i]);
    free(txdata[i]);
  }

  free(s_re);
  free(s_im);
  free(r_re);
  free(r_im);
  free(txdata);
  free(test_input_bit);
  free(estimated_output_bit);
  
  if (output_fd)
    fclose(output_fd);

  if (input_fd)
    fclose(input_fd);

  if (scg_fd)
    fclose(scg_fd);
  return(n_errors);
  
}
