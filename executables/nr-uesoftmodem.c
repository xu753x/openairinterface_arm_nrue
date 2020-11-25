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


#define _GNU_SOURCE             /* See feature_test_macros(7) */
#include <sched.h>

#include "T.h"
#include "assertions.h"
#include "PHY/types.h"
#include "PHY/defs_nr_UE.h"
#include "SCHED_NR_UE/defs.h"
#include "common/ran_context.h"
#include "common/config/config_userapi.h"
//#include "common/utils/threadPool/thread-pool.h"
#include "common/utils/load_module_shlib.h"
//#undef FRAME_LENGTH_COMPLEX_SAMPLES //there are two conflicting definitions, so we better make sure we don't use it at all

#include "../../ARCH/COMMON/common_lib.h"
#include "../../ARCH/ETHERNET/USERSPACE/LIB/if_defs.h"

//#undef FRAME_LENGTH_COMPLEX_SAMPLES //there are two conflicting definitions, so we better make sure we don't use it at all
#include "openair1/PHY/MODULATION/nr_modulation.h"
#include "PHY/phy_vars_nr_ue.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "SCHED/sched_common_vars.h"
#include "PHY/MODULATION/modulation_vars.h"
//#include "../../SIMU/USER/init_lte.h"

#include "LAYER2/MAC/mac_vars.h"
#include "RRC/LTE/rrc_vars.h"
#include "PHY_INTERFACE/phy_interface_vars.h"
#include "openair1/SIMULATION/TOOLS/sim.h"

#ifdef SMBV
#include "PHY/TOOLS/smbv.h"
unsigned short config_frames[4] = {2,9,11,13};
#endif
#include "common/utils/LOG/log.h"
#include "common/utils/LOG/vcd_signal_dumper.h"

#include "UTIL/OPT/opt.h"
#include "enb_config.h"
//#include "PHY/TOOLS/time_meas.h"

#ifndef OPENAIR2
  #include "UTIL/OTG/otg_vars.h"
#endif

#include "intertask_interface.h"

#include "PHY/INIT/phy_init.h"
#include "system.h"
#include <openair2/RRC/NR_UE/rrc_proto.h>
#include <openair2/LAYER2/NR_MAC_UE/mac_defs.h>
#include <openair2/LAYER2/NR_MAC_UE/mac_proto.h>
#include <openair2/NR_UE_PHY_INTERFACE/NR_IF_Module.h>
#include <openair1/SCHED_NR_UE/fapi_nr_ue_l1.h>

/* Callbacks, globals and object handlers */

//#include "stats.h"
// current status is that every UE has a DL scope for a SINGLE eNB (eNB_id=0)
#include "PHY/TOOLS/phy_scope_interface.h"
#include "PHY/TOOLS/nr_phy_scope.h"
#include <executables/nr-uesoftmodem.h>
#include "executables/softmodem-common.h"
#include "executables/thread-common.h"

extern const char *duplex_mode[];

// Thread variables
pthread_cond_t nfapi_sync_cond;
pthread_mutex_t nfapi_sync_mutex;
int nfapi_sync_var=-1; //!< protected by mutex \ref nfapi_sync_mutex
uint16_t sf_ahead=6; //??? value ???
pthread_cond_t sync_cond;
pthread_mutex_t sync_mutex;
int sync_var=-1; //!< protected by mutex \ref sync_mutex.
int config_sync_var=-1;
tpool_t *Tpool;
#ifdef UE_DLSCH_PARALLELISATION
  tpool_t *Tpool_dl;
#endif

RAN_CONTEXT_t RC;
volatile int             start_eNB = 0;
volatile int             start_UE = 0;
volatile int             oai_exit = 0;

// Command line options

extern int16_t  nr_dlsch_demod_shift;
static int      tx_max_power[MAX_NUM_CCs] = {0};

int      single_thread_flag = 1;
int         threequarter_fs = 0;
int         UE_scan_carrier = 0;
int      UE_fo_compensation = 0;
int UE_no_timing_correction = 0;
int                 N_RB_DL = 0;
int                 tddflag = 0;
int                 vcdflag = 0;
uint8_t       nb_antenna_tx = 1;
uint8_t       nb_antenna_rx = 1;
double          rx_gain_off = 0.0;
uint32_t     timing_advance = 0;
char             *usrp_args = NULL;
char       *rrc_config_path = NULL;
int               dumpframe = 0;

uint64_t        downlink_frequency[MAX_NUM_CCs][4];
int32_t         uplink_frequency_offset[MAX_NUM_CCs][4];
int             rx_input_level_dBm;

#if MAX_NUM_CCs == 1
rx_gain_t                rx_gain_mode[MAX_NUM_CCs][4] = {{max_gain,max_gain,max_gain,max_gain}};
double tx_gain[MAX_NUM_CCs][4] = {{20,0,0,0}};
double rx_gain[MAX_NUM_CCs][4] = {{110,0,0,0}};
#else
rx_gain_t                rx_gain_mode[MAX_NUM_CCs][4] = {{max_gain,max_gain,max_gain,max_gain},{max_gain,max_gain,max_gain,max_gain}};
double tx_gain[MAX_NUM_CCs][4] = {{20,0,0,0},{20,0,0,0}};
double rx_gain[MAX_NUM_CCs][4] = {{110,0,0,0},{20,0,0,0}};
#endif

// UE and OAI config variables

openair0_config_t openair0_cfg[MAX_CARDS];
int16_t           node_synch_ref[MAX_NUM_CCs];
int               otg_enabled;
double            cpuf;

runmode_t            mode = normal_txrx;
int               UE_scan = 0;
int          chain_offset = 0;
uint64_t num_missed_slots = 0; // counter for the number of missed slots
int     transmission_mode = 1;
int            numerology = 0;
int        usrp_tx_thread = 0;
int           oaisim_flag = 0;
int            emulate_rf = 0;

char uecap_xer[1024],uecap_xer_in=0;

/* see file openair2/LAYER2/MAC/main.c for why abstraction_flag is needed
 * this is very hackish - find a proper solution
 */
uint8_t abstraction_flag=0;

/*---------------------BMC: timespec helpers -----------------------------*/

struct timespec min_diff_time = { .tv_sec = 0, .tv_nsec = 0 };
struct timespec max_diff_time = { .tv_sec = 0, .tv_nsec = 0 };

struct timespec clock_difftime(struct timespec start, struct timespec end) {
  struct timespec temp;

  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }

  return temp;
}

void print_difftimes(void) {
  LOG_I(HW,"difftimes min = %lu ns ; max = %lu ns\n", min_diff_time.tv_nsec, max_diff_time.tv_nsec);
}

void exit_function(const char *file, const char *function, const int line, const char *s) {
  int CC_id;

  if (s != NULL) {
    printf("%s:%d %s() Exiting OAI softmodem: %s\n",file,line, function, s);
  }

  oai_exit = 1;

  if (PHY_vars_UE_g && PHY_vars_UE_g[0]) {
    for(CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
      if (PHY_vars_UE_g[0][CC_id] && PHY_vars_UE_g[0][CC_id]->rfdevice.trx_end_func)
        PHY_vars_UE_g[0][CC_id]->rfdevice.trx_end_func(&PHY_vars_UE_g[0][CC_id]->rfdevice);
    }
  }

  sleep(1); //allow lte-softmodem threads to exit first
  exit(1);
}

void *l2l1_task(void *arg) {
  MessageDef *message_p = NULL;
  int         result;
  itti_set_task_real_time(TASK_L2L1);
  itti_mark_task_ready(TASK_L2L1);

  do {
    // Wait for a message
    itti_receive_msg (TASK_L2L1, &message_p);

    switch (ITTI_MSG_ID(message_p)) {
      case TERMINATE_MESSAGE:
        oai_exit=1;
        itti_exit_task ();
        break;

      case ACTIVATE_MESSAGE:
        start_UE = 1;
        break;

      case DEACTIVATE_MESSAGE:
        start_UE = 0;
        break;

      case MESSAGE_TEST:
        LOG_I(EMU, "Received %s\n", ITTI_MSG_NAME(message_p));
        break;

      default:
        LOG_E(EMU, "Received unexpected message %s\n", ITTI_MSG_NAME(message_p));
        break;
    }

    result = itti_free (ITTI_MSG_ORIGIN_ID(message_p), message_p);
    AssertFatal (result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
  } while(!oai_exit);

  return NULL;
}

static void get_options(void) {

  char *loopfile=NULL;

  paramdef_t cmdline_params[] =CMDLINE_PARAMS_DESC_UE ;
  config_process_cmdline( cmdline_params,sizeof(cmdline_params)/sizeof(paramdef_t),NULL);
  paramdef_t cmdline_uemodeparams[] = CMDLINE_UEMODEPARAMS_DESC;
  paramdef_t cmdline_ueparams[] = CMDLINE_NRUEPARAMS_DESC;
  config_process_cmdline( cmdline_uemodeparams,sizeof(cmdline_uemodeparams)/sizeof(paramdef_t),NULL);
  config_process_cmdline( cmdline_ueparams,sizeof(cmdline_ueparams)/sizeof(paramdef_t),NULL);

  if ( (cmdline_uemodeparams[CMDLINE_CALIBUERX_IDX].paramflags &  PARAMFLAG_PARAMSET) != 0) mode = rx_calib_ue;

  if ( (cmdline_uemodeparams[CMDLINE_CALIBUERXMED_IDX].paramflags &  PARAMFLAG_PARAMSET) != 0) mode = rx_calib_ue_med;

  if ( (cmdline_uemodeparams[CMDLINE_CALIBUERXBYP_IDX].paramflags &  PARAMFLAG_PARAMSET) != 0) mode = rx_calib_ue_byp;

  if (cmdline_uemodeparams[CMDLINE_DEBUGUEPRACH_IDX].uptr)
    if ( *(cmdline_uemodeparams[CMDLINE_DEBUGUEPRACH_IDX].uptr) > 0) mode = debug_prach;

  if (cmdline_uemodeparams[CMDLINE_NOL2CONNECT_IDX].uptr)
    if ( *(cmdline_uemodeparams[CMDLINE_NOL2CONNECT_IDX].uptr) > 0)  mode = no_L2_connect;

  if (cmdline_uemodeparams[CMDLINE_CALIBPRACHTX_IDX].uptr)
    if ( *(cmdline_uemodeparams[CMDLINE_CALIBPRACHTX_IDX].uptr) > 0) mode = calib_prach_tx;

  if ((cmdline_uemodeparams[CMDLINE_DUMPMEMORY_IDX].paramflags & PARAMFLAG_PARAMSET) != 0)
    mode = rx_dump_frame;

  if (vcdflag > 0)
    ouput_vcd = 1;

  if ( !(CONFIG_ISFLAGSET(CONFIG_ABORT))  && (!(CONFIG_ISFLAGSET(CONFIG_NOOOPT))) ) {
    // Here the configuration file is the XER encoded UE capabilities
    // Read it in and store in asn1c data structures
    sprintf(uecap_xer,"%stargets/PROJECTS/GENERIC-LTE-EPC/CONF/UE_config.xml",getenv("OPENAIR_HOME"));
    printf("%s\n",uecap_xer);
    uecap_xer_in=1;
  } /* UE with config file  */
}

// set PHY vars from command line
void set_options(int CC_id, PHY_VARS_NR_UE *UE){

  // Init power variables
  tx_max_power[CC_id] = tx_max_power[0];
  rx_gain[0][CC_id]   = rx_gain[0][0];
  tx_gain[0][CC_id]   = tx_gain[0][0];

  // Set UE variables
  UE->UE_scan              = UE_scan;
  UE->UE_scan_carrier      = UE_scan_carrier;
  UE->UE_fo_compensation   = UE_fo_compensation;
  UE->no_timing_correction = UE_no_timing_correction;
  UE->mode                 = mode;
  UE->rx_total_gain_dB     = (int)rx_gain[CC_id][0] + rx_gain_off;
  UE->tx_total_gain_dB     = (int)tx_gain[CC_id][0];
  UE->tx_power_max_dBm     = tx_max_power[CC_id];

  LOG_I(PHY,"Set UE mode %d, UE_fo_compensation %d, UE_scan %d, UE_scan_carrier %d, UE_no_timing_correction %d \n", mode, UE_fo_compensation, UE_scan, UE_scan_carrier, UE_no_timing_correction);

  // Set FP variables
  NR_DL_FRAME_PARMS *fp = &UE->frame_parms;
  fp->nb_antennas_tx    = nb_antenna_tx;
  fp->nb_antennas_rx    = nb_antenna_rx;
  fp->threequarter_fs   = threequarter_fs;
  if (tddflag){
    fp->frame_type = TDD;
    LOG_I(PHY, "Set UE frame_type %d\n", fp->frame_type);
  }
  if (N_RB_DL){
    fp->N_RB_DL = N_RB_DL;
    LOG_I(PHY, "Set UE N_RB_DL %d\n", N_RB_DL);
  }

  LOG_I(PHY, "Set UE nb_rx_antenna %d, nb_tx_antenna %d, threequarter_fs %d\n", nb_antenna_rx, nb_antenna_tx, threequarter_fs);

}

void init_openair0(void) {
  int card;
  int freq_off = 0;
  NR_DL_FRAME_PARMS *frame_parms = &PHY_vars_UE_g[0][0]->frame_parms;

  for (card=0; card<MAX_CARDS; card++) {
    uint64_t dl_carrier, ul_carrier;
    openair0_cfg[card].configFilename = NULL;
    openair0_cfg[card].threequarter_fs = frame_parms->threequarter_fs;
    numerology = frame_parms->numerology_index;

    if(frame_parms->N_RB_DL == 66) {
      if (numerology==3) {
          openair0_cfg[card].sample_rate=122.88e6;
          openair0_cfg[card].samples_per_frame = 1228800;
        } else {
          LOG_E(PHY,"Unsupported numerology! FR2 supports only 120KHz SCS for now.\n");
          exit(-1);
        }
    }else if(frame_parms->N_RB_DL == 32) {
      if (numerology==3) {
          openair0_cfg[card].sample_rate=61.44e6;
          openair0_cfg[card].samples_per_frame = 614400;
        } else {
          LOG_E(PHY,"Unsupported numerology! FR2 supports only 120KHz SCS for now.\n");
          exit(-1);
        }
    }else if(frame_parms->N_RB_DL == 217) {
      if (numerology==1) {
        if (frame_parms->threequarter_fs) {
          openair0_cfg[card].sample_rate=92.16e6;
          openair0_cfg[card].samples_per_frame = 921600;
        }
        else {
          openair0_cfg[card].sample_rate=122.88e6;
          openair0_cfg[card].samples_per_frame = 1228800;
        }
      } else {
        LOG_E(PHY,"Unsupported numerology!\n");
        exit(-1);
      }
    } else if(frame_parms->N_RB_DL == 273) {
      if (numerology==1) {
        if (frame_parms->threequarter_fs) {
          AssertFatal(0 == 1,"three quarter sampling not supported for N_RB 273\n");
        }
        else {
          openair0_cfg[card].sample_rate=122.88e6;
          openair0_cfg[card].samples_per_frame = 1228800;
        }
      } else {
        LOG_E(PHY,"Unsupported numerology!\n");
        exit(-1);
      }
    } else if(frame_parms->N_RB_DL == 106) {
      if (numerology==0) {
        if (frame_parms->threequarter_fs) {
          openair0_cfg[card].sample_rate=23.04e6;
          openair0_cfg[card].samples_per_frame = 230400;
        } else {
          openair0_cfg[card].sample_rate=30.72e6;
          openair0_cfg[card].samples_per_frame = 307200;
        }
      } else if (numerology==1) {
        if (frame_parms->threequarter_fs) {
          openair0_cfg[card].sample_rate=46.08e6;
          openair0_cfg[card].samples_per_frame = 460800;
	}
	else {
          openair0_cfg[card].sample_rate=61.44e6;
          openair0_cfg[card].samples_per_frame = 614400;
        }
      } else if (numerology==2) {
        openair0_cfg[card].sample_rate=122.88e6;
        openair0_cfg[card].samples_per_frame = 1228800;
      } else {
        LOG_E(PHY,"Unsupported numerology!\n");
        exit(-1);
      }
    } else if(frame_parms->N_RB_DL == 50) {
      openair0_cfg[card].sample_rate=15.36e6;
      openair0_cfg[card].samples_per_frame = 153600;
    } else if (frame_parms->N_RB_DL == 25) {
      openair0_cfg[card].sample_rate=7.68e6;
      openair0_cfg[card].samples_per_frame = 76800;
    } else if (frame_parms->N_RB_DL == 6) {
      openair0_cfg[card].sample_rate=1.92e6;
      openair0_cfg[card].samples_per_frame = 19200;
    }
    else {
      LOG_E(PHY,"Unknown NB_RB %d!\n",frame_parms->N_RB_DL);
      exit(-1);
    }

    if (frame_parms->frame_type==TDD)
      openair0_cfg[card].duplex_mode = duplex_mode_TDD;
    else
      openair0_cfg[card].duplex_mode = duplex_mode_FDD;

    openair0_cfg[card].Mod_id = 0;
    openair0_cfg[card].num_rb_dl = frame_parms->N_RB_DL;
    openair0_cfg[card].clock_source = get_softmodem_params()->clock_source;
    openair0_cfg[card].time_source = get_softmodem_params()->timing_source;
    openair0_cfg[card].tx_num_channels = min(2, frame_parms->nb_antennas_tx);
    openair0_cfg[card].rx_num_channels = min(2, frame_parms->nb_antennas_rx);

    LOG_I(PHY, "HW: Configuring card %d, tx/rx num_channels %d/%d, duplex_mode %s\n",
      card,
      openair0_cfg[card].tx_num_channels,
      openair0_cfg[card].rx_num_channels,
      duplex_mode[openair0_cfg[card].duplex_mode]);

    nr_get_carrier_frequencies(frame_parms, &dl_carrier, &ul_carrier);

    nr_rf_card_config(&openair0_cfg[card], rx_gain_off, ul_carrier, dl_carrier, freq_off);

    openair0_cfg[card].configFilename = get_softmodem_params()->rf_config_file;

    if (usrp_args) openair0_cfg[card].sdr_addrs = usrp_args;

  }
}

void init_pdcp(void) {
  uint32_t pdcp_initmask = (!IS_SOFTMODEM_NOS1) ? LINK_ENB_PDCP_TO_GTPV1U_BIT : (LINK_ENB_PDCP_TO_GTPV1U_BIT | PDCP_USE_NETLINK_BIT | LINK_ENB_PDCP_TO_IP_DRIVER_BIT);

  /*if (IS_SOFTMODEM_BASICSIM || IS_SOFTMODEM_RFSIM || (nfapi_getmode()==NFAPI_UE_STUB_PNF)) {
    pdcp_initmask = pdcp_initmask | UE_NAS_USE_TUN_BIT;
  }*/

  if (IS_SOFTMODEM_NOKRNMOD)
    pdcp_initmask = pdcp_initmask | UE_NAS_USE_TUN_BIT;

  /*if (rlc_module_init() != 0) {
    LOG_I(RLC, "Problem at RLC initiation \n");
  }
  pdcp_layer_init();
  nr_DRB_preconfiguration();*/
  pdcp_module_init(pdcp_initmask);
  pdcp_set_rlc_data_req_func((send_rlc_data_req_func_t) rlc_data_req);
  pdcp_set_pdcp_data_ind_func((pdcp_data_ind_func_t) pdcp_data_ind);
  LOG_I(PDCP, "Before getting out from init_pdcp() \n");
}

// Stupid function addition because UE itti messages queues definition is common with eNB
void *rrc_enb_process_msg(void *notUsed) {
  return NULL;
}


int main( int argc, char **argv ) {
  //uint8_t beta_ACK=0,beta_RI=0,beta_CQI=2;
  PHY_VARS_NR_UE *UE[MAX_NUM_CCs];
  start_background_system();

  if ( load_configmodule(argc,argv,CONFIG_ENABLECMDLINEONLY) == NULL) {
    exit_fun("[SOFTMODEM] Error, configuration module init failed\n");
  }
  set_softmodem_sighandler();
  CONFIG_SETRTFLAG(CONFIG_NOEXITONHELP);
  memset(openair0_cfg,0,sizeof(openair0_config_t)*MAX_CARDS);
  memset(tx_max_power,0,sizeof(int)*MAX_NUM_CCs);
  // initialize logging
  logInit();
  // get options and fill parameters from configuration file
  get_options(); //Command-line options
  get_common_options(SOFTMODEM_5GUE_BIT );
#if T_TRACER
  T_Config_Init();
#endif
  //randominit (0);
  set_taus_seed (0);
  tpool_t pool;
  Tpool = &pool;
  char params[]="-1,-1";
  initTpool(params, Tpool, false);
#ifdef UE_DLSCH_PARALLELISATION
  tpool_t pool_dl;
  Tpool_dl = &pool_dl;
  char params_dl[]="-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1";
  initTpool(params_dl, Tpool_dl, false);
#endif
  cpuf=get_cpu_freq_GHz();
  itti_init(TASK_MAX, THREAD_MAX, MESSAGES_ID_MAX, tasks_info, messages_info);

  init_opt() ;
  load_nrLDPClib();

  if (ouput_vcd) {
    vcd_signal_dumper_init("/tmp/openair_dump_nrUE.vcd");
  }

  #ifndef PACKAGE_VERSION
#  define PACKAGE_VERSION "UNKNOWN-EXPERIMENTAL"
#endif
  LOG_I(HW, "Version: %s\n", PACKAGE_VERSION);

  init_NR_UE(1,rrc_config_path);
  if(IS_SOFTMODEM_NOS1)
	  init_pdcp();

  NB_UE_INST=1;
  NB_INST=1;
  PHY_vars_UE_g = malloc(sizeof(PHY_VARS_NR_UE **));
  PHY_vars_UE_g[0] = malloc(sizeof(PHY_VARS_NR_UE *)*MAX_NUM_CCs);

  if (get_softmodem_params()->do_ra)
    AssertFatal(get_softmodem_params()->phy_test == 0,"RA and phy_test are mutually exclusive\n");

  for (int CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {

    PHY_vars_UE_g[0][CC_id] = (PHY_VARS_NR_UE *)malloc(sizeof(PHY_VARS_NR_UE));
    UE[CC_id] = PHY_vars_UE_g[0][CC_id];
    memset(UE[CC_id],0,sizeof(PHY_VARS_NR_UE));

    set_options(CC_id, UE[CC_id]);

    NR_UE_MAC_INST_t *mac = get_mac_inst(0);
    if(mac->if_module != NULL && mac->if_module->phy_config_request != NULL)
      mac->if_module->phy_config_request(&mac->phy_config);

    fapi_nr_config_request_t *nrUE_config = &UE[CC_id]->nrUE_config;

    nr_init_frame_parms_ue(&UE[CC_id]->frame_parms, nrUE_config, *mac->scc->downlinkConfigCommon->frequencyInfoDL->frequencyBandList.list.array[0]);
    init_symbol_rotation(&UE[CC_id]->frame_parms, UE[CC_id]->frame_parms.dl_CarrierFreq);
    init_nr_ue_vars(UE[CC_id], 0, abstraction_flag);

    #ifdef FR2_TEST
    // Overwrite DL frequency (for FR2 testing)
    if (downlink_frequency[0][0]!=0){
      frame_parms[CC_id]->dl_CarrierFreq = downlink_frequency[0][0];
      if (frame_parms[CC_id]->frame_type == TDD)
        frame_parms[CC_id]->ul_CarrierFreq = downlink_frequency[0][0];
    }
    #endif
  }

  init_openair0();
  // init UE_PF_PO and mutex lock
  pthread_mutex_init(&ue_pf_po_mutex, NULL);
  memset (&UE_PF_PO[0][0], 0, sizeof(UE_PF_PO_t)*NUMBER_OF_UE_MAX*MAX_NUM_CCs);
  configure_linux();
  mlockall(MCL_CURRENT | MCL_FUTURE);
 
  if(IS_SOFTMODEM_DOFORMS) { 
    load_softscope("nr",PHY_vars_UE_g[0][0]);
  }     

  for(int CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
    PHY_vars_UE_g[0][CC_id]->rf_map.card=0;
    PHY_vars_UE_g[0][CC_id]->rf_map.chain=CC_id+chain_offset;
    PHY_vars_UE_g[0][CC_id]->timing_advance = timing_advance;
  }

  init_NR_UE_threads(1);
  printf("UE threads created by %ld\n", gettid());
  
  // wait for end of program
  printf("TYPE <CTRL-C> TO TERMINATE\n");

  while(true)
    sleep(3600);

  if (ouput_vcd)
    vcd_signal_dumper_close();

  return 0;
}
