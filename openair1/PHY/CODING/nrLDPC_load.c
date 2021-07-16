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

/*! \file openair1/PHY/CODING/coding_nr_load.c
 * \brief: load library implementing coding/decoding algorithms
 * \author Francois TABURET
 * \date 2020
 * \version 0.1
 * \company NOKIA BellLabs France
 * \email: francois.taburet@nokia-bell-labs.com
 * \note
 * \warning
 */
#define _GNU_SOURCE 
#include <sys/types.h>
#include <stdlib.h>
#include <malloc.h>
#include "assertions.h"
#include "common/utils/LOG/log.h"
#define LDPC_LOADER
#include "PHY/CODING/nrLDPC_extern.h"
#include "common/config/config_userapi.h" 
#include "common/utils/load_module_shlib.h" 
#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <hugetlbfs.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/time.h>
#include <dlfcn.h>
#include "openair1/PHY/NR_TRANSPORT/time_measure_diff.h"


/* function description array, to be used when loading the encoding/decoding shared lib */
static loader_shlibfunc_t shlib_fdesc[2];

char *arg[64]={"ldpctest","-O","cmdlineonly::dbgl0"};

#if 0
//nr_ulsch_unscrambling_optim_fpga_ldpc测解扰个别时间
double   unscramble_llr128_gettime_cur, unscramble_llr8_gettime_cur;
struct timespec unscramble_llr128_start, unscramble_llr8_start;
struct timespec unscramble_llr128_stop, unscramble_llr8_stop;

//nr_ulsch_decoding_fpga_ldpc测fpga时间
double   decode_clock_gettime_cur;
struct timespec decode_start;
struct timespec decode_stop;

//nr_ulsch_procedures_fpga_ldpc测解扰和译码时间
double   unscrambing_gettime_cur,decoding_gettime_cur;
struct timespec unscrambing_start,decoding_start;
struct timespec unscrambing_stop,decoding_stop;

//phy_procedures_gNB_uespec_RX测总时间
double   ulsch_clock_gettime_cur;
struct timespec ulsch_start;
struct timespec ulsch_stop;
#endif

int load_nrLDPClib(void) {
	 char *ptr = (char*)config_get_if();
     if ( ptr==NULL )  {// phy simulators, config module possibly not loaded
     	 load_configmodule(3,(char **)arg,CONFIG_ENABLECMDLINEONLY) ;
     	 logInit();
     }	 
     shlib_fdesc[0].fname = "nrLDPC_decod";
     shlib_fdesc[1].fname = "nrLDPC_encod";
     int ret=load_module_shlib("ldpc",shlib_fdesc,sizeof(shlib_fdesc)/sizeof(loader_shlibfunc_t),NULL);
     AssertFatal( (ret >= 0),"Error loading ldpc decoder");
     nrLDPC_decoder = (nrLDPC_decoderfunc_t)shlib_fdesc[0].fptr;
     nrLDPC_encoder = (nrLDPC_encoderfunc_t)shlib_fdesc[1].fptr;

     //手动加载指定位置的so动态库
     void *handle = dlopen("/home/witcomm/work/yihz_5gran/ran/yihz/FecDemo/cDemo2/ldpc_fpga_encode.so", RTLD_LAZY|RTLD_NODELETE|RTLD_GLOBAL);
     if(!handle){
          printf("open ldpc_fpga_encode error!\n");
          return -1;
     }
     
     //根据动态链接库操作句柄与符号，返回符号对应的地址
     // add = (LDPC_FPGA_EnTx_Test) dlsym(handle, "add");
     // if(!add){
     //      printf("FPGA loading add error!\n");
     //      dlclose(handle);
     //      return -1;
     // }
#if 1
     HugePage_Init = (LDPC_FPGA_HugePage_Init) dlsym(handle, "HugePage_Init");
     if(!HugePage_Init){
          printf("FPGA loading HugePage_Init error!\n");
          dlclose(handle);
          return -1;
     }
     int HP = HugePage_Init(1);
     if(HP != 0){
          printf("HugePage_Init error!\n");
     }
     LOG_D(PHY,"load_nrLDPClib \n");
     encoder_load = (LDPC_FPGA_EnTx) dlsym(handle, "encoder_load");
     if(!encoder_load){
          printf("FPGA loading encoder_load error!\n");
          dlclose(handle);
          return -1;
     }
     decoder_load = (LDPC_FPGA_DeTx) dlsym(handle, "decoder_load");
     if(!decoder_load){
          printf("FPGA loading decoder_load error!\n");
          dlclose(handle);
          return -1;
     }
#endif

return 0;
}

int load_nrLDPClib_ref(char *libversion, nrLDPC_encoderfunc_t * nrLDPC_encoder_ptr) {
	loader_shlibfunc_t shlib_encoder_fdesc;

     shlib_encoder_fdesc.fname = "nrLDPC_encod";
     char libpath[64];
     sprintf(libpath,"ldpc%s",libversion);
     int ret=load_module_shlib(libpath,&shlib_encoder_fdesc,1,NULL);
     AssertFatal( (ret >= 0),"Error loading ldpc encoder %s\n",libpath);
     *nrLDPC_encoder_ptr = (nrLDPC_encoderfunc_t)shlib_encoder_fdesc.fptr;
return 0;
}


