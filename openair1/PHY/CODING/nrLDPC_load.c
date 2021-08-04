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
#include <dlfcn.h>


/* function description array, to be used when loading the encoding/decoding shared lib */
static loader_shlibfunc_t shlib_fdesc[2];

char *arg[64]={"ldpctest","-O","cmdlineonly::dbgl0"};

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

int load_cuFFT(void) {
     //手动加载指定位置的so动态库
     void* handle = dlopen("../../../hs/cuFFT.so", RTLD_LAZY|RTLD_NODELETE|RTLD_GLOBAL);
     if(!handle){
          printf("open cuFFT.so error!\n");
          return -1;
     }
     //根据动态链接库操作句柄与符号，返回符号对应的地址
     cudft2048 = (cudft_EnTx) dlsym(handle, "_Z9cudft2048PsS_h");
     if(!cudft2048){
          printf("cuFFT.so cudft2048 error!\n");
          dlclose(handle);
          return -1;
     }
     load_cudft = (cudft_load) dlsym(handle, "_Z10load_cuFFTv");
     if(!load_cudft){
          printf("cuFFT.so load_cudft error!\n");
          dlclose(handle);
          return -1;
     }
     load_cudft();
return 0;
}

int load_cuFFT1(void) {
     //手动加载指定位置的so动态库
     void* handle1 = dlopen("../../../hs/cuFFT1.so", RTLD_LAZY|RTLD_NODELETE|RTLD_GLOBAL);
     if(!handle1){
          printf("open cuFFT1.so error!\n");
          return -1;
     }
     //根据动态链接库操作句柄与符号，返回符号对应的地址
     cudft20481 = (cudft_EnTx) dlsym(handle1, "_Z9cudft2048PsS_S_");
     if(!cudft20481){
          printf("cuFFT1.so cudft2048 error!\n");
          dlclose(handle1);
          return -1;
     }
     load_cudft1 = (cudft_load) dlsym(handle1, "_Z10load_cuFFTv");
     if(!load_cudft1){
          printf("cuFFT1.so load_cudft error!\n");
          dlclose(handle1);
          return -1;
     }
     cuda_rotate = (cudft_rotate) dlsym(handle1, "_Z22cuda_rotate_cpx_vectorPsS_S_jt");
     load_cudft1();
return 0;
}

