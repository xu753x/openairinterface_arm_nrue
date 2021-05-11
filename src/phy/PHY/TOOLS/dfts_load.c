


#define _GNU_SOURCE 
#include <sys/types.h>
#include <stdlib.h>
#include <malloc.h>
#include "assertions.h"
#include "common/utils/LOG/log.h"
#define OAIDFTS_LOADER
#include "tools_defs.h"
#include "common/config/config_userapi.h" 
#include "common/utils/load_module_shlib.h" 


/* function description array, to be used when loading the dfts/idfts lib */
static loader_shlibfunc_t shlib_fdesc[2];
static char *arg[64]={"phytest","-O","cmdlineonly::dbgl0"};


int load_dftslib(void) {
	 
	 char *ptr = (char*)config_get_if();
     if ( ptr==NULL )  {// phy simulators, config module possibly not loaded
     	 load_configmodule(3,(char **)arg,CONFIG_ENABLECMDLINEONLY) ;
     	 logInit();
     }	 
     shlib_fdesc[0].fname = "dft";
     shlib_fdesc[1].fname = "idft";
     int ret=load_module_shlib("dfts",shlib_fdesc,sizeof(shlib_fdesc)/sizeof(loader_shlibfunc_t),NULL);
     AssertFatal( (ret >= 0),"Error loading dftsc decoder");
     dft = (dftfunc_t)shlib_fdesc[0].fptr;
     idft = (idftfunc_t)shlib_fdesc[1].fptr;
return 0;
}


