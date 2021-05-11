

/*! \file common/utils/telnetsrv_proccmd.h
 * \brief: Include file defining telnet commands related to softmodem linux process
 * \author Francois TABURET
 * \date 2018
 * \version 0.1
 * \company NOKIA BellLabs France
 * \email: francois.taburet@nokia-bell-labs.com
 * \note
 * \warning
 */

#ifdef TELNETSRV_LOADER_MAIN

#include "common/utils/LOG/log.h"


#include "common/utils/load_module_shlib.h"


telnetshell_vardef_t loader_globalvardef[] = {
{"mainversion",TELNET_VARTYPE_STRING,&(loader_data.mainexec_buildversion)},
{"defpath",TELNET_VARTYPE_STRING,&(loader_data.shlibpath)},
{"maxshlibs",TELNET_VARTYPE_INT32,&(loader_data.maxshlibs)},
{"numshlibs",TELNET_VARTYPE_INT32,&(loader_data.numshlibs)},
{"",0,NULL}
};
telnetshell_vardef_t *loader_modulesvardef;

extern void add_loader_cmds(void);

#endif

/*-------------------------------------------------------------------------------------*/

