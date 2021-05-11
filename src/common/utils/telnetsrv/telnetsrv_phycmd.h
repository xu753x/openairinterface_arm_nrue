

/*! \file common/utils/telnetsrv_proccmd.h
 * \brief: Include file defining telnet commands related to softmodem linux process
 * \author Francois TABURET
 * \date 2017
 * \version 0.1
 * \company NOKIA BellLabs France
 * \email: francois.taburet@nokia-bell-labs.com
 * \note
 * \warning
 */

#ifndef __TELNETSRV_PHYCMD__H__
#define __TELNETSRV_PHYCMD__H__

#ifdef TELNETSRV_PHYCMD_MAIN

#include "common/utils/LOG/log.h"


#include "openair1/PHY/phy_extern.h"


#define TELNETVAR_PHYCC0    0
#define TELNETVAR_PHYCC1    1

telnetshell_vardef_t phy_vardef[] = {
//{"iqmax",TELNET_VARTYPE_INT16,NULL},
//{"iqmin",TELNET_VARTYPE_INT16,NULL},
//{"loglvl",TELNET_VARTYPE_INT32,NULL},
//{"sndslp",TELNET_VARTYPE_INT32,NULL},
//{"rxrescale",TELNET_VARTYPE_INT32,NULL},
//{"txshift",TELNET_VARTYPE_INT32,NULL},
//{"rachemin",TELNET_VARTYPE_INT32,NULL},
//{"rachdmax",TELNET_VARTYPE_INT32,NULL},
{"",0,NULL}
};

#else

extern void add_phy_cmds(void);

#endif

/*-------------------------------------------------------------------------------------*/

#endif
