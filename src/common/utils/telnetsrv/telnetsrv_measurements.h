


/*! \file common/utils/telnetsrv/telnetsrv_measurements.h
 * \brief: Include file defining constants, structures and function prototypes
 * \       used to implement the measurements functionality of the telnet server
 * \author Francois TABURET
 * \date 2019
 * \version 0.1
 * \company NOKIA BellLabs France
 * \email: francois.taburet@nokia-bell-labs.com
 * \note
 * \warning
 */

#ifndef __TELNETSRV_MEASUREMENTS__H__
#define __TELNETSRV_MEASUREMENTS__H__

#include <dlfcn.h>
#include "telnetsrv.h"
#include "openair1/PHY/defs_eNB.h"



#define TELNET_MAXMEASURNAME_LEN 30
#define TELNET_MAXMEASURGROUPS 10

#define PRINT_CPUMEAS_STATE  ((cpumeas(CPUMEAS_GETSTATE))?"enabled":"disabled")
typedef struct cpumeasurdef {
  char statname[TELNET_MAXMEASURNAME_LEN];
  time_stats_t *astatptr;
  unsigned int statemask;
} telnet_cpumeasurdef_t;

typedef struct ltemeasurdef {
  char statname[TELNET_MAXMEASURNAME_LEN];
  void     *vptr;
  char     vtyp;
  unsigned int statemask;
} telnet_ltemeasurdef_t;

#define GROUP_LTESTATS    0
#define GROUP_CPUSTATS    1
typedef void(*measur_dislayfunc_t)(telnet_printfunc_t prnt);
typedef struct mesurgroupdef {
  char groupname[TELNET_MAXMEASURNAME_LEN];
  unsigned char type;
  unsigned char size;
  measur_dislayfunc_t displayfunc;
  union {
    telnet_cpumeasurdef_t *cpustats;
    telnet_ltemeasurdef_t *ltestats;
  };
} telnet_measurgroupdef_t;

#define MACSTATS_NAME(valptr) #valptr
#define LTEMAC_MEASURGROUP_NAME  "ltemac"
#define PHYCPU_MEASURGROUP_NAME  "phycpu"

#ifdef TELNETSRV_MEASURMENTS_MAIN
int measurcmd_show(char *buf, int debug, telnet_printfunc_t prnt);
int measurcmd_cpustats(char *buf, int debug, telnet_printfunc_t prnt);
telnetshell_cmddef_t measur_cmdarray[] = {
  {"show", "groups | <group name>" , measurcmd_show},
  {"cpustats","[enable | disable]",measurcmd_cpustats},
  {"","",NULL}
};

telnetshell_vardef_t measur_vardef[] = {
  {"",0,NULL}
};
/* function to be implemented in any telnetsrv_xxx_measurements.c sources
   to allow common measurment code to access measurments data             */
extern int get_measurgroups(telnet_measurgroupdef_t **measurgroups);
/* */

#else
extern void     add_measur_cmds(void);
extern void     measurcmd_display_groups(telnet_printfunc_t prnt,telnet_measurgroupdef_t *measurgroups,int groups_size);
extern void     measurcmd_display_cpumeasures(telnet_printfunc_t prnt, telnet_cpumeasurdef_t  *cpumeasure, int cpumeasure_size) ;
extern uint64_t measurcmd_getstatvalue(telnet_ltemeasurdef_t *measur,telnet_printfunc_t prnt);
extern void     measurcmd_display_measures(telnet_printfunc_t prnt, telnet_ltemeasurdef_t  *statsptr, int stats_size);

#endif  /* TELNETSRV_MEASURCMD_MAIN */
#endif
