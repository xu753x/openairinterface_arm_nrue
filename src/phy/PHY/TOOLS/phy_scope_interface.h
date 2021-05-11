


#ifndef __PHY_SCOPE_INTERFACE_H__
#define __PHY_SCOPE_INTERFACE_H__
#include <openair1/PHY/defs_gNB.h>
typedef struct {
  int *argc;
  char **argv;
  RU_t *ru;
  PHY_VARS_gNB *gNB;
} scopeParms_t;


typedef struct scopeData_s {
  int *argc;
  char **argv;
  RU_t *ru;
  PHY_VARS_gNB *gNB;
  int32_t * rxdataF;
  void (*slotFunc)(int32_t* data, int slot,  void * scopeData);
} scopeData_t;

int load_softscope(char *exectype, void *initarg);
int end_forms(void) ;
#endif
