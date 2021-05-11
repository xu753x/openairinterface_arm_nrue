


#include <stdio.h>
#include "common/config/config_userapi.h"
#include "common/utils/load_module_shlib.h"
#include "phy_scope_interface.h"

#define SOFTSCOPE_ENDFUNC_IDX 0

static  loader_shlibfunc_t scope_fdesc[]= {{"end_forms",NULL}};

int load_softscope(char *exectype, void *initarg) {
  char libname[64];
  sprintf(libname,"%.10sscope",exectype);
  return load_module_shlib(libname,scope_fdesc,1,initarg);
}

int end_forms(void) {
  if (scope_fdesc[SOFTSCOPE_ENDFUNC_IDX].fptr) {
    scope_fdesc[SOFTSCOPE_ENDFUNC_IDX].fptr();
    return 0;
  }

  return -1;
}
