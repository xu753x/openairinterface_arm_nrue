

/*! \file common/config/libconfig/config_libconfig.c
 * \brief configuration module, include file for libconfig implementation 
 * \author Francois TABURET
 * \date 2017
 * \version 0.1
 * \company NOKIA BellLabs France
 * \email: francois.taburet@nokia-bell-labs.com
 * \note
 * \warning
 */
#ifndef INCLUDE_CONFIG_LIBCONFIG_H
#define INCLUDE_CONFIG_LIBCONFIG_H


#ifdef __cplusplus
extern "C"
{
#endif

#include "common/config/config_paramdesc.h"



 
typedef struct libconfig_privatedata {
      char *configfile;
      config_t cfg;
} libconfig_privatedata_t;
 
#ifdef __cplusplus
}
#endif
#endif  /* INCLUDE_CONFIG_LIBCONFIG_H */
