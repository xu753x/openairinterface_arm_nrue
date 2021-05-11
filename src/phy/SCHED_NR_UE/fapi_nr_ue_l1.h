



#ifndef __FAPI_NR_UE_L1_H__
#define __FAPI_NR_UE_L1_H__

#include "NR_IF_Module.h"

/**\brief NR UE FAPI-like P7 messages, scheduled response from L2 indicating L1
   \param scheduled_response including transmission config(dl_config, ul_config) and data transmission (tx_req)*/
int8_t nr_ue_scheduled_response(nr_scheduled_response_t *scheduled_response);

/**\brief NR UE FAPI-like P5 message, physical configuration from L2 to configure L1
   \param scheduled_response including transmission config(dl_config, ul_config) and data transmission (tx_req)*/
int8_t nr_ue_phy_config_request(nr_phy_config_t *phy_config);


#endif