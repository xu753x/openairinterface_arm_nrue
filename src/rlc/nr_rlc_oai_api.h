



#include "NR_RLC-BearerConfig.h"
#include "NR_RLC-Config.h"
#include "NR_LogicalChannelIdentity.h"
#include "NR_RadioBearerConfig.h"
#include "NR_CellGroupConfig.h"
#include "openair2/RRC/NR/nr_rrc_proto.h"

/* from OAI */
#include "pdcp.h"

struct NR_RLC_Config;
struct NR_LogicalChannelConfig;

void nr_rlc_bearer_init(NR_RLC_BearerConfig_t *RLC_BearerConfig, NR_RLC_BearerConfig__servedRadioBearer_PR rb_type);

void nr_drb_config(struct NR_RLC_Config *rlc_Config, NR_RLC_Config_PR rlc_config_pr);

void nr_rlc_bearer_init_ul_spec(struct NR_LogicalChannelConfig *mac_LogicalChannelConfig);
