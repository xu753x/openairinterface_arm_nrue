



//#include "defs.h"
#include "mac_proto.h"
#include "../../ARCH/COMMON/common_lib.h"
//#undef MALLOC
#include "assertions.h"
#include "PHY/types.h"
#include "PHY/defs_UE.h"
#include "openair2/LAYER2/nr_pdcp/nr_pdcp_entity.h"
#include "executables/softmodem-common.h"
#include "openair2/LAYER2/nr_pdcp/nr_pdcp.h"

static NR_UE_MAC_INST_t *nr_ue_mac_inst; 

NR_UE_MAC_INST_t * nr_l2_init_ue(NR_UE_RRC_INST_t* rrc_inst)
{
    //LOG_I(MAC, "[MAIN] MAC_INIT_GLOBAL_PARAM IN...\n");

    //LOG_I(MAC, "[MAIN] init UE MAC functions \n");
    
    //init mac here
    nr_ue_mac_inst = (NR_UE_MAC_INST_t *)calloc(sizeof(NR_UE_MAC_INST_t),NB_NR_UE_MAC_INST);
    for (int j=0;j<NB_NR_UE_MAC_INST;j++)
	    for (int i=0;i<NR_MAX_HARQ_PROCESSES;i++)
	      nr_ue_mac_inst[j].first_ul_tx[i]=1;

    if (rrc_inst && (rrc_inst->scell_group_config || rrc_inst->cell_group_config)) {

      if(rrc_inst->scell_group_config) {
        nr_rrc_mac_config_req_ue(0,0,0,NULL,NULL,NULL,rrc_inst->scell_group_config);
        //if (IS_SOFTMODEM_NOS1){
        //  AssertFatal(rlc_module_init(0) == 0, "%s: Could not initialize RLC layer\n", __FUNCTION__);
        //  nr_pdcp_layer_init();
        //  nr_DRB_preconfiguration(nr_ue_mac_inst->crnti);
        //}
      } else if (rrc_inst->cell_group_config) {
        nr_rrc_mac_config_req_ue(0,0,0,NULL,NULL,rrc_inst->cell_group_config,NULL);
        AssertFatal(rlc_module_init(0) == 0, "%s: Could not initialize RLC layer\n", __FUNCTION__);
        if (IS_SOFTMODEM_NOS1){
          pdcp_layer_init();
          nr_DRB_preconfiguration(nr_ue_mac_inst->crnti);
        }
      }

      // Allocate memory for ul_config_request in the mac instance. This is now a pointer and will
      // point to a list of structures (one for each UL slot) to store PUSCH scheduling parameters
      // received from UL DCI.
      if (nr_ue_mac_inst->scc) {
        int num_slots_ul = nr_ue_mac_inst->scc->tdd_UL_DL_ConfigurationCommon->pattern1.nrofUplinkSlots;
        if (nr_ue_mac_inst->scc->tdd_UL_DL_ConfigurationCommon->pattern1.nrofUplinkSymbols>0)
          num_slots_ul++;
        LOG_D(MAC, "Initializing ul_config_request. num_slots_ul = %d\n", num_slots_ul);
        nr_ue_mac_inst->ul_config_request = (fapi_nr_ul_config_request_t *)calloc(num_slots_ul, sizeof(fapi_nr_ul_config_request_t));
      }

    } else {
      LOG_I(MAC,"Running without CellGroupConfig\n");
      nr_rrc_mac_config_req_ue(0,0,0,NULL,NULL,NULL,NULL);
      if(get_softmodem_params()->sa == 1) {
        AssertFatal(rlc_module_init(0) == 0, "%s: Could not initialize RLC layer\n", __FUNCTION__);
      }
    }

    return (nr_ue_mac_inst);
}

NR_UE_MAC_INST_t *get_mac_inst(module_id_t module_id){
    return &nr_ue_mac_inst[(int)module_id];
}
