

#ifndef _NR_RLC_UE_MANAGER_H_
#define _NR_RLC_UE_MANAGER_H_

#include "nr_rlc_entity.h"

typedef void nr_rlc_ue_manager_t;

typedef struct nr_rlc_ue_t {
  int rnti;
  nr_rlc_entity_t *srb[3];
  nr_rlc_entity_t *drb[5];
} nr_rlc_ue_t;


/* manager functions                                                   */
/***********************************************************************/

nr_rlc_ue_manager_t *new_nr_rlc_ue_manager(int enb_flag);

int nr_rlc_manager_get_enb_flag(nr_rlc_ue_manager_t *m);

void nr_rlc_manager_lock(nr_rlc_ue_manager_t *m);
void nr_rlc_manager_unlock(nr_rlc_ue_manager_t *m);

nr_rlc_ue_t *nr_rlc_manager_get_ue(nr_rlc_ue_manager_t *m, int rnti);
void nr_rlc_manager_remove_ue(nr_rlc_ue_manager_t *m, int rnti);

/***********************************************************************/
/* ue functions                                                        */
/***********************************************************************/

void nr_rlc_ue_add_srb_rlc_entity(nr_rlc_ue_t *ue, int srb_id, nr_rlc_entity_t *entity);
void nr_rlc_ue_add_drb_rlc_entity(nr_rlc_ue_t *ue, int drb_id, nr_rlc_entity_t *entity);

#endif /* _NR_RLC_UE_MANAGER_H_ */
