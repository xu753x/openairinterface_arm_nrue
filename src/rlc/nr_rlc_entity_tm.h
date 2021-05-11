

#ifndef _NR_RLC_ENTITY_TM_H_
#define _NR_RLC_ENTITY_TM_H_

#include "nr_rlc_entity.h"
#include "nr_rlc_sdu.h"

typedef struct {
  nr_rlc_entity_t common;

  
  uint64_t t_current;

  /* tx management */
  nr_rlc_sdu_segment_t *tx_list;
  nr_rlc_sdu_segment_t *tx_end;
  int                  tx_size;
  int                  tx_maxsize;
} nr_rlc_entity_tm_t;

void nr_rlc_entity_tm_recv_sdu(nr_rlc_entity_t *entity,
                               char *buffer, int size,
                               int sdu_id);
void nr_rlc_entity_tm_recv_pdu(nr_rlc_entity_t *entity,
                               char *buffer, int size);
nr_rlc_entity_buffer_status_t nr_rlc_entity_tm_buffer_status(
    nr_rlc_entity_t *entity, int maxsize);
int nr_rlc_entity_tm_generate_pdu(nr_rlc_entity_t *entity,
                                  char *buffer, int size);
void nr_rlc_entity_tm_set_time(nr_rlc_entity_t *entity, uint64_t now);
void nr_rlc_entity_tm_discard_sdu(nr_rlc_entity_t *_entity, int sdu_id);
void nr_rlc_entity_tm_reestablishment(nr_rlc_entity_t *_entity);
void nr_rlc_entity_tm_delete(nr_rlc_entity_t *entity);

#endif /* _NR_RLC_ENTITY_TM_H_ */
