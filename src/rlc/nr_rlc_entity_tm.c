

#include "nr_rlc_entity_tm.h"

#include <stdlib.h>
#include <string.h>

#include "nr_rlc_pdu.h"

#include "LOG/log.h"


/* PDU RX functions                                                      */
/*************************************************************************/

void nr_rlc_entity_tm_recv_pdu(nr_rlc_entity_t *_entity,
                               char *buffer, int size)
{
  nr_rlc_entity_tm_t *entity = (nr_rlc_entity_tm_t *)_entity;
  entity->common.deliver_sdu(entity->common.deliver_sdu_data,
                             (nr_rlc_entity_t *)entity,
                             buffer, size);
}

/*************************************************************************/
/* TX functions                                                          */
/*************************************************************************/

static int generate_tx_pdu(nr_rlc_entity_tm_t *entity, char *buffer, int size)
{
  nr_rlc_sdu_segment_t *sdu;
  int ret;

  if (entity->tx_list == NULL)
    return 0;

  sdu = entity->tx_list;

  /* not enough room? do nothing */
  if (sdu->size > size)
    return 0;

  entity->tx_list = entity->tx_list->next;
  if (entity->tx_list == NULL)
    entity->tx_end = NULL;

  ret = sdu->size;

  memcpy(buffer, sdu->sdu->data, sdu->size);

  entity->tx_size -= sdu->size;
  nr_rlc_free_sdu_segment(sdu);

  return ret;
}

static int tx_list_size(nr_rlc_entity_tm_t *entity,
                        nr_rlc_sdu_segment_t *l, int maxsize)
{
  int ret = 0;

  while (l != NULL) {
    ret += l->size;
    l = l->next;
  }

  if (ret > maxsize) ret = maxsize;
  return ret;
}

nr_rlc_entity_buffer_status_t nr_rlc_entity_tm_buffer_status(
    nr_rlc_entity_t *_entity, int maxsize)
{
  nr_rlc_entity_tm_t *entity = (nr_rlc_entity_tm_t *)_entity;
  nr_rlc_entity_buffer_status_t ret;

  ret.status_size = 0;
  ret.tx_size = tx_list_size(entity, entity->tx_list, maxsize);
  ret.retx_size = 0;

  return ret;
}

int nr_rlc_entity_tm_generate_pdu(nr_rlc_entity_t *_entity,
                                  char *buffer, int size)
{
  nr_rlc_entity_tm_t *entity = (nr_rlc_entity_tm_t *)_entity;

  return generate_tx_pdu(entity, buffer, size);
}

/*************************************************************************/
/* SDU RX functions                                                      */
/*************************************************************************/

void nr_rlc_entity_tm_recv_sdu(nr_rlc_entity_t *_entity,
                               char *buffer, int size,
                               int sdu_id)
{
  nr_rlc_entity_tm_t *entity = (nr_rlc_entity_tm_t *)_entity;
  nr_rlc_sdu_segment_t *sdu;

  if (size > NR_SDU_MAX) {
    LOG_E(RLC, "%s:%d:%s: fatal: SDU size too big (%d bytes)\n",
          __FILE__, __LINE__, __FUNCTION__, size);
    exit(1);
  }

  if (entity->tx_size + size > entity->tx_maxsize) {
    LOG_D(RLC, "%s:%d:%s: warning: SDU rejected, SDU buffer full\n",
          __FILE__, __LINE__, __FUNCTION__);
    return;
  }

  entity->tx_size += size;

  sdu = nr_rlc_new_sdu(buffer, size, sdu_id);

  nr_rlc_sdu_segment_list_append(&entity->tx_list, &entity->tx_end, sdu);
}

/*************************************************************************/
/* time/timers                                                           */
/*************************************************************************/

void nr_rlc_entity_tm_set_time(nr_rlc_entity_t *_entity, uint64_t now)
{
  nr_rlc_entity_tm_t *entity = (nr_rlc_entity_tm_t *)_entity;

  entity->t_current = now;
}

/*************************************************************************/
/* discard/re-establishment/delete                                       */
/*************************************************************************/

void nr_rlc_entity_tm_discard_sdu(nr_rlc_entity_t *_entity, int sdu_id)
{
  /* nothing to do */
}

static void clear_entity(nr_rlc_entity_tm_t *entity)
{
  nr_rlc_free_sdu_segment_list(entity->tx_list);

  entity->tx_list         = NULL;
  entity->tx_end          = NULL;
  entity->tx_size         = 0;
}

void nr_rlc_entity_tm_reestablishment(nr_rlc_entity_t *_entity)
{
  nr_rlc_entity_tm_t *entity = (nr_rlc_entity_tm_t *)_entity;
  clear_entity(entity);
}

void nr_rlc_entity_tm_delete(nr_rlc_entity_t *_entity)
{
  nr_rlc_entity_tm_t *entity = (nr_rlc_entity_tm_t *)_entity;
  clear_entity(entity);
  free(entity);
}
