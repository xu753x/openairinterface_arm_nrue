

#include "nr_rlc_sdu.h"

#include <stdlib.h>
#include <string.h>

#include "LOG/log.h"

nr_rlc_sdu_segment_t *nr_rlc_new_sdu(
    char *buffer, int size,
    int upper_layer_id)
{
  nr_rlc_sdu_t *sdu         = calloc(1, sizeof(nr_rlc_sdu_t));
  nr_rlc_sdu_segment_t *ret = calloc(1, sizeof(nr_rlc_sdu_segment_t));
  if (sdu == NULL || ret == NULL)
    goto oom;

  sdu->ref_count      = 1;
  sdu->sn             = -1;                 
  sdu->upper_layer_id = upper_layer_id;
  sdu->data           = malloc(size);
  if (sdu->data == NULL)
    goto oom;
  memcpy(sdu->data, buffer, size);
  sdu->size           = size;
  sdu->retx_count     = -1;

  ret->sdu      = sdu;
  ret->size     = size;
  ret->so       = 0;
  ret->is_first = 1;
  ret->is_last  = 1;

  return ret;

oom:
  LOG_E(RLC, "%s:%d:%s: out of memory\n", __FILE__, __LINE__,  __FUNCTION__);
  exit(1);
}

void nr_rlc_free_sdu_segment(nr_rlc_sdu_segment_t *sdu)
{
  sdu->sdu->ref_count--;
  if (sdu->sdu->ref_count == 0) {
    free(sdu->sdu->data);
    free(sdu->sdu);
  }
  free(sdu);
}

void nr_rlc_sdu_segment_list_append(nr_rlc_sdu_segment_t **list,
                                    nr_rlc_sdu_segment_t **end,
                                    nr_rlc_sdu_segment_t *sdu)
{
  if (*list == NULL) {
    *list = sdu;
    *end = sdu;
    return;
  }

  (*end)->next = sdu;
  *end = sdu;
}

nr_rlc_sdu_segment_t *nr_rlc_sdu_segment_list_add(
    int (*sn_compare)(void *, int, int), void *sn_compare_data,
    nr_rlc_sdu_segment_t *list, nr_rlc_sdu_segment_t *sdu_segment)
{
  nr_rlc_sdu_segment_t head;
  nr_rlc_sdu_segment_t *cur;
  nr_rlc_sdu_segment_t *prev;

  head.next = list;
  cur = list;
  prev = &head;

  /* order is by 'sn', if 'sn' is the same then order is by 'so' */
  while (cur != NULL) {
    /* check if 'sdu_segment' is before 'cur' in the list */
    if (sn_compare(sn_compare_data, cur->sdu->sn, sdu_segment->sdu->sn) > 0 ||
        (cur->sdu->sn == sdu_segment->sdu->sn && cur->so > sdu_segment->so)) {
      break;
    }
    prev = cur;
    cur = cur->next;
  }
  prev->next = sdu_segment;
  sdu_segment->next = cur;
  return head.next;
}

void nr_rlc_free_sdu_segment_list(nr_rlc_sdu_segment_t *l)
{
  nr_rlc_sdu_segment_t *cur;

  while (l != NULL) {
    cur = l;
    l = l->next;
    nr_rlc_free_sdu_segment(cur);
  }
}
