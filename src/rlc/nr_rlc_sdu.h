

#ifndef _NR_RLC_SDU_H_
#define _NR_RLC_SDU_H_

typedef struct nr_rlc_sdu_t {
  int sn;
  int upper_layer_id;
  char *data;
  int size;
  int retx_count;

  int ref_count;      
} nr_rlc_sdu_t;

typedef struct nr_rlc_sdu_segment_t {
  nr_rlc_sdu_t *sdu;
  int size;
  int so;
  int is_first;
  int is_last;
  struct nr_rlc_sdu_segment_t *next;
} nr_rlc_sdu_segment_t;

nr_rlc_sdu_segment_t *nr_rlc_new_sdu(
    char *buffer, int size,
    int upper_layer_id);
void nr_rlc_free_sdu_segment(nr_rlc_sdu_segment_t *sdu);
void nr_rlc_sdu_segment_list_append(nr_rlc_sdu_segment_t **list,
                                    nr_rlc_sdu_segment_t **end,
                                    nr_rlc_sdu_segment_t *sdu);
nr_rlc_sdu_segment_t *nr_rlc_sdu_segment_list_add(
    int (*sn_compare)(void *, int, int), void *sn_compare_data,
    nr_rlc_sdu_segment_t *list, nr_rlc_sdu_segment_t *sdu_segment);
void nr_rlc_free_sdu_segment_list(nr_rlc_sdu_segment_t *l);

#endif /* _NR_RLC_SDU_H_ */
