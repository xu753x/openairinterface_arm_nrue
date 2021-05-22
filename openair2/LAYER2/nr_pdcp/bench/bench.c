#include "../nr_pdcp_entity.h"

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>

void deliver_sdu_ue(void *deliver_sdu_data, nr_pdcp_entity_t *entity,
                    char *buf, int size)
{
  int *count = deliver_sdu_data;
  (*count)++;
}

void deliver_pdu_ue(void *deliver_pdu_data, nr_pdcp_entity_t *entity,
                    char *buf, int size, int sdu_id)
{
  printf("deliver_pdu_ue %d id %d\n", size, sdu_id);
}

void deliver_sdu_gnb(void *deliver_sdu_data, nr_pdcp_entity_t *entity,
                     char *buf, int size)
{
  printf("deliver_sdu_gnb %d\n", size);
}

void deliver_pdu_gnb(void *deliver_pdu_data, nr_pdcp_entity_t *entity,
                     char *buf, int size, int sdu_id)
{
  nr_pdcp_entity_t *ue = deliver_pdu_data;
  ue->recv_pdu(ue, buf, size);
}

uint64_t get_time(void)
{
  struct timespec t;
  if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) {
    printf("error: clock_gettime failed\n");
    return 0;
  }
  return (uint64_t)t.tv_sec * (uint64_t)1000000 + (uint64_t)t.tv_nsec / 1000;
}

void run_test(int bytes, nr_pdcp_entity_t *gnb, char *sdu, int N, int *count)
{
  int i;
  uint64_t start_time;
  uint64_t stop_time;

  start_time = get_time();
  *count = 0;
  for (i = 0; i < N; i++)
    gnb->recv_sdu(gnb, sdu, bytes, i);
  if (*count != N)
    printf("error, only %d SDUs successfully transmitted instead of %d\n",
           *count, N);
  stop_time = get_time();
  printf("%d bytes: %"PRIu64" microseconds for %d loops\n",
         bytes, stop_time - start_time, N);
}

int main(void)
{
  char sdu[9000];
  int i;
  int N = 100000;     /* loop count for benchmarking */
  int count;

  unsigned char ciphering_key[16] = {
      0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
      0x09, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16
  };
  unsigned char integrity_key[16] = {
      0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
      0x19, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26
  };
  nr_pdcp_entity_t *ue;
  nr_pdcp_entity_t *gnb;

  ue = new_nr_pdcp_entity(NR_PDCP_DRB_AM, 0 /* is_gnb */, 1, 1, 0, 0, 0,
                          deliver_sdu_ue, &count,
                          deliver_pdu_ue, NULL,
                          18, /* sn size */
                          -1, /* t-reordering */
                          -1, /* discard timer */
                          2,  /* ciphering algorithm, 2 = NEA2 */
                          2,  /* integrity algorithm, 2 = NIA2 */
                          ciphering_key,
                          integrity_key);

  gnb = new_nr_pdcp_entity(NR_PDCP_DRB_AM, 1 /* is_gnb */, 1, 1, 0, 0, 0,
                           deliver_sdu_gnb, NULL,
                           deliver_pdu_gnb, ue,
                           18, /* sn size */
                           -1, /* t-reordering */
                           -1, /* discard timer */
                           2,  /* ciphering algorithm, 2 = NEA2 */
                           2,  /* integrity algorithm, 2 = NIA2 */
                           ciphering_key,
                           integrity_key);

  for (i = 0; i < 9000; i++) sdu[i] = i & 255;

  run_test(10, gnb, sdu, N, &count);
  run_test(100, gnb, sdu, N, &count);
  run_test(1000, gnb, sdu, N, &count);
  run_test(9000, gnb, sdu, N, &count);

  return 0;
}
