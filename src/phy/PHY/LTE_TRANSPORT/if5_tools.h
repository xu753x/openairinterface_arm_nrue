



#ifndef __IF5_TOOLS_H__
#define __IF5_TOOLS_H__

#include <stdint.h>
#include "PHY/defs_eNB.h"

#define IF5_RRH_GW_DL 0x0022
#define IF5_RRH_GW_UL 0x0023
#define IF5_MOBIPASS 0xbffe

struct IF5_mobipass_header {  
  /// 
  uint16_t flags; 
  /// 
  uint16_t fifo_status;
  /// 
  uint8_t seqno;
  ///
  uint8_t ack;
  ///
  uint32_t word0;
  /// 
  uint32_t time_stamp;
  
} __attribute__ ((__packed__));

typedef struct IF5_mobipass_header IF5_mobipass_header_t;
#define sizeof_IF5_mobipass_header_t 14

void send_IF5(RU_t *, openair0_timestamp, int, uint8_t*, uint16_t);

void recv_IF5(RU_t *ru, openair0_timestamp *proc_timestamp, int subframe, uint16_t packet_type);


void malloc_IF5_buffer(RU_t *ru);

#endif

