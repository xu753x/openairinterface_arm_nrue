



#ifndef __PHY_LTE_TRANSPORT_PRACH_EXTERN__H__
#define __PHY_LTE_TRANSPORT_PRACH_EXTERN__H__

#include "PHY/sse_intrin.h"
#include "PHY/defs_eNB.h"
#include "PHY/phy_extern.h"

//#define PRACH_DEBUG 1
//#define PRACH_WRITE_OUTPUT_DEBUG 1

extern uint16_t NCS_unrestricted[16];
extern uint16_t NCS_restricted[15];
extern uint16_t NCS_4[7];

extern int16_t ru[2*839]; // quantized roots of unity
extern uint32_t ZC_inv[839]; // multiplicative inverse for roots u
extern uint16_t du[838];



// This is table 5.7.1-4 from 36.211
extern PRACH_TDD_PREAMBLE_MAP tdd_preamble_map[64][7];




extern uint16_t prach_root_sequence_map0_3[838];
 

extern uint16_t prach_root_sequence_map4[138];

void dump_prach_config(LTE_DL_FRAME_PARMS *frame_parms,uint8_t subframe);


// This function computes the du
void fill_du(uint8_t prach_fmt);


uint8_t get_num_prach_tdd(module_id_t Mod_id);


uint8_t get_fid_prach_tdd(module_id_t Mod_id,uint8_t tdd_map_index);


uint8_t get_prach_fmt(uint8_t prach_ConfigIndex,lte_frame_type_t frame_type);


uint8_t get_prach_prb_offset(LTE_DL_FRAME_PARMS *frame_parms, 
			     uint8_t prach_ConfigIndex, 
			     uint8_t n_ra_prboffset,
			     uint8_t tdd_mapindex, uint16_t Nf); 


int is_prach_subframe0(LTE_DL_FRAME_PARMS *frame_parms,uint8_t prach_ConfigIndex,uint32_t frame, uint8_t subframe);

int is_prach_subframe(LTE_DL_FRAME_PARMS *frame_parms,uint32_t frame, uint8_t subframe);


void compute_prach_seq(uint16_t rootSequenceIndex,
		       uint8_t prach_ConfigIndex,
		       uint8_t zeroCorrelationZoneConfig,
		       uint8_t highSpeedFlag,
		       lte_frame_type_t frame_type,
		       uint32_t X_u[64][839]);

#endif
