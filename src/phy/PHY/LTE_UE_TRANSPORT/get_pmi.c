

#include "PHY/LTE_UE_TRANSPORT/transport_proto_ue.h"

uint8_t get_pmi(uint8_t N_RB_DL, MIMO_mode_t mode, uint32_t pmi_alloc,uint16_t rb)
{
  

  switch (N_RB_DL) {
    case 6:   // 1 PRB per subband
      if (mode <= PUSCH_PRECODING1)
        return((pmi_alloc>>(rb<<1))&3);
      else
        return((pmi_alloc>>rb)&1);

      break;

    default:
    case 25:  // 4 PRBs per subband
      if (mode <= PUSCH_PRECODING1)
        return((pmi_alloc>>((rb>>2)<<1))&3);
      else
        return((pmi_alloc>>(rb>>2))&1);

      break;

    case 50: // 6 PRBs per subband
      if (mode <= PUSCH_PRECODING1)
        return((pmi_alloc>>((rb/6)<<1))&3);
      else
        return((pmi_alloc>>(rb/6))&1);

      break;

    case 100: // 8 PRBs per subband
      if (mode <= PUSCH_PRECODING1)
        return((pmi_alloc>>((rb>>3)<<1))&3);
      else
        return((pmi_alloc>>(rb>>3))&1);

      break;
  }
}
