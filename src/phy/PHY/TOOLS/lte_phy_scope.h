



#ifndef FD_lte_scope_h_
#define FD_lte_scope_h_

#include <forms.h>
#include "PHY/defs_eNB.h"
#include "PHY/defs_UE.h"
#include "PHY/impl_defs_top.h"


/* Forms and Objects */
typedef struct {
  FL_FORM   * lte_phy_scope_enb;
  FL_OBJECT * rxsig_t;
  FL_OBJECT * chest_f;
  FL_OBJECT * chest_t;
  FL_OBJECT * pusch_comp;
  FL_OBJECT * pucch_comp;
  FL_OBJECT * pucch_comp1;
  FL_OBJECT * pusch_llr;
  FL_OBJECT * pusch_tput;
  FL_OBJECT * button_0;
} FD_lte_phy_scope_enb;

typedef struct {
    FL_FORM   * lte_phy_scope_ue;
    FL_OBJECT * rxsig_t;
    FL_OBJECT * chest_f;
    FL_OBJECT * chest_t;
    FL_OBJECT * pbch_comp;
    FL_OBJECT * pbch_llr;
    FL_OBJECT * pdcch_comp;
    FL_OBJECT * pdcch_llr;
    FL_OBJECT * pdsch_comp;
    FL_OBJECT * pdsch_llr;
    FL_OBJECT * pdsch_comp1;
    FL_OBJECT * pdsch_llr1;
    FL_OBJECT * pdsch_tput;
    FL_OBJECT * button_0;

} FD_lte_phy_scope_ue;

FD_lte_phy_scope_enb * create_lte_phy_scope_enb( void );
FD_lte_phy_scope_ue * create_lte_phy_scope_ue( void );

void phy_scope_eNB(FD_lte_phy_scope_enb *form,
                   PHY_VARS_eNB *phy_vars_enb,
                   int UE_id);

void phy_scope_UE(FD_lte_phy_scope_ue *form,
                  PHY_VARS_UE *phy_vars_ue,
                  int eNB_id,
                  int UE_id,
                  uint8_t subframe);




#endif /* FD_lte_scope_h_ */
