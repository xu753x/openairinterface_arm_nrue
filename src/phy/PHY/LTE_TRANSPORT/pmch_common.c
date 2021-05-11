

#include "PHY/defs_eNB.h"
#include "PHY/phy_extern.h"






int is_fembms_cas_subframe(uint32_t frame, int subframe, LTE_DL_FRAME_PARMS *frame_parms) {
  //uint32_t period;

  //if(frame_parms->NonMBSFN_config_flag ) {
  //  period = 4<<frame_parms->NonMBSFN_config.radioframeAllocationPeriod;

  //  if ((frame % period) == frame_parms->NonMBSFN_config.radioframeAllocationOffset) {
  //    switch (subframe) {
  //      case 0:
  //        return(1); //This should be CAS
  //        break;
  //    }
  //  }
  //}
  if(frame_parms->NonMBSFN_config_flag || frame_parms->FeMBMS_active){
   if ((frame&3)==0) {
      switch (subframe) {
        case 0:
                return(1); //This should be CAS 
           break;
      }
   }
	
  }

  return (0);
}

int is_fembms_nonMBSFN_subframe(uint32_t frame, int subframe, LTE_DL_FRAME_PARMS *frame_parms){
   uint32_t period;
   if(frame_parms->NonMBSFN_config_flag ) {
      period = 4<<frame_parms->NonMBSFN_config.radioframeAllocationPeriod;
      if ((frame % period) == frame_parms->NonMBSFN_config.radioframeAllocationOffset) {
        switch (subframe) {
        case 0:
                return(1); //This should be CAS
           break;
        case 1:
           if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x100) > 0)
                return(1);
           break;
        case 2:
           if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x80) > 0)
                return(1);
           break;
        case 3:
           if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x40) > 0)
                return(1);
           break;
        case 4:
           if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x20) > 0)
                return(1);
           break;
        case 5:
           if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x10) > 0)
                return(1);
           break;
        case 6:
           if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x8) > 0)
                return(1);
           break;
        case 7:
           if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x4) > 0)
                return(1);
           break;
        case 8:
           if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x2) > 0)
                return(1);
           break;
        case 9:
           if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x1) > 0)
                return(1);
           break;
        }
      }
  }
    return (0);
}

int is_fembms_pmch_subframe(uint32_t frame, int subframe, LTE_DL_FRAME_PARMS *frame_parms) {
  uint32_t period;

  if(frame_parms->NonMBSFN_config_flag ) {
    period = 4<<frame_parms->NonMBSFN_config.radioframeAllocationPeriod;

    if ((frame % period) == frame_parms->NonMBSFN_config.radioframeAllocationOffset) {
      switch (subframe) {
 	case 0:
            return(0);
  	  break;
        case 1:
          if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x100) > 0)
            return(0);

          break;

        case 2:
          if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x80) > 0)
            return(0);

          break;

        case 3:
          if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x40) > 0)
            return(0);

          break;

        case 4:
          if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x20) > 0)
            return(0);

          break;

        case 5:
          if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x10) > 0)
            return(0);

          break;

        case 6:
          if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x8) > 0)
            return(0);

          break;

        case 7:
          if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x4) > 0)
            return(0);

          break;

        case 8:
          if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x2) > 0)
            return(0);

          break;

        case 9:
          if ((frame_parms->NonMBSFN_config.non_mbsfn_SubframeConfig & 0x1) > 0)
            return(0);

          break;
      }
    } else if((frame_parms->FeMBMS_active == 1 || frame_parms->NonMBSFN_config_flag == 1 ) && !is_fembms_cas_subframe(frame,subframe,frame_parms) ) { //Then regular MBSFN FeMBMS subframe
      return(1);
    }
    return(1);
  }

  return(0);
}


int is_pmch_subframe(uint32_t frame, int subframe, LTE_DL_FRAME_PARMS *frame_parms) {
  uint32_t period;
  uint8_t i;
  uint8_t j;

  //  LOG_D(PHY,"is_pmch_subframe: frame %d, subframe %d, num_MBSFN_config %d\n",
  //  frame,subframe,frame_parms->num_MBSFN_config);

  for (i=0; i<frame_parms->num_MBSFN_config; i++) {  // we have at least one MBSFN configuration
    period = 1<<frame_parms->MBSFN_config[i].radioframeAllocationPeriod;

    if (frame_parms->MBSFN_config[i].fourFrames_flag == 0) {
      if ((frame % period) == frame_parms->MBSFN_config[i].radioframeAllocationOffset) {
        if (frame_parms->frame_type == FDD) {
          switch (subframe) {
            case 1:
              if ((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig & MBSFN_FDD_SF1) > 0)
                return(1);

              break;

            case 2:
              if ((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig & MBSFN_FDD_SF2) > 0)
                return(1);

              break;

            case 3:
              if ((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig & MBSFN_FDD_SF3) > 0)
                return(1);

              break;

            case 6:
              if ((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig & MBSFN_FDD_SF6) > 0)
                return(1);

              break;

            case 7:
              if ((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig & MBSFN_FDD_SF7) > 0)
                return(1);

              break;

            case 8:
              if ((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig & MBSFN_FDD_SF8) > 0)
                return(1);

              break;
          }
        } else  {
          switch (subframe) {
            case 3:
              if ((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig & MBSFN_TDD_SF3) > 0)
                return(1);

              break;

            case 4:
              if ((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig & MBSFN_TDD_SF4) > 0)
                return(1);

              break;

            case 7:
              if ((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig & MBSFN_TDD_SF7) > 0)
                return(1);

              break;

            case 8:
              if ((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig & MBSFN_TDD_SF8) > 0)
                return(1);

              break;

            case 9:
              if ((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig & MBSFN_TDD_SF9) > 0)
                return(1);

              break;
          }
        }
      }
    } else { // handle 4 frames case
      for(j=0; j<4; j++) {
        if ((frame % period) == (frame_parms->MBSFN_config[i].radioframeAllocationOffset + j)) {
          if (frame_parms->frame_type == FDD) {
            switch (subframe) {
              case 1:
                if (((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig)  & (0x800000>>(j*6))) > 0) {
                  //LOG_E(PHY,"SubframeConfig<<6(%x),MBSFN_FDD_SF1(%x)\n",frame_parms->MBSFN_config[i].mbsfn_SubframeConfig,0x800000>>(j*6));
                  return(1);
                }

                break;

              case 2:
                if (((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig) & (0x400000>>(j*6))) > 0) {
                  //LOG_E(PHY,"SubframeConfig<<6(%x),MBSFN_FDD_SF1(%x)\n",frame_parms->MBSFN_config[i].mbsfn_SubframeConfig,0x400000>>(j*6));
                  return(1);
                }

                break;

              case 3:
                if (((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig) & (0x200000>>(j*6))) > 0) {
                  //LOG_E(PHY,"SubframeConfig<<6(%x),MBSFN_FDD_SF1(%x)\n",frame_parms->MBSFN_config[i].mbsfn_SubframeConfig,0x200000>>(j*6));
                  return(1);
                }

                break;

              case 6:
                if (((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig) & (0x100000>>(j*6))) > 0) {
                  //LOG_E(PHY,"SubframeConfig<<6(%x),MBSFN_FDD_SF1(%x)\n",frame_parms->MBSFN_config[i].mbsfn_SubframeConfig,0x100000>>(j*6));
                  return(1);
                }

                break;

              case 7:
                if (((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig) & (0x80000>>(j*6))) > 0) {
                  //LOG_E(PHY,"SubframeConfig<<6(%x),MBSFN_FDD_SF1(%x)\n",frame_parms->MBSFN_config[i].mbsfn_SubframeConfig,0x80000>>(j*6));
                  return(1);
                }

                break;

              case 8:
                if (((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig) & (0x40000>>(j*6))) > 0) {
                  //LOG_E(PHY,"SubframeConfig<<6(%x),MBSFN_FDD_SF1(%x)\n",frame_parms->MBSFN_config[i].mbsfn_SubframeConfig,0x40000>>(j*6));
                  return(1);
                }

                break;
            }
          } else {
            switch (subframe) {
              case 3:
                if (((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig>>j*6) & MBSFN_TDD_SF3) > 0)
                  return(1);

                break;

              case 4:
                if (((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig>>j*6) & MBSFN_TDD_SF4) > 0)
                  return(1);

                break;

              case 7:
                if (((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig>>j*6) & MBSFN_TDD_SF7) > 0)
                  return(1);

                break;

              case 8:
                if (((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig>>j*6) & MBSFN_TDD_SF8) > 0)
                  return(1);

                break;

              case 9:
                if (((frame_parms->MBSFN_config[i].mbsfn_SubframeConfig>>j*6) & MBSFN_TDD_SF9) > 0)
                  return(1);

                break;
            }
          }
        }
      }
    }
  }

  return(0);
}
