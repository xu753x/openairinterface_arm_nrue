



#include "PHY/defs_UE.h"
#include "PHY/phy_extern_ue.h"
#include "SCHED_UE/sched_UE.h"
#ifdef DEBUG_DCI_TOOLS
#include "PHY/phy_vars.h"
#endif
#include "assertions.h"


//#define DEBUG_HARQ


#include "LAYER2/MAC/mac.h"

//#define DEBUG_DCI


#include "../LTE_TRANSPORT/dci_tools_common_extern.h"
#include "../LTE_TRANSPORT/transport_proto.h"
#include "transport_proto_ue.h"
#include "../LTE_TRANSPORT/transport_common_proto.h"
#include "SCHED/sched_common.h"

/*
#undef LOG_D
#define LOG_D(A,B...) printf(B)
#undef LOG_I
#define LOG_I(A,B...) printf(B)
*/

extern uint16_t beta_cqi[16];
extern uint16_t beta_ri[16];
extern uint16_t beta_ack[16];

void extract_dci1A_info(uint8_t N_RB_DL, lte_frame_type_t frame_type, void *dci_pdu, DCI_INFO_EXTRACTED_t *pdci_info_extarcted)
{
    uint8_t harq_pid=0;
    uint32_t rballoc=0;
    uint8_t vrb_type=0;
    uint8_t mcs=0;
    uint8_t rv=0;
    uint8_t ndi=0;
    uint8_t TPC=0;

    uint8_t dai=0;

    switch (N_RB_DL) {
    case 6:
        if (frame_type == TDD) {
            vrb_type = ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->vrb_type;
            mcs      = ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->mcs;
            rballoc  = ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->rballoc;
            rv       = ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->rv;
            ndi      = ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->ndi;
            TPC      = ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->TPC;
            harq_pid = ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->harq_pid;
            dai      = ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->dai;
            //  printf("TDD 1A: mcs %d, rballoc %x,rv %d, TPC %d\n",mcs,rballoc,rv,TPC);
        } else {
            vrb_type = ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->vrb_type;
            mcs      = ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->mcs;
            rballoc  = ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->rballoc;
            rv       = ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->rv;
            ndi      = ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->ndi;
            TPC      = ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->TPC;
            harq_pid  = ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->harq_pid;
            //printf("FDD 1A: mcs %d, rballoc %x,rv %d, TPC %d\n",mcs,rballoc,rv,TPC);
        }
        break;

    case 25:

        if (frame_type == TDD) {
            vrb_type = ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->vrb_type;
            mcs      = ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->mcs;
            rballoc  = ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->rballoc;
            rv       = ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->rv;
            ndi      = ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->ndi;
            TPC      = ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->TPC;
            harq_pid = ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->harq_pid;
            dai      = ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->dai;
            //printf("TDD 1A: mcs %d, rballoc %x,rv %d, TPC %d\n",mcs,rballoc,rv,TPC);
        } else {
            vrb_type = ((DCI1A_5MHz_FDD_t *)dci_pdu)->vrb_type;
            mcs      = ((DCI1A_5MHz_FDD_t *)dci_pdu)->mcs;
            rballoc  = ((DCI1A_5MHz_FDD_t *)dci_pdu)->rballoc;
            rv       = ((DCI1A_5MHz_FDD_t *)dci_pdu)->rv;
            ndi      = ((DCI1A_5MHz_FDD_t *)dci_pdu)->ndi;
            TPC      = ((DCI1A_5MHz_FDD_t *)dci_pdu)->TPC;
            harq_pid  = ((DCI1A_5MHz_FDD_t *)dci_pdu)->harq_pid;
            //printf("FDD 1A: mcs %d, rballoc %x,rv %d, TPC %d\n",mcs,rballoc,rv,TPC);
        }

        break;

    case 50:
        if (frame_type == TDD) {
            vrb_type = ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->vrb_type;
            mcs      = ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->mcs;
            rballoc  = ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->rballoc;
            rv       = ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->rv;
            ndi      = ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->ndi;
            TPC      = ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->TPC;
            harq_pid = ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->harq_pid;
            dai      = ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->dai;
            //  printf("TDD 1A: mcs %d, rballoc %x,rv %d, TPC %d\n",mcs,rballoc,rv,TPC);
        } else {
            vrb_type = ((DCI1A_10MHz_FDD_t *)dci_pdu)->vrb_type;
            mcs      = ((DCI1A_10MHz_FDD_t *)dci_pdu)->mcs;
            rballoc  = ((DCI1A_10MHz_FDD_t *)dci_pdu)->rballoc;
            rv       = ((DCI1A_10MHz_FDD_t *)dci_pdu)->rv;
            ndi      = ((DCI1A_10MHz_FDD_t *)dci_pdu)->ndi;
            TPC      = ((DCI1A_10MHz_FDD_t *)dci_pdu)->TPC;
            harq_pid  = ((DCI1A_10MHz_FDD_t *)dci_pdu)->harq_pid;
            //printf("FDD 1A: mcs %d, vrb_type %d, rballoc %x,ndi %d, rv %d, TPC %d\n",mcs,vrb_type,rballoc,ndi,rv,TPC);
        }
        break;

    case 100:
        if (frame_type == TDD) {
            vrb_type = ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->vrb_type;
            mcs      = ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->mcs;
            rballoc  = ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->rballoc;
            rv       = ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->rv;
            ndi      = ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->ndi;
            TPC      = ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->TPC;
            harq_pid = ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->harq_pid;
            dai      = ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->dai;
            //  printf("TDD 1A: mcs %d, rballoc %x,rv %d, TPC %d\n",mcs,rballoc,rv,TPC);
        } else {
            vrb_type = ((DCI1A_20MHz_FDD_t *)dci_pdu)->vrb_type;
            mcs      = ((DCI1A_20MHz_FDD_t *)dci_pdu)->mcs;
            rballoc  = ((DCI1A_20MHz_FDD_t *)dci_pdu)->rballoc;
            rv       = ((DCI1A_20MHz_FDD_t *)dci_pdu)->rv;
            ndi      = ((DCI1A_20MHz_FDD_t *)dci_pdu)->ndi;
            TPC      = ((DCI1A_20MHz_FDD_t *)dci_pdu)->TPC;
            harq_pid = ((DCI1A_20MHz_FDD_t *)dci_pdu)->harq_pid;
            //printf("FDD 1A: mcs %d, rballoc %x,rv %d, TPC %d\n",mcs,rballoc,rv,TPC);
        }
        break;
    }

    pdci_info_extarcted->vrb_type = vrb_type;
    pdci_info_extarcted->mcs1     = mcs;
    pdci_info_extarcted->rballoc  = rballoc;
    pdci_info_extarcted->rv1      = rv;
    pdci_info_extarcted->ndi1     = ndi;
    pdci_info_extarcted->TPC      = TPC;
    pdci_info_extarcted->harq_pid = harq_pid;
    pdci_info_extarcted->dai      = dai;
}

void extract_dci1C_info(uint8_t N_RB_DL, lte_frame_type_t frame_type, void *dci_pdu, DCI_INFO_EXTRACTED_t *pdci_info_extarcted)
{

    uint32_t rballoc=0;
    uint8_t mcs=0;
    uint8_t Ngap=0;

    switch (N_RB_DL) {
        case 6:
          mcs             = ((DCI1C_1_5MHz_t *)dci_pdu)->mcs;
          rballoc         = conv_1C_RIV(((DCI1C_1_5MHz_t *)dci_pdu)->rballoc, 6);
          break;

        case 25:
          mcs             = ((DCI1C_5MHz_t *)dci_pdu)->mcs;
          rballoc         = conv_1C_RIV(((DCI1C_5MHz_t *)dci_pdu)->rballoc, 25);
          break;

        case 50:
          mcs             = ((DCI1C_10MHz_t *)dci_pdu)->mcs;
          rballoc         = conv_1C_RIV(((DCI1C_10MHz_t *)dci_pdu)->rballoc, 50);
          Ngap            = ((DCI1C_10MHz_t *)dci_pdu)->Ngap;
          break;

        case 100:
          mcs             = ((DCI1C_20MHz_t *)dci_pdu)->mcs;
          rballoc         = conv_1C_RIV(((DCI1C_20MHz_t *)dci_pdu)->rballoc, 100);
          Ngap            = ((DCI1C_20MHz_t *)dci_pdu)->Ngap;
          break;

        default:
          AssertFatal(0,"Format 1C: Unknown N_RB_DL %d\n",N_RB_DL);
          break;
        }

    pdci_info_extarcted->mcs1     = mcs;
    pdci_info_extarcted->rballoc  = rballoc;
    pdci_info_extarcted->Ngap     = Ngap;
}

void extract_dci1_info(uint8_t N_RB_DL, lte_frame_type_t frame_type, void *dci_pdu, DCI_INFO_EXTRACTED_t *pdci_info_extarcted)
{

    uint32_t rballoc=0;
    uint8_t mcs=0;
    uint8_t rah=0;
    uint8_t rv=0;
    uint8_t TPC=0;
    uint8_t ndi=0;
    uint8_t harq_pid=0;

    switch (N_RB_DL) {
    case 6:
        if (frame_type == TDD) {
            mcs       = ((DCI1_1_5MHz_TDD_t *)dci_pdu)->mcs;
            rballoc   = ((DCI1_1_5MHz_TDD_t *)dci_pdu)->rballoc;
            rah       = ((DCI1_1_5MHz_TDD_t *)dci_pdu)->rah;
            rv        = ((DCI1_1_5MHz_TDD_t *)dci_pdu)->rv;
            TPC       = ((DCI1_1_5MHz_TDD_t *)dci_pdu)->TPC;
            ndi       = ((DCI1_1_5MHz_TDD_t *)dci_pdu)->ndi;
            harq_pid  = ((DCI1_1_5MHz_TDD_t *)dci_pdu)->harq_pid;
        } else {
            mcs      = ((DCI1_1_5MHz_FDD_t *)dci_pdu)->mcs;
            rah      = ((DCI1_1_5MHz_FDD_t *)dci_pdu)->rah;
            rballoc  = ((DCI1_1_5MHz_FDD_t *)dci_pdu)->rballoc;
            rv       = ((DCI1_1_5MHz_FDD_t *)dci_pdu)->rv;
            TPC       = ((DCI1_1_5MHz_FDD_t *)dci_pdu)->TPC;
            ndi      = ((DCI1_1_5MHz_FDD_t *)dci_pdu)->ndi;
            harq_pid = ((DCI1_1_5MHz_FDD_t *)dci_pdu)->harq_pid;
        }

        break;

    case 25:
        if (frame_type == TDD) {
            mcs       = ((DCI1_5MHz_TDD_t *)dci_pdu)->mcs;
            rballoc   = ((DCI1_5MHz_TDD_t *)dci_pdu)->rballoc;
            rah       = ((DCI1_5MHz_TDD_t *)dci_pdu)->rah;
            rv        = ((DCI1_5MHz_TDD_t *)dci_pdu)->rv;
            TPC       = ((DCI1_5MHz_TDD_t *)dci_pdu)->TPC;
            ndi       = ((DCI1_5MHz_TDD_t *)dci_pdu)->ndi;
            harq_pid  = ((DCI1_5MHz_TDD_t *)dci_pdu)->harq_pid;
        } else {
            mcs      = ((DCI1_5MHz_FDD_t *)dci_pdu)->mcs;
            rah      = ((DCI1_5MHz_FDD_t *)dci_pdu)->rah;
            rballoc  = ((DCI1_5MHz_FDD_t *)dci_pdu)->rballoc;
            rv       = ((DCI1_5MHz_FDD_t *)dci_pdu)->rv;
            TPC      = ((DCI1_5MHz_FDD_t *)dci_pdu)->TPC;
            ndi      = ((DCI1_5MHz_FDD_t *)dci_pdu)->ndi;
            harq_pid = ((DCI1_5MHz_FDD_t *)dci_pdu)->harq_pid;
        }

        break;

    case 50:
        if (frame_type == TDD) {
            mcs       = ((DCI1_10MHz_TDD_t *)dci_pdu)->mcs;
            rballoc   = ((DCI1_10MHz_TDD_t *)dci_pdu)->rballoc;
            rah       = ((DCI1_10MHz_TDD_t *)dci_pdu)->rah;
            rv        = ((DCI1_10MHz_TDD_t *)dci_pdu)->rv;
            TPC       = ((DCI1_10MHz_TDD_t *)dci_pdu)->TPC;
            ndi       = ((DCI1_10MHz_TDD_t *)dci_pdu)->ndi;
            harq_pid  = ((DCI1_10MHz_TDD_t *)dci_pdu)->harq_pid;
        } else {
            mcs      = ((DCI1_10MHz_FDD_t *)dci_pdu)->mcs;
            rah      = ((DCI1_10MHz_FDD_t *)dci_pdu)->rah;
            rballoc  = ((DCI1_10MHz_FDD_t *)dci_pdu)->rballoc;
            rv       = ((DCI1_10MHz_FDD_t *)dci_pdu)->rv;
            TPC      = ((DCI1_10MHz_FDD_t *)dci_pdu)->TPC;
            ndi      = ((DCI1_10MHz_FDD_t *)dci_pdu)->ndi;
            harq_pid = ((DCI1_10MHz_FDD_t *)dci_pdu)->harq_pid;
        }

        break;

    case 100:
        if (frame_type == TDD) {
            mcs       = ((DCI1_20MHz_TDD_t *)dci_pdu)->mcs;
            rballoc   = ((DCI1_20MHz_TDD_t *)dci_pdu)->rballoc;
            rah       = ((DCI1_20MHz_TDD_t *)dci_pdu)->rah;
            rv        = ((DCI1_20MHz_TDD_t *)dci_pdu)->rv;
            TPC        = ((DCI1_20MHz_TDD_t *)dci_pdu)->TPC;
            ndi       = ((DCI1_20MHz_TDD_t *)dci_pdu)->ndi;
            harq_pid  = ((DCI1_20MHz_TDD_t *)dci_pdu)->harq_pid;
        } else {
            mcs      = ((DCI1_20MHz_FDD_t *)dci_pdu)->mcs;
            rah      = ((DCI1_20MHz_FDD_t *)dci_pdu)->rah;
            rballoc  = ((DCI1_20MHz_FDD_t *)dci_pdu)->rballoc;
            rv       = ((DCI1_20MHz_FDD_t *)dci_pdu)->rv;
            TPC      = ((DCI1_20MHz_FDD_t *)dci_pdu)->TPC;
            ndi      = ((DCI1_20MHz_FDD_t *)dci_pdu)->ndi;
            harq_pid = ((DCI1_20MHz_FDD_t *)dci_pdu)->harq_pid;
        }

        break;
    }

    pdci_info_extarcted->mcs1     = mcs;
    pdci_info_extarcted->rah      = rah;
    pdci_info_extarcted->rballoc  = rballoc;
    pdci_info_extarcted->rv1      = rv;
    pdci_info_extarcted->TPC      = TPC;
    pdci_info_extarcted->ndi1     = ndi;
    pdci_info_extarcted->harq_pid = harq_pid;

}

void extract_dci2_info(uint8_t N_RB_DL, lte_frame_type_t frame_type, uint8_t nb_antenna_ports_eNB, void *dci_pdu, DCI_INFO_EXTRACTED_t *pdci_info_extarcted)
{

    uint32_t rballoc=0;
    uint8_t rah=0;
    uint8_t mcs1=0;
    uint8_t mcs2=0;
    uint8_t rv1=0;
    uint8_t rv2=0;
    uint8_t ndi1=0;
    uint8_t ndi2=0;
    uint8_t tbswap=0;
    uint8_t tpmi=0;
    uint8_t harq_pid=0;
    uint8_t TPC=0;

    switch (N_RB_DL) {

    case 6:
        if (nb_antenna_ports_eNB == 2) {
            if (frame_type == TDD) {
                rah       = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->rah;
                mcs1      = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->mcs1;
                mcs2      = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->mcs2;
                rballoc   = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->rballoc;
                rv1       = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->rv1;
                rv2       = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->rv2;
                harq_pid  = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->harq_pid;
                tbswap    = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->tb_swap;
                tpmi      = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->tpmi;
                TPC       = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->TPC;
                ndi1      = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->ndi1;
                ndi2      = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->ndi2;
            } else {
                rah       = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->rah;
                mcs1      = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->mcs1;
                mcs2      = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->mcs2;
                rballoc   = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->rballoc;
                rv1       = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->rv1;
                rv2       = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->rv2;
                harq_pid  = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->harq_pid;
                tbswap    = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->tb_swap;
                tpmi      = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->tpmi;
                TPC       = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->TPC;
                ndi1      = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->ndi1;
                ndi2      = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->ndi2;
            }
        } else if (nb_antenna_ports_eNB == 4) {
            if (frame_type == TDD) {
                rah       = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->rah;
                mcs1      = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->mcs1;
                mcs2      = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->mcs2;
                rballoc   = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->rballoc;
                rv1       = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->rv1;
                rv2       = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->rv2;
                harq_pid  = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->harq_pid;
                tbswap    = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->tb_swap;
                tpmi      = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->tpmi;
                TPC       = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->TPC;
                ndi1      = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->ndi1;
                ndi2      = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->ndi2;
            } else {
                rah       = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->rah;
                mcs1      = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->mcs1;
                mcs2      = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->mcs2;
                rballoc   = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->rballoc;
                rv1       = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->rv1;
                rv2       = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->rv2;
                harq_pid  = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->harq_pid;
                tbswap    = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->tb_swap;
                tpmi      = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->tpmi;
                TPC       = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->TPC;
                ndi1      = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->ndi1;
                ndi2      = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->ndi2;
            }
        } else {
            LOG_E(PHY,"UE: Format2 DCI: unsupported number of TX antennas %d\n",nb_antenna_ports_eNB);
        }

        break;

    case 25:
        if (nb_antenna_ports_eNB == 2) {
            if (frame_type == TDD) {
                rah       = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->rah;
                mcs1      = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->mcs1;
                mcs2      = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->mcs2;
                rballoc   = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->rballoc;
                rv1       = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->rv1;
                rv2       = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->rv2;
                harq_pid  = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->harq_pid;
                tbswap    = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->tb_swap;
                tpmi      = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->tpmi;
                TPC       = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->TPC;
                ndi1      = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->ndi1;
                ndi2      = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->ndi2;
            } else {
                rah       = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->rah;
                mcs1      = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->mcs1;
                mcs2      = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->mcs2;
                rballoc   = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->rballoc;
                rv1       = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->rv1;
                rv2       = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->rv2;
                harq_pid  = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->harq_pid;
                tbswap    = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->tb_swap;
                tpmi      = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->tpmi;
                TPC       = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->TPC;
                ndi1      = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->ndi1;
                ndi2      = ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->ndi2;
            }
        } else if (nb_antenna_ports_eNB == 4) {
            if (frame_type == TDD) {
                rah       = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->rah;
                mcs1      = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->mcs1;
                mcs2      = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->mcs2;
                rballoc   = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->rballoc;
                rv1       = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->rv1;
                rv2       = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->rv2;
                harq_pid  = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->harq_pid;
                tbswap    = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->tb_swap;
                tpmi      = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->tpmi;
                TPC       = ((DCI2_5MHz_4A_TDD_t *)dci_pdu)->TPC;
                ndi1      = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->ndi1;
                ndi2      = ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->ndi2;
            } else {
                rah       = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->rah;
                mcs1      = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->mcs1;
                mcs2      = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->mcs2;
                rballoc   = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->rballoc;
                rv1       = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->rv1;
                rv2       = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->rv2;
                harq_pid  = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->harq_pid;
                tbswap    = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->tb_swap;
                tpmi      = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->tpmi;
                TPC       = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->TPC;
                ndi1      = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->ndi1;
                ndi2      = ((DCI2_5MHz_4A_FDD_t *)dci_pdu)->ndi2;
            }
        } else {
            LOG_E(PHY,"UE: Format2 DCI: unsupported number of TX antennas %d\n",nb_antenna_ports_eNB);
        }

        break;

    case 50:
        if (nb_antenna_ports_eNB == 2) {
            if (frame_type == TDD) {
                rah       = ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->rah;
                mcs1      = ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->mcs1;
                mcs2      = ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->mcs2;
                rballoc   = ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->rballoc;
                rv1       = ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->rv1;
                rv2       = ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->rv2;
                harq_pid  = ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->harq_pid;
                tbswap    = ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->tb_swap;
                tpmi      = ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->tpmi;
                TPC       = ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->TPC;
                ndi1      = ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->ndi1;
                ndi2      = ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->ndi2;
            } else {
                rah       = ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->rah;
                mcs1      = ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->mcs1;
                mcs2      = ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->mcs2;
                rballoc   = ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->rballoc;
                rv1       = ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->rv1;
                rv2       = ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->rv2;
                harq_pid  = ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->harq_pid;
                tbswap    = ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->tb_swap;
                tpmi      = ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->tpmi;
                TPC       = ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->TPC;
                ndi1      = ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->ndi1;
                ndi2      = ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->ndi2;
            }
        } else if (nb_antenna_ports_eNB == 4) {
            if (frame_type == TDD) {
                rah       = ((DCI2_10MHz_4A_TDD_t *)dci_pdu)->rah;
                mcs1      = ((DCI2_10MHz_4A_TDD_t *)dci_pdu)->mcs1;
                mcs2      = ((DCI2_10MHz_4A_TDD_t *)dci_pdu)->mcs2;
                rballoc   = ((DCI2_10MHz_4A_TDD_t *)dci_pdu)->rballoc;
                rv1       = ((DCI2_10MHz_4A_TDD_t *)dci_pdu)->rv1;
                rv2       = ((DCI2_10MHz_4A_TDD_t *)dci_pdu)->rv2;
                harq_pid  = ((DCI2_10MHz_4A_TDD_t *)dci_pdu)->harq_pid;
                tbswap    = ((DCI2_10MHz_4A_TDD_t *)dci_pdu)->tb_swap;
                tpmi      = ((DCI2_10MHz_4A_TDD_t *)dci_pdu)->tpmi;
                TPC       = ((DCI2_10MHz_4A_TDD_t *)dci_pdu)->TPC;
                ndi1      = ((DCI2_10MHz_4A_TDD_t *)dci_pdu)->ndi1;
                ndi2      = ((DCI2_10MHz_4A_TDD_t *)dci_pdu)->ndi2;
            } else {
                rah       = ((DCI2_10MHz_4A_FDD_t *)dci_pdu)->rah;
                mcs1      = ((DCI2_10MHz_4A_FDD_t *)dci_pdu)->mcs1;
                mcs2      = ((DCI2_10MHz_4A_FDD_t *)dci_pdu)->mcs2;
                rballoc   = ((DCI2_10MHz_4A_FDD_t *)dci_pdu)->rballoc;
                rv1       = ((DCI2_10MHz_4A_FDD_t *)dci_pdu)->rv1;
                rv2       = ((DCI2_10MHz_4A_FDD_t *)dci_pdu)->rv2;
                harq_pid  = ((DCI2_10MHz_4A_FDD_t *)dci_pdu)->harq_pid;
                tbswap    = ((DCI2_10MHz_4A_FDD_t *)dci_pdu)->tb_swap;
                tpmi      = ((DCI2_10MHz_4A_FDD_t *)dci_pdu)->tpmi;
                TPC       = ((DCI2_10MHz_4A_FDD_t *)dci_pdu)->TPC;
                ndi1      = ((DCI2_10MHz_4A_FDD_t *)dci_pdu)->ndi1;
                ndi2      = ((DCI2_10MHz_4A_FDD_t *)dci_pdu)->ndi2;
            }
        } else {
            LOG_E(PHY,"UE: Format2A DCI: unsupported number of TX antennas %d\n",nb_antenna_ports_eNB);
        }

        break;

    case 100:
        if (nb_antenna_ports_eNB == 2) {
            if (frame_type == TDD) {
                rah       = ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->rah;
                mcs1      = ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->mcs1;
                mcs2      = ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->mcs2;
                rballoc   = ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->rballoc;
                rv1       = ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->rv1;
                rv2       = ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->rv2;
                harq_pid  = ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->harq_pid;
                tbswap    = ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->tb_swap;
                tpmi      = ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->tpmi;
                TPC       = ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->TPC;
                ndi1      = ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->ndi1;
                ndi2      = ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->ndi2;
            } else {
                rah       = ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->rah;
                mcs1      = ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->mcs1;
                mcs2      = ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->mcs2;
                rballoc   = ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->rballoc;
                rv1       = ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->rv1;
                rv2       = ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->rv2;
                harq_pid  = ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->harq_pid;
                tbswap    = ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->tb_swap;
                tpmi      = ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->tpmi;
                TPC       = ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->TPC;
                ndi1      = ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->ndi1;
                ndi2      = ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->ndi2;
            }
        } else if (nb_antenna_ports_eNB == 4) {
            if (frame_type == TDD) {
                rah       = ((DCI2_20MHz_4A_TDD_t *)dci_pdu)->rah;
                mcs1      = ((DCI2_20MHz_4A_TDD_t *)dci_pdu)->mcs1;
                mcs2      = ((DCI2_20MHz_4A_TDD_t *)dci_pdu)->mcs2;
                rballoc   = ((DCI2_20MHz_4A_TDD_t *)dci_pdu)->rballoc;
                rv1       = ((DCI2_20MHz_4A_TDD_t *)dci_pdu)->rv1;
                rv2       = ((DCI2_20MHz_4A_TDD_t *)dci_pdu)->rv2;
                harq_pid  = ((DCI2_20MHz_4A_TDD_t *)dci_pdu)->harq_pid;
                tbswap    = ((DCI2_20MHz_4A_TDD_t *)dci_pdu)->tb_swap;
                tpmi      = ((DCI2_20MHz_4A_TDD_t *)dci_pdu)->tpmi;
                TPC       = ((DCI2_20MHz_4A_TDD_t *)dci_pdu)->TPC;
                ndi1      = ((DCI2_20MHz_4A_TDD_t *)dci_pdu)->ndi1;
                ndi2      = ((DCI2_20MHz_4A_TDD_t *)dci_pdu)->ndi2;
            } else {
                rah       = ((DCI2_20MHz_4A_FDD_t *)dci_pdu)->rah;
                mcs1      = ((DCI2_20MHz_4A_FDD_t *)dci_pdu)->mcs1;
                mcs2      = ((DCI2_20MHz_4A_FDD_t *)dci_pdu)->mcs2;
                rballoc   = ((DCI2_20MHz_4A_FDD_t *)dci_pdu)->rballoc;
                rv1       = ((DCI2_20MHz_4A_FDD_t *)dci_pdu)->rv1;
                rv2       = ((DCI2_20MHz_4A_FDD_t *)dci_pdu)->rv2;
                harq_pid  = ((DCI2_20MHz_4A_FDD_t *)dci_pdu)->harq_pid;
                tbswap    = ((DCI2_20MHz_4A_FDD_t *)dci_pdu)->tb_swap;
                tpmi      = ((DCI2_20MHz_4A_FDD_t *)dci_pdu)->tpmi;
                TPC       = ((DCI2_20MHz_4A_FDD_t *)dci_pdu)->TPC;
                ndi1      = ((DCI2_20MHz_4A_FDD_t *)dci_pdu)->ndi1;
                ndi2      = ((DCI2_20MHz_4A_FDD_t *)dci_pdu)->ndi2;
            }
        } else {
            LOG_E(PHY,"UE: Format2A DCI: unsupported number of TX antennas %d\n",nb_antenna_ports_eNB);
        }

        break;
    }

    pdci_info_extarcted->rah      = rah;
    pdci_info_extarcted->mcs1     = mcs1;
    pdci_info_extarcted->mcs2     = mcs2;
    pdci_info_extarcted->rv1      = rv1;
    pdci_info_extarcted->rv2      = rv2;
    pdci_info_extarcted->harq_pid = harq_pid;
    pdci_info_extarcted->rballoc  = rballoc;
    pdci_info_extarcted->tb_swap  = tbswap;
    pdci_info_extarcted->tpmi     = tpmi;
    pdci_info_extarcted->TPC      = TPC;
    pdci_info_extarcted->ndi1     = ndi1;
    pdci_info_extarcted->ndi2     = ndi2;

}

void extract_dci2A_info(uint8_t N_RB_DL, lte_frame_type_t frame_type, uint8_t nb_antenna_ports_eNB, void *dci_pdu, DCI_INFO_EXTRACTED_t *pdci_info_extarcted)
{

    uint32_t rballoc=0;
    uint8_t rah=0;
    uint8_t mcs1=0;
    uint8_t mcs2=0;
    uint8_t rv1=0;
    uint8_t rv2=0;
    uint8_t ndi1=0;
    uint8_t ndi2=0;
    uint8_t tbswap=0;
    uint8_t tpmi=0;
    uint8_t harq_pid=0;
    uint8_t TPC=0;

    AssertFatal( (nb_antenna_ports_eNB == 2) || (nb_antenna_ports_eNB == 4), "unsupported nb_antenna_ports_eNB %d\n", nb_antenna_ports_eNB);
    switch (N_RB_DL) {

    case 6:
      if (nb_antenna_ports_eNB == 2) {
        if (frame_type == TDD) {
          mcs1      = ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->mcs1;
          mcs2      = ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->mcs2;
          rballoc   = ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->rballoc;
          rv1       = ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->rv1;
          rv2       = ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->rv2;
          ndi1      = ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->ndi1;
          ndi2      = ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->ndi2;
          harq_pid  = ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->harq_pid;
          tbswap    = ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->tb_swap;
          TPC       = ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->TPC;
        } else {
          mcs1      = ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->mcs1;
          mcs2      = ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->mcs2;
          rballoc   = ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->rballoc;
          rv1       = ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->rv1;
          rv2       = ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->rv2;
          ndi1      = ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->ndi1;
          ndi2      = ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->ndi2;
          harq_pid  = ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->harq_pid;
          tbswap    = ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->tb_swap;
          TPC       = ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->TPC;
        }
      } else if (nb_antenna_ports_eNB == 4) {
        if (frame_type == TDD) {
          mcs1      = ((DCI2A_1_5MHz_4A_TDD_t *)dci_pdu)->mcs1;
          mcs2      = ((DCI2A_1_5MHz_4A_TDD_t *)dci_pdu)->mcs2;
          rballoc   = ((DCI2A_1_5MHz_4A_TDD_t *)dci_pdu)->rballoc;
          rv1       = ((DCI2A_1_5MHz_4A_TDD_t *)dci_pdu)->rv1;
          rv2       = ((DCI2A_1_5MHz_4A_TDD_t *)dci_pdu)->rv2;
          ndi1      = ((DCI2A_1_5MHz_4A_TDD_t *)dci_pdu)->ndi1;
          ndi2      = ((DCI2A_1_5MHz_4A_TDD_t *)dci_pdu)->ndi2;
          harq_pid  = ((DCI2A_1_5MHz_4A_TDD_t *)dci_pdu)->harq_pid;
          tbswap    = ((DCI2A_1_5MHz_4A_TDD_t *)dci_pdu)->tb_swap;
          tpmi      = ((DCI2A_1_5MHz_4A_TDD_t *)dci_pdu)->tpmi;
          TPC       = ((DCI2A_1_5MHz_4A_TDD_t *)dci_pdu)->TPC;
        } else {
          mcs1      = ((DCI2A_1_5MHz_4A_FDD_t *)dci_pdu)->mcs1;
          mcs2      = ((DCI2A_1_5MHz_4A_FDD_t *)dci_pdu)->mcs2;
          rballoc   = ((DCI2A_1_5MHz_4A_FDD_t *)dci_pdu)->rballoc;
          rv1       = ((DCI2A_1_5MHz_4A_FDD_t *)dci_pdu)->rv1;
          rv2       = ((DCI2A_1_5MHz_4A_FDD_t *)dci_pdu)->rv2;
          ndi1      = ((DCI2A_1_5MHz_4A_FDD_t *)dci_pdu)->ndi1;
          ndi2      = ((DCI2A_1_5MHz_4A_FDD_t *)dci_pdu)->ndi2;
          harq_pid  = ((DCI2A_1_5MHz_4A_FDD_t *)dci_pdu)->harq_pid;
          tbswap    = ((DCI2A_1_5MHz_4A_FDD_t *)dci_pdu)->tb_swap;
          tpmi      = ((DCI2A_1_5MHz_4A_FDD_t *)dci_pdu)->tpmi;
          TPC       = ((DCI2A_1_5MHz_4A_FDD_t *)dci_pdu)->TPC;
        }
      }

      break;

    case 25:
      if (nb_antenna_ports_eNB == 2) {
        if (frame_type == TDD) {
          mcs1      = ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->mcs1;
          mcs2      = ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->mcs2;
          rballoc   = ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->rballoc;
          rah       = ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->rah;
          rv1       = ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->rv1;
          rv2       = ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->rv2;
          ndi1      = ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->ndi1;
          ndi2      = ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->ndi2;
          harq_pid  = ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->harq_pid;
          tbswap    = ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->tb_swap;
          TPC       = ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->TPC;
        } else {
          mcs1      = ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->mcs1;
          mcs2      = ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->mcs2;
          rballoc   = ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->rballoc;
          rah       = ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->rah;
          rv1       = ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->rv1;
          rv2       = ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->rv2;
          ndi1      = ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->ndi1;
          ndi2      = ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->ndi2;
          harq_pid  = ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->harq_pid;
          tbswap    = ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->tb_swap;
          TPC       = ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->TPC;
        }
      } else if (nb_antenna_ports_eNB == 4) {
        if (frame_type == TDD) {
          mcs1      = ((DCI2A_5MHz_4A_TDD_t *)dci_pdu)->mcs1;
          mcs2      = ((DCI2A_5MHz_4A_TDD_t *)dci_pdu)->mcs2;
          rballoc   = ((DCI2A_5MHz_4A_TDD_t *)dci_pdu)->rballoc;
          rah       = ((DCI2A_5MHz_4A_TDD_t *)dci_pdu)->rah;
          rv1       = ((DCI2A_5MHz_4A_TDD_t *)dci_pdu)->rv1;
          rv2       = ((DCI2A_5MHz_4A_TDD_t *)dci_pdu)->rv2;
          ndi1      = ((DCI2A_5MHz_4A_TDD_t *)dci_pdu)->ndi1;
          ndi2      = ((DCI2A_5MHz_4A_TDD_t *)dci_pdu)->ndi2;
          harq_pid  = ((DCI2A_5MHz_4A_TDD_t *)dci_pdu)->harq_pid;
          tbswap    = ((DCI2A_5MHz_4A_TDD_t *)dci_pdu)->tb_swap;
          tpmi      = ((DCI2A_5MHz_4A_TDD_t *)dci_pdu)->tpmi;
          TPC       = ((DCI2A_5MHz_4A_TDD_t *)dci_pdu)->TPC;
        } else {
          mcs1      = ((DCI2A_5MHz_4A_FDD_t *)dci_pdu)->mcs1;
          mcs2      = ((DCI2A_5MHz_4A_FDD_t *)dci_pdu)->mcs2;
          rballoc   = ((DCI2A_5MHz_4A_FDD_t *)dci_pdu)->rballoc;
          rah       = ((DCI2A_5MHz_4A_FDD_t *)dci_pdu)->rah;
          rv1       = ((DCI2A_5MHz_4A_FDD_t *)dci_pdu)->rv1;
          rv2       = ((DCI2A_5MHz_4A_FDD_t *)dci_pdu)->rv2;
          ndi1      = ((DCI2A_5MHz_4A_TDD_t *)dci_pdu)->ndi1;
          ndi2      = ((DCI2A_5MHz_4A_TDD_t *)dci_pdu)->ndi2;
          harq_pid  = ((DCI2A_5MHz_4A_FDD_t *)dci_pdu)->harq_pid;
          tbswap    = ((DCI2A_5MHz_4A_FDD_t *)dci_pdu)->tb_swap;
          tpmi      = ((DCI2A_5MHz_4A_FDD_t *)dci_pdu)->tpmi;
          TPC       = ((DCI2A_5MHz_4A_FDD_t *)dci_pdu)->TPC;
        }
      }
      break;

    case 50:
      if (nb_antenna_ports_eNB == 2) {
        if (frame_type == TDD) {
          mcs1      = ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->mcs1;
          mcs2      = ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->mcs2;
          rballoc   = ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->rballoc;
          rah       = ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->rah;
          rv1       = ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->rv1;
          rv2       = ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->rv2;
          ndi1      = ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->ndi1;
          ndi2      = ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->ndi2;
          harq_pid  = ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->harq_pid;
          tbswap    = ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->tb_swap;
          TPC       = ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->TPC;
        } else {
          mcs1      = ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->mcs1;
          mcs2      = ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->mcs2;
          rballoc   = ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->rballoc;
          rah       = ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->rah;
          rv1       = ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->rv1;
          rv2       = ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->rv2;
          ndi1      = ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->ndi1;
          ndi2      = ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->ndi2;
          harq_pid  = ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->harq_pid;
          tbswap    = ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->tb_swap;
          TPC       = ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->TPC;
        }
      } else if (nb_antenna_ports_eNB == 4) {
        if (frame_type == TDD) {
          mcs1      = ((DCI2A_10MHz_4A_TDD_t *)dci_pdu)->mcs1;
          mcs2      = ((DCI2A_10MHz_4A_TDD_t *)dci_pdu)->mcs2;
          rballoc   = ((DCI2A_10MHz_4A_TDD_t *)dci_pdu)->rballoc;
          rah       = ((DCI2A_10MHz_4A_TDD_t *)dci_pdu)->rah;
          rv1       = ((DCI2A_10MHz_4A_TDD_t *)dci_pdu)->rv1;
          rv2       = ((DCI2A_10MHz_4A_TDD_t *)dci_pdu)->rv2;
          ndi1      = ((DCI2A_10MHz_4A_TDD_t *)dci_pdu)->ndi1;
          ndi2      = ((DCI2A_10MHz_4A_TDD_t *)dci_pdu)->ndi2;
          harq_pid  = ((DCI2A_10MHz_4A_TDD_t *)dci_pdu)->harq_pid;
          tbswap    = ((DCI2A_10MHz_4A_TDD_t *)dci_pdu)->tb_swap;
          tpmi      = ((DCI2A_10MHz_4A_TDD_t *)dci_pdu)->tpmi;
          TPC       = ((DCI2A_10MHz_4A_TDD_t *)dci_pdu)->TPC;
        } else {
          mcs1      = ((DCI2A_10MHz_4A_FDD_t *)dci_pdu)->mcs1;
          mcs2      = ((DCI2A_10MHz_4A_FDD_t *)dci_pdu)->mcs2;
          rballoc   = ((DCI2A_10MHz_4A_FDD_t *)dci_pdu)->rballoc;
          rah       = ((DCI2A_10MHz_4A_FDD_t *)dci_pdu)->rah;
          rv1       = ((DCI2A_10MHz_4A_FDD_t *)dci_pdu)->rv1;
          rv2       = ((DCI2A_10MHz_4A_FDD_t *)dci_pdu)->rv2;
          ndi1      = ((DCI2A_10MHz_4A_FDD_t *)dci_pdu)->ndi1;
          ndi2      = ((DCI2A_10MHz_4A_FDD_t *)dci_pdu)->ndi2;
          harq_pid  = ((DCI2A_10MHz_4A_FDD_t *)dci_pdu)->harq_pid;
          tbswap    = ((DCI2A_10MHz_4A_FDD_t *)dci_pdu)->tb_swap;
          tpmi      = ((DCI2A_10MHz_4A_FDD_t *)dci_pdu)->tpmi;
          TPC       = ((DCI2A_10MHz_4A_FDD_t *)dci_pdu)->TPC;
        }
      }

      break;

    case 100:
      if (nb_antenna_ports_eNB == 2) {
        if (frame_type == TDD) {
          mcs1      = ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->mcs1;
          mcs2      = ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->mcs2;
          rballoc   = ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->rballoc;
          rah       = ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->rah;
          rv1       = ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->rv1;
          rv2       = ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->rv2;
          ndi1      = ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->ndi1;
          ndi2      = ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->ndi2;
          harq_pid  = ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->harq_pid;
          tbswap    = ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->tb_swap;
          TPC       = ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->TPC;
        } else {
          mcs1      = ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->mcs1;
          mcs2      = ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->mcs2;
          rballoc   = ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->rballoc;
          rah       = ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->rah;
          rv1       = ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->rv1;
          rv2       = ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->rv2;
          ndi1      = ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->ndi1;
          ndi2      = ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->ndi2;
          harq_pid  = ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->harq_pid;
          tbswap    = ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->tb_swap;
          TPC       = ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->TPC;
        }
      } else if (nb_antenna_ports_eNB == 4) {
        if (frame_type == TDD) {
          mcs1      = ((DCI2A_20MHz_4A_TDD_t *)dci_pdu)->mcs1;
          mcs2      = ((DCI2A_20MHz_4A_TDD_t *)dci_pdu)->mcs2;
          rballoc   = ((DCI2A_20MHz_4A_TDD_t *)dci_pdu)->rballoc;
          rah       = ((DCI2A_20MHz_4A_TDD_t *)dci_pdu)->rah;
          rv1       = ((DCI2A_20MHz_4A_TDD_t *)dci_pdu)->rv1;
          rv2       = ((DCI2A_20MHz_4A_TDD_t *)dci_pdu)->rv2;
          ndi1      = ((DCI2A_20MHz_4A_TDD_t *)dci_pdu)->ndi1;
          ndi2      = ((DCI2A_20MHz_4A_TDD_t *)dci_pdu)->ndi2;
          harq_pid  = ((DCI2A_20MHz_4A_TDD_t *)dci_pdu)->harq_pid;
          tbswap    = ((DCI2A_20MHz_4A_TDD_t *)dci_pdu)->tb_swap;
          tpmi      = ((DCI2A_20MHz_4A_TDD_t *)dci_pdu)->tpmi;
          TPC       = ((DCI2A_20MHz_4A_TDD_t *)dci_pdu)->TPC;
        } else {
          mcs1      = ((DCI2A_20MHz_4A_FDD_t *)dci_pdu)->mcs1;
          mcs2      = ((DCI2A_20MHz_4A_FDD_t *)dci_pdu)->mcs2;
          rballoc   = ((DCI2A_20MHz_4A_FDD_t *)dci_pdu)->rballoc;
          rah       = ((DCI2A_20MHz_4A_FDD_t *)dci_pdu)->rah;
          rv1       = ((DCI2A_20MHz_4A_FDD_t *)dci_pdu)->rv1;
          rv2       = ((DCI2A_20MHz_4A_FDD_t *)dci_pdu)->rv2;
          ndi1      = ((DCI2A_20MHz_4A_FDD_t *)dci_pdu)->ndi1;
          ndi2      = ((DCI2A_20MHz_4A_FDD_t *)dci_pdu)->ndi2;
          harq_pid  = ((DCI2A_20MHz_4A_FDD_t *)dci_pdu)->harq_pid;
          tbswap    = ((DCI2A_20MHz_4A_FDD_t *)dci_pdu)->tb_swap;
          tpmi    = ((DCI2A_20MHz_4A_FDD_t *)dci_pdu)->tpmi;
          TPC     = ((DCI2A_20MHz_4A_FDD_t *)dci_pdu)->TPC;
        }
      }

      break;
    }

    pdci_info_extarcted->mcs1     = mcs1;
    pdci_info_extarcted->mcs2     = mcs2;
    pdci_info_extarcted->rballoc  = rballoc;
    pdci_info_extarcted->rah      = rah;
    pdci_info_extarcted->rv1      = rv1;
    pdci_info_extarcted->rv2      = rv2;
    pdci_info_extarcted->ndi1     = ndi1;
    pdci_info_extarcted->ndi2     = ndi2;
    pdci_info_extarcted->harq_pid = harq_pid;
    pdci_info_extarcted->tb_swap  = tbswap;
    pdci_info_extarcted->TPC      = TPC;
    pdci_info_extarcted->tpmi     = tpmi;
}

int check_dci_format1_1a_coherency(DCI_format_t dci_format,
        uint8_t N_RB_DL,
        uint16_t rnti,
        uint16_t tc_rnti,
        uint16_t si_rnti,
        uint16_t ra_rnti,
        uint16_t p_rnti,
        uint32_t frame,
        uint8_t  subframe,
        DCI_INFO_EXTRACTED_t *pdci_info_extarcted,
        LTE_DL_UE_HARQ_t *pdlsch0_harq)
{
    uint8_t  harq_pid  = pdci_info_extarcted->harq_pid;
    uint32_t rballoc   = pdci_info_extarcted->rballoc;
    uint8_t  mcs1      = pdci_info_extarcted->mcs1;
    uint8_t  TPC       = pdci_info_extarcted->TPC;
    uint8_t  rah       = pdci_info_extarcted->rah;
#ifdef DEBUG_DCI
    uint8_t  rv1       = pdci_info_extarcted->rv1;
    uint8_t  ndi1      = pdci_info_extarcted->ndi1;
#endif

    uint8_t  NPRB    = 0;
    long long int RIV_max = 0;

#ifdef DEBUG_DCI
    LOG_I(PHY,"[DCI-FORMAT-1-1A] AbsSubframe %d.%d dci_format %d\n", frame, subframe, dci_format);
    LOG_I(PHY,"[DCI-FORMAT-1-1A] rnti       %x\n",  rnti);
    LOG_I(PHY,"[DCI-FORMAT-1-1A] harq_pid   %d\n", harq_pid);
    LOG_I(PHY,"[DCI-FORMAT-1-1A] rah        %d\n", rah);
    LOG_I(PHY,"[DCI-FORMAT-1-1A] rballoc    %x\n", rballoc);
    LOG_I(PHY,"[DCI-FORMAT-1-1A] mcs1       %d\n", mcs1);
    LOG_I(PHY,"[DCI-FORMAT-1-1A] rv1        %d\n", rv1);
    LOG_I(PHY,"[DCI-FORMAT-1-1A] ndi1       %d\n", ndi1);
    LOG_I(PHY,"[DCI-FORMAT-1-1A] TPC        %d\n", TPC);
#endif


    // I- check dci content minimum coherency
    if( ((rnti==si_rnti) || (rnti==p_rnti) || (rnti==ra_rnti)) && harq_pid > 0)
    {
        return(0);
    }

    if(harq_pid>=8)
    {
      //        LOG_I(PHY,"bad harq id \n");
        return(0);
    }

    if(dci_format == format1 && ((rnti==si_rnti) || (rnti==p_rnti) || (rnti==ra_rnti)) )
    {
      //        LOG_I(PHY,"bad dci format \n");
        return(0);
    }


    if( mcs1 > 28)
    {
        if(pdlsch0_harq->round == 0)
        {
	  //            LOG_I(PHY,"bad dci mcs + round \n");
            return(0);
        }

        if((rnti==si_rnti) || (rnti==p_rnti) || (rnti==ra_rnti))
        {
	  //            LOG_I(PHY,"bad dci mcs + rnti  \n");
            return(0);
        }
    }

    if (dci_format == format1A && ((rnti==si_rnti) || (rnti==p_rnti) || (rnti==ra_rnti)))
    {
        NPRB = (TPC&1) + 2;
        switch (N_RB_DL) {
        case 6:
            RIV_max  = RIV_max6;
            break;
        case 25:
            RIV_max  = RIV_max25;
            break;
        case 50:
            RIV_max  = RIV_max50;
            break;
        case 100:
            RIV_max  = RIV_max100;
            break;
        }
    } 
    else if (dci_format == format1A)
    {
        switch (N_RB_DL) {
        case 6:
            NPRB     = RIV2nb_rb_LUT6[rballoc];//NPRB;
            if(rah)
              RIV_max  = RIV_max6;
            else
              RIV_max  = 0x3F;
            break;
        case 25:
            NPRB     = RIV2nb_rb_LUT25[rballoc];//NPRB;
            if(rah)
              RIV_max  = RIV_max25;
            else
              RIV_max  = 0x1FFF;
            break;
        case 50:
            NPRB     = RIV2nb_rb_LUT50[rballoc];//NPRB;
            if(rah)
              RIV_max  = RIV_max50;
            else
              RIV_max  = 0x1FFFF;
            break;
        case 100:
            NPRB     = RIV2nb_rb_LUT100[rballoc];//NPRB;
            if(rah)
              RIV_max  = RIV_max100;
            else
              RIV_max  =  0x1FFFFFF;
            break;
        }
    }


    else if (dci_format == format1)
    {
        NPRB = conv_nprb(rah, rballoc, N_RB_DL);
    }


    if(dci_format == format1A && rballoc > RIV_max)
    {
      //        LOG_I(PHY,"bad dci rballoc rballoc %d  RIV_max %lld \n",rballoc, RIV_max);
        // DCI false detection
        return(0);
    }

    if(NPRB == 0)
    {
        // DCI false detection
      //        LOG_I(PHY,"bad NPRB = 0 \n");
        return(0);
    }

    // this a retransmission
    if(pdlsch0_harq->round>0)
    {
        // compare old TBS to new TBS
        if((mcs1<29) && (pdlsch0_harq->TBS != TBStable[get_I_TBS(mcs1)][NPRB-1]))
        {
            // this is an eNB issue
            // retransmisison but old and new TBS are different !!!
            // work around, consider it as a new transmission
            LOG_E(PHY,"Format1A Retransmission but TBS are different: consider it as new transmission !!! \n");
            pdlsch0_harq->round = 0;
            //return(0); // ?? to cross check
        }
    }

    return(1);
}

int check_dci_format1c_coherency(uint8_t N_RB_DL,
                                 DCI_INFO_EXTRACTED_t *pdci_info_extarcted,
                                 uint16_t rnti,
                                 uint16_t si_rnti,
                                 uint16_t ra_rnti,
                                 uint16_t p_rnti,
                                 LTE_DL_UE_HARQ_t *pdlsch0_harq)
{
    uint32_t rballoc = pdci_info_extarcted->rballoc;

    uint8_t  NPRB    = 0;
    uint32_t RIV_max = 0;

    // I- check dci content minimum coherency

    if((rnti!=si_rnti) && (rnti!=p_rnti) && (rnti!=ra_rnti))
      return(0);

    switch (N_RB_DL) {
    case 6:
      NPRB     = RIV2nb_rb_LUT6[rballoc];//NPRB;
      RIV_max  = RIV_max6;
    break;
    case 25:
      NPRB     = RIV2nb_rb_LUT25[rballoc];//NPRB;
      RIV_max  = RIV_max25;
    break;
    case 50:
      NPRB     = RIV2nb_rb_LUT50[rballoc];//NPRB;
      RIV_max  = RIV_max50;
    break;
    case 100:
      NPRB     = RIV2nb_rb_LUT100[rballoc];//NPRB;
      RIV_max  = RIV_max100;
    break;
    }

   if(rballoc > RIV_max)
   {
      // DCI false detection
      return(0);
   }

   if(NPRB == 0)
   {
      // DCI false detection
      return(0);
   }

   return(1);
}

int check_dci_format2_2a_coherency(DCI_format_t dci_format,
                                   uint8_t N_RB_DL,
                                   DCI_INFO_EXTRACTED_t *pdci_info_extarcted,
                                   uint16_t rnti,
                                   uint16_t si_rnti,
                                   uint16_t ra_rnti,
                                   uint16_t p_rnti,
                                   LTE_DL_UE_HARQ_t *pdlsch0_harq,
                                   LTE_DL_UE_HARQ_t *pdlsch1_harq)
{
    uint8_t  rah  = pdci_info_extarcted->rah;
    uint8_t  mcs1 = pdci_info_extarcted->mcs1;
    uint8_t  mcs2 = pdci_info_extarcted->mcs2;
    uint8_t  rv1  = pdci_info_extarcted->rv1;
    uint8_t  rv2  = pdci_info_extarcted->rv2;
    uint8_t  harq_pid = pdci_info_extarcted->harq_pid;
    uint32_t rballoc  = pdci_info_extarcted->rballoc;

#ifdef DEBUG_DCI
    uint8_t  ndi1     = pdci_info_extarcted->ndi1;
    uint8_t  ndi2     = pdci_info_extarcted->ndi2;
#endif

    uint8_t  NPRB    = 0;
    long long RIV_max = 0;

#ifdef DEBUG_DCI
    LOG_I(PHY, "extarcted dci - dci_format %d \n", dci_format);
    LOG_I(PHY, "extarcted dci - rnti       %d \n", rnti);
    LOG_I(PHY, "extarcted dci - rah        %d \n", rah);
    LOG_I(PHY, "extarcted dci - mcs1       %d \n", mcs1);
    LOG_I(PHY, "extarcted dci - mcs2       %d \n", mcs2);
    LOG_I(PHY, "extarcted dci - rv1        %d \n", rv1);
    LOG_I(PHY, "extarcted dci - rv2        %d \n", rv2);
    //LOG_I(PHY, "extarcted dci - ndi1       %d \n", ndi1);
    //LOG_I(PHY, "extarcted dci - ndi2       %d \n", ndi2);
    LOG_I(PHY, "extarcted dci - rballoc    %x \n", rballoc);
    LOG_I(PHY, "extarcted dci - harq pid   %d \n", harq_pid);
    LOG_I(PHY, "extarcted dci - round0     %d \n", pdlsch0_harq->round);
    LOG_I(PHY, "extarcted dci - round1     %d \n", pdlsch1_harq->round);
#endif

    // I- check dci content minimum coherency
    if(harq_pid>=8)
    {
      //        LOG_I(PHY,"bad harq pid\n");
      return(0);
    }

    if( (rnti==si_rnti) || (rnti==p_rnti) || (rnti==ra_rnti) )
    {
      //        LOG_I(PHY,"bad rnti\n");
        return(0);
    }


    if( mcs1 > 28)
    {
      if(pdlsch0_harq->round == 0)
      {
	//          LOG_I(PHY,"bad mcs1\n");
        return(0);
      }
    }

    if( mcs2 > 28)
    {
      if(pdlsch1_harq->round == 0)
      {
	//          LOG_I(PHY,"bad mcs2\n");
          return(0);
      }
    }


    if((pdlsch0_harq->round == 0) && (rv1 > 0) && (mcs1 != 0))
    {
      // DCI false detection
      //        LOG_I(PHY,"bad rv1\n");
      return(0);
    }

    if((pdlsch1_harq->round == 0) && (rv2 > 0) && (mcs2 != 0))
    {
      // DCI false detection
      //        LOG_I(PHY,"bad rv2\n");
      return(0);
    }


    switch (N_RB_DL) {
    case 6:
        if (rah == 0)
        {
            //RBG = 1;
            RIV_max = 0x3F;
        }
        else
        {
            RIV_max  = RIV_max6;
        }
        break;
    case 25:
        if (rah == 0)
        {
            //RBG = 2;
            RIV_max = 0x1FFF;
        }
        else
        {
            RIV_max  = RIV_max25;
        }
        break;
    case 50:
        if (rah == 0)
        {
            //RBG = 3;
            RIV_max = 0x1FFFF;
        }
        else
        {
            RIV_max  = RIV_max50;
        }
        break;
    case 100:
        if (rah == 0)
        {
            //RBG = 4;
            RIV_max  = 0x1FFFFFF;
        }
        else
        {
            RIV_max  = RIV_max100;
        }
        break;
    }

    NPRB = conv_nprb(rah,
                     rballoc,
                     N_RB_DL);



   if( (rballoc > RIV_max) && (rah == 1) )
   {
      // DCI false detection
     //       LOG_I(PHY,"bad rballoc %d RIV_max %lld\n", rballoc, RIV_max);
      return(0);
   }

   if(NPRB == 0)
   {
      // DCI false detection
     //       LOG_I(PHY,"bad NPRB\n");
      return(0);
   }

   return(1);
}

void compute_llr_offset(LTE_DL_FRAME_PARMS *frame_parms,
                        LTE_UE_PDCCH *pdcch_vars,
                        LTE_UE_PDSCH *pdsch_vars,
                        LTE_DL_UE_HARQ_t *dlsch0_harq,
                        uint8_t nb_rb_alloc,
                        uint8_t subframe)
{
    uint32_t pbch_pss_sss_re;
    uint32_t crs_re;
    uint32_t granted_re;
    uint32_t data_re;
    uint32_t llr_offset;
    uint8_t symbol;
    uint8_t symbol_mod;

    pdsch_vars->llr_offset[pdcch_vars->num_pdcch_symbols] = 0;

    LOG_D(PHY,"compute_llr_offset:  nb RB %d - Qm %d \n", nb_rb_alloc, dlsch0_harq->Qm);

    //dlsch0_harq->rb_alloc_even;
    //dlsch0_harq->rb_alloc_odd;

    for(symbol=pdcch_vars->num_pdcch_symbols; symbol<frame_parms->symbols_per_tti; symbol++)
    {
        symbol_mod = (symbol >= (7-frame_parms->Ncp))? (symbol-(7-frame_parms->Ncp)) : symbol;
        if((symbol_mod == 0) || symbol_mod == (4-frame_parms->Ncp))
        {
	  if (frame_parms->nb_antenna_ports_eNB == 2)
	    crs_re = 4;
	  else
	    crs_re = 2;
        }
        else
        {
            crs_re = 0;
        }

        granted_re = nb_rb_alloc * (12-crs_re);
        pbch_pss_sss_re = adjust_G2(frame_parms,dlsch0_harq->rb_alloc_even,dlsch0_harq->Qm,subframe,symbol);
        pbch_pss_sss_re = (double)pbch_pss_sss_re * ((double)(12-crs_re)/12);
        data_re = granted_re - pbch_pss_sss_re;
        llr_offset = data_re * dlsch0_harq->Qm * 2;

        pdsch_vars->llr_length[symbol]   = data_re;
        if(symbol < (frame_parms->symbols_per_tti-1))
          pdsch_vars->llr_offset[symbol+1] = pdsch_vars->llr_offset[symbol] + llr_offset;

	LOG_D(PHY,"Granted Re subframe %d / symbol %d => %d (%d RBs)\n", subframe, symbol_mod, granted_re,dlsch0_harq->nb_rb);
	LOG_D(PHY,"Pbch/PSS/SSS Re subframe %d / symbol %d => %d \n", subframe, symbol_mod, pbch_pss_sss_re);
	LOG_D(PHY,"CRS Re Per PRB subframe %d / symbol %d => %d \n", subframe, symbol_mod, crs_re);
	LOG_D(PHY,"Data Re subframe %d / symbol %d => %d \n", subframe, symbol_mod, data_re);



        LOG_D(PHY,"Data Re subframe %d-symbol %d => llr length %d, llr offset %d \n", subframe, symbol,
              pdsch_vars->llr_length[symbol], pdsch_vars->llr_offset[symbol]);
    }
}
void prepare_dl_decoding_format1_1A(DCI_format_t dci_format,
                                    uint8_t N_RB_DL,
                                    DCI_INFO_EXTRACTED_t *pdci_info_extarcted,
                                    LTE_DL_FRAME_PARMS *frame_parms,
                                    LTE_UE_PDCCH *pdcch_vars,
                                    LTE_UE_PDSCH *pdsch_vars,
                                    uint8_t  subframe,
                                    uint16_t rnti,
									uint16_t tc_rnti,
                                    uint16_t si_rnti,
                                    uint16_t ra_rnti,
                                    uint16_t p_rnti,
                                    LTE_DL_UE_HARQ_t *pdlsch0_harq,
                                    LTE_UE_DLSCH_t *pdlsch0)
{

    uint8_t  harq_pid  = pdci_info_extarcted->harq_pid;
    uint8_t  vrb_type  = pdci_info_extarcted->vrb_type;
    uint32_t rballoc   = pdci_info_extarcted->rballoc;
    uint8_t  mcs1      = pdci_info_extarcted->mcs1;
    uint8_t  rv1       = pdci_info_extarcted->rv1;
    uint8_t  ndi1      = pdci_info_extarcted->ndi1;
    uint8_t  TPC       = pdci_info_extarcted->TPC;
    uint8_t  rah       = pdci_info_extarcted->rah;
    uint8_t  dai       = pdci_info_extarcted->dai;


    uint8_t  NPRB      = 0;
    uint8_t  NPRB4TBS  = 0;

    if(dci_format == format1A)
    {
      switch (N_RB_DL) {
      case 6:
	NPRB     = RIV2nb_rb_LUT6[rballoc];
	break;
      case 25:
	NPRB     = RIV2nb_rb_LUT25[rballoc];
	break;
      case 50:
	NPRB     = RIV2nb_rb_LUT50[rballoc];
	break;
      case 100:
	NPRB     = RIV2nb_rb_LUT100[rballoc];
	break;
      }
      if ((rnti==si_rnti) || (rnti==p_rnti) || (rnti==ra_rnti))
        {
	  NPRB4TBS = (TPC&1) + 2;
        }
      else
        {
	  NPRB4TBS = NPRB;
	  /*
            switch (N_RB_DL) {
            case 6:
                NPRB     = RIV2nb_rb_LUT6[rballoc];//NPRB;
                break;
            case 25:
                NPRB     = RIV2nb_rb_LUT25[rballoc];//NPRB;
                break;
            case 50:
                NPRB     = RIV2nb_rb_LUT50[rballoc];//NPRB;
                break;
            case 100:
                NPRB     = RIV2nb_rb_LUT100[rballoc];//NPRB;
                break;
            }

	  */
        }
    }
    else // format1
    {
        NPRB = conv_nprb(rah, rballoc, N_RB_DL);
	NPRB4TBS=NPRB;
    }

    pdlsch0->current_harq_pid = harq_pid;
    pdlsch0->active           = 1;
    pdlsch0->rnti             = rnti;
    if(dci_format == format1A)
        pdlsch0->harq_ack[subframe].vDAI_DL = dai+1;

    if ((rnti==si_rnti) || (rnti==p_rnti) || (rnti==ra_rnti))
    {
        pdlsch0_harq->round    = 0;
        pdlsch0_harq->status   = ACTIVE;
    }
    else //CRNTI
    {
    	if (rnti == tc_rnti) {
			//fix for standalone Contention Resolution Id
	  pdlsch0_harq->DCINdi = (uint8_t)-1;
	  LOG_D(PHY,"UE (%x/%d): Format1A DCI: C-RNTI is temporary. Set NDI = %d and to be ignored\n",
		rnti,harq_pid,pdlsch0_harq->DCINdi);
    	}

        // NDI has been toggled or this is the first transmission
        if ((ndi1!=pdlsch0_harq->DCINdi) || (pdlsch0_harq->first_tx==1))
        {
            pdlsch0_harq->round    = 0;
            pdlsch0_harq->first_tx = 0;
            pdlsch0_harq->status   = ACTIVE;

        }else if (rv1  != 0 )
            //NDI has not been toggled but rv was increased by eNB: retransmission
        {
            if (pdlsch0_harq->status == SCH_IDLE)
                //packet was actually decoded in previous transmission (ACK was missed by eNB)
                //However, the round is not a good check as it might have been decoded in a retransmission prior to this one.
            {
	      //                LOG_D(PHY,"skip pdsch decoding and report ack\n");
                // skip pdsch decoding and report ack
                //pdlsch0_harq->status   = SCH_IDLE;
                pdlsch0->active       = 0;
                pdlsch0->harq_ack[subframe].ack = 1;
                pdlsch0->harq_ack[subframe].harq_id = harq_pid;
                pdlsch0->harq_ack[subframe].send_harq_status = 1;

                //pdlsch0_harq->first_tx = 0;
            }
            else  //normal retransmission
            {
                // nothing special to do
            }
        }
        else
        {
            pdlsch0_harq->status   = ACTIVE;
        }
    }

    pdlsch0_harq->DCINdi = ndi1;
    pdlsch0_harq->mcs    = mcs1;
    pdlsch0_harq->rvidx  = rv1;
    pdlsch0_harq->nb_rb  = NPRB;

    pdlsch0_harq->codeword     = 0;
    pdlsch0_harq->Nl           = 1;
    pdlsch0_harq->mimo_mode    = frame_parms->nb_antenna_ports_eNB == 1 ?SISO : ALAMOUTI;
    pdlsch0_harq->dl_power_off = 1; //no power offset
    pdlsch0_harq->delta_PUCCH  = delta_PUCCH_lut[TPC &3];

    // compute resource allocation
    if(dci_format == format1A)
    {
        switch (N_RB_DL) {
        case 6:
            if (vrb_type == LOCALIZED) {
                pdlsch0_harq->rb_alloc_even[0] = localRIV2alloc_LUT6[rballoc];
                pdlsch0_harq->rb_alloc_odd[0]  = localRIV2alloc_LUT6[rballoc];
            }
            else {
                pdlsch0_harq->rb_alloc_even[0] = distRIV2alloc_even_LUT6[rballoc];
                pdlsch0_harq->rb_alloc_odd[0]  = distRIV2alloc_odd_LUT6[rballoc];
            }
            break;

        case 25:
            if (vrb_type == LOCALIZED) {
                pdlsch0_harq->rb_alloc_even[0] = localRIV2alloc_LUT25[rballoc];
                pdlsch0_harq->rb_alloc_odd[0]  = localRIV2alloc_LUT25[rballoc];
            }
            else {
                pdlsch0_harq->rb_alloc_even[0] = distRIV2alloc_even_LUT25[rballoc];
                pdlsch0_harq->rb_alloc_odd[0]  = distRIV2alloc_odd_LUT25[rballoc];
            }
            break;

        case 50:
            if (vrb_type == LOCALIZED) {
                pdlsch0_harq->rb_alloc_even[0] = localRIV2alloc_LUT50_0[rballoc];
                pdlsch0_harq->rb_alloc_even[1] = localRIV2alloc_LUT50_1[rballoc];
                pdlsch0_harq->rb_alloc_odd[0]  = localRIV2alloc_LUT50_0[rballoc];
                pdlsch0_harq->rb_alloc_odd[1]  = localRIV2alloc_LUT50_1[rballoc];
            } else { // DISTRIBUTED
                if ((rballoc&(1<<10)) == 0) {
                    rballoc = rballoc&(~(1<<10));
                    pdlsch0_harq->rb_alloc_even[0] = distRIV2alloc_gap0_even_LUT50_0[rballoc];
                    pdlsch0_harq->rb_alloc_even[1] = distRIV2alloc_gap0_even_LUT50_1[rballoc];
                    pdlsch0_harq->rb_alloc_odd[0]  = distRIV2alloc_gap0_odd_LUT50_0[rballoc];
                    pdlsch0_harq->rb_alloc_odd[1]  = distRIV2alloc_gap0_odd_LUT50_1[rballoc];
                }
                else {
                    rballoc = rballoc&(~(1<<10));
                    pdlsch0_harq->rb_alloc_even[0] = distRIV2alloc_gap0_even_LUT50_0[rballoc];
                    pdlsch0_harq->rb_alloc_even[1] = distRIV2alloc_gap0_even_LUT50_1[rballoc];
                    pdlsch0_harq->rb_alloc_odd[0]  = distRIV2alloc_gap0_odd_LUT50_0[rballoc];
                    pdlsch0_harq->rb_alloc_odd[1]  = distRIV2alloc_gap0_odd_LUT50_1[rballoc];
                }
            }
            break;

        case 100:
            if (vrb_type == LOCALIZED) {
                pdlsch0_harq->rb_alloc_even[0] = localRIV2alloc_LUT100_0[rballoc];
                pdlsch0_harq->rb_alloc_even[1] = localRIV2alloc_LUT100_1[rballoc];
                pdlsch0_harq->rb_alloc_even[2] = localRIV2alloc_LUT100_2[rballoc];
                pdlsch0_harq->rb_alloc_even[3] = localRIV2alloc_LUT100_3[rballoc];
                pdlsch0_harq->rb_alloc_odd[0]  = localRIV2alloc_LUT100_0[rballoc];
                pdlsch0_harq->rb_alloc_odd[1]  = localRIV2alloc_LUT100_1[rballoc];
                pdlsch0_harq->rb_alloc_odd[2]  = localRIV2alloc_LUT100_2[rballoc];
                pdlsch0_harq->rb_alloc_odd[3]  = localRIV2alloc_LUT100_3[rballoc];
            } else {
                if ((rballoc&(1<<10)) == 0) { //Gap 1
                    rballoc = rballoc&(~(1<<12));
                    pdlsch0_harq->rb_alloc_even[0] = distRIV2alloc_gap0_even_LUT100_0[rballoc];
                    pdlsch0_harq->rb_alloc_even[1] = distRIV2alloc_gap0_even_LUT100_1[rballoc];
                    pdlsch0_harq->rb_alloc_even[2] = distRIV2alloc_gap0_even_LUT100_2[rballoc];
                    pdlsch0_harq->rb_alloc_even[3] = distRIV2alloc_gap0_even_LUT100_3[rballoc];
                    pdlsch0_harq->rb_alloc_odd[0]  = distRIV2alloc_gap0_odd_LUT100_0[rballoc];
                    pdlsch0_harq->rb_alloc_odd[1]  = distRIV2alloc_gap0_odd_LUT100_1[rballoc];
                    pdlsch0_harq->rb_alloc_odd[2]  = distRIV2alloc_gap0_odd_LUT100_2[rballoc];
                    pdlsch0_harq->rb_alloc_odd[3]  = distRIV2alloc_gap0_odd_LUT100_3[rballoc];
                }
                else { //Gap 2
                    rballoc = rballoc&(~(1<<12));
                    pdlsch0_harq->rb_alloc_even[0] = distRIV2alloc_gap1_even_LUT100_0[rballoc];
                    pdlsch0_harq->rb_alloc_even[1] = distRIV2alloc_gap1_even_LUT100_1[rballoc];
                    pdlsch0_harq->rb_alloc_even[2] = distRIV2alloc_gap1_even_LUT100_2[rballoc];
                    pdlsch0_harq->rb_alloc_even[3] = distRIV2alloc_gap1_even_LUT100_3[rballoc];
                    pdlsch0_harq->rb_alloc_odd[0]  = distRIV2alloc_gap1_odd_LUT100_0[rballoc];
                    pdlsch0_harq->rb_alloc_odd[1]  = distRIV2alloc_gap1_odd_LUT100_1[rballoc];
                    pdlsch0_harq->rb_alloc_odd[2]  = distRIV2alloc_gap1_odd_LUT100_2[rballoc];
                    pdlsch0_harq->rb_alloc_odd[3]  = distRIV2alloc_gap1_odd_LUT100_3[rballoc];
                }
            }
            break;
        }
    }
    else // format1
    {
        conv_rballoc(rah,rballoc,frame_parms->N_RB_DL,pdlsch0_harq->rb_alloc_even);
        pdlsch0_harq->rb_alloc_odd[0]= pdlsch0_harq->rb_alloc_even[0];
        pdlsch0_harq->rb_alloc_odd[1]= pdlsch0_harq->rb_alloc_even[1];
        pdlsch0_harq->rb_alloc_odd[2]= pdlsch0_harq->rb_alloc_even[2];
        pdlsch0_harq->rb_alloc_odd[3]= pdlsch0_harq->rb_alloc_even[3];
    }
    if ((rnti==si_rnti) || (rnti==p_rnti) || (rnti==ra_rnti))
    {
        pdlsch0_harq->TBS = TBStable[mcs1][NPRB4TBS-1];
        pdlsch0_harq->Qm  = 2;
    }
    else
    {
        if(mcs1 < 29)
        {
            pdlsch0_harq->TBS = TBStable[get_I_TBS(mcs1)][NPRB4TBS-1];
            pdlsch0_harq->Qm  = get_Qm(mcs1);
        }
    }

    compute_llr_offset(frame_parms,
                       pdcch_vars,
                       pdsch_vars,
                       pdlsch0_harq,
                       NPRB,
                       subframe);
}

void prepare_dl_decoding_format1C(uint8_t N_RB_DL,
                                  DCI_INFO_EXTRACTED_t *pdci_info_extarcted,
                                  LTE_DL_FRAME_PARMS *frame_parms,
                                  LTE_UE_PDCCH *pdcch_vars,
                                  LTE_UE_PDSCH *pdsch_vars,
                                  uint32_t rnti,
                                  uint32_t si_rnti,
                                  uint32_t ra_rnti,
                                  uint32_t p_rnti,
                                  uint32_t frame,
                                  uint8_t  subframe,
                                  LTE_DL_UE_HARQ_t *pdlsch0_harq,
                                  LTE_UE_DLSCH_t *pdlsch0)
{

    uint8_t  harq_pid  = pdci_info_extarcted->harq_pid;
    uint32_t rballoc   = pdci_info_extarcted->rballoc;
    uint8_t  mcs1      = pdci_info_extarcted->mcs1;
    uint8_t  Ngap      = pdci_info_extarcted->Ngap;

      pdlsch0_harq->round     = 0;
      pdlsch0_harq->first_tx  = 1;
      pdlsch0_harq->vrb_type  = DISTRIBUTED;

      if (rnti==si_rnti) { // rule from Section 5.3.1 of 36.321
        if (((frame&1) == 0) && (subframe == 5))
           pdlsch0_harq->rvidx = (((3*((frame>>1)&3))+1)>>1)&3;  // SIB1
        else
           pdlsch0_harq->rvidx = (((3*(subframe&3))+1)>>1)&3;  // other SIBs
      }
      else if ((rnti==p_rnti) || (rnti==ra_rnti)) { // Section 7.1.7.3
        pdlsch0_harq->rvidx = 0;
      }


      pdlsch0_harq->Nl           = 1;
      pdlsch0_harq->mimo_mode    = frame_parms->nb_antenna_ports_eNB == 1 ?SISO : ALAMOUTI;
      pdlsch0_harq->dl_power_off = 1; //no power offset

      pdlsch0_harq->codeword = 0;
      pdlsch0_harq->mcs      = mcs1;
      pdlsch0_harq->TBS      = TBStable1C[mcs1];
      pdlsch0_harq->Qm       = 2;


      pdlsch0->current_harq_pid = harq_pid;
      pdlsch0->active = 1;
      pdlsch0->rnti   = rnti;

    switch (N_RB_DL) {
    case 6:
        pdlsch0_harq->nb_rb            = RIV2nb_rb_LUT6[rballoc];
        pdlsch0_harq->rb_alloc_even[0] = distRIV2alloc_even_LUT6[rballoc];
        pdlsch0_harq->rb_alloc_odd[0]  = distRIV2alloc_odd_LUT6[rballoc];

        break;

    case 25:
        pdlsch0_harq->nb_rb            = RIV2nb_rb_LUT25[rballoc];
        pdlsch0_harq->rb_alloc_even[0] = distRIV2alloc_even_LUT25[rballoc];
        pdlsch0_harq->rb_alloc_odd[0]  = distRIV2alloc_odd_LUT25[rballoc];
        break;

    case 50:
        pdlsch0_harq->nb_rb            = RIV2nb_rb_LUT50[rballoc];
        if (Ngap == 0) {
            pdlsch0_harq->rb_alloc_even[0] = distRIV2alloc_gap0_even_LUT50_0[rballoc];
            pdlsch0_harq->rb_alloc_odd[0]  = distRIV2alloc_gap0_odd_LUT50_0[rballoc];
            pdlsch0_harq->rb_alloc_even[1] = distRIV2alloc_gap0_even_LUT50_1[rballoc];
            pdlsch0_harq->rb_alloc_odd[1]  = distRIV2alloc_gap0_odd_LUT50_1[rballoc];
        }
        else {
            pdlsch0_harq->rb_alloc_even[0] = distRIV2alloc_gap1_even_LUT50_0[rballoc];
            pdlsch0_harq->rb_alloc_odd[0]  = distRIV2alloc_gap1_odd_LUT50_0[rballoc];
            pdlsch0_harq->rb_alloc_even[1] = distRIV2alloc_gap1_even_LUT50_1[rballoc];
            pdlsch0_harq->rb_alloc_odd[1]  = distRIV2alloc_gap1_odd_LUT50_1[rballoc];
        }
        break;

    case 100:
        pdlsch0_harq->nb_rb       = RIV2nb_rb_LUT100[rballoc];
        if (Ngap==0) {
            pdlsch0_harq->rb_alloc_even[0] = distRIV2alloc_gap0_even_LUT100_0[rballoc];
            pdlsch0_harq->rb_alloc_odd[0]  = distRIV2alloc_gap0_odd_LUT100_0[rballoc];
            pdlsch0_harq->rb_alloc_even[1] = distRIV2alloc_gap0_even_LUT100_1[rballoc];
            pdlsch0_harq->rb_alloc_odd[1]  = distRIV2alloc_gap0_odd_LUT100_1[rballoc];
            pdlsch0_harq->rb_alloc_even[2] = distRIV2alloc_gap0_even_LUT100_2[rballoc];
            pdlsch0_harq->rb_alloc_odd[2]  = distRIV2alloc_gap0_odd_LUT100_2[rballoc];
            pdlsch0_harq->rb_alloc_even[3] = distRIV2alloc_gap0_even_LUT100_3[rballoc];
            pdlsch0_harq->rb_alloc_odd[3]  = distRIV2alloc_gap0_odd_LUT100_3[rballoc];
        }
        else {
            pdlsch0_harq->rb_alloc_even[0] = distRIV2alloc_gap1_even_LUT100_0[rballoc];
            pdlsch0_harq->rb_alloc_odd[0]  = distRIV2alloc_gap1_odd_LUT100_0[rballoc];
            pdlsch0_harq->rb_alloc_even[1] = distRIV2alloc_gap1_even_LUT100_1[rballoc];
            pdlsch0_harq->rb_alloc_odd[1]  = distRIV2alloc_gap1_odd_LUT100_1[rballoc];
            pdlsch0_harq->rb_alloc_even[2] = distRIV2alloc_gap1_even_LUT100_2[rballoc];
            pdlsch0_harq->rb_alloc_odd[2]  = distRIV2alloc_gap1_odd_LUT100_2[rballoc];
            pdlsch0_harq->rb_alloc_even[3] = distRIV2alloc_gap1_even_LUT100_3[rballoc];
            pdlsch0_harq->rb_alloc_odd[3]  = distRIV2alloc_gap1_odd_LUT100_3[rballoc];
        }
        break;

    default:
        AssertFatal(0,"Format 1C: Unknown N_RB_DL %d\n",frame_parms->N_RB_DL);
        break;
    }

    compute_llr_offset(frame_parms,
                       pdcch_vars,
                       pdsch_vars,
                       pdlsch0_harq,
                       pdlsch0_harq->nb_rb,
                       subframe);

}

void compute_precoding_info_2cw(uint8_t tpmi, uint8_t tbswap, uint16_t pmi_alloc, LTE_DL_FRAME_PARMS *frame_parms, LTE_DL_UE_HARQ_t *dlsch0_harq, LTE_DL_UE_HARQ_t *dlsch1_harq)
{

switch (tpmi) {
          case 0:
            dlsch0_harq->mimo_mode   = DUALSTREAM_UNIFORM_PRECODING1;
            dlsch1_harq->mimo_mode   = DUALSTREAM_UNIFORM_PRECODING1;
            dlsch0_harq->pmi_alloc   = pmi_extend(frame_parms,0, 1);
            dlsch1_harq->pmi_alloc   = pmi_extend(frame_parms,0, 1);
          break;
          case 1:
            dlsch0_harq->mimo_mode   = DUALSTREAM_UNIFORM_PRECODINGj;
            dlsch1_harq->mimo_mode   = DUALSTREAM_UNIFORM_PRECODINGj;
            dlsch0_harq->pmi_alloc   = pmi_extend(frame_parms,1, 1);
            dlsch1_harq->pmi_alloc   = pmi_extend(frame_parms,1, 1);
          break;
          case 2: // PUSCH precoding
            dlsch0_harq->mimo_mode   = DUALSTREAM_PUSCH_PRECODING;
            dlsch1_harq->mimo_mode   = DUALSTREAM_PUSCH_PRECODING;
            if (tbswap==0){
              dlsch0_harq->pmi_alloc   = pmi_alloc;
              dlsch1_harq->pmi_alloc   = pmi_alloc^0x1555;
            } else {
                dlsch1_harq->pmi_alloc   = pmi_alloc;
                dlsch0_harq->pmi_alloc   = pmi_alloc^0x1555;
            }
#ifdef DEBUG_HARQ
              printf ("\n \n compute_precoding_info_2cw pmi_alloc_new = %d\n", dlsch0_harq->pmi_alloc);
  #endif
          break;
          default:
          break;
        }
}

void compute_precoding_info_1cw(uint8_t tpmi, uint16_t pmi_alloc, LTE_DL_FRAME_PARMS *frame_parms, LTE_DL_UE_HARQ_t *dlsch_harq)
{

switch (tpmi) {
            case 0 :
              dlsch_harq->mimo_mode   = ALAMOUTI;
            break;
            case 1:
              dlsch_harq->mimo_mode   = UNIFORM_PRECODING11;
              dlsch_harq->pmi_alloc   = pmi_extend(frame_parms,0, 0);
            break;
            case 2:
              dlsch_harq->mimo_mode   = UNIFORM_PRECODING1m1;
              dlsch_harq->pmi_alloc   = pmi_extend(frame_parms,1, 0);
            break;
            case 3:
              dlsch_harq->mimo_mode   = UNIFORM_PRECODING1j;
              dlsch_harq->pmi_alloc   = pmi_extend(frame_parms,2, 0);
            break;
            case 4:
              dlsch_harq->mimo_mode   = UNIFORM_PRECODING1mj;
              dlsch_harq->pmi_alloc   = pmi_extend(frame_parms,3, 0);
            break;
            case 5:
              dlsch_harq->mimo_mode   = PUSCH_PRECODING0;
              dlsch_harq->pmi_alloc   = pmi_alloc;//pmi_convert(frame_parms,dlsch0->pmi_alloc,0);
            break;
            case 6:
              dlsch_harq->mimo_mode   = PUSCH_PRECODING1;
              dlsch_harq->pmi_alloc   = pmi_alloc;//pmi_convert(frame_parms,dlsch0->pmi_alloc,1);
            break;
            }
  #ifdef DEBUG_HARQ
              printf ("[DCI UE] I am calling from the UE side pmi_alloc_new = %d with tpmi %d\n", dlsch_harq->pmi_alloc, tpmi);
  #endif
            }

void compute_precoding_info_format2A(uint8_t tpmi,
                                     uint8_t nb_antenna_ports_eNB,
                                     uint8_t tb0_active,
                                     uint8_t tb1_active,
                                     LTE_DL_UE_HARQ_t *dlsch0_harq,
                                     LTE_DL_UE_HARQ_t *dlsch1_harq)
{

    dlsch0_harq->dl_power_off = 0;
    dlsch1_harq->dl_power_off = 0;

    if (nb_antenna_ports_eNB == 2) {
        if ((tb0_active==1) && (tb1_active==1)) {
          dlsch0_harq->mimo_mode = LARGE_CDD;
          dlsch1_harq->mimo_mode = LARGE_CDD;
          dlsch0_harq->dl_power_off = 1;
          dlsch1_harq->dl_power_off = 1;
        } else {
          dlsch0_harq->mimo_mode   = ALAMOUTI;
          dlsch1_harq->mimo_mode   = ALAMOUTI;
        }
      } else if (nb_antenna_ports_eNB == 4) { // 4 antenna case
        if ((tb0_active==1) && (tb1_active==1)) {
          switch (tpmi) {
          case 0: // one layer per transport block
            dlsch0_harq->mimo_mode   = LARGE_CDD;
            dlsch1_harq->mimo_mode   = LARGE_CDD;
            dlsch0_harq->dl_power_off = 1;
            dlsch1_harq->dl_power_off = 1;
            break;

          case 1: // one-layers on TB 0, two on TB 1
            dlsch0_harq->mimo_mode   = LARGE_CDD;
            dlsch1_harq->mimo_mode   = LARGE_CDD;
            dlsch1_harq->Nl          = 2;
            dlsch0_harq->dl_power_off = 1;
            dlsch1_harq->dl_power_off = 1;
            break;

          case 2: // two-layers on TB 0, two on TB 1
            dlsch0_harq->mimo_mode   = LARGE_CDD;
            dlsch1_harq->mimo_mode   = LARGE_CDD;
            dlsch0_harq->Nl          = 2;
            dlsch0_harq->dl_power_off = 1;
            dlsch1_harq->dl_power_off = 1;
            break;

          case 3: //
            LOG_E(PHY,"Illegal value (3) for TPMI in Format 2A DCI\n");
            break;
          }
        } else if (tb0_active == 1) {
          switch (tpmi) {
          case 0: // one layer per transport block
            dlsch0_harq->mimo_mode   = ALAMOUTI;
            dlsch1_harq->mimo_mode   = ALAMOUTI;
            break;

          case 1: // two-layers on TB 0
            dlsch0_harq->mimo_mode   = LARGE_CDD;
            dlsch0_harq->Nl          = 2;
            dlsch0_harq->dl_power_off = 1;
            break;

          case 2: // two-layers on TB 0, two on TB 1
          case 3: //
            LOG_E(PHY,"Illegal value %d for TPMI in Format 2A DCI with one transport block enabled\n",tpmi);
            break;
          }
        } else if (tb1_active == 1) {
          switch (tpmi) {
          case 0: // one layer per transport block
            dlsch0_harq->mimo_mode   = ALAMOUTI;
            dlsch1_harq->mimo_mode   = ALAMOUTI;
            break;

          case 1: // two-layers on TB 0
            dlsch1_harq->mimo_mode   = LARGE_CDD;
            dlsch1_harq->Nl          = 2;
            dlsch0_harq->dl_power_off = 1;
            break;

          case 2: // two-layers on TB 0, two on TB 1
          case 3: //
            LOG_E(PHY,"Illegal value %d for TPMI in Format 2A DCI with one transport block enabled\n",tpmi);
            break;
          }
        }
      }
      //    printf("Format 2A: NPRB=%d (rballoc %x,mcs1 %d, mcs2 %d, frame_type %d N_RB_DL %d,active %d/%d)\n",NPRB,rballoc,mcs1,mcs2,frame_parms->frame_type,frame_parms->N_RB_DL,dlsch0->active,dlsch1->active);
      //printf("UE (%x/%d): Subframe %d Format2A DCI: ndi1 %d, old_ndi1 %d, ndi2 %d, old_ndi2 %d (first tx1 %d, first tx2 %d) harq_status1 %d, harq_status2 %d\n",dlsch0->rnti,harq_pid,subframe,ndi,dlsch0_harq->DCINdi,
      //    dlsch0_harq->first_tx,dlsch1_harq->first_tx,dlsch0_harq->status,dlsch1_harq->status);
      //printf("TBS0 %d, TBS1 %d\n",dlsch0_harq->TBS,dlsch1_harq->TBS);

}

void prepare_dl_decoding_format2_2A(DCI_format_t dci_format,
                                    DCI_INFO_EXTRACTED_t *pdci_info_extarcted,
                                    LTE_DL_FRAME_PARMS *frame_parms,
                                    LTE_UE_PDCCH *pdcch_vars,
                                    LTE_UE_PDSCH *pdsch_vars,
                                    uint16_t rnti,
                                    uint8_t subframe,
                                    LTE_DL_UE_HARQ_t *dlsch0_harq,
                                    LTE_DL_UE_HARQ_t *dlsch1_harq,
                                    LTE_UE_DLSCH_t *pdlsch0,
                                    LTE_UE_DLSCH_t *pdlsch1)
{

    uint8_t  rah  = pdci_info_extarcted->rah;
    uint8_t  mcs1 = pdci_info_extarcted->mcs1;
    uint8_t  mcs2 = pdci_info_extarcted->mcs2;
    uint8_t  rv1  = pdci_info_extarcted->rv1;
    uint8_t  rv2  = pdci_info_extarcted->rv2;
    uint8_t  harq_pid = pdci_info_extarcted->harq_pid;
    uint32_t rballoc  = pdci_info_extarcted->rballoc;
    uint8_t  tbswap   = pdci_info_extarcted->tb_swap;
    uint8_t  tpmi     = pdci_info_extarcted->tpmi;
    uint8_t  TPC      = pdci_info_extarcted->TPC;
    uint8_t  ndi1     = pdci_info_extarcted->ndi1;
    uint8_t  ndi2     = pdci_info_extarcted->ndi2;

    uint8_t TB0_active = 1;
    uint8_t TB1_active = 1;

   // printf("inside prepare pdlsch1->pmi_alloc %d \n",pdlsch1->pmi_alloc);


      if ((rv1 == 1) && (mcs1 == 0)) {
        TB0_active=0;
      }
      if ((rv2 == 1) && (mcs2 == 0)) {
        TB1_active=0;
      }

#ifdef DEBUG_HARQ
      printf("[DCI UE]: TB0 status %d , TB1 status %d\n", TB0_active, TB1_active);
#endif

        dlsch0_harq->mcs      = mcs1;
        dlsch1_harq->mcs      = mcs2;
        dlsch0_harq->rvidx    = rv1;
        dlsch1_harq->rvidx    = rv2;
        dlsch0_harq->DCINdi   = ndi1;
        dlsch1_harq->DCINdi   = ndi2;

        dlsch0_harq->codeword = 0;
        dlsch1_harq->codeword = 1;
        dlsch0_harq->Nl       = 1;
        dlsch1_harq->Nl       = 1;
        dlsch0_harq->delta_PUCCH  = delta_PUCCH_lut[TPC&3];
        dlsch1_harq->delta_PUCCH  = delta_PUCCH_lut[TPC&3];
        dlsch0_harq->dl_power_off = 1;
        dlsch1_harq->dl_power_off = 1;

        pdlsch0->current_harq_pid = harq_pid;
        pdlsch0->harq_ack[subframe].harq_id     = harq_pid;
        pdlsch1->current_harq_pid = harq_pid;
        pdlsch1->harq_ack[subframe].harq_id     = harq_pid;

        // assume two CW are active
        dlsch0_harq->status   = ACTIVE;
        dlsch1_harq->status   = ACTIVE;
        pdlsch0->active = 1;
        pdlsch1->active = 1;
        pdlsch0->rnti = rnti;
        pdlsch1->rnti = rnti;


      if (TB0_active && TB1_active && tbswap==1) {
        dlsch0_harq->codeword = 1;
        dlsch1_harq->codeword = 0;
      }


      if (!TB0_active && TB1_active){
        dlsch1_harq->codeword = 0;
      }

      if (TB0_active && !TB1_active){
        dlsch0_harq->codeword = 0;
      }


      if (TB0_active==0) {
        dlsch0_harq->status = SCH_IDLE;
        pdlsch0->active     = 0;
  #ifdef DEBUG_HARQ
        printf("[DCI UE]: TB0 is deactivated, retransmit TB1 transmit in TM6\n");
  #endif
      }

      if (TB1_active==0) {
        dlsch1_harq->status = SCH_IDLE;
        pdlsch1->active     = 0;
      }

#ifdef DEBUG_HARQ
      printf("[DCI UE]: dlsch0_harq status %d , dlsch1_harq status %d\n", dlsch0_harq->status, dlsch1_harq->status);
#endif

      // compute resource allocation
      if (TB0_active == 1){

        dlsch0_harq->nb_rb = conv_nprb(rah,
                                       rballoc,
                                       frame_parms->N_RB_DL);
        conv_rballoc(rah,
                     rballoc,
                     frame_parms->N_RB_DL,
                     dlsch0_harq->rb_alloc_even);

        dlsch0_harq->rb_alloc_odd[0]= dlsch0_harq->rb_alloc_even[0];
        dlsch0_harq->rb_alloc_odd[1]= dlsch0_harq->rb_alloc_even[1];
        dlsch0_harq->rb_alloc_odd[2]= dlsch0_harq->rb_alloc_even[2];
        dlsch0_harq->rb_alloc_odd[3]= dlsch0_harq->rb_alloc_even[3];

        if (TB1_active == 1){
          dlsch1_harq->rb_alloc_even[0]= dlsch0_harq->rb_alloc_even[0];
          dlsch1_harq->rb_alloc_even[1]= dlsch0_harq->rb_alloc_even[1];
          dlsch1_harq->rb_alloc_even[2]= dlsch0_harq->rb_alloc_even[2];
          dlsch1_harq->rb_alloc_even[3]= dlsch0_harq->rb_alloc_even[3];
          dlsch1_harq->rb_alloc_odd[0] = dlsch0_harq->rb_alloc_odd[0];
          dlsch1_harq->rb_alloc_odd[1] = dlsch0_harq->rb_alloc_odd[1];
          dlsch1_harq->rb_alloc_odd[2] = dlsch0_harq->rb_alloc_odd[2];
          dlsch1_harq->rb_alloc_odd[3] = dlsch0_harq->rb_alloc_odd[3];

          dlsch1_harq->nb_rb = dlsch0_harq->nb_rb;

          //dlsch0_harq->Nl       = 1;
          //dlsch1_harq->Nl       = 1;
        }
      } else if ((TB0_active == 0) && (TB1_active == 1)){

          conv_rballoc(rah,
                       rballoc,
                       frame_parms->N_RB_DL,
                       dlsch1_harq->rb_alloc_even);

          dlsch1_harq->rb_alloc_odd[0]= dlsch1_harq->rb_alloc_even[0];
          dlsch1_harq->rb_alloc_odd[1]= dlsch1_harq->rb_alloc_even[1];
          dlsch1_harq->rb_alloc_odd[2]= dlsch1_harq->rb_alloc_even[2];
          dlsch1_harq->rb_alloc_odd[3]= dlsch1_harq->rb_alloc_even[3];
          dlsch1_harq->nb_rb = conv_nprb(rah,
                                         rballoc,
                                         frame_parms->N_RB_DL);
        }


      // compute precoding matrix + mimo mode
      if(dci_format == format2)
      {
      if ((TB0_active) && (TB1_active)){  //two CW active
        compute_precoding_info_2cw(tpmi, tbswap, pdlsch0->pmi_alloc,frame_parms, dlsch0_harq, dlsch1_harq);

   //   printf("[DCI UE 1]: dlsch0_harq status %d , dlsch1_harq status %d\n", dlsch0_harq->status, dlsch1_harq->status);
      } else if ((TB0_active) && (!TB1_active))  { // only CW 0 active
        compute_precoding_info_1cw(tpmi, pdlsch0->pmi_alloc, frame_parms, dlsch0_harq);
      } else {
        compute_precoding_info_1cw(tpmi, pdlsch1->pmi_alloc, frame_parms, dlsch1_harq);
       // printf("I am doing compute_precoding_info_1cw with tpmi %d \n", tpmi);
      }
      //printf(" UE DCI harq0 MIMO mode = %d\n", dlsch0_harq->mimo_mode);
      if ((frame_parms->nb_antenna_ports_eNB == 1) && (TB0_active))
        dlsch0_harq->mimo_mode   = SISO;
      }
      else
      {
        compute_precoding_info_format2A( tpmi,
                                      frame_parms->nb_antenna_ports_eNB,
                                      TB0_active,
                                      TB1_active,
                                      dlsch0_harq,
                                      dlsch1_harq);
      }
  //    printf("[DCI UE 2]: dlsch0_harq status %d , dlsch1_harq status %d\n", dlsch0_harq->status, dlsch1_harq->status);
      // reset round + compute Qm
      if (TB0_active) {
       // printf("TB0 ndi1 =%d, dlsch0_harq->DCINdi =%d, dlsch0_harq->first_tx = %d\n", ndi1, dlsch0_harq->DCINdi, dlsch0_harq->first_tx);
        if ((ndi1!=dlsch0_harq->DCINdi) || (dlsch0_harq->first_tx==1))  {
          dlsch0_harq->round = 0;
           dlsch0_harq->status = ACTIVE;
           dlsch0_harq->DCINdi = ndi1;

          //LOG_I(PHY,"[UE] DLSCH: New Data Indicator CW0 subframe %d (pid %d, round %d)\n",
          //           subframe,harq_pid,dlsch0_harq->round);
          if ( dlsch0_harq->first_tx==1) {
	    //            LOG_D(PHY,"Format 2 DCI First TX0: Clearing flag\n");
            dlsch0_harq->first_tx = 0;
          }
        }
	/*else if (rv1  != 0 )
	  //NDI has not been toggled but rv was increased by eNB: retransmission
	  {
	    if(dlsch0_harq->status == SCH_IDLE) {
            // skip pdsch decoding and report ack
	      //dlsch0_harq->status   = SCH_IDLE;
            pdlsch0->active       = 0;
            pdlsch0->harq_ack[subframe].ack = 1;
            pdlsch0->harq_ack[subframe].harq_id = harq_pid;
            pdlsch0->harq_ack[subframe].send_harq_status = 1;
	    }*/

        // if Imcs in [29..31] TBS is assumed to be as determined from DCI transported in the latest
        // PDCCH for the same trasport block using Imcs in [0 .. 28]
        if(dlsch0_harq->mcs <= 28)
        {
            dlsch0_harq->TBS = TBStable[get_I_TBS(dlsch0_harq->mcs)][dlsch0_harq->nb_rb-1];
            LOG_D(PHY,"[UE] DLSCH: New TBS CW0 subframe %d (pid %d, round %d) TBS %d \n",
                       subframe,harq_pid,dlsch0_harq->round, dlsch0_harq->TBS);
        }
        else
        {
            LOG_D(PHY,"[UE] DLSCH: Keep the same TBS CW0 subframe %d (pid %d, round %d) TBS %d \n",
                       subframe,harq_pid,dlsch0_harq->round, dlsch0_harq->TBS);
        }
        //if(dlsch0_harq->Nl == 2)
        //dlsch0_harq->TBS = TBStable[get_I_TBS(dlsch0_harq->mcs)][(dlsch0_harq->nb_rb<<1)-1];
        if (mcs1 <= 28)
            dlsch0_harq->Qm = get_Qm(mcs1);
        else if (mcs1<=31)
            dlsch0_harq->Qm = (mcs1-28)<<1;
      }

   //   printf("[DCI UE 3]: dlsch0_harq status %d , dlsch1_harq status %d\n", dlsch0_harq->status, dlsch1_harq->status);

      if (TB1_active) {
       // printf("TB1 ndi2 =%d, dlsch1_harq->DCINdi =%d, dlsch1_harq->first_tx = %d\n", ndi2, dlsch1_harq->DCINdi, dlsch1_harq->first_tx);
        if ((ndi2!=dlsch1_harq->DCINdi) || (dlsch1_harq->first_tx==1)) {
          dlsch1_harq->round = 0;
          dlsch1_harq->status = ACTIVE;
          dlsch1_harq->DCINdi = ndi2;
          //LOG_I(PHY,"[UE] DLSCH: New Data Indicator CW1 subframe %d (pid %d, round %d)\n",
          //           subframe,harq_pid,dlsch0_harq->round);
          if (dlsch1_harq->first_tx==1) {
	    //            LOG_D(PHY,"Format 2 DCI First TX1: Clearing flag\n");
            dlsch1_harq->first_tx = 0;
          }
        }
	/*else if (rv1  != 0 )
	//NDI has not been toggled but rv was increased by eNB: retransmission
	  {
	    if(dlsch1_harq->status == SCH_IDLE) {
            // skip pdsch decoding and report ack
	      //dlsch1_harq->status   = SCH_IDLE;
            pdlsch1->active       = 0;
            pdlsch1->harq_ack[subframe].ack = 1;
            pdlsch1->harq_ack[subframe].harq_id = harq_pid;
            pdlsch1->harq_ack[subframe].send_harq_status = 1;
         }
	  }*/

        // if Imcs in [29..31] TBS is assumed to be as determined from DCI transported in the latest
        // PDCCH for the same trasport block using Imcs in [0 .. 28]
        if(dlsch1_harq->mcs <= 28)
        {
            dlsch1_harq->TBS = TBStable[get_I_TBS(dlsch1_harq->mcs)][dlsch1_harq->nb_rb-1];
            LOG_D(PHY,"[UE] DLSCH: New TBS CW1 subframe %d (pid %d, round %d) TBS %d \n",
                       subframe,harq_pid,dlsch1_harq->round, dlsch1_harq->TBS);
        }
        else
        {
            LOG_D(PHY,"[UE] DLSCH: Keep the same TBS CW1 subframe %d (pid %d, round %d) TBS %d \n",
                       subframe,harq_pid,dlsch1_harq->round, dlsch1_harq->TBS);
        }
        if (mcs2 <= 28)
            dlsch1_harq->Qm = get_Qm(mcs2);
        else if (mcs1<=31)
            dlsch1_harq->Qm = (mcs2-28)<<1;
      }


      compute_llr_offset(frame_parms,
                         pdcch_vars,
                         pdsch_vars,
                         dlsch0_harq,
                         dlsch0_harq->nb_rb,
                         subframe);


 /* #ifdef DEBUG_HARQ
      printf("[DCI UE]: dlsch0_harq status %d , dlsch1_harq status %d\n", dlsch0_harq->status, dlsch1_harq->status);
      printf("[DCI UE]: TB0_active %d , TB1_active %d\n", TB0_active, TB1_active);
      if (dlsch0 != NULL && dlsch1 != NULL)
        printf("[DCI UE] dlsch0_harq status = %d, dlsch1_harq status = %d\n", dlsch0_harq->status, dlsch1_harq->status);
      else if (dlsch0 == NULL && dlsch1 != NULL)
        printf("[DCI UE] dlsch0_harq NULL dlsch1_harq status = %d\n", dlsch1_harq->status);
      else if (dlsch0 != NULL && dlsch1 == NULL)
        printf("[DCI UE] dlsch1_harq NULL dlsch0_harq status = %d\n", dlsch0_harq->status);
  #endif*/
}

int generate_ue_dlsch_params_from_dci(int frame,
                                      uint8_t subframe,
                                      void *dci_pdu,
                                      uint16_t rnti,
                                      DCI_format_t dci_format,
                                      LTE_UE_PDCCH *pdcch_vars,
                                      LTE_UE_PDSCH *pdsch_vars,
                                      LTE_UE_DLSCH_t **dlsch,
                                      LTE_DL_FRAME_PARMS *frame_parms,
                                      PDSCH_CONFIG_DEDICATED *pdsch_config_dedicated,
                                      uint16_t si_rnti,
                                      uint16_t ra_rnti,
                                      uint16_t p_rnti,
                                      uint8_t beamforming_mode,
                                      uint16_t tc_rnti)
{
    uint8_t frame_type=frame_parms->frame_type;
    uint8_t tpmi=0;
    LTE_UE_DLSCH_t *dlsch0=NULL,*dlsch1=NULL;
    LTE_DL_UE_HARQ_t *dlsch0_harq=NULL,*dlsch1_harq=NULL;

    DCI_INFO_EXTRACTED_t dci_info_extarcted;
    uint8_t status=0;

    if (!dlsch[0]) return -1;

  #ifdef DEBUG_DCI
    LOG_D(PHY,"dci_tools.c: Filling ue dlsch params -> rnti %x, SFN/SF %d/%d, dci_format %s\n",
        rnti,
        frame%1024,
        subframe,
        (dci_format==format0?  "Format 0":(
         dci_format==format1?  "format 1":(
         dci_format==format1A? "format 1A":(
         dci_format==format1B? "format 1B":(
         dci_format==format1C? "format 1C":(
         dci_format==format1D? "format 1D":(
         dci_format==format1E_2A_M10PRB? "format 1E_2A_M10PRB":(
         dci_format==format2?  "format 2":(
         dci_format==format2A? "format 2A":(
         dci_format==format2B? "format 2B":(
         dci_format==format2C? "format 2C":(
         dci_format==format2D? "format 2D":(
         dci_format==format3?  "format 3": "UNKNOWN"
         ))))))))))))));
  #endif

    memset(&dci_info_extarcted,0,sizeof(dci_info_extarcted));
    switch (dci_format) {

    case format0:   // This is an ULSCH allocation so nothing here, inform MAC
      LOG_E(PHY,"format0 not possible\n");
      return(-1);
      break;

    case format1A:
    {
      // extract dci infomation
#ifdef DEBUG_DCI
      LOG_I(PHY,"[DCI-FORMAT-1A] AbsSubframe %d.%d extarct dci info \n", frame, subframe);
#endif
      extract_dci1A_info(frame_parms->N_RB_DL,
                         frame_type,
                         dci_pdu,
                         &dci_info_extarcted);


      // check dci content
      dlsch0 = dlsch[0];
      dlsch0->active = 0;
      dlsch0_harq   = dlsch[0]->harq_processes[dci_info_extarcted.harq_pid];
#ifdef DEBUG_DCI
      LOG_I(PHY,"[DCI-FORMAT-1A] AbsSubframe %d.%d check dci coherency \n", frame, subframe);
#endif
      status = check_dci_format1_1a_coherency(format1A,
                                              frame_parms->N_RB_DL,
                                              rnti,
                                              tc_rnti,
                                              si_rnti,
                                              ra_rnti,
                                              p_rnti,frame,subframe,
                                              &dci_info_extarcted,
                                              dlsch0_harq);
      if(status == 0)
      {
        printf("bad DCI 1A !!! \n");
        return(-1);
      }

      // dci is correct ==> update internal structure and prepare dl decoding
#ifdef DEBUG_DCI
      LOG_I(PHY,"[DCI-FORMAT-1A] AbsSubframe %d.%d prepare dl decoding \n", frame, subframe);
#endif
      prepare_dl_decoding_format1_1A(format1A,
                                     frame_parms->N_RB_DL,
                                     &dci_info_extarcted,
                                     frame_parms,
                                     pdcch_vars,
                                     pdsch_vars,
                                     subframe,
                                     rnti,
									 tc_rnti,
                                     si_rnti,
                                     ra_rnti,
                                     p_rnti,
                                     dlsch0_harq,
                                     dlsch0);



      break;
    }
    case format1C:
    {
      // extract dci infomation
#ifdef DEBUG_DL_DECODING
      LOG_I(PHY,"[DCI Format-1C] extact dci information \n");
#endif
      extract_dci1C_info(frame_parms->N_RB_DL,
                         frame_type,
                         dci_pdu,
                         &dci_info_extarcted);


      // check dci content
#ifdef DEBUG_DL_DECODING
      LOG_I(PHY,"[DCI Format-1C] check dci content \n");
#endif
      dlsch0 = dlsch[0];
      dlsch0->active = 0;
      dlsch0_harq = dlsch[0]->harq_processes[dci_info_extarcted.harq_pid];

      status = check_dci_format1c_coherency(frame_parms->N_RB_DL,
                                            &dci_info_extarcted,
                                            rnti,
                                            si_rnti,
                                            ra_rnti,
                                            p_rnti,
                                            dlsch0_harq);
      if(status == 0)
        return(-1);

      // dci is correct ==> update internal structure and prepare dl decoding
#ifdef DEBUG_DL_DECODING
      LOG_I(PHY,"[DCI Format-1C] prepare downlink decoding \n");
#endif
      prepare_dl_decoding_format1C(frame_parms->N_RB_DL,
                                   &dci_info_extarcted,
                                   frame_parms,
                                   pdcch_vars,
                                   pdsch_vars,
                                   rnti,
                                   si_rnti,
                                   ra_rnti,
                                   p_rnti,
                                   frame,
                                   subframe,
                                   dlsch0_harq,
                                   dlsch0);

      break;
    }

    case format1:
    {
      // extract dci infomation
#ifdef DEBUG_DCI
      LOG_I(PHY,"[DCI-FORMAT-1] AbsSubframe %d.%d extarct dci info \n", frame, subframe);
#endif
      extract_dci1_info(frame_parms->N_RB_DL,
                         frame_type,
                         dci_pdu,
                         &dci_info_extarcted);

      // check dci content
      dlsch0 = dlsch[0];
      dlsch0->active = 0;
      dlsch0_harq = dlsch[0]->harq_processes[dci_info_extarcted.harq_pid];

#ifdef DEBUG_DCI
      LOG_I(PHY,"[DCI-FORMAT-1] AbsSubframe %d.%d check dci coherency \n", frame, subframe);
#endif
      status = check_dci_format1_1a_coherency(format1,
                                              frame_parms->N_RB_DL,
                                              rnti,
                                              tc_rnti,
                                              si_rnti,
                                              ra_rnti,
                                              p_rnti,frame,subframe,
                                              &dci_info_extarcted,
                                              dlsch0_harq);
      if(status == 0)
      {
          printf("bad DCI 1 !!! \n");
          return(-1);
      }


      // dci is correct ==> update internal structure and prepare dl decoding
#ifdef DEBUG_DCI
      LOG_I(PHY,"[DCI-FORMAT-1] AbsSubframe %d.%d prepare dl decoding \n", frame, subframe);
#endif
      prepare_dl_decoding_format1_1A(format1,
                                     frame_parms->N_RB_DL,
                                     &dci_info_extarcted,
                                     frame_parms,
                                     pdcch_vars,
                                     pdsch_vars,
                                     subframe,
                                     rnti,
									 tc_rnti,
                                     si_rnti,
                                     ra_rnti,
                                     p_rnti,
                                     dlsch0_harq,
                                     dlsch0);
      break;
    }

    case format2:
    {
        // extract dci infomation
        //LOG_I(PHY,"[DCI-format2] AbsSubframe %d.%d extract dci infomation \n", frame, subframe);
        extract_dci2_info(frame_parms->N_RB_DL,
                frame_type,
                frame_parms->nb_antenna_ports_eNB,
                dci_pdu,
                &dci_info_extarcted);


        // check dci content
        dlsch[0]->active = 1;
        dlsch[1]->active = 1;

            dlsch0 = dlsch[0];
            dlsch1 = dlsch[1];

    dlsch0_harq = dlsch0->harq_processes[dci_info_extarcted.harq_pid];
    dlsch1_harq = dlsch1->harq_processes[dci_info_extarcted.harq_pid];
   // printf("before coherency dlsch[1]->pmi_alloc %d\n",dlsch[1]->pmi_alloc);
   // printf("before coherency dlsch1->pmi_alloc %d\n",dlsch1->pmi_alloc);
   // printf("before coherency dlsch1_harq->pmi_alloc %d\n",dlsch1_harq->pmi_alloc);

        //LOG_I(PHY,"[DCI-format2] check dci content \n");
        status = check_dci_format2_2a_coherency(format2,
                frame_parms->N_RB_DL,
                &dci_info_extarcted,
                rnti,
                si_rnti,
                ra_rnti,
                p_rnti,
                dlsch0_harq,
                dlsch1_harq);
        if(status == 0)
            return(-1);


        // dci is correct ==> update internal structure and prepare dl decoding
        //LOG_I(PHY,"[DCI-format2] update internal structure and prepare dl decoding \n");
        prepare_dl_decoding_format2_2A(format2,
                &dci_info_extarcted,
                frame_parms,
                pdcch_vars,
                pdsch_vars,
                rnti,
                subframe,
                dlsch0_harq,
                dlsch1_harq,
                dlsch0,
                dlsch1);
    }
    break;

    case format2A:
    {
    // extract dci infomation
    LOG_I(PHY,"[DCI-format2] AbsSubframe %d.%d extract dci infomation \n", frame%1024, subframe);
    extract_dci2A_info(frame_parms->N_RB_DL,
                       frame_type,
                       frame_parms->nb_antenna_ports_eNB,
                       dci_pdu,
                       &dci_info_extarcted);

    // check dci content
    //LOG_I(PHY,"[DCI-format2A] check dci content \n");
    //LOG_I(PHY,"[DCI-format2A] tb_swap %d harq_pid %d\n", dci_info_extarcted.tb_swap, dci_info_extarcted.harq_pid);
      //dlsch[0]->active = 0;
      //dlsch[1]->active = 0;

    if (dci_info_extarcted.tb_swap == 0) {
      dlsch0 = dlsch[0];
      dlsch1 = dlsch[1];
    } else {
      dlsch0 = dlsch[1];
      dlsch1 = dlsch[0];
    }
    dlsch0_harq = dlsch0->harq_processes[dci_info_extarcted.harq_pid];
    dlsch1_harq = dlsch1->harq_processes[dci_info_extarcted.harq_pid];

    //LOG_I(PHY,"[DCI-format2A] check dci content \n");
    status = check_dci_format2_2a_coherency(format2A,
                                              frame_parms->N_RB_DL,
                                              &dci_info_extarcted,
                                              rnti,
                                              si_rnti,
                                              ra_rnti,
                                              p_rnti,
                                              dlsch0_harq,
                                              dlsch1_harq);
    if(status == 0)
      return(-1);

    // dci is correct ==> update internal structure and prepare dl decoding
    //LOG_I(PHY,"[DCI-format2A] update internal structure and prepare dl decoding \n");
    prepare_dl_decoding_format2_2A(format2A,
                                   &dci_info_extarcted,
                                   frame_parms,
                                   pdcch_vars,
                                   pdsch_vars,
                                   rnti,
                                   subframe,
                                   dlsch0_harq,
                                   dlsch1_harq,
                                   dlsch0,
                                   dlsch1);
    }
      break;

    case format1E_2A_M10PRB:
      if (!dlsch[0]) return -1;

      dci_info_extarcted.harq_pid  = ((DCI1E_5MHz_2A_M10PRB_TDD_t *)dci_pdu)->harq_pid;

      if (dci_info_extarcted.harq_pid>=8) {
        LOG_E(PHY,"Format 1E_2A_M10PRB: harq_pid=%d >= 8\n", dci_info_extarcted.harq_pid);
        return(-1);
      }

      dlsch[0]->current_harq_pid = dci_info_extarcted.harq_pid;
      dlsch[0]->harq_ack[subframe].harq_id = dci_info_extarcted.harq_pid;

      /*
        tbswap = ((DCI1E_5MHz_2A_M10PRB_TDD_t *)dci_pdu)->tb_swap;
        if (tbswap == 0) {
        dlsch0 = dlsch[0];
        dlsch1 = dlsch[1];
        }
        else{
        dlsch0 = dlsch[1];
        dlsch1 = dlsch[0];
        }
      */
      dlsch0 = dlsch[0];

      dlsch0_harq = dlsch[0]->harq_processes[dci_info_extarcted.harq_pid];
      // Needs to be checked
      dlsch0_harq->codeword=0;
      conv_rballoc(((DCI1E_5MHz_2A_M10PRB_TDD_t *)dci_pdu)->rah,
                   ((DCI1E_5MHz_2A_M10PRB_TDD_t *)dci_pdu)->rballoc,frame_parms->N_RB_DL,
                   dlsch0_harq->rb_alloc_even);

      dlsch0_harq->rb_alloc_odd[0]                         = dlsch0_harq->rb_alloc_even[0];
      dlsch0_harq->rb_alloc_odd[1]                         = dlsch0_harq->rb_alloc_even[1];
      dlsch0_harq->rb_alloc_odd[2]                         = dlsch0_harq->rb_alloc_even[2];
      dlsch0_harq->rb_alloc_odd[3]                         = dlsch0_harq->rb_alloc_even[3];
      /*
      dlsch1_harq->rb_alloc_even[0]                         = dlsch0_harq->rb_alloc_even[0];
      dlsch1_harq->rb_alloc_even[1]                         = dlsch0_harq->rb_alloc_even[1];
      dlsch1_harq->rb_alloc_even[2]                         = dlsch0_harq->rb_alloc_even[2];
      dlsch1_harq->rb_alloc_even[3]                         = dlsch0_harq->rb_alloc_even[3];
      */
      dlsch0_harq->nb_rb                               = conv_nprb(((DCI1E_5MHz_2A_M10PRB_TDD_t *)dci_pdu)->rah,
								   ((DCI1E_5MHz_2A_M10PRB_TDD_t *)dci_pdu)->rballoc,
								   frame_parms->N_RB_DL);
      //dlsch1_harq->nb_rb                               = dlsch0_harq->nb_rb;

      dlsch0_harq->mcs             = ((DCI1E_5MHz_2A_M10PRB_TDD_t *)dci_pdu)->mcs;
      dlsch0_harq->delta_PUCCH     = delta_PUCCH_lut[((DCI1E_5MHz_2A_M10PRB_TDD_t *)dci_pdu)->TPC&3];



      /*
        if (dlsch0_harq->mcs>20) {
        printf("dci_tools.c: mcs > 20 disabled for now (asked %d)\n",dlsch0_harq->mcs);
        return(-1);
        }
      */

      //dlsch1_harq->mcs       = ((DCI1E_5MHz_2A_M10PRB_TDD_t *)dci_pdu)->mcs2;
      dlsch0_harq->rvidx     = ((DCI1E_5MHz_2A_M10PRB_TDD_t *)dci_pdu)->rv;
      //dlsch1_harq->rvidx     = ((DCI1E_5MHz_2A_M10PRB_TDD_t *)dci_pdu)->rv2;

      // check if either TB is disabled (see 36-213 V8.6 p. 26)

      if ((dlsch0_harq->rvidx == 1) && (dlsch0_harq->mcs == 0)) {
        dlsch0_harq->status = DISABLED;
      }

      //if ((dlsch1_harq->rvidx == 1) && (dlsch1_harq->mcs == 0)) {
      //dlsch1_harq->status = DISABLED;
      //}
      dlsch0_harq->Nl        = 1;

      //dlsch0->layer_index                         = tbswap;
      //dlsch1->layer_index                         = 1-tbswap;


      // Fix this
      tpmi = ((DCI1E_5MHz_2A_M10PRB_TDD_t *)dci_pdu)->tpmi;
      //    printf("ue: tpmi %d\n",tpmi);

      switch (tpmi) {
      case 0 :
        dlsch0_harq->mimo_mode   = ALAMOUTI;
        break;

      case 1:
        dlsch0_harq->mimo_mode   = UNIFORM_PRECODING11;
        dlsch0_harq->pmi_alloc   = pmi_extend(frame_parms,0,0);
        break;

      case 2:
        dlsch0_harq->mimo_mode   = UNIFORM_PRECODING1m1;
        dlsch0_harq->pmi_alloc   = pmi_extend(frame_parms,1, 0);
        break;

      case 3:
        dlsch0_harq->mimo_mode   = UNIFORM_PRECODING1j;
        dlsch0_harq->pmi_alloc   = pmi_extend(frame_parms,2, 0);

        break;

      case 4:
        dlsch0_harq->mimo_mode   = UNIFORM_PRECODING1mj;
        dlsch0_harq->pmi_alloc   = pmi_extend(frame_parms,3, 0);
        break;

      case 5:
        dlsch0_harq->mimo_mode   = PUSCH_PRECODING0;
        // pmi stored from ulsch allocation routine
        dlsch0_harq->pmi_alloc   = dlsch0->pmi_alloc;
        //LOG_I(PHY,"XXX using PMI %x\n",pmi2hex_2Ar1(dlsch0_harq->pmi_alloc));
        break;


      case 6:
        dlsch0_harq->mimo_mode   = PUSCH_PRECODING1;
        LOG_E(PHY,"Unsupported TPMI\n");
        return(-1);
        break;
      }


      if (frame_parms->nb_antenna_ports_eNB == 1)
        dlsch0_harq->mimo_mode   = SISO;


      if ((dlsch0_harq->DCINdi != ((DCI1E_5MHz_2A_M10PRB_TDD_t *)dci_pdu)->ndi) ||
	(dlsch0_harq->first_tx==1)) {

        dlsch0_harq->round = 0;
	dlsch0_harq->first_tx = 0;
        dlsch0_harq->status = ACTIVE;
      }
      /*
	else if (dlsch0_harq->status == SCH_IDLE) { // we got same ndi for a previously decoded process,
        // this happens if either another harq process in the same
        // is NAK or an ACK was not received

        dlsch0->harq_ack[subframe].ack              = 1;
        dlsch0->harq_ack[subframe].harq_id          = dci_info_extarcted.harq_pid;
        dlsch0->harq_ack[subframe].send_harq_status = 1;
        dlsch0->active = 0;
        return(0);
      }
      */

      dlsch0_harq->DCINdi = ((DCI1E_5MHz_2A_M10PRB_TDD_t *)dci_pdu)->ndi;
      dlsch0_harq->mcs    = ((DCI1E_5MHz_2A_M10PRB_TDD_t *)dci_pdu)->mcs;

      if (dlsch0_harq->nb_rb>1) {
        dlsch0_harq->TBS         = TBStable[get_I_TBS(dlsch0_harq->mcs)][dlsch0_harq->nb_rb-1];
      } else
        dlsch0_harq->TBS         =0;

      dlsch0->rnti = rnti;
      //dlsch1->rnti = rnti;

      dlsch0->active = 1;
      //dlsch1->active = 1;

      dlsch0_harq->dl_power_off = ((DCI1E_5MHz_2A_M10PRB_TDD_t *)dci_pdu)->dl_power_off;
      //dlsch1_harq->dl_power_off = ((DCI1E_5MHz_2A_M10PRB_TDD_t *)dci_pdu)->dl_power_off;


      break;

    default:
      LOG_E(PHY,"format %d not yet implemented\n",dci_format);
      return(-1);
      break;
    }

#ifdef UE_DEBUG_TRACE

    if (dlsch[0] && (dlsch[0]->rnti != 0xffff)) {
        LOG_I(PHY,"dci_format:%d Abssubframe: %d.%d \n",dci_format,frame%1024,subframe);
        LOG_I(PHY,"PDSCH dlsch0 UE: rnti     %x\n",dlsch[0]->rnti);
        LOG_D(PHY,"PDSCH dlsch0 UE: NBRB     %d\n",dlsch0_harq->nb_rb);
        LOG_D(PHY,"PDSCH dlsch0 UE: rballoc  %x\n",dlsch0_harq->rb_alloc_even[0]);
        LOG_I(PHY,"PDSCH dlsch0 UE: harq_pid %d\n",dci_info_extarcted.harq_pid);
        LOG_I(PHY,"PDSCH dlsch0 UE: g        %d\n",dlsch[0]->g_pucch);
        LOG_D(PHY,"PDSCH dlsch0 UE: round    %d\n",dlsch0_harq->round);
        LOG_D(PHY,"PDSCH dlsch0 UE: DCINdi   %d\n",dlsch0_harq->DCINdi);
        LOG_D(PHY,"PDSCH dlsch0 UE: rvidx    %d\n",dlsch0_harq->rvidx);
        LOG_D(PHY,"PDSCH dlsch0 UE: TBS      %d\n",dlsch0_harq->TBS);
        LOG_D(PHY,"PDSCH dlsch0 UE: mcs      %d\n",dlsch0_harq->mcs);
        LOG_D(PHY,"PDSCH dlsch0 UE: pwr_off  %d\n",dlsch0_harq->dl_power_off);
    }
#endif

#if T_TRACER
    if( (dlsch[0]->rnti != si_rnti) && (dlsch[0]->rnti != ra_rnti) && (dlsch[0]->rnti != p_rnti))
    {
      T(T_UE_PHY_DLSCH_UE_DCI, T_INT(0), T_INT(frame%1024), T_INT(subframe),
        T_INT(dlsch[0]->rnti), T_INT(dci_format),
        T_INT(dci_info_extarcted.harq_pid),
        T_INT(dlsch0_harq->mcs),
        T_INT(dlsch0_harq->TBS));
    }
#endif


    // compute DL power control parameters
    if (dlsch0_harq != NULL){
      computeRhoA_UE(pdsch_config_dedicated, dlsch[0],dlsch0_harq->dl_power_off, frame_parms->nb_antenna_ports_eNB);
      computeRhoB_UE(pdsch_config_dedicated,&(frame_parms->pdsch_config_common),frame_parms->nb_antenna_ports_eNB,dlsch[0],dlsch0_harq->dl_power_off);
    }

    if (dlsch1_harq != NULL) {
      computeRhoA_UE(pdsch_config_dedicated, dlsch[1],dlsch1_harq->dl_power_off, frame_parms->nb_antenna_ports_eNB);
      computeRhoB_UE(pdsch_config_dedicated,&(frame_parms->pdsch_config_common),frame_parms->nb_antenna_ports_eNB,dlsch[1],dlsch1_harq->dl_power_off);
    }


    return(0);
}



int32_t pmi_convert_rank1_from_rank2(uint16_t pmi_alloc, int tpmi, int nb_rb)
{
  int nb_subbands = 0;
  int32_t pmi_alloc_new = 0, pmi_new = 0, pmi_old = 0;
  int i;

  switch (nb_rb) {
    case 6:
      nb_subbands = 6;
      break;
    default:
    case 25:
      nb_subbands = 7;
      break;
    case 50:
      nb_subbands = 9;
      break;
    case 100:
      nb_subbands = 13;
      break;
    }

  for (i = 0; i < nb_subbands; i++) {
    pmi_old = (pmi_alloc >> i)&1;

    if (pmi_old == 0)
      if (tpmi == 5)
        pmi_new = 0;
      else
        pmi_new = 1;
    else
      if (tpmi == 5)
        pmi_new = 2;
      else
        pmi_new = 3;

    pmi_alloc_new|=pmi_new<<(2*i);

  }
#ifdef DEBUG_HARQ
printf("  [DCI UE] pmi_alloc_old %d, pmi_alloc_new %d pmi_old %d , pmi_new %d\n", pmi_alloc, pmi_alloc_new,pmi_old, pmi_new );
#endif
return(pmi_alloc_new);

}

uint16_t quantize_subband_pmi(PHY_MEASUREMENTS *meas,uint8_t eNB_id,int nb_rb)
{

  int i, aarx;
  uint16_t pmiq=0;
  uint32_t pmivect = 0;
  uint8_t rank = meas->rank[eNB_id];
  int pmi_re,pmi_im;
  int  nb_subbands=0;


  switch (nb_rb) {
    case 6:
      nb_subbands = 6;
      break;
    default:
    case 25:
      nb_subbands = 7;
      break;
    case 50:
      nb_subbands = 9;
      break;
    case 100:
      nb_subbands = 13;
      break;
    }


  for (i=0; i<nb_subbands; i++) {
    pmi_re = 0;
    pmi_im = 0;

    if (rank == 0) {
      for (aarx=0; aarx<meas->nb_antennas_rx; aarx++) {
        pmi_re += meas->subband_pmi_re[eNB_id][i][aarx];
        pmi_im += meas->subband_pmi_im[eNB_id][i][aarx];
      }

      //  pmi_re = meas->subband_pmi_re[eNB_id][i][meas->selected_rx_antennas[eNB_id][i]];
      //  pmi_im = meas->subband_pmi_im[eNB_id][i][meas->selected_rx_antennas[eNB_id][i]];

      //      printf("pmi => (%d,%d)\n",pmi_re,pmi_im);
      if ((pmi_re > pmi_im) && (pmi_re > -pmi_im))
        pmiq = PMI_2A_11;
      else if ((pmi_re < pmi_im) && (pmi_re > -pmi_im))
        pmiq = PMI_2A_1j;
      else if ((pmi_re < pmi_im) && (pmi_re < -pmi_im))
        pmiq = PMI_2A_1m1;
      else if ((pmi_re > pmi_im) && (pmi_re < -pmi_im))
        pmiq = PMI_2A_1mj;

      //      printf("subband %d, pmi%d \n",i,pmiq);
      pmivect |= (pmiq<<(2*i));
    }

    else if (rank==1) {
      for (aarx=0; aarx<meas->nb_antennas_rx; aarx++) {
        pmi_re += meas->subband_pmi_re[eNB_id][i][aarx];
  //printf("meas->subband_pmi_re[eNB_id][i][%d]=%d\n", aarx, meas->subband_pmi_re[eNB_id][i][aarx]);
        pmi_im += meas->subband_pmi_im[eNB_id][i][aarx];
  //printf("meas->subband_pmi_im[eNB_id][i][%d]=%d\n",aarx, meas->subband_pmi_im[eNB_id][i][aarx]);
      }
     if (pmi_re >= pmi_im) // this is not orthogonal
     // this is orthogonal
     //if (((pmi_re >= pmi_im) && (pmi_re >= -pmi_im)) || ((pmi_re <= pmi_im) && (pmi_re >= -pmi_im)))
       pmiq = PMI_2A_R1_11;
     else
       pmiq = PMI_2A_R1_1j;

     // printf("subband %d, pmi_re %d, pmi_im %d, pmiq %d \n",i,pmi_re,pmi_im,pmiq);
     // printf("subband %d, pmi%d \n",i,pmiq);
      //According to Section 7.2.4 of 36.213

      pmivect |= ((pmiq-1)<<(i)); //shift 1 since only one bit
    }
    else {
      LOG_E(PHY,"PMI feedback for rank>1 not supported!\n");
      pmivect = 0;
    }

  }
#ifdef DEBUG_HARQ
    printf( "quantize_subband_pmi pmivect %d \n", pmivect);
#endif
  return(pmivect);
}

uint16_t quantize_subband_pmi2(PHY_MEASUREMENTS *meas,uint8_t eNB_id,uint8_t a_id,int nb_subbands)
{

  uint8_t i;
  uint16_t pmiq=0;
  uint16_t pmivect = 0;
  uint8_t rank = meas->rank[eNB_id];
  int pmi_re,pmi_im;

  for (i=0; i<nb_subbands; i++) {

    if (rank == 0) {
      pmi_re = meas->subband_pmi_re[eNB_id][i][a_id];
      pmi_im = meas->subband_pmi_im[eNB_id][i][a_id];

      if ((pmi_re > pmi_im) && (pmi_re > -pmi_im))
        pmiq = PMI_2A_11;
      else if ((pmi_re < pmi_im) && (pmi_re > -pmi_im))
        pmiq = PMI_2A_1j;
      else if ((pmi_re < pmi_im) && (pmi_re < -pmi_im))
        pmiq = PMI_2A_1m1;
      else if ((pmi_re > pmi_im) && (pmi_re < -pmi_im))
        pmiq = PMI_2A_1mj;

      pmivect |= (pmiq<<(2*i));
    } else {
      // This needs to be done properly!!!
      pmivect = 0;
    }
  }

  return(pmivect);
}

uint16_t quantize_wideband_pmi(PHY_MEASUREMENTS *meas,uint8_t eNB_id)
{

  uint16_t pmiq=0;
  uint8_t rank = meas->rank[eNB_id];
  //int pmi;
  int pmi_re,pmi_im;

  if (rank == 1) {
    //pmi =
    pmi_re = meas->wideband_pmi_re[eNB_id][meas->selected_rx_antennas[eNB_id][0]];
    pmi_im = meas->wideband_pmi_im[eNB_id][meas->selected_rx_antennas[eNB_id][0]];

    if ((pmi_re > pmi_im) && (pmi_re > -pmi_im))
      pmiq = PMI_2A_11;
    else if ((pmi_re < pmi_im) && (pmi_re > -pmi_im))
      pmiq = PMI_2A_1j;
    else if ((pmi_re < pmi_im) && (pmi_re < -pmi_im))
      pmiq = PMI_2A_1m1;
    else if ((pmi_re > pmi_im) && (pmi_re < -pmi_im))
      pmiq = PMI_2A_1mj;

  } else {
    // This needs to be done properly!
    pmiq = PMI_2A_11;
  }


  return(pmiq);
}
/*
  uint8_t sinr2cqi(int sinr) {
  if (sinr<-3)
  return(0);
  if (sinr>14)
  return(10);
  else
  return(3+(sinr>>1));
  }
*/

uint8_t sinr2cqi(double sinr,uint8_t trans_mode)
{
  // int flag_LA=0;

  uint8_t retValue = 0;

  if(flag_LA==0) {
    // Ideal Channel Estimation
    if (sinr<=-4.89)
      retValue = (0);
    else if (sinr < -3.53)
      retValue = (3);
    else if (sinr <= -1.93)
      retValue = (4);
    else if (sinr <= -0.43)
      retValue = (5);
    else if (sinr <= 1.11)
      retValue = (6);
    else if (sinr <= 3.26)
      retValue = (7);
    else if (sinr <= 5.0)
      retValue = (8);
    else if (sinr <= 7.0)
      retValue = (9);
    else if (sinr <= 9.0)
      retValue = (10);
    else if (sinr <= 11.0)
      retValue = (11);
    else if (sinr <= 13.0)
      retValue = (12);
    else if (sinr <= 15.5)
      retValue = (13);
    else if (sinr <= 17.5)
      retValue = (14);
    else
      retValue = (15);
  } else {
    int h=0;
    int trans_mode_tmp;

    if (trans_mode ==5)
      trans_mode_tmp=2;
    else if(trans_mode ==6)
      trans_mode_tmp=3;
    else
      trans_mode_tmp = trans_mode-1;

    for(h=0; h<16; h++) {
      if(sinr<=sinr_to_cqi[trans_mode_tmp][h])
        retValue = (h);
    }
  }

  LOG_D(PHY, "sinr=%f trans_mode=%d cqi=%d\n", sinr, trans_mode, retValue);
  return retValue;
}
//uint32_t fill_subband_cqi(PHY_MEASUREMENTS *meas,uint8_t eNB_id) {
//
//  uint8_t i;
////  uint16_t cqivect = 0;
//  uint32_t cqivect = 0;
//
////  char diff_cqi;
//  int diff_cqi=0;
//
//  for (i=0;i<NUMBER_OF_SUBBANDS;i++) {
//
//    diff_cqi = -sinr2cqi(meas->wideband_cqi_dB[eNB_id][0]) + sinr2cqi(meas->subband_cqi_dB[eNB_id][0][i]);
//
//    // Note, this is Table 7.2.1-2 from 36.213
//    if (diff_cqi<=-1)
//      diff_cqi = 3;
//    else if (diff_cqi>2)
//      diff_cqi = 2;
//    cqivect |= (diff_cqi<<(2*i));
//
//  }
//
//  return(cqivect);
//}


uint32_t fill_subband_cqi(PHY_MEASUREMENTS *meas,uint8_t eNB_id,uint8_t trans_mode,int nb_subbands)
{

  uint8_t i;

  uint32_t cqivect = 0,offset=0;


  int diff_cqi=0;

  for (i=0; i<nb_subbands; i++) {

    diff_cqi = -sinr2cqi(meas->wideband_cqi_avg[eNB_id],trans_mode) + sinr2cqi(meas->subband_cqi_tot_dB[eNB_id][i],trans_mode);

    // Note, this is Table 7.2.1-2 from 36.213
    if (diff_cqi<=-1)
      offset = 3;
    else if (diff_cqi>=2)
      offset = 2;
    else
      offset=(uint32_t)diff_cqi;

    cqivect |= (offset<<(2*i));

  }

  return(cqivect);
}

void fill_CQI(LTE_UE_ULSCH_t *ulsch,PHY_MEASUREMENTS *meas,uint8_t eNB_id,uint8_t harq_pid,int N_RB_DL,uint16_t rnti, uint8_t trans_mode, double sinr_eff)
{

  //  printf("[PHY][UE] Filling CQI for eNB %d, meas->wideband_cqi_tot[%d] %d\n",
  //      eNB_id,eNB_id,meas->wideband_cqi_tot[eNB_id]);
  double sinr_tmp;
  uint8_t *o = ulsch->o;
  UCI_format_t uci_format = ulsch->uci_format;

  if(flag_LA==1)
    sinr_tmp = sinr_eff;
  else
    sinr_tmp = (double) meas->wideband_cqi_avg[eNB_id];



  //LOG_I(PHY,"[UE][UCI] Filling CQI format %d for eNB %d N_RB_DL %d\n",uci_format,eNB_id,N_RB_DL);

  switch (N_RB_DL) {

  case 6:
    switch (uci_format) {
    case wideband_cqi_rank1_2A:
      ((wideband_cqi_rank1_2A_1_5MHz *)o)->cqi1 = sinr2cqi(sinr_tmp,trans_mode);
      ((wideband_cqi_rank1_2A_1_5MHz *)o)->pmi  = quantize_subband_pmi(meas,eNB_id,6);
      break;

    case wideband_cqi_rank2_2A:
      ((wideband_cqi_rank2_2A_1_5MHz *)o)->cqi1 = sinr2cqi(sinr_tmp,trans_mode); //FIXME: calculate rank2 cqi
      ((wideband_cqi_rank2_2A_1_5MHz *)o)->cqi2 = sinr2cqi(sinr_tmp,trans_mode); //FIXME: calculate rank2 cqi
      ((wideband_cqi_rank2_2A_1_5MHz *)o)->pmi  = quantize_subband_pmi(meas,eNB_id,6);
      break;

    case HLC_subband_cqi_nopmi:
      ((HLC_subband_cqi_nopmi_1_5MHz *)o)->cqi1     = sinr2cqi(sinr_tmp,trans_mode);
      ((HLC_subband_cqi_nopmi_1_5MHz *)o)->diffcqi1 = fill_subband_cqi(meas,eNB_id,trans_mode,6);
      break;

    case HLC_subband_cqi_rank1_2A:
      ((HLC_subband_cqi_rank1_2A_1_5MHz *)o)->cqi1     = sinr2cqi(sinr_tmp,trans_mode);
      ((HLC_subband_cqi_rank1_2A_1_5MHz *)o)->diffcqi1 = fill_subband_cqi(meas,eNB_id,trans_mode,6);
      ((HLC_subband_cqi_rank1_2A_1_5MHz *)o)->pmi      = quantize_wideband_pmi(meas,eNB_id);
      break;

    case HLC_subband_cqi_rank2_2A:
      // This has to be improved!!!
      ((HLC_subband_cqi_rank2_2A_1_5MHz *)o)->cqi1     = sinr2cqi(sinr_tmp,trans_mode);
      ((HLC_subband_cqi_rank2_2A_1_5MHz *)o)->diffcqi1 = fill_subband_cqi(meas,eNB_id,trans_mode,6);
      ((HLC_subband_cqi_rank2_2A_1_5MHz *)o)->cqi2     = sinr2cqi(sinr_tmp,trans_mode);
      ((HLC_subband_cqi_rank2_2A_1_5MHz *)o)->diffcqi2 = fill_subband_cqi(meas,eNB_id,trans_mode,6);
      ((HLC_subband_cqi_rank2_2A_1_5MHz *)o)->pmi      = quantize_subband_pmi(meas,eNB_id,6);
      break;

    case HLC_subband_cqi_mcs_CBA:
      // this is the cba mcs uci for cba transmission
      ((HLC_subband_cqi_mcs_CBA_1_5MHz *)o)->mcs     = ulsch->harq_processes[harq_pid]->mcs;
      ((HLC_subband_cqi_mcs_CBA_1_5MHz *)o)->crnti  = rnti;
      LOG_D(PHY,"fill uci for cba rnti %x, mcs %d \n", rnti, ulsch->harq_processes[harq_pid]->mcs);
      break;

    case ue_selected:
      LOG_E(PHY,"fill_CQI ue_selected CQI not supported yet!!!\n");
      AssertFatal(1==0,"fill_CQI ue_selected CQI not supported yet!!!");
      break;

    default:
      LOG_E(PHY,"unsupported CQI mode (%d)!!!\n",uci_format);
      AssertFatal(1==0,"unsupported CQI mode !!!");
      break;

    }

    break;

  case 25:
    switch (uci_format) {
    case wideband_cqi_rank1_2A:
      ((wideband_cqi_rank1_2A_5MHz *)o)->cqi1 = sinr2cqi(sinr_tmp,trans_mode);
      ((wideband_cqi_rank1_2A_5MHz *)o)->pmi  = quantize_subband_pmi(meas,eNB_id,7);
      break;

    case wideband_cqi_rank2_2A:
      ((wideband_cqi_rank2_2A_5MHz *)o)->cqi1 = sinr2cqi(sinr_tmp,trans_mode); //FIXME: calculate rank2 cqi
      ((wideband_cqi_rank2_2A_5MHz *)o)->cqi2 = sinr2cqi(sinr_tmp,trans_mode); //FIXME: calculate rank2 cqi
      ((wideband_cqi_rank2_2A_5MHz *)o)->pmi  = quantize_subband_pmi(meas,eNB_id,7);
      break;

    case HLC_subband_cqi_nopmi:
      ((HLC_subband_cqi_nopmi_5MHz *)o)->cqi1     = sinr2cqi(sinr_tmp,trans_mode);
      ((HLC_subband_cqi_nopmi_5MHz *)o)->diffcqi1 = fill_subband_cqi(meas,eNB_id,trans_mode,7);
      break;

    case HLC_subband_cqi_rank1_2A:
      ((HLC_subband_cqi_rank1_2A_5MHz *)o)->cqi1     = sinr2cqi(sinr_tmp,trans_mode);
      ((HLC_subband_cqi_rank1_2A_5MHz *)o)->diffcqi1 = fill_subband_cqi(meas,eNB_id,trans_mode,7);
      ((HLC_subband_cqi_rank1_2A_5MHz *)o)->pmi      = quantize_wideband_pmi(meas,eNB_id);
      break;

    case HLC_subband_cqi_rank2_2A:
      // This has to be improved!!!
      ((HLC_subband_cqi_rank2_2A_5MHz *)o)->cqi1     = sinr2cqi(sinr_tmp,trans_mode);
      ((HLC_subband_cqi_rank2_2A_5MHz *)o)->diffcqi1 = fill_subband_cqi(meas,eNB_id,trans_mode,7);
      ((HLC_subband_cqi_rank2_2A_5MHz *)o)->cqi2     = sinr2cqi(sinr_tmp,trans_mode);
      ((HLC_subband_cqi_rank2_2A_5MHz *)o)->diffcqi2 = fill_subband_cqi(meas,eNB_id,trans_mode,7);
      ((HLC_subband_cqi_rank2_2A_5MHz *)o)->pmi      = quantize_subband_pmi(meas,eNB_id,7);
      break;

    case HLC_subband_cqi_mcs_CBA:
      // this is the cba mcs uci for cba transmission
      ((HLC_subband_cqi_mcs_CBA_5MHz *)o)->mcs     = ulsch->harq_processes[harq_pid]->mcs;
      ((HLC_subband_cqi_mcs_CBA_5MHz *)o)->crnti  = rnti;
      LOG_I(PHY,"fill uci for cba rnti %x, mcs %d \n", rnti, ulsch->harq_processes[harq_pid]->mcs);
      break;

    case ue_selected:
      LOG_E(PHY,"fill_CQI ue_selected CQI not supported yet!!!\n");
      AssertFatal(1==0,"fill_CQI ue_selected CQI not supported yet!!!");
      break;

    default:
      LOG_E(PHY,"unsupported CQI mode (%d)!!!\n",uci_format);
      AssertFatal(1==0,"unsupported CQI mode !!!");
      break;

    }

    break;

  case 50:
    switch (uci_format) {
    case wideband_cqi_rank1_2A:
      ((wideband_cqi_rank1_2A_10MHz *)o)->cqi1 = sinr2cqi(sinr_tmp,trans_mode);
      ((wideband_cqi_rank1_2A_10MHz *)o)->pmi  = quantize_subband_pmi(meas,eNB_id,9);
      break;

    case wideband_cqi_rank2_2A:
      ((wideband_cqi_rank2_2A_10MHz *)o)->cqi1 = sinr2cqi(sinr_tmp,trans_mode); //FIXME: calculate rank2 cqi
      ((wideband_cqi_rank2_2A_10MHz *)o)->cqi2 = sinr2cqi(sinr_tmp,trans_mode); //FIXME: calculate rank2 cqi
      ((wideband_cqi_rank2_2A_10MHz *)o)->pmi  = quantize_subband_pmi(meas,eNB_id,9);
      break;

    case HLC_subband_cqi_nopmi:
      ((HLC_subband_cqi_nopmi_10MHz *)o)->cqi1     = sinr2cqi(sinr_tmp,trans_mode);
      ((HLC_subband_cqi_nopmi_10MHz *)o)->diffcqi1 = fill_subband_cqi(meas,eNB_id,trans_mode,9);
      break;

    case HLC_subband_cqi_rank1_2A:
      ((HLC_subband_cqi_rank1_2A_10MHz *)o)->cqi1     = sinr2cqi(sinr_tmp,trans_mode);
      ((HLC_subband_cqi_rank1_2A_10MHz *)o)->diffcqi1 = fill_subband_cqi(meas,eNB_id,trans_mode,9);
      ((HLC_subband_cqi_rank1_2A_10MHz *)o)->pmi      = quantize_wideband_pmi(meas,eNB_id);
      break;

    case HLC_subband_cqi_rank2_2A:
      // This has to be improved!!!
      ((HLC_subband_cqi_rank2_2A_10MHz *)o)->cqi1     = sinr2cqi(sinr_tmp,trans_mode);
      ((HLC_subband_cqi_rank2_2A_10MHz *)o)->diffcqi1 = fill_subband_cqi(meas,eNB_id,trans_mode,9);
      ((HLC_subband_cqi_rank2_2A_10MHz *)o)->cqi2     = sinr2cqi(sinr_tmp,trans_mode);
      ((HLC_subband_cqi_rank2_2A_10MHz *)o)->diffcqi2 = fill_subband_cqi(meas,eNB_id,trans_mode,9);
      ((HLC_subband_cqi_rank2_2A_10MHz *)o)->pmi      = quantize_subband_pmi(meas,eNB_id,9);
      break;

    case HLC_subband_cqi_mcs_CBA:
      // this is the cba mcs uci for cba transmission
      ((HLC_subband_cqi_mcs_CBA_10MHz *)o)->mcs     = ulsch->harq_processes[harq_pid]->mcs;
      ((HLC_subband_cqi_mcs_CBA_10MHz *)o)->crnti  = rnti;
      LOG_I(PHY,"fill uci for cba rnti %x, mcs %d \n", rnti, ulsch->harq_processes[harq_pid]->mcs);
      break;

    case ue_selected:
      LOG_E(PHY,"fill_CQI ue_selected CQI not supported yet!!!\n");
      AssertFatal(1==0,"fill_CQI ue_selected CQI not supported yet!!!");
      break;

    default:
      LOG_E(PHY,"unsupported CQI mode (%d)!!!\n",uci_format);
      AssertFatal(1==0,"unsupported CQI mode !!!");
      break;

    }

    break;

  case 100:
    switch (uci_format) {
    case wideband_cqi_rank1_2A:
      ((wideband_cqi_rank1_2A_20MHz *)o)->cqi1 = sinr2cqi(sinr_tmp,trans_mode);
      ((wideband_cqi_rank1_2A_20MHz *)o)->pmi  = quantize_subband_pmi(meas,eNB_id,13);
      break;

    case wideband_cqi_rank2_2A:
      ((wideband_cqi_rank2_2A_20MHz *)o)->cqi1 = sinr2cqi(sinr_tmp,trans_mode); //FIXME: calculate rank2 cqi
      ((wideband_cqi_rank2_2A_20MHz *)o)->cqi2 = sinr2cqi(sinr_tmp,trans_mode); //FIXME: calculate rank2 cqi
      ((wideband_cqi_rank2_2A_20MHz *)o)->pmi  = quantize_subband_pmi(meas,eNB_id,13);
      break;

    case HLC_subband_cqi_nopmi:
      ((HLC_subband_cqi_nopmi_20MHz *)o)->cqi1     = sinr2cqi(sinr_tmp,trans_mode);
      ((HLC_subband_cqi_nopmi_20MHz *)o)->diffcqi1 = fill_subband_cqi(meas,eNB_id,trans_mode,13);
      break;

    case HLC_subband_cqi_rank1_2A:
      ((HLC_subband_cqi_rank1_2A_20MHz *)o)->cqi1     = sinr2cqi(sinr_tmp,trans_mode);
      ((HLC_subband_cqi_rank1_2A_20MHz *)o)->diffcqi1 = fill_subband_cqi(meas,eNB_id,trans_mode,13);
      ((HLC_subband_cqi_rank1_2A_20MHz *)o)->pmi      = quantize_wideband_pmi(meas,eNB_id);
      break;

    case HLC_subband_cqi_rank2_2A:
      // This has to be improved!!!
      ((HLC_subband_cqi_rank2_2A_20MHz *)o)->cqi1     = sinr2cqi(sinr_tmp,trans_mode);
      ((HLC_subband_cqi_rank2_2A_20MHz *)o)->diffcqi1 = fill_subband_cqi(meas,eNB_id,trans_mode,13);
      ((HLC_subband_cqi_rank2_2A_20MHz *)o)->cqi2     = sinr2cqi(sinr_tmp,trans_mode);
      ((HLC_subband_cqi_rank2_2A_20MHz *)o)->diffcqi2 = fill_subband_cqi(meas,eNB_id,trans_mode,13);
      ((HLC_subband_cqi_rank2_2A_20MHz *)o)->pmi      = quantize_subband_pmi(meas,eNB_id,13);
      break;

    case HLC_subband_cqi_mcs_CBA:
      // this is the cba mcs uci for cba transmission
      ((HLC_subband_cqi_mcs_CBA_20MHz *)o)->mcs     = ulsch->harq_processes[harq_pid]->mcs;
      ((HLC_subband_cqi_mcs_CBA_20MHz *)o)->crnti  = rnti;
      LOG_I(PHY,"fill uci for cba rnti %x, mcs %d \n", rnti, ulsch->harq_processes[harq_pid]->mcs);
      break;

    case ue_selected:
      LOG_E(PHY,"fill_CQI ue_selected CQI not supported yet!!!\n");
      AssertFatal(1==0,"fill_CQI ue_selected CQI not supported yet!!!");
      break;

    default:
      LOG_E(PHY,"unsupported CQI mode (%d)!!!\n",uci_format);
      AssertFatal(1==0,"unsupported CQI mode !!!");
      break;

    }

    break;

  }


}

void reset_cba_uci(void *o)
{
  // this is the cba mcs uci for cba transmission
  ((HLC_subband_cqi_mcs_CBA_5MHz *)o)->mcs     = 0; //fixme
  ((HLC_subband_cqi_mcs_CBA_5MHz *)o)->crnti  = 0x0;
}





int generate_ue_ulsch_params_from_dci(void *dci_pdu,
                                      uint16_t rnti,
                                      uint8_t subframe,
                                      DCI_format_t dci_format,
                                      PHY_VARS_UE *ue,
                                      UE_rxtx_proc_t *proc,
                                      uint16_t si_rnti,
                                      uint16_t ra_rnti,
                                      uint16_t p_rnti,
                                      uint16_t cba_rnti,
                                      uint8_t eNB_id,
                                      uint8_t use_srs)
{

  uint8_t harq_pid;
  uint8_t transmission_mode = ue->transmission_mode[eNB_id];
  ANFBmode_t AckNackFBMode;
  LTE_UE_ULSCH_t *ulsch = ue->ulsch[eNB_id];
  LTE_UE_DLSCH_t **dlsch = ue->dlsch[ue->current_thread_id[subframe]][0];
  PHY_MEASUREMENTS *meas = &ue->measurements;
  LTE_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;
  //  uint32_t current_dlsch_cqi = ue->current_dlsch_cqi[eNB_id];

  if(frame_parms->frame_type == TDD)
  {
      AckNackFBMode = ue->pucch_config_dedicated[eNB_id].tdd_AckNackFeedbackMode;
  }
  else
  {
      AckNackFBMode = 1; // 1: multiplexing for FDD
  }

  uint32_t cqi_req;
  uint32_t dai=3;
  uint32_t cshift;
  uint32_t TPC;
  uint32_t ndi;
  uint32_t mcs;
  uint32_t rballoc,RIV_max;
  uint16_t* RIV2first_rb_LUT;
  uint16_t* RIV2nb_rb_LUT;

  //  uint32_t hopping;
  //  uint32_t type;

  if (dci_format == format0) {

    if (!ulsch)
      return -1;

    if (rnti == ra_rnti)
      harq_pid = 0;
    else
      harq_pid = subframe2harq_pid(frame_parms,
                                   pdcch_alloc2ul_frame(frame_parms,proc->frame_rx,subframe),
                                   pdcch_alloc2ul_subframe(frame_parms,subframe));
    LOG_D(PHY,"Frame %d, Subframe %d: Programming ULSCH for (%d.%d) => harq_pid %d\n",
	  proc->frame_rx,subframe,
	  pdcch_alloc2ul_frame(frame_parms,proc->frame_rx,subframe),
	  pdcch_alloc2ul_subframe(frame_parms,subframe), harq_pid);

    if (harq_pid == 255) {
      LOG_E(PHY, "frame %d, subframe %d, rnti %x, format %d: illegal harq_pid!\n",
            proc->frame_rx, subframe, rnti, dci_format);
      return(-1);
    }

    switch (frame_parms->N_RB_DL) {
    case 6:
      if (frame_parms->frame_type == TDD) {
        cqi_req = ((DCI0_1_5MHz_TDD_1_6_t *)dci_pdu)->cqi_req;
        dai     = ((DCI0_1_5MHz_TDD_1_6_t *)dci_pdu)->dai;
        cshift  = ((DCI0_1_5MHz_TDD_1_6_t *)dci_pdu)->cshift;
        TPC     = ((DCI0_1_5MHz_TDD_1_6_t *)dci_pdu)->TPC;
        ndi     = ((DCI0_1_5MHz_TDD_1_6_t *)dci_pdu)->ndi;
        mcs     = ((DCI0_1_5MHz_TDD_1_6_t *)dci_pdu)->mcs;
        rballoc = ((DCI0_1_5MHz_TDD_1_6_t *)dci_pdu)->rballoc;
        //  hopping = ((DCI0_1_5MHz_TDD_1_6_t *)dci_pdu)->hopping=hopping;
        //  type    = ((DCI0_1_5MHz_TDD_1_6_t *)dci_pdu)->type;
      } else {
        cqi_req = ((DCI0_1_5MHz_FDD_t *)dci_pdu)->cqi_req;
        cshift  = ((DCI0_1_5MHz_FDD_t *)dci_pdu)->cshift;
        TPC     = ((DCI0_1_5MHz_FDD_t *)dci_pdu)->TPC;
        ndi     = ((DCI0_1_5MHz_FDD_t *)dci_pdu)->ndi;
        mcs     = ((DCI0_1_5MHz_FDD_t *)dci_pdu)->mcs;
        rballoc = ((DCI0_1_5MHz_FDD_t *)dci_pdu)->rballoc;
        //  hopping = ((DCI0_1_5MHz_FDD_t *)dci_pdu)->hopping=hopping;
        //  type    = ((DCI0_1_5MHz_FDD_t *)dci_pdu)->type;
      }

      RIV_max = RIV_max6;
      RIV2first_rb_LUT = RIV2first_rb_LUT6;
      RIV2nb_rb_LUT = RIV2nb_rb_LUT6;

      break;

    case 25:
      if (frame_parms->frame_type == TDD) {
        cqi_req = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->cqi_req;
        dai     = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->dai;
        cshift  = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->cshift;
        TPC     = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->TPC;
        ndi     = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->ndi;
        mcs     = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->mcs;
        rballoc = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->rballoc;
        //  hopping = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->hopping=hopping;
        //  type    = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->type;
      } else {
        cqi_req = ((DCI0_5MHz_FDD_t *)dci_pdu)->cqi_req;
        cshift  = ((DCI0_5MHz_FDD_t *)dci_pdu)->cshift;
        TPC     = ((DCI0_5MHz_FDD_t *)dci_pdu)->TPC;
        ndi     = ((DCI0_5MHz_FDD_t *)dci_pdu)->ndi;
        mcs     = ((DCI0_5MHz_FDD_t *)dci_pdu)->mcs;
        rballoc = ((DCI0_5MHz_FDD_t *)dci_pdu)->rballoc;
        //  hopping = ((DCI0_5MHz_FDD_t *)dci_pdu)->hopping=hopping;
        //  type    = ((DCI0_5MHz_FDD_t *)dci_pdu)->type;
      }

      RIV_max = RIV_max25;
      RIV2first_rb_LUT = RIV2first_rb_LUT25;
      RIV2nb_rb_LUT = RIV2nb_rb_LUT25;
      //      printf("***********rballoc %d, first_rb %d, nb_rb %d (dci %p)\n",rballoc,ulsch->harq_processes[harq_pid]->first_rb,ulsch->harq_processes[harq_pid]->nb_rb,dci_pdu);
      break;

    case 50:
      if (frame_parms->frame_type == TDD) {
        cqi_req = ((DCI0_10MHz_TDD_1_6_t *)dci_pdu)->cqi_req;
        dai     = ((DCI0_10MHz_TDD_1_6_t *)dci_pdu)->dai;
        cshift  = ((DCI0_10MHz_TDD_1_6_t *)dci_pdu)->cshift;
        TPC     = ((DCI0_10MHz_TDD_1_6_t *)dci_pdu)->TPC;
        ndi     = ((DCI0_10MHz_TDD_1_6_t *)dci_pdu)->ndi;
        mcs     = ((DCI0_10MHz_TDD_1_6_t *)dci_pdu)->mcs;
        rballoc = ((DCI0_10MHz_TDD_1_6_t *)dci_pdu)->rballoc;
        //  hopping = ((DCI0_10MHz_TDD_1_6_t *)dci_pdu)->hopping=hopping;
        //  type    = ((DCI0_10MHz_TDD_1_6_t *)dci_pdu)->type;
      } else {
        cqi_req = ((DCI0_10MHz_FDD_t *)dci_pdu)->cqi_req;
        cshift  = ((DCI0_10MHz_FDD_t *)dci_pdu)->cshift;
        TPC     = ((DCI0_10MHz_FDD_t *)dci_pdu)->TPC;
        ndi     = ((DCI0_10MHz_FDD_t *)dci_pdu)->ndi;
        mcs     = ((DCI0_10MHz_FDD_t *)dci_pdu)->mcs;
        rballoc = ((DCI0_10MHz_FDD_t *)dci_pdu)->rballoc;
        //  hopping = ((DCI0_10MHz_FDD_t *)dci_pdu)->hopping=hopping;
        //  type    = ((DCI0_10MHz_FDD_t *)dci_pdu)->type;
      }

      RIV_max = RIV_max50;
      RIV2first_rb_LUT = RIV2first_rb_LUT50;
      RIV2nb_rb_LUT = RIV2nb_rb_LUT50;

      break;

    case 100:
      if (frame_parms->frame_type == TDD) {
        cqi_req = ((DCI0_20MHz_TDD_1_6_t *)dci_pdu)->cqi_req;
        dai     = ((DCI0_20MHz_TDD_1_6_t *)dci_pdu)->dai;
        cshift  = ((DCI0_20MHz_TDD_1_6_t *)dci_pdu)->cshift;
        TPC     = ((DCI0_20MHz_TDD_1_6_t *)dci_pdu)->TPC;
        ndi     = ((DCI0_20MHz_TDD_1_6_t *)dci_pdu)->ndi;
        mcs     = ((DCI0_20MHz_TDD_1_6_t *)dci_pdu)->mcs;
        rballoc = ((DCI0_20MHz_TDD_1_6_t *)dci_pdu)->rballoc;
        //  hopping = ((DCI0_20MHz_TDD_1_6_t *)dci_pdu)->hopping=hopping;
        //  type    = ((DCI0_20MHz_TDD_1_6_t *)dci_pdu)->type;
      } else {
        cqi_req = ((DCI0_20MHz_FDD_t *)dci_pdu)->cqi_req;
        cshift  = ((DCI0_20MHz_FDD_t *)dci_pdu)->cshift;
        TPC     = ((DCI0_20MHz_FDD_t *)dci_pdu)->TPC;
        ndi     = ((DCI0_20MHz_FDD_t *)dci_pdu)->ndi;
        mcs     = ((DCI0_20MHz_FDD_t *)dci_pdu)->mcs;
        rballoc = ((DCI0_20MHz_FDD_t *)dci_pdu)->rballoc;
        //  hopping = ((DCI0_20MHz_FDD_t *)dci_pdu)->hopping=hopping;
        //  type    = ((DCI0_20MHz_FDD_t *)dci_pdu)->type;
      }

      RIV_max = RIV_max100;
      RIV2first_rb_LUT = RIV2first_rb_LUT100;
      RIV2nb_rb_LUT = RIV2nb_rb_LUT100;

      //      printf("rb_alloc (20 MHz dci) %d\n",rballoc);
      break;

    default:
      LOG_E(PHY,"Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
      DevParam (frame_parms->N_RB_DL, 0, 0);
      break;
    }


    if (rballoc > RIV_max) {
      LOG_E(PHY,"frame %d, subframe %d, rnti %x, format %d: FATAL ERROR: generate_ue_ulsch_params_from_dci, rb_alloc[%d] > RIV_max[%d]\n",
            proc->frame_rx, subframe, rnti, dci_format,rballoc,RIV_max);
      LOG_E(PHY,"Wrong DCI0 detection, do not transmit PUSCH for HARQID: %d\n",harq_pid);
      ulsch->harq_processes[harq_pid]->subframe_scheduling_flag = 0;
      return(-1);
    }


    // indicate that this process is to be serviced in subframe n+4
    if ((rnti >= cba_rnti) && (rnti < p_rnti))
      ulsch->harq_processes[harq_pid]->subframe_cba_scheduling_flag = 1; //+=1 this indicates the number of dci / cba group: not supported in the data struct
    else
    {
        ulsch->harq_processes[harq_pid]->subframe_scheduling_flag = 1;
        //LOG_I(PHY,"[HARQ-UL harqId: %d] DCI0 ==> subframe_scheduling_flag = %d round: %d\n", harq_pid, ulsch->harq_processes[harq_pid]->subframe_scheduling_flag, ulsch->harq_processes[harq_pid]->round);

    }

    ulsch->harq_processes[harq_pid]->TPC                                   = TPC;
    ulsch->harq_processes[harq_pid]->first_rb                              = RIV2first_rb_LUT[rballoc];
    ulsch->harq_processes[harq_pid]->nb_rb                                 = RIV2nb_rb_LUT[rballoc];

    if (ue->ul_power_control_dedicated[eNB_id].accumulationEnabled == 1) {
      LOG_D(PHY,"[UE %d][PUSCH %d] Frame %d subframe %d: f_pusch (ACC) %d, adjusting by %d (TPC %d)\n",
            ue->Mod_id,harq_pid,proc->frame_rx,subframe,ulsch->f_pusch,
            delta_PUSCH_acc[ue->ulsch[eNB_id]->harq_processes[harq_pid]->TPC],
            ue->ulsch[eNB_id]->harq_processes[harq_pid]->TPC);
      ulsch->f_pusch += delta_PUSCH_acc[ue->ulsch[eNB_id]->harq_processes[harq_pid]->TPC];
    } else {
      LOG_D(PHY,"[UE %d][PUSCH %d] Frame %d subframe %d: f_pusch (ABS) %d, adjusting to %d (TPC %d)\n",
            ue->Mod_id,harq_pid,proc->frame_rx,subframe,ulsch->f_pusch,
            delta_PUSCH_abs[ue->ulsch[eNB_id]->harq_processes[harq_pid]->TPC],
            ue->ulsch[eNB_id]->harq_processes[harq_pid]->TPC);
      ulsch->f_pusch = delta_PUSCH_abs[ue->ulsch[eNB_id]->harq_processes[harq_pid]->TPC];
    }

    if (ulsch->harq_processes[harq_pid]->first_tx==1) {
      //      ulsch->harq_processes[harq_pid]->Ndi                                   = 1;
      ulsch->harq_processes[harq_pid]->first_tx=0;
      ulsch->harq_processes[harq_pid]->DCINdi= ndi;
      ulsch->harq_processes[harq_pid]->round = 0;
    } else {
      if (ulsch->harq_processes[harq_pid]->DCINdi!=ndi) { // new SDU opportunity
        //  ulsch->harq_processes[harq_pid]->Ndi = 1;
        ulsch->harq_processes[harq_pid]->DCINdi= ndi;
        ulsch->harq_processes[harq_pid]->round = 0;
      } else {
        //  ulsch->harq_processes[harq_pid]->Ndi = 0;
        //ulsch->harq_processes[harq_pid]->round++;  // This is done in phich RX

        //#ifdef DEBUG_PHICH
        //LOG_I(PHY,"[UE  %d][PUSCH %d] Frame %d subframe %d Adaptative Retrans, NDI not toggled => Nack. maxHARQ_Tx %d \n",
        //      ue->Mod_id,harq_pid,
        //      proc->frame_rx,
        //      subframe,
        //      ulsch->Mlimit);
        //#endif
/*
        if (ulsch->harq_processes[harq_pid]->round > 0) // NACK detected on phich
        {
            // ulsch->harq_processes[harq_pid]->round++; already done on phich_rx
            // ulsch->harq_processes[harq_pid] = ulsch->harq_processes[8];
            // LOG_I(PHY,"          Adaptative retransmission - copy temporary harq Process to current harq process. [harqId %d round %d] \n",harq_pid, ulsch->harq_processes[8]->round);

            if (ulsch->harq_processes[harq_pid]->round >= ulsch->Mlimit) //UE_mac_inst[eNB_id].scheduling_info.maxHARQ_Tx)
            {
                ulsch->harq_processes[harq_pid]->subframe_scheduling_flag = 0;
                ulsch->harq_processes[harq_pid]->round  = 0;
                ulsch->harq_processes[harq_pid]->status = IDLE;
                //LOG_I(PHY,"          PUSCH MAX Retransmission acheived ==> flush harq buff (%d) \n",harq_pid);
                //LOG_I(PHY,"          [HARQ-UL harqId: %d] Adaptative retransmission NACK MAX RETRANS(%d) ==> subframe_scheduling_flag = %d round: %d\n", harq_pid, UE_mac_inst[eNB_id].scheduling_info.maxHARQ_Tx, ulsch->harq_processes[harq_pid]->subframe_scheduling_flag, ulsch->harq_processes[harq_pid]->round);
            }
            else
            {
                // ulsch->harq_processes[harq_pid]->subframe_scheduling_flag = 1;
                uint8_t rv_table[4] = {0, 2, 3, 1};
                ulsch->harq_processes[harq_pid]->rvidx = rv_table[ulsch->harq_processes[harq_pid]->round&3];
                ulsch->O_RI = 0;
                ulsch->O    = 0;
                ulsch->uci_format = HLC_subband_cqi_nopmi;
                //LOG_I(PHY,"          [HARQ-UL harqId: %d] Adaptative retransmission NACK ==> subframe_scheduling_flag = %d round: %d\n", harq_pid, ulsch->harq_processes[harq_pid]->subframe_scheduling_flag,ulsch->harq_processes[harq_pid]->round);
            }
        }
*/
      }
    }

    ulsch->harq_processes[harq_pid]->n_DMRS                                = cshift;

    //printf("nb_rb %d, first_rb %d (RIV %d)\n",ulsch->harq_processes[harq_pid]->nb_rb,ulsch->harq_processes[harq_pid]->first_rb,rballoc);
    if ((rnti >= cba_rnti) && (rnti < p_rnti)) {
      // ulsch->cba_rnti[0]=rnti;
    } else {
      ulsch->rnti = rnti;
    }

    //    printf("[PHY][UE] DCI format 0: harq_pid %d nb_rb %d, rballoc %d\n",harq_pid,ulsch->harq_processes[harq_pid]->nb_rb,
    //     ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->rballoc);
    //Mapping of cyclic shift field in DCI format0 to n_DMRS2 (3GPP 36.211, Table 5.5.2.1.1-1)
    if(cshift == 0)
      ulsch->harq_processes[harq_pid]->n_DMRS2 = 0;
    else if(cshift == 1)
      ulsch->harq_processes[harq_pid]->n_DMRS2 = 6;
    else if(cshift == 2)
      ulsch->harq_processes[harq_pid]->n_DMRS2 = 3;
    else if(cshift == 3)
      ulsch->harq_processes[harq_pid]->n_DMRS2 = 4;
    else if(cshift == 4)
      ulsch->harq_processes[harq_pid]->n_DMRS2 = 2;
    else if(cshift == 5)
      ulsch->harq_processes[harq_pid]->n_DMRS2 = 8;
    else if(cshift == 6)
      ulsch->harq_processes[harq_pid]->n_DMRS2 = 10;
    else if(cshift == 7)
      ulsch->harq_processes[harq_pid]->n_DMRS2 = 9;


    //reserved for cooperative communication
    /*
      if(ulsch->n_DMRS2 == 6)
      ulsch->cooperation_flag = 2;
      else
      ulsch->cooperation_flag = 0;
    */

    if ((ulsch->harq_processes[harq_pid]->nb_rb>0) && (ulsch->harq_processes[harq_pid]->nb_rb < 25))
      ulsch->power_offset = ue_power_offsets[ulsch->harq_processes[harq_pid]->nb_rb-1];

    //    if (ulsch->harq_processes[harq_pid]->Ndi == 1)
    //    ulsch->harq_processes[harq_pid]->status = ACTIVE;


    if (cqi_req == 1) {

      if( (LTE_AntennaInfoDedicated__transmissionMode_tm3 == transmission_mode) || (LTE_AntennaInfoDedicated__transmissionMode_tm4 == transmission_mode) )
      {
          ulsch->O_RI = 1;
      }
      else
      {
          ulsch->O_RI = 0;
      }
      //ulsch->O_RI = 0; //we only support 2 antenna ports, so this is always 1 according to 3GPP 36.213 Table

      switch(transmission_mode) {
        // The aperiodic CQI reporting mode is fixed for every transmission mode instead of being configured by higher layer signaling
      case 1:
        if ((rnti >= cba_rnti) && (rnti < p_rnti)) {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_20MHz;
            break;
          }

          ulsch->uci_format                          = HLC_subband_cqi_mcs_CBA;
          ulsch->o_RI[0]                             = 0;
        } else  if(meas->rank[eNB_id] == 0) {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_20MHz;
            break;
          }

          ulsch->uci_format                          = HLC_subband_cqi_nopmi;
          ulsch->o_RI[0]                             = 0;
        } else {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_20MHz;
            break;
          }

          ulsch->uci_format                          = HLC_subband_cqi_nopmi;
          ulsch->o_RI[0]                             = 1;
        }

        break;

      case 2:
        if ((rnti >= cba_rnti) && (rnti < p_rnti)) {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_20MHz;
            break;
          }

          ulsch->uci_format                          = HLC_subband_cqi_mcs_CBA;
          ulsch->o_RI[0]                             = 0;
        } else if(meas->rank[eNB_id] == 0) {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_20MHz;
            break;
          }

          ulsch->uci_format                          = HLC_subband_cqi_nopmi;
          ulsch->o_RI[0]                             = 0;
        } else {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_20MHz;
            break;
          }

          ulsch->uci_format                          = HLC_subband_cqi_nopmi;
          ulsch->o_RI[0]                             = 1;
        }

        break;

      case 3:
        if ((rnti >= cba_rnti) && (rnti < p_rnti)) {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_20MHz;
            break;
          }

          ulsch->uci_format                          = HLC_subband_cqi_mcs_CBA;
          ulsch->o_RI[0]                             = 0;
        } else if(meas->rank[eNB_id] == 0) {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_20MHz;
            break;
          }

          ulsch->uci_format                          = HLC_subband_cqi_nopmi;
          ulsch->o_RI[0]                             = 0;
        } else {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_20MHz;
            break;
          }

          ulsch->uci_format                          = HLC_subband_cqi_nopmi;
          ulsch->o_RI[0]                             = 1;
        }

        break;

      case 4:
        if ((rnti >= cba_rnti) && (rnti < p_rnti)) {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_20MHz;
            break;
          }

          ulsch->uci_format                          = HLC_subband_cqi_mcs_CBA;
          ulsch->o_RI[0]                             = 0;
        } else if(meas->rank[eNB_id] == 0) {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_wideband_cqi_rank1_2A_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_wideband_cqi_rank1_2A_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_wideband_cqi_rank1_2A_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_wideband_cqi_rank1_2A_20MHz;
            break;
          }

          ulsch->uci_format                          = wideband_cqi_rank1_2A;
          ulsch->o_RI[0]                             = 0;
        } else {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_wideband_cqi_rank2_2A_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_wideband_cqi_rank2_2A_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_wideband_cqi_rank2_2A_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_wideband_cqi_rank2_2A_20MHz;
            break;
          }

          ulsch->uci_format                          = wideband_cqi_rank2_2A;
          ulsch->o_RI[0]                             = 1;
        }

        break;

      case 5:
        if ((rnti >= cba_rnti) && (rnti < p_rnti)) {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_20MHz;
            break;
          }

          ulsch->uci_format                          = HLC_subband_cqi_mcs_CBA;
          ulsch->o_RI[0]                             = 0;
        } else if(meas->rank[eNB_id] == 0) {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_wideband_cqi_rank1_2A_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_wideband_cqi_rank1_2A_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_wideband_cqi_rank1_2A_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_wideband_cqi_rank1_2A_20MHz;
            break;
          }

          ulsch->uci_format                          = wideband_cqi_rank1_2A;
          ulsch->o_RI[0]                             = 0;
        } else {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_wideband_cqi_rank2_2A_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_wideband_cqi_rank2_2A_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_wideband_cqi_rank2_2A_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_wideband_cqi_rank2_2A_20MHz;
            break;
          }

          ulsch->uci_format                          = wideband_cqi_rank2_2A;
          ulsch->o_RI[0]                             = 1;
        }

        break;

      case 6:
        if ((rnti >= cba_rnti) && (rnti < p_rnti)) {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_20MHz;
            break;
          }

          ulsch->uci_format                          = HLC_subband_cqi_mcs_CBA;
          ulsch->o_RI[0]                             = 0;
        } else if(meas->rank[eNB_id] == 0) {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_wideband_cqi_rank1_2A_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_wideband_cqi_rank1_2A_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_wideband_cqi_rank1_2A_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_wideband_cqi_rank1_2A_20MHz;
            break;
          }

          ulsch->uci_format                          = wideband_cqi_rank1_2A;
          ulsch->o_RI[0]                             = 0;
        } else {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_wideband_cqi_rank2_2A_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_wideband_cqi_rank2_2A_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_wideband_cqi_rank2_2A_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_wideband_cqi_rank2_2A_20MHz;
            break;
          }

          ulsch->uci_format                          = wideband_cqi_rank2_2A;
          ulsch->o_RI[0]                             = 1;
        }

        break;

      case 7:
        if ((rnti >= cba_rnti) && (rnti < p_rnti)) {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_HLC_subband_cqi_mcs_CBA_20MHz;
            break;
          }

          ulsch->uci_format                          = HLC_subband_cqi_mcs_CBA;
          ulsch->o_RI[0]                             = 0;
        } else if(meas->rank[eNB_id] == 0) {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_20MHz;
            break;
          }

          ulsch->uci_format                          = HLC_subband_cqi_nopmi;
          ulsch->o_RI[0]                             = 0;
        } else {
          switch (ue->frame_parms.N_RB_DL) {
          case 6:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_1_5MHz;
            break;

          case 25:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_5MHz;
            break;

          case 50:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_10MHz;
            break;

          case 100:
            ulsch->O                                   = sizeof_HLC_subband_cqi_nopmi_20MHz;
            break;
          }

          ulsch->uci_format                          = HLC_subband_cqi_nopmi;
          ulsch->o_RI[0]                             = 1;
        }

        break;

      default:
        LOG_E(PHY,"Incorrect Transmission Mode \n");
        break;
      }
    } else {
      ulsch->O_RI = 0;
      ulsch->O                                   = 0;
      ulsch->uci_format                          = HLC_subband_cqi_nopmi;
    }

    print_CQI(ulsch->o,ulsch->uci_format,eNB_id,ue->frame_parms.N_RB_DL);

    ulsch->bundling = 1-AckNackFBMode;

    if (frame_parms->frame_type == FDD) {
      //int dl_subframe = (subframe<4) ? (subframe+6) : (subframe-4);
      int dl_subframe = subframe;

      if (ue->dlsch[ue->current_thread_id[subframe]][eNB_id][0]->harq_ack[dl_subframe].send_harq_status>0) { // we have downlink transmission
        ulsch->harq_processes[harq_pid]->O_ACK = 1;
      } else {
        ulsch->harq_processes[harq_pid]->O_ACK = 0;
      }
      /*LOG_I(PHY,"DCI 0 Processing: dl_subframe %d send_harq_status Odd %d send_harq_status Even %d harq_pid %d O_ACK %d\n", dl_subframe,
              ue->dlsch[0][eNB_id][0]->harq_ack[dl_subframe].send_harq_status,
              ue->dlsch[1][eNB_id][0]->harq_ack[dl_subframe].send_harq_status,
              harq_pid,
              ulsch->harq_processes[harq_pid]->O_ACK);*/

    } else {
      if (ulsch->bundling)
        ulsch->harq_processes[harq_pid]->O_ACK = (dai == 3)? 0 : 1;
      else
        ulsch->harq_processes[harq_pid]->O_ACK = (dai >= 2)? 2 : (dai+1)&3; //(dai+1)&3;

      //      ulsch->harq_processes[harq_pid]->V_UL_DAI = dai+1;
    }

    dlsch[0]->harq_ack[subframe].vDAI_UL = dai+1;


    LOG_D(PHY, "[PUSCH %d] Format0 DCI %s, CQI_req=%d, cshift=%d, TPC=%d, DAI=%d, vDAI_UL[sf#%d]=%d, NDI=%d, MCS=%d, RBalloc=%d, first_rb=%d, harq_pid=%d, nb_rb=%d, subframe_scheduling_flag=%d"
            "   ulsch->bundling %d, O_ACK %d \n",
        harq_pid,
        (frame_parms->frame_type == TDD? "TDD" : "FDD"),
        cqi_req, cshift, TPC, dai, subframe, dlsch[0]->harq_ack[subframe].vDAI_UL, ndi, mcs, rballoc,
        ulsch->harq_processes[harq_pid]->first_rb, harq_pid, ulsch->harq_processes[harq_pid]->nb_rb,
        ulsch->harq_processes[harq_pid]->subframe_scheduling_flag,
        ulsch->bundling,
        ulsch->harq_processes[harq_pid]->O_ACK);

    LOG_D(PHY,"Setting beta_offset_cqi_times8 to %d, index %d\n",
	  beta_cqi[ue->pusch_config_dedicated[eNB_id].betaOffset_CQI_Index],
	  ue->pusch_config_dedicated[eNB_id].betaOffset_CQI_Index);

    ulsch->beta_offset_cqi_times8                = beta_cqi[ue->pusch_config_dedicated[eNB_id].betaOffset_CQI_Index];//18;
    ulsch->beta_offset_ri_times8                 = beta_ri[ue->pusch_config_dedicated[eNB_id].betaOffset_RI_Index];//10;
    ulsch->beta_offset_harqack_times8            = beta_ack[ue->pusch_config_dedicated[eNB_id].betaOffset_ACK_Index];//16;

    ulsch->Nsymb_pusch                             = 12-(frame_parms->Ncp<<1)-(use_srs==0?0:1);
    ulsch->srs_active                              = use_srs;

    if ((rnti >= cba_rnti) && (rnti < p_rnti))
      ulsch->harq_processes[harq_pid]->status = CBA_ACTIVE;
    else
      ulsch->harq_processes[harq_pid]->status = ACTIVE;

    ulsch->harq_processes[harq_pid]->rvidx = 0;

    //      ulsch->harq_processes[harq_pid]->calibration_flag =0;
    if (mcs < 29) {
      ulsch->harq_processes[harq_pid]->mcs = mcs;
      // ulsch->harq_processes[harq_pid]->round = 0;
    } else {
      ulsch->harq_processes[harq_pid]->rvidx = mcs - 28;
      if (ulsch->harq_processes[harq_pid]->round == 0) {
        LOG_W(PHY,"PUSCH::mcs = %d and DCI0::mcs(%d) > 28 and round == %d\n", ulsch->harq_processes[harq_pid]->mcs, mcs, ulsch->harq_processes[harq_pid]->round);
      } else {
        LOG_D(PHY,"PUSCH::mcs = %d and DCI0::mcs(%d) > 28 and round == %d\n", ulsch->harq_processes[harq_pid]->mcs, mcs, ulsch->harq_processes[harq_pid]->round);
      }
      //LOG_E(PHY,"Fatal: mcs(%d) > 28!!! and round == 0\n", mcs);
    }
    ulsch->harq_processes[harq_pid]->TBS = TBStable[get_I_TBS_UL(ulsch->harq_processes[harq_pid]->mcs)][ulsch->harq_processes[harq_pid]->nb_rb-1];

    /*
       else if (ulsch->harq_processes[harq_pid]->mcs == 29) {
       ulsch->harq_processes[harq_pid]->mcs = 4;
       ulsch->harq_processes[harq_pid]->TBS         = TBStable[get_I_TBS_UL(ulsch->harq_processes[harq_pid]->mcs)][ulsch->harq_processes[harq_pid]->nb_rb-1];
    // ulsch->harq_processes[harq_pid]->calibration_flag =1;
    // printf("Auto-Calibration (UE): mcs %d, TBS %d, nb_rb %d\n",ulsch->harq_processes[harq_pid]->mcs,ulsch->harq_processes[harq_pid]->TBS,ulsch->harq_processes[harq_pid]->nb_rb);
    }*/
    ulsch->harq_processes[harq_pid]->Msc_initial   = 12*ulsch->harq_processes[harq_pid]->nb_rb;
    ulsch->harq_processes[harq_pid]->Nsymb_initial = ulsch->Nsymb_pusch;

    // a Ndi=1 automatically acknowledges previous PUSCH transmission
    if (ue->ulsch_Msg3_active[eNB_id] == 1)
      ue->ulsch_Msg3_active[eNB_id] = 0;

    LOG_D(PHY,"[UE %d][PUSCH %d] Frame %d, subframe %d : Programming PUSCH with n_DMRS2 %d (cshift %d), nb_rb %d, first_rb %d, mcs %d, round %d, rv %d, ulsch_ue_Msg3_active %d, cqi_req %d => O %d\n",
        ue->Mod_id,harq_pid,
        proc->frame_rx,subframe,ulsch->harq_processes[harq_pid]->n_DMRS2,cshift,ulsch->harq_processes[harq_pid]->nb_rb,ulsch->harq_processes[harq_pid]->first_rb,
	  ulsch->harq_processes[harq_pid]->mcs,ulsch->harq_processes[harq_pid]->round,ulsch->harq_processes[harq_pid]->rvidx, ue->ulsch_Msg3_active[eNB_id],cqi_req,ulsch->O);

  // ulsch->n_DMRS2 = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->cshift;

#ifdef UE_DEBUG_TRACE

    LOG_D(PHY,"Format 0 DCI : ulsch (ue): AbsSubframe %d.%d\n",proc->frame_rx%1024,subframe);
    LOG_D(PHY,"Format 0 DCI : ulsch (ue): NBRB        %d\n",ulsch->harq_processes[harq_pid]->nb_rb);
    LOG_D(PHY,"Format 0 DCI :ulsch (ue): first_rb    %d\n",ulsch->harq_processes[harq_pid]->first_rb);
    LOG_D(PHY,"Format 0 DCI :ulsch (ue): rballoc     %d\n",rballoc);
    LOG_D(PHY,"Format 0 DCI :ulsch (ue): harq_pid    %d\n",harq_pid);
    LOG_D(PHY,"Format 0 DCI :ulsch (ue): first_tx       %d\n",ulsch->harq_processes[harq_pid]->first_tx);
    LOG_D(PHY,"Format 0 DCI :ulsch (ue): DCINdi       %d\n",ulsch->harq_processes[harq_pid]->DCINdi);
    LOG_D(PHY,"Format 0 DCI :ulsch (ue): round       %d\n",ulsch->harq_processes[harq_pid]->round);
    //LOG_I(PHY,"Format 0 DCI :ulsch (ue): TBS         %d\n",ulsch->harq_processes[harq_pid]->TBS);
    LOG_D(PHY,"Format 0 DCI :ulsch (ue): mcs         %d\n",ulsch->harq_processes[harq_pid]->mcs);
    //LOG_I(PHY,"Format 0 DCI :ulsch (ue): O           %d\n",ulsch->O);
    //LOG_I(PHY,"Format 0 DCI :ulsch (ue): cqiReq      %d\n",cqi_req);
    //if (frame_parms->frame_type == TDD)
    //  LOG_I(PHY,"Format 0 DCI :ulsch (ue): O_ACK/DAI   %d/%d\n",ulsch->harq_processes[harq_pid]->O_ACK,dai);
    //else
    //  LOG_I(PHY,"Format 0 DCI :ulsch (ue): O_ACK       %d\n",ulsch->harq_processes[harq_pid]->O_ACK);

    LOG_D(PHY,"Format 0 DCI :ulsch (ue): Nsymb_pusch   %d\n",ulsch->Nsymb_pusch);
    LOG_D(PHY,"Format 0 DCI :ulsch (ue): cshift        %d\n",ulsch->harq_processes[harq_pid]->n_DMRS2);
    LOG_D(PHY,"Format 0 DCI :ulsch (ue): phich status  %d\n",ulsch->harq_processes[harq_pid]->status);
#else
    UNUSED_VARIABLE(dai);
#endif
    return(0);
  } else {
    LOG_E(PHY,"frame %d, subframe %d: FATAL ERROR, generate_ue_ulsch_params_from_dci, Illegal dci_format %d\n",
          proc->frame_rx, subframe,dci_format);
    return(-1);
  }

}

/*
int generate_eNB_ulsch_params_from_dci(PHY_VARS_eNB *eNB,
                                       L1_rxtx_proc_t *proc,
                                       void *dci_pdu,
                                       uint16_t rnti,
                                       DCI_format_t dci_format,
                                       uint8_t UE_id,
                                       uint16_t si_rnti,
                                       uint16_t ra_rnti,
                                       uint16_t p_rnti,
                                       uint16_t cba_rnti,
                                       uint8_t use_srs)
{

  uint8_t harq_pid;
  uint32_t rb_alloc;
  uint8_t transmission_mode=eNB->transmission_mode[UE_id];
  ANFBmode_t AckNackFBMode = eNB->pucch_config_dedicated[UE_id].tdd_AckNackFeedbackMode;
  LTE_eNB_ULSCH_t *ulsch=eNB->ulsch[UE_id];
  LTE_DL_FRAME_PARMS *frame_parms = &eNB->frame_parms;
  int subframe = proc->subframe_tx;

  uint32_t cqi_req = 0;
  uint32_t dai = 0;
  uint32_t cshift = 0;
  uint32_t TPC = 0;
  uint32_t mcs = 0;
  uint32_t rballoc = UINT32_MAX;
  uint32_t RIV_max = 0;
  //  uint32_t hopping;
  //  uint32_t type;

#ifdef DEBUG_DCI
  printf("filling eNB ulsch params for rnti %x, dci format %d, dci %x, subframe %d\n",
        rnti,dci_format,*(uint32_t*)dci_pdu,subframe);
#endif

  if (dci_format == format0) {

    harq_pid = subframe2harq_pid(frame_parms,
                                 pdcch_alloc2ul_frame(frame_parms,
                                                      proc->frame_tx,
                                                      subframe),
                                 pdcch_alloc2ul_subframe(frame_parms,subframe));
    switch (frame_parms->N_RB_DL) {
    case 6:
      if (frame_parms->frame_type == TDD) {
        cqi_req = ((DCI0_1_5MHz_TDD_1_6_t *)dci_pdu)->cqi_req;
        dai     = ((DCI0_1_5MHz_TDD_1_6_t *)dci_pdu)->dai;
        cshift  = ((DCI0_1_5MHz_TDD_1_6_t *)dci_pdu)->cshift;
        TPC     = ((DCI0_1_5MHz_TDD_1_6_t *)dci_pdu)->TPC;
        mcs     = ((DCI0_1_5MHz_TDD_1_6_t *)dci_pdu)->mcs;
        rballoc = ((DCI0_1_5MHz_TDD_1_6_t *)dci_pdu)->rballoc;
        //  hopping = ((DCI0_1_5MHz_TDD_1_6_t *)dci_pdu)->hopping=hopping;
        //  type    = ((DCI0_1_5MHz_TDD_1_6_t *)dci_pdu)->type;
      } else {
        cqi_req = ((DCI0_1_5MHz_FDD_t *)dci_pdu)->cqi_req;
        cshift  = ((DCI0_1_5MHz_FDD_t *)dci_pdu)->cshift;
        TPC     = ((DCI0_1_5MHz_FDD_t *)dci_pdu)->TPC;
        mcs     = ((DCI0_1_5MHz_FDD_t *)dci_pdu)->mcs;
        rballoc = ((DCI0_1_5MHz_FDD_t *)dci_pdu)->rballoc;
        //  hopping = ((DCI0_1_5MHz_FDD_t *)dci_pdu)->hopping=hopping;
        //  type    = ((DCI0_1_5MHz_FDD_t *)dci_pdu)->type;
      }
      
      RIV_max = RIV_max6;
      ulsch->harq_processes[harq_pid]->first_rb                              = RIV2first_rb_LUT6[rballoc];
      ulsch->harq_processes[harq_pid]->nb_rb                                 = RIV2nb_rb_LUT6[rballoc];

      break;

    case 25:
      if (frame_parms->frame_type == TDD) {
        cqi_req = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->cqi_req;
        dai     = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->dai;
        cshift  = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->cshift;
        TPC     = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->TPC;
        mcs     = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->mcs;
        rballoc = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->rballoc;
        //  hopping = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->hopping;
        //  type    = ((DCI0_5MHz_TDD_1_6_t *)dci_pdu)->type;
      } else {
        cqi_req = ((DCI0_5MHz_FDD_t *)dci_pdu)->cqi_req;
        cshift  = ((DCI0_5MHz_FDD_t *)dci_pdu)->cshift;
        TPC     = ((DCI0_5MHz_FDD_t *)dci_pdu)->TPC;
        mcs     = ((DCI0_5MHz_FDD_t *)dci_pdu)->mcs;
        rballoc = ((DCI0_5MHz_FDD_t *)dci_pdu)->rballoc;
        //  hopping = ((DCI0_5MHz_FDD_t *)dci_pdu)->hopping;
        //  type    = ((DCI0_5MHz_FDD_t *)dci_pdu)->type;
      }

      RIV_max = RIV_max25;
      ulsch->harq_processes[harq_pid]->first_rb                              = RIV2first_rb_LUT25[rballoc];
      ulsch->harq_processes[harq_pid]->nb_rb                                 = RIV2nb_rb_LUT25[rballoc];

      break;

    case 50:
      if (frame_parms->frame_type == TDD) {
        cqi_req = ((DCI0_10MHz_TDD_1_6_t *)dci_pdu)->cqi_req;
        dai     = ((DCI0_10MHz_TDD_1_6_t *)dci_pdu)->dai;
        cshift  = ((DCI0_10MHz_TDD_1_6_t *)dci_pdu)->cshift;
        TPC     = ((DCI0_10MHz_TDD_1_6_t *)dci_pdu)->TPC;
        mcs     = ((DCI0_10MHz_TDD_1_6_t *)dci_pdu)->mcs;
        rballoc = ((DCI0_10MHz_TDD_1_6_t *)dci_pdu)->rballoc;
        //  hopping = ((DCI0_10MHz_TDD_1_6_t *)dci_pdu)->hopping;
        //  type    = ((DCI0_10MHz_TDD_1_6_t *)dci_pdu)->type;
      } else {
        cqi_req = ((DCI0_10MHz_FDD_t *)dci_pdu)->cqi_req;
        cshift  = ((DCI0_10MHz_FDD_t *)dci_pdu)->cshift;
        TPC     = ((DCI0_10MHz_FDD_t *)dci_pdu)->TPC;
        mcs     = ((DCI0_10MHz_FDD_t *)dci_pdu)->mcs;
        rballoc = ((DCI0_10MHz_FDD_t *)dci_pdu)->rballoc;
        //  hopping = ((DCI0_10MHz_FDD_t *)dci_pdu)->hopping;
        //  type    = ((DCI0_10MHz_FDD_t *)dci_pdu)->type;
      }

      RIV_max = RIV_max50;
      ulsch->harq_processes[harq_pid]->first_rb                              = RIV2first_rb_LUT50[rballoc];
      ulsch->harq_processes[harq_pid]->nb_rb                                 = RIV2nb_rb_LUT50[rballoc];

      break;

    case 100:
      if (frame_parms->frame_type == TDD) {
        cqi_req = ((DCI0_20MHz_TDD_1_6_t *)dci_pdu)->cqi_req;
        dai     = ((DCI0_20MHz_TDD_1_6_t *)dci_pdu)->dai;
        cshift  = ((DCI0_20MHz_TDD_1_6_t *)dci_pdu)->cshift;
        TPC     = ((DCI0_20MHz_TDD_1_6_t *)dci_pdu)->TPC;
        mcs     = ((DCI0_20MHz_TDD_1_6_t *)dci_pdu)->mcs;
        rballoc = ((DCI0_20MHz_TDD_1_6_t *)dci_pdu)->rballoc;
        //  hopping = ((DCI0_20MHz_TDD_1_6_t *)dci_pdu)->hopping;
        //  type    = ((DCI0_20MHz_TDD_1_6_t *)dci_pdu)->type;
      } else {
        cqi_req = ((DCI0_20MHz_FDD_t *)dci_pdu)->cqi_req;
        cshift  = ((DCI0_20MHz_FDD_t *)dci_pdu)->cshift;
        TPC     = ((DCI0_20MHz_FDD_t *)dci_pdu)->TPC;
        mcs     = ((DCI0_20MHz_FDD_t *)dci_pdu)->mcs;
        rballoc = ((DCI0_20MHz_FDD_t *)dci_pdu)->rballoc;
        //  hopping = ((DCI0_20MHz_FDD_t *)dci_pdu)->hopping;
        //  type    = ((DCI0_20MHz_FDD_t *)dci_pdu)->type;
      }

      RIV_max = RIV_max100;
      ulsch->harq_processes[harq_pid]->first_rb                              = RIV2first_rb_LUT100[rballoc];
      ulsch->harq_processes[harq_pid]->nb_rb                                 = RIV2nb_rb_LUT100[rballoc];

      //printf("eNB: rb_alloc (20 MHz dci) %d\n",rballoc);
      break;

    default:
      LOG_E(PHY,"Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
      DevParam (frame_parms->N_RB_DL, 0, 0);
      break;
    }


    rb_alloc = rballoc;
    AssertFatal(rb_alloc>RIV_max,
		"Format 0: rb_alloc (%d) > RIV_max (%d)\n",rb_alloc,RIV_max);
#ifdef DEBUG_DCI
    printf("generate_eNB_ulsch_params_from_dci: subframe %d, rnti %x,harq_pid %d,cqi_req %d\n",subframe,rnti,harq_pid,cqi_req);
#endif

    ulsch->harq_processes[harq_pid]->dci_alloc                             = 1;
    ulsch->harq_processes[harq_pid]->rar_alloc                             = 0;
    ulsch->harq_processes[harq_pid]->TPC                                   = TPC;
    ulsch->harq_processes[harq_pid]->n_DMRS                                = cshift;


    if (cqi_req == 1) {
      // 36.213 7.2.1 (release 10) says:
      // "RI is only reported for transmission modes 3 and 4,
      // as well as transmission modes 8 and 9 with PMI/RI reporting"
      // This is for aperiodic reporting.
      // TODO: deal with TM 8&9 correctly when they are implemented.
      // TODO: deal with periodic reporting if we implement it.
      //
      if (transmission_mode == 3 || transmission_mode == 4)
        ulsch->harq_processes[harq_pid]->O_RI = 1; //we only support 2 antenna ports, so this is always 1 according to 3GPP 36.213 Table
      else
        ulsch->harq_processes[harq_pid]->O_RI = 0;

      switch(transmission_mode) {
        // The aperiodic CQI reporting mode is fixed for every transmission mode instead of being configured by higher layer signaling
      case 1:
        if ((rnti >= cba_rnti) && (rnti < p_rnti)) {
          ulsch->harq_processes[harq_pid]->Or2                                   = 0;

          switch (frame_parms->N_RB_DL) {
          case 6:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_1_5MHz;
            break;

          case 25:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_5MHz;
            break;

          case 50:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_10MHz;
            break;

          case 100:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_20MHz;
            break;
          }

          ulsch->harq_processes[harq_pid]->uci_format                            = HLC_subband_cqi_mcs_CBA;
        } else {
          ulsch->harq_processes[harq_pid]->Or2                                   = 0;

          switch (frame_parms->N_RB_DL) {
          case 6:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_nopmi_1_5MHz;
            break;

          case 25:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_nopmi_5MHz;
            break;

          case 50:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_nopmi_10MHz;
            break;

          case 100:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_nopmi_20MHz;
            break;
          }

          ulsch->harq_processes[harq_pid]->uci_format                            = HLC_subband_cqi_nopmi;
        }

        break;

      case 2:
        if ((rnti >= cba_rnti) && (rnti < p_rnti)) {
          ulsch->harq_processes[harq_pid]->Or2                                   = 0;

          switch (frame_parms->N_RB_DL) {
          case 6:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_1_5MHz;
            break;

          case 25:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_5MHz;
            break;

          case 50:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_10MHz;
            break;

          case 100:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_20MHz;
            break;
          }

          ulsch->harq_processes[harq_pid]->uci_format                            = HLC_subband_cqi_mcs_CBA;
        } else {
          ulsch->harq_processes[harq_pid]->Or2                                   = 0;

          switch (frame_parms->N_RB_DL) {
          case 6:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_nopmi_1_5MHz;
            break;

          case 25:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_nopmi_5MHz;
            break;

          case 50:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_nopmi_10MHz;
            break;

          case 100:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_nopmi_20MHz;
            break;
          }

          ulsch->harq_processes[harq_pid]->uci_format                            = HLC_subband_cqi_nopmi;
        }

        break;

      case 3:
        if ((rnti >= cba_rnti) && (rnti < p_rnti)) {
          ulsch->harq_processes[harq_pid]->Or2                                   = 0;

          switch (frame_parms->N_RB_DL) {
          case 6:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_1_5MHz;
            break;

          case 25:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_5MHz;
            break;

          case 50:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_10MHz;
            break;

          case 100:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_20MHz;
            break;
          }

          ulsch->harq_processes[harq_pid]->uci_format                            = HLC_subband_cqi_mcs_CBA;
        } else {
          ulsch->harq_processes[harq_pid]->Or2                                   = 0;

          switch (frame_parms->N_RB_DL) {
          case 6:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_nopmi_1_5MHz;
            break;

          case 25:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_nopmi_5MHz;
            break;

          case 50:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_nopmi_10MHz;
            break;

          case 100:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_nopmi_20MHz;
            break;
          }

          ulsch->harq_processes[harq_pid]->uci_format                            = HLC_subband_cqi_nopmi;
        }

        break;

      case 4:
        if ((rnti >= cba_rnti) && (rnti < p_rnti)) {
          ulsch->harq_processes[harq_pid]->Or2                                   = 0;

          switch (frame_parms->N_RB_DL) {
          case 6:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_1_5MHz;
            break;

          case 25:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_5MHz;
            break;

          case 50:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_10MHz;
            break;

          case 100:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_20MHz;
            break;
          }

          ulsch->harq_processes[harq_pid]->uci_format                            = HLC_subband_cqi_mcs_CBA;
        } else {
          switch (frame_parms->N_RB_DL) {
          case 6:
            ulsch->harq_processes[harq_pid]->Or2                                 = sizeof_wideband_cqi_rank2_2A_1_5MHz;
            ulsch->harq_processes[harq_pid]->Or1                                 = sizeof_wideband_cqi_rank1_2A_1_5MHz;
            break;

          case 25:
            ulsch->harq_processes[harq_pid]->Or2                                 = sizeof_wideband_cqi_rank2_2A_5MHz;
            ulsch->harq_processes[harq_pid]->Or1                                 = sizeof_wideband_cqi_rank1_2A_5MHz;
            break;

          case 50:
            ulsch->harq_processes[harq_pid]->Or2                                 = sizeof_wideband_cqi_rank2_2A_10MHz;
            ulsch->harq_processes[harq_pid]->Or1                                 = sizeof_wideband_cqi_rank1_2A_10MHz;
            break;

          case 100:
            ulsch->harq_processes[harq_pid]->Or2                                 = sizeof_wideband_cqi_rank2_2A_20MHz;
            ulsch->harq_processes[harq_pid]->Or1                                 = sizeof_wideband_cqi_rank1_2A_20MHz;
            break;

          }

          ulsch->harq_processes[harq_pid]->uci_format                          = wideband_cqi_rank1_2A;
        }

        break;

      case 5:
        if ((rnti >= cba_rnti) && (rnti < p_rnti)) {
          ulsch->harq_processes[harq_pid]->Or2                                   = 0;

          switch (frame_parms->N_RB_DL) {
          case 6:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_1_5MHz;
            break;

          case 25:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_5MHz;
            break;

          case 50:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_10MHz;
            break;

          case 100:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_20MHz;
            break;
          }

          ulsch->harq_processes[harq_pid]->uci_format                            = HLC_subband_cqi_mcs_CBA;
        } else {
          switch (frame_parms->N_RB_DL) {
          case 6:
            ulsch->harq_processes[harq_pid]->Or2                                 = sizeof_wideband_cqi_rank2_2A_1_5MHz;
            ulsch->harq_processes[harq_pid]->Or1                                 = sizeof_wideband_cqi_rank1_2A_1_5MHz;
            break;

          case 25:
            ulsch->harq_processes[harq_pid]->Or2                                 = sizeof_wideband_cqi_rank2_2A_5MHz;
            ulsch->harq_processes[harq_pid]->Or1                                 = sizeof_wideband_cqi_rank1_2A_5MHz;
            break;

          case 50:
            ulsch->harq_processes[harq_pid]->Or2                                 = sizeof_wideband_cqi_rank2_2A_10MHz;
            ulsch->harq_processes[harq_pid]->Or1                                 = sizeof_wideband_cqi_rank1_2A_10MHz;
            break;

          case 100:
            ulsch->harq_processes[harq_pid]->Or2                                 = sizeof_wideband_cqi_rank2_2A_20MHz;
            ulsch->harq_processes[harq_pid]->Or1                                 = sizeof_wideband_cqi_rank1_2A_20MHz;
            break;
          }

          ulsch->harq_processes[harq_pid]->uci_format                          = wideband_cqi_rank1_2A;
        }

        break;

      case 6:
        if ((rnti >= cba_rnti) && (rnti < p_rnti)) {
          ulsch->harq_processes[harq_pid]->Or2                                   = 0;

          switch (frame_parms->N_RB_DL) {
          case 6:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_1_5MHz;
            break;

          case 25:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_5MHz;
            break;

          case 50:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_10MHz;
            break;

          case 100:
            ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_mcs_CBA_20MHz;
            break;
          }

          ulsch->harq_processes[harq_pid]->uci_format                            = HLC_subband_cqi_mcs_CBA;
        } else {
          switch (frame_parms->N_RB_DL) {
          case 6:
            ulsch->harq_processes[harq_pid]->Or2                                 = sizeof_wideband_cqi_rank2_2A_1_5MHz;
            ulsch->harq_processes[harq_pid]->Or1                                 = sizeof_wideband_cqi_rank1_2A_1_5MHz;
            break;

          case 25:
            ulsch->harq_processes[harq_pid]->Or2                                 = sizeof_wideband_cqi_rank2_2A_5MHz;
            ulsch->harq_processes[harq_pid]->Or1                                 = sizeof_wideband_cqi_rank1_2A_5MHz;
            break;

          case 50:
            ulsch->harq_processes[harq_pid]->Or2                                 = sizeof_wideband_cqi_rank2_2A_10MHz;
            ulsch->harq_processes[harq_pid]->Or1                                 = sizeof_wideband_cqi_rank1_2A_10MHz;
            break;

          case 100:
            ulsch->harq_processes[harq_pid]->Or2                                 = sizeof_wideband_cqi_rank2_2A_20MHz;
            ulsch->harq_processes[harq_pid]->Or1                                 = sizeof_wideband_cqi_rank1_2A_20MHz;
            break;
          }

          ulsch->harq_processes[harq_pid]->uci_format                          = wideband_cqi_rank1_2A;
        }

        break;

      case 7:
        ulsch->harq_processes[harq_pid]->Or2                                   = 0;

        switch (frame_parms->N_RB_DL) {
        case 6:
          ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_nopmi_1_5MHz;
          break;

        case 25:
          ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_nopmi_5MHz;
          break;

        case 50:
          ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_nopmi_10MHz;
          break;

        case 100:
          ulsch->harq_processes[harq_pid]->Or1                                   = sizeof_HLC_subband_cqi_nopmi_20MHz;
          break;
        }

        ulsch->harq_processes[harq_pid]->uci_format                            = HLC_subband_cqi_nopmi;
        break;

      default:
        LOG_E(PHY,"Incorrect Transmission Mode \n");
        break;
      }
    } else {
      ulsch->harq_processes[harq_pid]->O_RI = 0;
      ulsch->harq_processes[harq_pid]->Or2                                   = 0;
      ulsch->harq_processes[harq_pid]->Or1                                   = 0;
      ulsch->harq_processes[harq_pid]->uci_format                            = HLC_subband_cqi_nopmi;
    }

    ulsch->bundling = 1-AckNackFBMode;

    if (frame_parms->frame_type == FDD) {
      int dl_subframe = (subframe<4) ? (subframe+6) : (subframe-4);

      if (eNB->dlsch[UE_id][0]->subframe_tx[dl_subframe]>0) { // we have downlink transmission
        ulsch->harq_processes[harq_pid]->O_ACK = 1;
      } else {
        ulsch->harq_processes[harq_pid]->O_ACK = 0;
      }
    } else {
      if (ulsch->bundling)
        ulsch->harq_processes[harq_pid]->O_ACK = (dai == 3)? 0 : 1;
      else
        ulsch->harq_processes[harq_pid]->O_ACK = (dai+1)&3;

      ulsch->harq_processes[harq_pid]->V_UL_DAI = dai+1;
    }

    ulsch->beta_offset_cqi_times8                = beta_cqi[eNB->pusch_config_dedicated[UE_id].betaOffset_CQI_Index];//18;
    ulsch->beta_offset_ri_times8                 = beta_ri[eNB->pusch_config_dedicated[UE_id].betaOffset_RI_Index];//10;
    ulsch->beta_offset_harqack_times8            = beta_ack[eNB->pusch_config_dedicated[UE_id].betaOffset_ACK_Index];//16;

    ulsch->harq_processes[harq_pid]->Nsymb_pusch                             = 12-(frame_parms->Ncp<<1)-(use_srs==0?0:1);
    ulsch->harq_processes[harq_pid]->srs_active                            = use_srs;

    //Mapping of cyclic shift field in DCI format0 to n_DMRS2 (3GPP 36.211, Table 5.5.2.1.1-1)
    if(cshift == 0)
      ulsch->harq_processes[harq_pid]->n_DMRS2 = 0;
    else if(cshift == 1)
      ulsch->harq_processes[harq_pid]->n_DMRS2 = 6;
    else if(cshift == 2)
      ulsch->harq_processes[harq_pid]->n_DMRS2 = 3;
    else if(cshift == 3)
      ulsch->harq_processes[harq_pid]->n_DMRS2 = 4;
    else if(cshift == 4)
      ulsch->harq_processes[harq_pid]->n_DMRS2 = 2;
    else if(cshift == 5)
      ulsch->harq_processes[harq_pid]->n_DMRS2 = 8;
    else if(cshift == 6)
      ulsch->harq_processes[harq_pid]->n_DMRS2 = 10;
    else if(cshift == 7)
      ulsch->harq_processes[harq_pid]->n_DMRS2 = 9;


    LOG_D(PHY,"[eNB %d][PUSCH %d] Frame %d, subframe %d : Programming PUSCH with n_DMRS2 %d (cshift %d)\n",
          eNB->Mod_id,harq_pid,proc->frame_tx,subframe,ulsch->harq_processes[harq_pid]->n_DMRS2,cshift);



    if (ulsch->harq_processes[harq_pid]->round == 0) {
      if ((rnti >= cba_rnti) && (rnti < p_rnti))
        ulsch->harq_processes[harq_pid]->status = CBA_ACTIVE;
      else
        ulsch->harq_processes[harq_pid]->status = ACTIVE;

      ulsch->harq_processes[harq_pid]->rvidx = 0;
      ulsch->harq_processes[harq_pid]->mcs         = mcs;
      //      ulsch->harq_processes[harq_pid]->calibration_flag = 0;
      //if (ulsch->harq_processes[harq_pid]->mcs)
      //
      //if (ulsch->harq_processes[harq_pid]->mcs == 29) {
      //ulsch->harq_processes[harq_pid]->mcs = 4;
      // ulsch->harq_processes[harq_pid]->calibration_flag = 1;
      // printf("Auto-Calibration (eNB): mcs %d, nb_rb %d\n",ulsch->harq_processes[harq_pid]->mcs,ulsch->harq_processes[harq_pid]->nb_rb);
      //}
      
      ulsch->harq_processes[harq_pid]->TBS         = TBStable[get_I_TBS_UL(ulsch->harq_processes[harq_pid]->mcs)][ulsch->harq_processes[harq_pid]->nb_rb-1];

      ulsch->harq_processes[harq_pid]->Msc_initial   = 12*ulsch->harq_processes[harq_pid]->nb_rb;
      ulsch->harq_processes[harq_pid]->Nsymb_initial = ulsch->harq_processes[harq_pid]->Nsymb_pusch;
      ulsch->harq_processes[harq_pid]->round = 0;
    } else {
      if (mcs>28)
        ulsch->harq_processes[harq_pid]->rvidx = mcs - 28;
      else {
        ulsch->harq_processes[harq_pid]->rvidx = 0;
        ulsch->harq_processes[harq_pid]->mcs = mcs;
      }

      //      ulsch->harq_processes[harq_pid]->round++;
    }

    if ((rnti >= cba_rnti) && (rnti < p_rnti)) {
      ulsch->cba_rnti[0] = rnti;
    } else {
      ulsch->rnti = rnti;
    }

    //ulsch->n_DMRS2 = cshift;

#ifdef DEBUG_DCI
    printf("ulsch (eNB): NBRB          %d\n",ulsch->harq_processes[harq_pid]->nb_rb);
    printf("ulsch (eNB): first_rb      %d\n",ulsch->harq_processes[harq_pid]->first_rb);
    printf("ulsch (eNB): harq_pid      %d\n",harq_pid);
    printf("ulsch (eNB): round         %d\n",ulsch->harq_processes[harq_pid]->round);
    printf("ulsch (eNB): TBS           %d\n",ulsch->harq_processes[harq_pid]->TBS);
    printf("ulsch (eNB): mcs           %d\n",ulsch->harq_processes[harq_pid]->mcs);
    printf("ulsch (eNB): Or1           %d\n",ulsch->harq_processes[harq_pid]->Or1);
    printf("ulsch (eNB): Nsymb_pusch   %d\n",ulsch->harq_processes[harq_pid]->Nsymb_pusch);
    printf("ulsch (eNB): cshift        %d\n",ulsch->harq_processes[harq_pid]->n_DMRS2);
#else
    UNUSED_VARIABLE(dai);
#endif
    return(0);
  } else {
    LOG_E(PHY,"generate_eNB_ulsch_params_from_dci, Illegal dci_format %d\n",dci_format);
    return(-1);
  }

}
*/

double sinr_eff_cqi_calc(PHY_VARS_UE *ue, uint8_t eNB_id, uint8_t subframe)
{
  uint8_t transmission_mode = ue->transmission_mode[eNB_id];
  PHY_MEASUREMENTS *meas = &ue->measurements;
  LTE_DL_FRAME_PARMS *frame_parms =  &ue->frame_parms;
  int32_t **dl_channel_est = ue->common_vars.common_vars_rx_data_per_thread[ue->current_thread_id[subframe]].dl_ch_estimates[eNB_id];
  double *s_dB;
  s_dB = ue->sinr_CQI_dB;
  //  LTE_UE_ULSCH_t *ulsch  = ue->ulsch[eNB_id];
  //for the calculation of SINR_eff for CQI calculation
  int count,a_rx,a_tx;
  double abs_channel=0;
  double channelx=0;
  double channely=0;
  double channelx_i=0;
  double channely_i=0;
  uint16_t q = quantize_subband_pmi(meas,eNB_id,7);
  uint8_t qq;

  switch(transmission_mode) {
  case 1:
    for (count=0; count<frame_parms->N_RB_DL*12; count++) {
      for(a_tx=0; a_tx<frame_parms->nb_antenna_ports_eNB; a_tx++) {
        for (a_rx=0; a_rx<frame_parms->nb_antennas_rx; a_rx++) {
          s_dB[count] = 10*log10(pow(((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2],2) + pow(((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2],
                                 2)) - meas->n0_power_avg_dB;
        }
      }
    }

    break;

  case 2:
    for (count=0; count<frame_parms->N_RB_DL*12; count++) {
      abs_channel=0;

      for(a_tx=0; a_tx<frame_parms->nb_antenna_ports_eNB; a_tx++) {
        for (a_rx=0; a_rx<frame_parms->nb_antennas_rx; a_rx++) {
          abs_channel += (pow(((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2],2) + pow(((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2],2));
        }
      }

      s_dB[count] = 10*log10(abs_channel/2) - meas->n0_power_avg_dB;
    }

    break;

  case 5:
    for (count=0; count<frame_parms->N_RB_DL*12; count++) {
      channelx=0;
      channely=0;
      channelx_i=0;
      channely_i=0;
      qq = (q>>(((count/12)>>2)<<1))&3;

      //printf("pmi_alloc %d: rb %d, pmi %d\n",q,count/12,qq);
      for(a_tx=0; a_tx<frame_parms->nb_antenna_ports_eNB; a_tx++) {
        for (a_rx=0; a_rx<frame_parms->nb_antennas_rx; a_rx++) {
          switch(qq) {
          case 0:
            if (channelx==0 || channely==0) {
              channelx = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
              channelx_i = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely_i = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
            } else {
              channelx += ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely += ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
              channelx_i -= ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely_i -= ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
            }

            break;

          case 1:
            if (channelx==0 || channely==0) {
              channelx = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
              channelx_i = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely_i = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
            } else {
              channelx -= ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely -= ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
              channelx_i += ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely_i += ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
            }

            break;

          case 2:
            if (channelx==0 || channely==0) {
              channelx = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
              channelx_i = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely_i = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
            } else {
              channelx -= ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
              channely += ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channelx_i += ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
              channely_i -= ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
            }

            break;

          case 3:
            if (channelx==0 || channely==0) {
              channelx = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
              channelx_i = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely_i = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
            } else {
              channelx += ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
              channely -= ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channelx_i -= ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
              channely_i += ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
            }

            break;

          default:
            printf("Problem in SINR Calculation for TM5 \n");
            break;
          }//switch(qq)
        }//a_rx
      }//a_tx

      s_dB[count] =  10 * log10 ((pow(channelx,2) + pow(channely,2))/2) - 10 * log10 ((pow(channelx_i,2) + pow(channely_i,2))/2) - meas->n0_power_avg_dB;
    }//count

    break;

  case 6:
    for (count=0; count<frame_parms->N_RB_DL*12; count++) {
      channelx=0;
      channely=0;
      qq = (q>>(((count/12)>>2)<<1))&3;

      //printf("pmi_alloc %d: rb %d, pmi %d\n",q,count/12,qq);
      for(a_tx=0; a_tx<frame_parms->nb_antenna_ports_eNB; a_tx++) {
        for (a_rx=0; a_rx<frame_parms->nb_antennas_rx; a_rx++) {
          switch(qq) {
          case 0:
            if (channelx==0 || channely==0) {
              channelx = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
            } else {
              channelx += ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely += ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
            }

            break;

          case 1:
            if (channelx==0 || channely==0) {
              channelx = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
            } else {
              channelx -= ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely -= ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
            }

            break;

          case 2:
            if (channelx==0 || channely==0) {
              channelx = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
            } else {
              channelx -= ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
              channely += ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
            }

            break;

          case 3:
            if (channelx==0 || channely==0) {
              channelx = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
              channely = ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
            } else {
              channelx += ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+1+(LTE_CE_FILTER_LENGTH)*2];
              channely -= ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count+(LTE_CE_FILTER_LENGTH)*2];
            }

            break;

          default:
            printf("Problem in SINR Calculation for TM6 \n");
            break;
          }//switch(qq)
        }//a_rx
      }//a_tx

      s_dB[count] =  10 * log10 ((pow(channelx,2) + pow(channely,2))/2) - meas->n0_power_avg_dB;
    }//count

    break;

  default:
    printf("Problem in SINR Calculation for CQI \n");
    break;
  }

  int ii;
  double sinr_eff = 0;
  double sinr_eff_qpsk=0;
  double sinr_eff_qam16=0;
  double sinr_eff_qam64=0;
  double x = 0;
  double I_qpsk=0;
  double I_qam16=0;
  double I_qam64=0;
  double I_qpsk_avg=0;
  double I_qam16_avg=0;
  double I_qam64_avg=0;
  double qpsk_max=12.2;
  double qam16_max=19.2;
  double qam64_max=25.2;
  double sinr_min = -20;
  int offset=0;


  for (offset = 0; offset <= 24; offset++) {
    for(ii=0; ii<12; ii++) {
      //x is the sinr_dB in dB
      x = s_dB[(offset*12)+ii];

      if(x<sinr_min) {
        I_qpsk +=0;
        I_qam16 +=0;
        I_qam64 +=0;
      } else {
        if(x>qpsk_max)
          I_qpsk += 1;
        else
          I_qpsk += (q_qpsk[0]*pow(x,7) + q_qpsk[1]*pow(x,6) + q_qpsk[2]*pow(x,5) + q_qpsk[3]*pow(x,4) + q_qpsk[4]*pow(x,3) + q_qpsk[5]*pow(x,2) + q_qpsk[6]*x + q_qpsk[7]);

        if(x>qam16_max)
          I_qam16 += 1;
        else
          I_qam16 += (q_qam16[0]*pow(x,7) + q_qam16[1]*pow(x,6) + q_qam16[2]*pow(x,5) + q_qam16[3]*pow(x,4) + q_qam16[4]*pow(x,3) + q_qam16[5]*pow(x,2) + q_qam16[6]*x + q_qam16[7]);

        if(x>qam64_max)
          I_qam64 += 1;
        else
          I_qam64 += (q_qam64[0]*pow(x,7) + q_qam64[1]*pow(x,6) + q_qam64[2]*pow(x,5) + q_qam64[3]*pow(x,4) + q_qam64[4]*pow(x,3) + q_qam64[5]*pow(x,2) + q_qam64[6]*x + q_qam64[7]);

      }
    }
  }

  // averaging of accumulated MI
  I_qpsk_avg = I_qpsk/(12*frame_parms->N_RB_DL);
  I_qam16_avg = I_qam16/(12*frame_parms->N_RB_DL);
  I_qam64_avg = I_qam64/(12*frame_parms->N_RB_DL);

  // I->SINR_effective Mapping

  sinr_eff_qpsk = (p_qpsk[0]*pow(I_qpsk_avg,7) + p_qpsk[1]*pow(I_qpsk_avg,6) + p_qpsk[2]*pow(I_qpsk_avg,5) + p_qpsk[3]*pow(I_qpsk_avg,4) + p_qpsk[4]*pow(I_qpsk_avg,3) + p_qpsk[5]*pow(I_qpsk_avg,
                   2) + p_qpsk[6]*I_qpsk_avg + p_qpsk[7]);

  sinr_eff_qam16 = (p_qam16[0]*pow(I_qam16_avg,7) + p_qam16[1]*pow(I_qam16_avg,6) + p_qam16[2]*pow(I_qam16_avg,5) + p_qam16[3]*pow(I_qam16_avg,4) + p_qam16[4]*pow(I_qam16_avg,
                    3) + p_qam16[5]*pow(I_qam16_avg,2) + p_qam16[6]*I_qam16_avg + p_qam16[7]);

  sinr_eff_qam64 = (p_qam64[0]*pow(I_qam64_avg,7) + p_qam64[1]*pow(I_qam64_avg,6) + p_qam64[2]*pow(I_qam64_avg,5) + p_qam64[3]*pow(I_qam64_avg,4) + p_qam64[4]*pow(I_qam64_avg,
                    3) + p_qam64[5]*pow(I_qam64_avg,2) + p_qam64[6]*I_qam64_avg + p_qam64[7]);
  sinr_eff = cmax3(sinr_eff_qpsk,sinr_eff_qam16,sinr_eff_qam64);

  //printf("SINR_Eff = %e\n",sinr_eff);

  return(sinr_eff);
}
//



#ifdef DEBUG_DLSCH_TOOLS
main()
{

  int i;
  uint8_t rah;
  uint32_t rballoc;

  generate_RIV_tables();

  rah = 0;
  rballoc = 0x1fff;
  printf("rballoc 0 %x => %x\n",rballoc,conv_rballoc(rah,rballoc));
  rah = 1;

  rballoc = 0x1678;
  printf("rballoc 1 %x => %x\n",rballoc,conv_rballoc(rah,rballoc));

  rballoc = 0xfffc;
  printf("rballoc 1 %x => %x\n",rballoc,conv_rballoc(rah,rballoc));
  rballoc = 0xfffd;
  printf("rballoc 1 %x => %x\n",rballoc,conv_rballoc(rah,rballoc));
  rballoc = 0xffff;
  printf("rballoc 1 %x => %x\n",rballoc,conv_rballoc(rah,rballoc));
  rballoc = 0xfffe;
  printf("rballoc 1 %x => %x\n",rballoc,conv_rballoc(rah,rballoc));
}

#endif

