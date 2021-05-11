



#include "PHY/defs_eNB.h"
#include "PHY/phy_extern.h"
#include "SCHED/sched_eNB.h"
#ifdef DEBUG_DCI_TOOLS
#include "PHY/phy_vars.h"
#endif
#include "assertions.h"
#include "nfapi_interface.h"
#include "transport_common_proto.h"
#include "SCHED/sched_common.h"
//#define DEBUG_HARQ


#include "LAYER2/MAC/mac.h"

//#define DEBUG_DCI

uint32_t localRIV2alloc_LUT6[32];
uint32_t distRIV2alloc_even_LUT6[32];
uint32_t distRIV2alloc_odd_LUT6[32];
uint16_t RIV2nb_rb_LUT6[32];
uint16_t RIV2first_rb_LUT6[32];
uint16_t RIV_max6=0;

uint32_t localRIV2alloc_LUT25[512];
uint32_t distRIV2alloc_even_LUT25[512];
uint32_t distRIV2alloc_odd_LUT25[512];
uint16_t RIV2nb_rb_LUT25[512];
uint16_t RIV2first_rb_LUT25[512];
uint16_t RIV_max25=0;


uint32_t localRIV2alloc_LUT50_0[1600];
uint32_t localRIV2alloc_LUT50_1[1600];
uint32_t distRIV2alloc_gap0_even_LUT50_0[1600];
uint32_t distRIV2alloc_gap0_odd_LUT50_0[1600];
uint32_t distRIV2alloc_gap0_even_LUT50_1[1600];
uint32_t distRIV2alloc_gap0_odd_LUT50_1[1600];
uint32_t distRIV2alloc_gap1_even_LUT50_0[1600];
uint32_t distRIV2alloc_gap1_odd_LUT50_0[1600];
uint32_t distRIV2alloc_gap1_even_LUT50_1[1600];
uint32_t distRIV2alloc_gap1_odd_LUT50_1[1600];
uint16_t RIV2nb_rb_LUT50[1600];
uint16_t RIV2first_rb_LUT50[1600];
uint16_t RIV_max50=0;

uint32_t localRIV2alloc_LUT100_0[6000];
uint32_t localRIV2alloc_LUT100_1[6000];
uint32_t localRIV2alloc_LUT100_2[6000];
uint32_t localRIV2alloc_LUT100_3[6000];
uint32_t distRIV2alloc_gap0_even_LUT100_0[6000];
uint32_t distRIV2alloc_gap0_odd_LUT100_0[6000];
uint32_t distRIV2alloc_gap0_even_LUT100_1[6000];
uint32_t distRIV2alloc_gap0_odd_LUT100_1[6000];
uint32_t distRIV2alloc_gap0_even_LUT100_2[6000];
uint32_t distRIV2alloc_gap0_odd_LUT100_2[6000];
uint32_t distRIV2alloc_gap0_even_LUT100_3[6000];
uint32_t distRIV2alloc_gap0_odd_LUT100_3[6000];
uint32_t distRIV2alloc_gap1_even_LUT100_0[6000];
uint32_t distRIV2alloc_gap1_odd_LUT100_0[6000];
uint32_t distRIV2alloc_gap1_even_LUT100_1[6000];
uint32_t distRIV2alloc_gap1_odd_LUT100_1[6000];
uint32_t distRIV2alloc_gap1_even_LUT100_2[6000];
uint32_t distRIV2alloc_gap1_odd_LUT100_2[6000];
uint32_t distRIV2alloc_gap1_even_LUT100_3[6000];
uint32_t distRIV2alloc_gap1_odd_LUT100_3[6000];
uint16_t RIV2nb_rb_LUT100[6000];
uint16_t RIV2first_rb_LUT100[6000];
uint16_t RIV_max100=0;

extern RAN_CONTEXT_t RC;

extern uint32_t current_dlsch_cqi;

// Table 8.6.3-3 36.213
uint16_t beta_cqi[16] = {0,   //reserved
                         0,   //reserved
                         9,   //1.125
                         10,  //1.250
                         11,  //1.375
                         13,  //1.625
                         14,  //1.750
                         16,  //2.000
                         18,  //2.250
                         20,  //2.500
                         23,  //2.875
                         25,  //3.125
                         28,  //3.500
                         32,  //4.000
                         40,  //5.000
                         50
                        }; //6.250

// Table 8.6.3-2 36.213
uint16_t beta_ri[16] = {10,   //1.250
                        13,   //1.625
                        16,   //2.000
                        20,   //2.500
                        25,   //3.125
                        32,   //4.000
                        40,   //5.000
                        50,   //6.250
                        64,   //8.000
                        80,   //10.000
                        101,  //12.625
                        127,  //15.875
                        160,  //20.000
                        0,    //reserved
                        0,    //reserved
                        0
                       };   //reserved

// Table 8.6.3-2 36.213
uint16_t beta_ack[16] = {16,  //2.000
                         20,  //2.500
                         25,  //3.125
                         32,  //4.000
                         40,  //5.000
                         50,  //6.250
                         64,  //8.000
                         80,  //10.000
                         101, //12.625
                         127, //15.875
                         160, //20.000
                         248, //31.000
                         400, //50.000
                         640, //80.000
                         808
                        };//126.00

int8_t delta_PUSCH_abs[4] = {-4,-1,1,4};
int8_t delta_PUSCH_acc[4] = {-1,0,1,3};

int8_t *delta_PUCCH_lut = delta_PUSCH_acc;

uint32_t check_phich_reg(LTE_DL_FRAME_PARMS *frame_parms,uint32_t kprime,uint8_t lprime,uint8_t mi)
{

  uint16_t i;
  uint16_t Ngroup_PHICH = (frame_parms->phich_config_common.phich_resource*frame_parms->N_RB_DL)/48;
  uint16_t mprime;
  uint16_t *pcfich_reg = frame_parms->pcfich_reg;

  if ((lprime>0) && (frame_parms->Ncp==0) )
    return(0);

  //  printf("check_phich_reg : mi %d\n",mi);

  // compute REG based on symbol
  if ((lprime == 0)||
      ((lprime==1)&&(frame_parms->nb_antenna_ports_eNB == 4)))
    mprime = kprime/6;
  else
    mprime = kprime>>2;

  // check if PCFICH uses mprime
  if ((lprime==0) &&
      ((mprime == pcfich_reg[0]) ||
       (mprime == pcfich_reg[1]) ||
       (mprime == pcfich_reg[2]) ||
       (mprime == pcfich_reg[3]))) {
#ifdef DEBUG_DCI_ENCODING
    printf("[PHY] REG %d allocated to PCFICH\n",mprime);
#endif
    return(1);
  }

  // handle Special subframe case for TDD !!!

  //  printf("Checking phich_reg %d\n",mprime);
  if (mi > 0) {
    if (((frame_parms->phich_config_common.phich_resource*frame_parms->N_RB_DL)%48) > 0)
      Ngroup_PHICH++;

    if (frame_parms->Ncp == 1) {
      Ngroup_PHICH<<=1;
    }



    for (i=0; i<Ngroup_PHICH; i++) {
      if ((mprime == frame_parms->phich_reg[i][0]) ||
          (mprime == frame_parms->phich_reg[i][1]) ||
          (mprime == frame_parms->phich_reg[i][2]))  {
#ifdef DEBUG_DCI_ENCODING
        printf("[PHY] REG %d (lprime %d) allocated to PHICH\n",mprime,lprime);
#endif
        return(1);
      }
    }
  }

  return(0);
}

uint16_t get_nquad(uint8_t num_pdcch_symbols,LTE_DL_FRAME_PARMS *frame_parms,uint8_t mi)
{

  uint16_t Nreg=0;
  uint8_t Ngroup_PHICH = (frame_parms->phich_config_common.phich_resource*frame_parms->N_RB_DL)/48;

  if (((frame_parms->phich_config_common.phich_resource*frame_parms->N_RB_DL)%48) > 0)
    Ngroup_PHICH++;

  if (frame_parms->Ncp == 1) {
    Ngroup_PHICH<<=1;
  }

  Ngroup_PHICH*=mi;

  if ((num_pdcch_symbols>0) && (num_pdcch_symbols<4))
    switch (frame_parms->N_RB_DL) {
    case 6:
      Nreg=12+(num_pdcch_symbols-1)*18;
      break;

    case 25:
      Nreg=50+(num_pdcch_symbols-1)*75;
      break;

    case 50:
      Nreg=100+(num_pdcch_symbols-1)*150;
      break;

    case 100:
      Nreg=200+(num_pdcch_symbols-1)*300;
      break;

    default:
      return(0);
    }

  //   printf("Nreg %d (%d)\n",Nreg,Nreg - 4 - (3*Ngroup_PHICH));
  return(Nreg - 4 - (3*Ngroup_PHICH));
}

uint16_t get_nCCE(uint8_t num_pdcch_symbols,LTE_DL_FRAME_PARMS *frame_parms,uint8_t mi)
{
  return(get_nquad(num_pdcch_symbols,frame_parms,mi)/9);
}



uint16_t get_nCCE_mac(uint8_t Mod_id,uint8_t CC_id,int num_pdcch_symbols,int subframe)
{

  // check for eNB only !
  return(get_nCCE(num_pdcch_symbols,
		  &RC.eNB[Mod_id][CC_id]->frame_parms,
		  get_mi(&RC.eNB[Mod_id][CC_id]->frame_parms,subframe)));
}


void conv_eMTC_rballoc (uint16_t resource_block_coding, uint32_t N_RB_DL, uint32_t * rb_alloc)
{


  int             RIV = resource_block_coding&31;
  int             narrowband = resource_block_coding>>5;
  int             N_NB_DL = N_RB_DL / 6;
  int             i0 = (N_RB_DL >> 1) - (3 * N_NB_DL);
  int             first_rb = (6 * narrowband) + i0;
  int             alloc = localRIV2alloc_LUT6[RIV];
  int             ind = first_rb >> 5;
  int             ind_mod = first_rb & 31;

  AssertFatal(RIV<32,"RIV is %d > 31\n",RIV);

  if (((N_RB_DL & 1) > 0) && (narrowband >= (N_NB_DL >> 1)))
    first_rb++;
  rb_alloc[0] = 0;
  rb_alloc[1] = 0;
  rb_alloc[2] = 0;
  rb_alloc[3] = 0;
  rb_alloc[ind] = alloc << ind_mod;
  if (ind_mod > 26)
    rb_alloc[ind + 1] = alloc >> (6 - (ind_mod - 26));
}


void conv_rballoc(uint8_t ra_header,uint32_t rb_alloc,uint32_t N_RB_DL,uint32_t *rb_alloc2)
{

  uint32_t i,shift,subset;
  rb_alloc2[0] = 0;
  rb_alloc2[1] = 0;
  rb_alloc2[2] = 0;
  rb_alloc2[3] = 0;

  //  printf("N_RB_DL %d, ra_header %d, rb_alloc %x\n",N_RB_DL,ra_header,rb_alloc);

  switch (N_RB_DL) {

  case 6:
    rb_alloc2[0] = rb_alloc&0x3f;
    break;

  case 25:
    if (ra_header == 0) {// Type 0 Allocation

      for (i=12; i>0; i--) {
        if ((rb_alloc&(1<<i)) != 0)
          rb_alloc2[0] |= (3<<((2*(12-i))));

        //      printf("rb_alloc2 (type 0) %x\n",rb_alloc2);
      }

      if ((rb_alloc&1) != 0)
        rb_alloc2[0] |= (1<<24);
    } else {
      subset = rb_alloc&1;
      shift  = (rb_alloc>>1)&1;

      for (i=0; i<11; i++) {
        if ((rb_alloc&(1<<(i+2))) != 0)
          rb_alloc2[0] |= (1<<(2*i));

        //printf("rb_alloc2 (type 1) %x\n",rb_alloc2);
      }

      if ((shift == 0) && (subset == 1))
        rb_alloc2[0]<<=1;
      else if ((shift == 1) && (subset == 0))
        rb_alloc2[0]<<=4;
      else if ((shift == 1) && (subset == 1))
        rb_alloc2[0]<<=3;
    }

    break;

  case 50:
    if (ra_header==0) {

      for (i=16; i>0; i--) {
	if ((rb_alloc&(1<<i)) != 0)
	  rb_alloc2[(3*(16-i))>>5] |= (7<<((3*(16-i))%32));
      }

      // bit mask across
      if ((rb_alloc2[0]>>31)==1)
	rb_alloc2[1] |= 1;

      if ((rb_alloc&1) != 0)
	rb_alloc2[1] |= (3<<16);
    }
    else {
      LOG_W(PHY,"resource type 1 not supported for  N_RB_DL=50\n");
    }
    break;

  case 100:
    if (ra_header==0) {
      for (i=0; i<25; i++) {
	if ((rb_alloc&(1<<(24-i))) != 0)
	  rb_alloc2[(4*i)>>5] |= (0xf<<((4*i)%32));

	//  printf("rb_alloc2[%d] (type 0) %x (%d)\n",(4*i)>>5,rb_alloc2[(4*i)>>5],rb_alloc&(1<<i));
      }
    }
    else {
      LOG_W(PHY,"resource type 1 not supported for  N_RB_DL=100\n");
    }

    break;

  default:
    LOG_E(PHY,"Invalid N_RB_DL %d\n", N_RB_DL);
    DevParam (N_RB_DL, 0, 0);
    break;
  }

}



uint32_t conv_nprb(uint8_t ra_header,uint32_t rb_alloc,int N_RB_DL)
{

  uint32_t nprb=0,i;

  switch (N_RB_DL) {
  case 6:
    for (i=0; i<6; i++) {
      if ((rb_alloc&(1<<i)) != 0)
        nprb += 1;
    }

    break;

  case 25:
    if (ra_header == 0) {// Type 0 Allocation

      for (i=12; i>0; i--) {
        if ((rb_alloc&(1<<i)) != 0)
          nprb += 2;
      }

      if ((rb_alloc&1) != 0)
        nprb += 1;
    } else {
      for (i=0; i<11; i++) {
        if ((rb_alloc&(1<<(i+2))) != 0)
          nprb += 1;
      }
    }

    break;

  case 50:
    if (ra_header == 0) {// Type 0 Allocation

      for (i=0; i<16; i++) {
        if ((rb_alloc&(1<<(16-i))) != 0)
          nprb += 3;
      }

      if ((rb_alloc&1) != 0)
        nprb += 2;

    } else {
      for (i=0; i<17; i++) {
        if ((rb_alloc&(1<<(i+2))) != 0)
          nprb += 1;
      }
    }

    break;

  case 100:
    if (ra_header == 0) {// Type 0 Allocation

      for (i=0; i<25; i++) {
        if ((rb_alloc&(1<<(24-i))) != 0)
          nprb += 4;
      }
    } else {
      for (i=0; i<25; i++) {
        if ((rb_alloc&(1<<(i+2))) != 0)
          nprb += 1;
      }
    }

    break;

  default:
    LOG_E(PHY,"Invalide N_RB_DL %d\n", N_RB_DL);
    DevParam (N_RB_DL, 0, 0);
    break;
  }

  return(nprb);
}

uint16_t computeRIV(uint16_t N_RB_DL,uint16_t RBstart,uint16_t Lcrbs)
{

  uint16_t RIV;

  if (Lcrbs<=(1+(N_RB_DL>>1)))
    RIV = (N_RB_DL*(Lcrbs-1)) + RBstart;
  else
    RIV = (N_RB_DL*(N_RB_DL+1-Lcrbs)) + (N_RB_DL-1-RBstart);

  return(RIV);
}

// Convert a DCI Format 1C RIV to a Format 1A RIV
// This extracts the start and length in PRBs from the 1C rballoc and
// recomputes the RIV as if it were the 1A rballoc

uint32_t conv_1C_RIV(int32_t rballoc,uint32_t N_RB_DL) {

  int NpDLVRB,N_RB_step,LpCRBsm1,RBpstart;

  switch (N_RB_DL) {

  case 6: // N_RB_step = 2, NDLVRB = 6, NpDLVRB = 3
    NpDLVRB   = 3;
    N_RB_step = 2;
    break;
  case 25: // N_RB_step = 2, NDLVRB = 24, NpDLVRB = 12
    NpDLVRB   = 12;
    N_RB_step = 2;
    break;
  case 50: // N_RB_step = 4, NDLVRB = 46, NpDLVRB = 11
    NpDLVRB   = 11;
    N_RB_step = 4;
    break;
  case 100: // N_RB_step = 4, NDLVRB = 96, NpDLVRB = 24
    NpDLVRB   = 24;
    N_RB_step = 4;
    break;
  default:
    NpDLVRB   = 24;
    N_RB_step = 4;
    break;
  }

  // This is the 1C part from 7.1.6.3 in 36.213
  LpCRBsm1 = rballoc/NpDLVRB;
  //  printf("LpCRBs = %d\n",LpCRBsm1+1);

  if (LpCRBsm1 <= (NpDLVRB/2)) {
    RBpstart = rballoc % NpDLVRB;
  }
  else {
    LpCRBsm1 = NpDLVRB-LpCRBsm1;
    RBpstart = NpDLVRB-(rballoc%NpDLVRB);
  }
  //  printf("RBpstart %d\n",RBpstart);
  return(computeRIV(N_RB_DL,N_RB_step*RBpstart,N_RB_step*(LpCRBsm1+1)));

}

uint32_t get_prb(int N_RB_DL,int odd_slot,int vrb,int Ngap) {

  int offset;

  switch (N_RB_DL) {

  case 6:
  // N_RB_DL = tildeN_RB_DL = 6
  // Ngap = 4 , P=1, Nrow = 2, Nnull = 2

    switch (vrb) {
    case 0:  // even: 0->0, 1->2, odd: 0->3, 1->5
    case 1:
      return ((3*odd_slot) + 2*(vrb&3))%6;
      break;
    case 2:  // even: 2->3, 3->5, odd: 2->0, 3->2
    case 3:
      return ((3*odd_slot) + 2*(vrb&3) + 5)%6;
      break;
    case 4:  // even: 4->1, odd: 4->4
      return ((3*odd_slot) + 1)%6;
    case 5:  // even: 5->4, odd: 5->1
      return ((3*odd_slot) + 4)%6;
      break;
    }
    break;

  case 15:
    if (vrb<12) {
      if ((vrb&3) < 2)     // even: 0->0, 1->4, 4->1, 5->5, 8->2, 9->6 odd: 0->7, 1->11
  return(((7*odd_slot) + 4*(vrb&3) + (vrb>>2))%14) + 14*(vrb/14);
      else if (vrb < 12) // even: 2->7, 3->11, 6->8, 7->12, 10->9, 11->13
  return (((7*odd_slot) + 4*(vrb&3) + (vrb>>2) +13 )%14) + 14*(vrb/14);
    }
    if (vrb==12)
      return (3+(7*odd_slot)) % 14;
    if (vrb==13)
      return (10+(7*odd_slot)) % 14;
    return 14;
    break;

  case 25:
    return (((12*odd_slot) + 6*(vrb&3) + (vrb>>2))%24) + 24*(vrb/24);
    break;

  case 50: // P=3
    if (Ngap==0) {
      // Nrow=12,Nnull=2,NVRBDL=46,Ngap1= 27
      if (vrb>=23)
  offset=4;
      else
  offset=0;
      if (vrb<44) {
  if ((vrb&3)>=2)
    return offset+((23*odd_slot) + 12*(vrb&3) + (vrb>>2) + 45)%46;
  else
    return offset+((23*odd_slot) + 12*(vrb&3) + (vrb>>2))%46;
      }
      if (vrb==44)  // even: 44->11, odd: 45->34
  return offset+((23*odd_slot) + 22-12+1);
      if (vrb==45)  // even: 45->10, odd: 45->33
  return offset+((23*odd_slot) + 22+12);
      if (vrb==46)
  return offset+46+((23*odd_slot) + 23-12+1) % 46;
      if (vrb==47)
  return offset+46+((23*odd_slot) + 23+12) % 46;
      if (vrb==48)
  return offset+46+((23*odd_slot) + 23-12+1) % 46;
      if (vrb==49)
  return offset+46+((23*odd_slot) + 23+12) % 46;
    }
    else {
      // Nrow=6,Nnull=6,NVRBDL=18,Ngap1= 27
      if (vrb>=9)
  offset=18;
      else
  offset=0;

      if (vrb<12) {
  if ((vrb&3)>=2)
    return offset+((9*odd_slot) + 6*(vrb&3) + (vrb>>2) + 17)%18;
  else
    return offset+((9*odd_slot) + 6*(vrb&3) + (vrb>>2))%18;
      }
      else {
  return offset+((9*odd_slot) + 12*(vrb&1)+(vrb>>1) )%18 + 18*(vrb/18);
      }
    }
    break;
  case 75:
    // Ngap1 = 32, NVRBRL=64, P=4, Nrow= 16, Nnull=0
    if (Ngap ==0) {
      return ((32*odd_slot) + 16*(vrb&3) + (vrb>>2))%64 + (vrb/64);
    } else {
      // Ngap2 = 16, NVRBDL=32, Nrow=8, Nnull=0
      return ((16*odd_slot) + 8*(vrb&3) + (vrb>>2))%32 + (vrb/32);
    }
    break;
  case 100:
    // Ngap1 = 48, NVRBDL=96, Nrow=24, Nnull=0
    if (Ngap ==0) {
      return ((48*odd_slot) + 24*(vrb&3) + (vrb>>2))%96 + (vrb/96);
    } else {
      // Ngap2 = 16, NVRBDL=32, Nrow=8, Nnull=0
      return ((16*odd_slot) + 8*(vrb&3) + (vrb>>2))%32 + (vrb/32);
    }
    break;
  default:
    LOG_E(PHY,"Unknown N_RB_DL %d\n",N_RB_DL);
    return 0;
  }
  return 0;

}


void generate_RIV_tables(void)
{

  // 6RBs localized RIV
  uint8_t Lcrbs,RBstart;
  uint16_t RIV;
  uint32_t alloc0,allocdist0_0_even,allocdist0_0_odd,allocdist0_1_even,allocdist0_1_odd;
  uint32_t alloc1,allocdist1_0_even,allocdist1_0_odd,allocdist1_1_even,allocdist1_1_odd;
  uint32_t alloc2,allocdist2_0_even,allocdist2_0_odd,allocdist2_1_even,allocdist2_1_odd;
  uint32_t alloc3,allocdist3_0_even,allocdist3_0_odd,allocdist3_1_even,allocdist3_1_odd;
  uint32_t nVRB,nVRB_even_dist,nVRB_odd_dist;

  for (RBstart=0; RBstart<6; RBstart++) {
    alloc0 = 0;
    allocdist0_0_even = 0;
    allocdist0_0_odd  = 0;
    for (Lcrbs=1; Lcrbs<=(6-RBstart); Lcrbs++) {
      //printf("RBstart %d, len %d --> ",RBstart,Lcrbs);
      nVRB             = Lcrbs-1+RBstart;
      alloc0          |= (1<<nVRB);
      allocdist0_0_even |= (1<<get_prb(6,0,nVRB,0));
      allocdist0_0_odd  |= (1<<get_prb(6,1,nVRB,0));
      RIV=computeRIV(6,RBstart,Lcrbs);

      if (RIV>RIV_max6)
        RIV_max6 = RIV;

      //      printf("RIV %d (%d) : first_rb %d NBRB %d\n",RIV,localRIV2alloc_LUT25[RIV],RBstart,Lcrbs);
      localRIV2alloc_LUT6[RIV] = alloc0;
      distRIV2alloc_even_LUT6[RIV]  = allocdist0_0_even;
      distRIV2alloc_odd_LUT6[RIV]  = allocdist0_0_odd;
      RIV2nb_rb_LUT6[RIV]      = Lcrbs;
      RIV2first_rb_LUT6[RIV]   = RBstart;
    }
  }


  for (RBstart=0; RBstart<25; RBstart++) {
    alloc0 = 0;
    allocdist0_0_even = 0;
    allocdist0_0_odd  = 0;
    for (Lcrbs=1; Lcrbs<=(25-RBstart); Lcrbs++) {
      nVRB = Lcrbs-1+RBstart;
      //printf("RBstart %d, len %d --> ",RBstart,Lcrbs);
      alloc0     |= (1<<nVRB);
      allocdist0_0_even |= (1<<get_prb(25,0,nVRB,0));
      allocdist0_0_odd  |= (1<<get_prb(25,1,nVRB,0));

      //printf("alloc 0 %x, allocdist0_even %x, allocdist0_odd %x\n",alloc0,allocdist0_0_even,allocdist0_0_odd);
      RIV=computeRIV(25,RBstart,Lcrbs);

      if (RIV>RIV_max25)
        RIV_max25 = RIV;;


      localRIV2alloc_LUT25[RIV]      = alloc0;
      distRIV2alloc_even_LUT25[RIV]  = allocdist0_0_even;
      distRIV2alloc_odd_LUT25[RIV]   = allocdist0_0_odd;
      RIV2nb_rb_LUT25[RIV]           = Lcrbs;
      RIV2first_rb_LUT25[RIV]        = RBstart;
    }
  }


  for (RBstart=0; RBstart<50; RBstart++) {
    alloc0 = 0;
    alloc1 = 0;
    allocdist0_0_even=0;
    allocdist1_0_even=0;
    allocdist0_0_odd=0;
    allocdist1_0_odd=0;
    allocdist0_1_even=0;
    allocdist1_1_even=0;
    allocdist0_1_odd=0;
    allocdist1_1_odd=0;

    for (Lcrbs=1; Lcrbs<=(50-RBstart); Lcrbs++) {

      nVRB = Lcrbs-1+RBstart;


      if (nVRB<32)
        alloc0 |= (1<<nVRB);
      else
        alloc1 |= (1<<(nVRB-32));

      // Distributed Gap1, even slot
      nVRB_even_dist = get_prb(50,0,nVRB,0);
      if (nVRB_even_dist<32)
        allocdist0_0_even |= (1<<nVRB_even_dist);
      else
        allocdist1_0_even |= (1<<(nVRB_even_dist-32));

      // Distributed Gap1, odd slot
      nVRB_odd_dist = get_prb(50,1,nVRB,0);
      if (nVRB_odd_dist<32)
        allocdist0_0_odd |= (1<<nVRB_odd_dist);
      else
        allocdist1_0_odd |= (1<<(nVRB_odd_dist-32));

      // Distributed Gap2, even slot
      nVRB_even_dist = get_prb(50,0,nVRB,1);
      if (nVRB_even_dist<32)
        allocdist0_1_even |= (1<<nVRB_even_dist);
      else
        allocdist1_1_even |= (1<<(nVRB_even_dist-32));

      // Distributed Gap2, odd slot
      nVRB_odd_dist = get_prb(50,1,nVRB,1);
      if (nVRB_odd_dist<32)
        allocdist0_1_odd |= (1<<nVRB_odd_dist);
      else
        allocdist1_1_odd |= (1<<(nVRB_odd_dist-32));

      RIV=computeRIV(50,RBstart,Lcrbs);

      if (RIV>RIV_max50)
        RIV_max50 = RIV;

      //      printf("RIV %d : first_rb %d NBRB %d\n",RIV,RBstart,Lcrbs);
      localRIV2alloc_LUT50_0[RIV]      = alloc0;
      localRIV2alloc_LUT50_1[RIV]      = alloc1;
      distRIV2alloc_gap0_even_LUT50_0[RIV]  = allocdist0_0_even;
      distRIV2alloc_gap0_even_LUT50_1[RIV]  = allocdist1_0_even;
      distRIV2alloc_gap0_odd_LUT50_0[RIV]   = allocdist0_0_odd;
      distRIV2alloc_gap0_odd_LUT50_1[RIV]   = allocdist1_0_odd;
      distRIV2alloc_gap1_even_LUT50_0[RIV]  = allocdist0_1_even;
      distRIV2alloc_gap1_even_LUT50_1[RIV]  = allocdist1_1_even;
      distRIV2alloc_gap1_odd_LUT50_0[RIV]   = allocdist0_1_odd;
      distRIV2alloc_gap1_odd_LUT50_1[RIV]   = allocdist1_1_odd;
      RIV2nb_rb_LUT50[RIV]        = Lcrbs;
      RIV2first_rb_LUT50[RIV]     = RBstart;
    }
  }


  for (RBstart=0; RBstart<100; RBstart++) {
    alloc0 = 0;
    alloc1 = 0;
    alloc2 = 0;
    alloc3 = 0;
    allocdist0_0_even=0;
    allocdist1_0_even=0;
    allocdist2_0_even=0;
    allocdist3_0_even=0;
    allocdist0_0_odd=0;
    allocdist1_0_odd=0;
    allocdist2_0_odd=0;
    allocdist3_0_odd=0;
    allocdist0_1_even=0;
    allocdist1_1_even=0;
    allocdist2_1_even=0;
    allocdist3_1_even=0;
    allocdist0_1_odd=0;
    allocdist1_1_odd=0;
    allocdist2_1_odd=0;
    allocdist3_1_odd=0;

    for (Lcrbs=1; Lcrbs<=(100-RBstart); Lcrbs++) {

      nVRB = Lcrbs-1+RBstart;

      if (nVRB<32)
        alloc0 |= (1<<nVRB);
      else if (nVRB<64)
        alloc1 |= (1<<(nVRB-32));
      else if (nVRB<96)
        alloc2 |= (1<<(nVRB-64));
      else
        alloc3 |= (1<<(nVRB-96));

      // Distributed Gap1, even slot
      nVRB_even_dist = get_prb(100,0,nVRB,0);

//      if ((RBstart==0) && (Lcrbs<=8))
//  printf("nVRB %d => nVRB_even_dist %d\n",nVRB,nVRB_even_dist);


      if (nVRB_even_dist<32)
        allocdist0_0_even |= (1<<nVRB_even_dist);
      else if (nVRB_even_dist<64)
        allocdist1_0_even |= (1<<(nVRB_even_dist-32));
      else if (nVRB_even_dist<96)
  allocdist2_0_even |= (1<<(nVRB_even_dist-64));
      else
  allocdist3_0_even |= (1<<(nVRB_even_dist-96));
/*      if ((RBstart==0) && (Lcrbs<=8))
  printf("rballoc =>(%08x.%08x.%08x.%08x)\n",
         allocdist0_0_even,
         allocdist1_0_even,
         allocdist2_0_even,
         allocdist3_0_even
         );
*/
      // Distributed Gap1, odd slot
      nVRB_odd_dist = get_prb(100,1,nVRB,0);
      if (nVRB_odd_dist<32)
        allocdist0_0_odd |= (1<<nVRB_odd_dist);
      else if (nVRB_odd_dist<64)
        allocdist1_0_odd |= (1<<(nVRB_odd_dist-32));
      else if (nVRB_odd_dist<96)
  allocdist2_0_odd |= (1<<(nVRB_odd_dist-64));
      else
  allocdist3_0_odd |= (1<<(nVRB_odd_dist-96));


      // Distributed Gap2, even slot
      nVRB_even_dist = get_prb(100,0,nVRB,1);
      if (nVRB_even_dist<32)
        allocdist0_1_even |= (1<<nVRB_even_dist);
      else if (nVRB_even_dist<64)
        allocdist1_1_even |= (1<<(nVRB_even_dist-32));
      else if (nVRB_even_dist<96)
  allocdist2_1_even |= (1<<(nVRB_even_dist-64));
      else
  allocdist3_1_even |= (1<<(nVRB_even_dist-96));


      // Distributed Gap2, odd slot
      nVRB_odd_dist = get_prb(100,1,nVRB,1);
      if (nVRB_odd_dist<32)
        allocdist0_1_odd |= (1<<nVRB_odd_dist);
      else if (nVRB_odd_dist<64)
        allocdist1_1_odd |= (1<<(nVRB_odd_dist-32));
      else if (nVRB_odd_dist<96)
  allocdist2_1_odd |= (1<<(nVRB_odd_dist-64));
      else
  allocdist3_1_odd |= (1<<(nVRB_odd_dist-96));


      RIV=computeRIV(100,RBstart,Lcrbs);

      if (RIV>RIV_max100)
        RIV_max100 = RIV;

      //      printf("RIV %d : first_rb %d NBRB %d\n",RIV,RBstart,Lcrbs);
      localRIV2alloc_LUT100_0[RIV] = alloc0;
      localRIV2alloc_LUT100_1[RIV] = alloc1;
      localRIV2alloc_LUT100_2[RIV] = alloc2;
      localRIV2alloc_LUT100_3[RIV] = alloc3;
      distRIV2alloc_gap0_even_LUT100_0[RIV]  = allocdist0_0_even;
      distRIV2alloc_gap0_even_LUT100_1[RIV]  = allocdist1_0_even;
      distRIV2alloc_gap0_even_LUT100_2[RIV]  = allocdist2_0_even;
      distRIV2alloc_gap0_even_LUT100_3[RIV]  = allocdist3_0_even;
      distRIV2alloc_gap0_odd_LUT100_0[RIV]   = allocdist0_0_odd;
      distRIV2alloc_gap0_odd_LUT100_1[RIV]   = allocdist1_0_odd;
      distRIV2alloc_gap0_odd_LUT100_2[RIV]   = allocdist2_0_odd;
      distRIV2alloc_gap0_odd_LUT100_3[RIV]   = allocdist3_0_odd;
      distRIV2alloc_gap1_even_LUT100_0[RIV]  = allocdist0_1_even;
      distRIV2alloc_gap1_even_LUT100_1[RIV]  = allocdist1_1_even;
      distRIV2alloc_gap1_even_LUT100_2[RIV]  = allocdist2_1_even;
      distRIV2alloc_gap1_even_LUT100_3[RIV]  = allocdist3_1_even;
      distRIV2alloc_gap1_odd_LUT100_0[RIV]   = allocdist0_1_odd;
      distRIV2alloc_gap1_odd_LUT100_1[RIV]   = allocdist1_1_odd;
      distRIV2alloc_gap1_odd_LUT100_2[RIV]   = allocdist2_1_odd;
      distRIV2alloc_gap1_odd_LUT100_3[RIV]   = allocdist3_1_odd;

      RIV2nb_rb_LUT100[RIV]      = Lcrbs;
      RIV2first_rb_LUT100[RIV]   = RBstart;
    }
  }
}

// Ngap = 3, N_VRB_DL=6, P=1, N_row=2, N_null=4*2-6=2
// permutation for even slots :
//    n_PRB'(0,2,4) = (0,1,2), n_PRB'(1,3,5) = (4,5,6)
//    n_PRB''(0,1,2,3) = (0,2,4,6)
//    => n_tilde_PRB(5) = (4)
//       n_tilde_PRB(4) = (1)
//       n_tilde_PRB(2,3) = (3,5)
//       n_tilde_PRB(0,1) = (0,2)

uint32_t get_rballoc(vrb_t vrb_type,uint16_t rb_alloc_dci)
{

  return(localRIV2alloc_LUT25[rb_alloc_dci]);

}

uint8_t subframe2harq_pid(LTE_DL_FRAME_PARMS *frame_parms,uint32_t frame,uint8_t subframe)
{
  uint8_t ret = 255;

  if (frame_parms->frame_type == FDD) {
    ret = (((frame*10)+subframe)&7);
  } else {

    switch (frame_parms->tdd_config) {
    case 1:
      switch (subframe) {
      case 2:
      case 3:
        ret = (subframe-2);
        break;

      case 7:
      case 8:
        ret = (subframe-5);
        break;

      default:
        LOG_E(PHY,"subframe2_harq_pid, Illegal subframe %d for TDD mode %d\n",subframe,frame_parms->tdd_config);
        ret = (255);
        break;
      }

      break;

    case 2:
      if ((subframe!=2) && (subframe!=7)) {
	LOG_E(PHY,"subframe2_harq_pid, Illegal subframe %d for TDD mode %d\n",subframe,frame_parms->tdd_config);
	ret=255;
      }
      else ret = (subframe/7);
      break;

    case 3:
      if ((subframe<2) || (subframe>4)) {
        LOG_E(PHY,"subframe2_harq_pid, Illegal subframe %d for TDD mode %d\n",subframe,frame_parms->tdd_config);
        ret = (255);
      }
      else ret = (subframe-2);
      break;

    case 4:
      if ((subframe<2) || (subframe>3)) {
        LOG_E(PHY,"subframe2_harq_pid, Illegal subframe %d for TDD mode %d\n",subframe,frame_parms->tdd_config);
        ret = (255);
      }
      else ret = (subframe-2);
      break;

    case 5:
      if (subframe!=2) {
        LOG_E(PHY,"subframe2_harq_pid, Illegal subframe %d for TDD mode %d\n",subframe,frame_parms->tdd_config);
        ret = (255);
      }
      else ret = (subframe-2);
      break;

    default:
      LOG_E(PHY,"subframe2_harq_pid, Unsupported TDD mode %d\n",frame_parms->tdd_config);
      ret = (255);
    }
  }

  AssertFatal(ret!=255,
	      "invalid harq_pid(%d) at SFN/SF = %d/%d\n", (int8_t)ret, frame, subframe);
  return ret;
}

uint8_t pdcch_alloc2ul_subframe(LTE_DL_FRAME_PARMS *frame_parms,uint8_t n)
{
  uint8_t ul_subframe = 255;

  if ((frame_parms->frame_type == TDD) &&
      (frame_parms->tdd_config == 1)) {
    if ((n==1)||(n==6)) { // tdd_config 0,1 SF 1,5
      ul_subframe = ((n+6)%10);
    } else if ((n==4)||(n==9)) {
      ul_subframe = ((n+4)%10);
    }
  } else if ((frame_parms->frame_type == TDD) &&
           (frame_parms->tdd_config == 6) &&
           ((n==0)||(n==1)||(n==5)||(n==6)))
    ul_subframe = ((n+7)%10);
  else if ((frame_parms->frame_type == TDD) &&
           (frame_parms->tdd_config == 6) &&
           (n==9)) // tdd_config 6 SF 9
    ul_subframe = ((n+5)%10);
  else
    ul_subframe = ((n+4)%10);

  if ( (subframe_select(frame_parms,ul_subframe) != SF_UL) && (frame_parms->frame_type == TDD)) return(255);

  LOG_D(PHY, "subframe %d: PUSCH subframe = %d\n", n, ul_subframe);
  return ul_subframe;
}

uint8_t ul_subframe2pdcch_alloc_subframe(LTE_DL_FRAME_PARMS *frame_parms,uint8_t n)
{
  if ((frame_parms->frame_type == TDD) &&
      (frame_parms->tdd_config == 1)) {
    if ((n==7)||(n==2)) { // tdd_config 0,1 SF 1,5
      return((n==7)? 1 : 6);
    } else if ((n==3)||(n==8)) {
      return((n==3)? 9 : 4);
    }
  } else if ((frame_parms->frame_type == TDD) &&
           (frame_parms->tdd_config == 6) &&
           ((n==7)||(n==8)||(n==2)||(n==3)))
    return((n+3)%10);
  else if ((frame_parms->frame_type == TDD) &&
           (frame_parms->tdd_config == 6) &&
           (n==4)) // tdd_config 6 SF 9
    return(9);
  else
    return((n+6)%10);

  LOG_E(PHY, "%s %s:%i pdcch allocation error\n",__FUNCTION__,__FILE__,__LINE__);
  return 0;
}

uint32_t pdcch_alloc2ul_frame(LTE_DL_FRAME_PARMS *frame_parms,uint32_t frame, uint8_t n)
{
  uint32_t ul_frame;

  if ((frame_parms->frame_type == TDD) &&
      (frame_parms->tdd_config == 1) &&
      ((n==1)||(n==6)||(n==4)||(n==9))) { // tdd_config 0,1 SF 1,5
      ul_frame = (frame + (n < 5 ? 0 : 1));
  } else if ((frame_parms->frame_type == TDD) &&
           (frame_parms->tdd_config == 6) &&
           ((n==0)||(n==1)||(n==5)||(n==6)))
    ul_frame = (frame + (n>=5 ? 1 : 0));
  else if ((frame_parms->frame_type == TDD) &&
           (frame_parms->tdd_config == 6) &&
           (n==9)) // tdd_config 6 SF 9
    ul_frame = (frame+1);
  else
    ul_frame = (frame+(n>=6 ? 1 : 0));

  LOG_D(PHY, "frame %d subframe %d: PUSCH frame = %d\n", frame, n, ul_frame);
  return ul_frame % 1024;
}

uint32_t pmi_extend(LTE_DL_FRAME_PARMS *frame_parms,uint8_t wideband_pmi, uint8_t rank)
{

  uint8_t i,wideband_pmi2;
  uint32_t pmi_ex = 0;

  if (frame_parms->N_RB_DL!=25) {
    LOG_E(PHY,"pmi_extend not yet implemented for anything else than 25PRB\n");
    return(-1);
  }

  if (rank==0) {
    wideband_pmi2=wideband_pmi&3;
    for (i=0; i<14; i+=2)
      pmi_ex|=(wideband_pmi2<<i);
  }
  else if (rank==1) {
    wideband_pmi2=wideband_pmi&1;
    for (i=0; i<7; i++)
      pmi_ex|=(wideband_pmi2<<i);
  }
  else {
    LOG_E(PHY,"unsupported rank\n");
    return(-1);
  }

  return(pmi_ex);
}

uint64_t pmi2hex_2Ar1(uint32_t pmi)
{

  uint64_t pmil = (uint64_t)pmi;

  return ((pmil&3) + (((pmil>>2)&3)<<4) + (((pmil>>4)&3)<<8) + (((pmil>>6)&3)<<12) +
          (((pmil>>8)&3)<<16) + (((pmil>>10)&3)<<20) + (((pmil>>12)&3)<<24) +
          (((pmil>>14)&3)<<28) + (((pmil>>16)&3)<<32) + (((pmil>>18)&3)<<36) +
          (((pmil>>20)&3)<<40) + (((pmil>>22)&3)<<44) + (((pmil>>24)&3)<<48));
}

uint64_t pmi2hex_2Ar2(uint32_t pmi)
{

  uint64_t pmil = (uint64_t)pmi;
  return ((pmil&1) + (((pmil>>1)&1)<<4) + (((pmil>>2)&1)<<8) + (((pmil>>3)&1)<<12) +
          (((pmil>>4)&1)<<16) + (((pmil>>5)&1)<<20) + (((pmil>>6)&1)<<24) +
          (((pmil>>7)&1)<<28) + (((pmil>>8)&1)<<32) + (((pmil>>9)&1)<<36) +
          (((pmil>>10)&1)<<40) + (((pmil>>11)&1)<<44) + (((pmil>>12)&1)<<48));
}


int dump_dci(LTE_DL_FRAME_PARMS *frame_parms, DCI_ALLOC_t *dci)
{
  switch (dci->format) {

  case format0:   // This is an UL SCH allocation so nothing here, inform MAC
    if ((frame_parms->frame_type == TDD) &&
        (frame_parms->tdd_config>0))
      switch(frame_parms->N_RB_DL) {
      case 6:
        LOG_D(PHY,"DCI format0 (TDD, 1.5MHz), rnti %x (%x): hopping %d, rb_alloc %x, mcs %d, ndi %d, TPC %d, cshift %d, dai %d, cqi_req %d\n",
              dci->rnti,
              ((uint32_t*)&dci->dci_pdu[0])[0],
              ((DCI0_1_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->hopping,
              ((DCI0_1_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->rballoc,
              ((DCI0_1_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->mcs,
              ((DCI0_1_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->ndi,
              ((DCI0_1_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->TPC,
              ((DCI0_1_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->cshift,
              ((DCI0_1_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->dai,
              ((DCI0_1_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->cqi_req);
        break;

      case 25:
        LOG_D(PHY,"DCI format0 (TDD1-6, 5MHz), rnti %x (%x): hopping %d, rb_alloc %x, mcs %d, ndi %d, TPC %d, cshift %d, dai %d, cqi_req %d\n",
              dci->rnti,
              ((uint32_t*)&dci->dci_pdu[0])[0],
              ((DCI0_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->hopping,
              ((DCI0_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->rballoc,
              ((DCI0_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->mcs,
              ((DCI0_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->ndi,
              ((DCI0_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->TPC,
              ((DCI0_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->cshift,
              ((DCI0_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->dai,
              ((DCI0_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->cqi_req);
        break;

      case 50:
        LOG_D(PHY,"DCI format0 (TDD1-6, 10MHz), rnti %x (%x): hopping %d, rb_alloc %x, mcs %d, ndi %d, TPC %d, cshift %d, dai %d, cqi_req %d\n",
              dci->rnti,
              ((uint32_t*)&dci->dci_pdu[0])[0],
              ((DCI0_10MHz_TDD_1_6_t *)&dci->dci_pdu[0])->hopping,
              ((DCI0_10MHz_TDD_1_6_t *)&dci->dci_pdu[0])->rballoc,
              ((DCI0_10MHz_TDD_1_6_t *)&dci->dci_pdu[0])->mcs,
              ((DCI0_10MHz_TDD_1_6_t *)&dci->dci_pdu[0])->ndi,
              ((DCI0_10MHz_TDD_1_6_t *)&dci->dci_pdu[0])->TPC,
              ((DCI0_10MHz_TDD_1_6_t *)&dci->dci_pdu[0])->cshift,
              ((DCI0_10MHz_TDD_1_6_t *)&dci->dci_pdu[0])->dai,
              ((DCI0_10MHz_TDD_1_6_t *)&dci->dci_pdu[0])->cqi_req);
        break;

      case 100:
        LOG_D(PHY,"DCI format0 (TDD1-6, 20MHz), rnti %x (%x): hopping %d, rb_alloc %x, mcs %d, ndi %d, TPC %d, cshift %d, dai %d, cqi_req %d\n",
              dci->rnti,
              ((uint32_t*)&dci->dci_pdu[0])[0],
              ((DCI0_20MHz_TDD_1_6_t *)&dci->dci_pdu[0])->hopping,
              ((DCI0_20MHz_TDD_1_6_t *)&dci->dci_pdu[0])->rballoc,
              ((DCI0_20MHz_TDD_1_6_t *)&dci->dci_pdu[0])->mcs,
              ((DCI0_20MHz_TDD_1_6_t *)&dci->dci_pdu[0])->ndi,
              ((DCI0_20MHz_TDD_1_6_t *)&dci->dci_pdu[0])->TPC,
              ((DCI0_20MHz_TDD_1_6_t *)&dci->dci_pdu[0])->cshift,
              ((DCI0_20MHz_TDD_1_6_t *)&dci->dci_pdu[0])->dai,
              ((DCI0_20MHz_TDD_1_6_t *)&dci->dci_pdu[0])->cqi_req);
        break;

      default:
        LOG_E(PHY,"Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
        DevParam (frame_parms->N_RB_DL, 0, 0);
        break;
      }
    else if (frame_parms->frame_type == FDD)
      switch(frame_parms->N_RB_DL) {
      case 6:
        LOG_D(PHY,"DCI format0 (FDD, 1.5MHz), rnti %x (%x): hopping %d, rb_alloc %x, mcs %d, ndi %d, TPC %d, cshift %d, cqi_req %d\n",
              dci->rnti,
              ((uint32_t*)&dci->dci_pdu[0])[0],
              ((DCI0_1_5MHz_FDD_t *)&dci->dci_pdu[0])->hopping,
              ((DCI0_1_5MHz_FDD_t *)&dci->dci_pdu[0])->rballoc,
              ((DCI0_1_5MHz_FDD_t *)&dci->dci_pdu[0])->mcs,
              ((DCI0_1_5MHz_FDD_t *)&dci->dci_pdu[0])->ndi,
              ((DCI0_1_5MHz_FDD_t *)&dci->dci_pdu[0])->TPC,
              ((DCI0_1_5MHz_FDD_t *)&dci->dci_pdu[0])->cshift,
              ((DCI0_1_5MHz_FDD_t *)&dci->dci_pdu[0])->cqi_req);
        break;

      case 25:
        LOG_D(PHY,"DCI format0 (FDD, 5MHz), rnti %x (%x): hopping %d, rb_alloc %x, mcs %d, ndi %d, TPC %d, cshift %d, cqi_req %d\n",
              dci->rnti,
              ((uint32_t*)&dci->dci_pdu[0])[0],
              ((DCI0_5MHz_FDD_t *)&dci->dci_pdu[0])->hopping,
              ((DCI0_5MHz_FDD_t *)&dci->dci_pdu[0])->rballoc,
              ((DCI0_5MHz_FDD_t *)&dci->dci_pdu[0])->mcs,
              ((DCI0_5MHz_FDD_t *)&dci->dci_pdu[0])->ndi,
              ((DCI0_5MHz_FDD_t *)&dci->dci_pdu[0])->TPC,
              ((DCI0_5MHz_FDD_t *)&dci->dci_pdu[0])->cshift,
              ((DCI0_5MHz_FDD_t *)&dci->dci_pdu[0])->cqi_req);
        break;

      case 50:
        LOG_D(PHY,"DCI format0 (FDD, 10MHz), rnti %x (%x): hopping %d, rb_alloc %x, mcs %d, ndi %d, TPC %d, cshift %d, cqi_req %d\n",
              dci->rnti,
              ((uint32_t*)&dci->dci_pdu[0])[0],
              ((DCI0_10MHz_FDD_t *)&dci->dci_pdu[0])->hopping,
              ((DCI0_10MHz_FDD_t *)&dci->dci_pdu[0])->rballoc,
              ((DCI0_10MHz_FDD_t *)&dci->dci_pdu[0])->mcs,
              ((DCI0_10MHz_FDD_t *)&dci->dci_pdu[0])->ndi,
              ((DCI0_10MHz_FDD_t *)&dci->dci_pdu[0])->TPC,
              ((DCI0_10MHz_FDD_t *)&dci->dci_pdu[0])->cshift,
              ((DCI0_10MHz_FDD_t *)&dci->dci_pdu[0])->cqi_req);
        break;

      case 100:
        LOG_D(PHY,"DCI format0 (FDD, 20MHz), rnti %x (%x): hopping %d, rb_alloc %x, mcs %d, ndi %d, TPC %d, cshift %d, cqi_req %d\n",
              dci->rnti,
              ((uint32_t*)&dci->dci_pdu[0])[0],
              ((DCI0_20MHz_FDD_t *)&dci->dci_pdu[0])->hopping,
              ((DCI0_20MHz_FDD_t *)&dci->dci_pdu[0])->rballoc,
              ((DCI0_20MHz_FDD_t *)&dci->dci_pdu[0])->mcs,
              ((DCI0_20MHz_FDD_t *)&dci->dci_pdu[0])->ndi,
              ((DCI0_20MHz_FDD_t *)&dci->dci_pdu[0])->TPC,
              ((DCI0_20MHz_FDD_t *)&dci->dci_pdu[0])->cshift,
              ((DCI0_20MHz_FDD_t *)&dci->dci_pdu[0])->cqi_req);
        break;

      default:
        LOG_E(PHY,"Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
        DevParam (frame_parms->N_RB_DL, 0, 0);
        break;
      }
    else
      LOG_E(PHY,"Don't know how to handle TDD format 0 yet\n");

    break;

  case format1:
    if ((frame_parms->frame_type == TDD) &&
        (frame_parms->tdd_config>0))

      switch(frame_parms->N_RB_DL) {
      case 6:
        LOG_D(PHY,"DCI format1 (TDD 1.5 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d, harq_pid %d, ndi %d, RV %d, TPC %d, dai %d\n",
              dci->rnti,
              ((uint32_t*)&dci->dci_pdu)[0],
              ((DCI1_1_5MHz_TDD_t *)&dci->dci_pdu[0])->rah,
              ((DCI1_1_5MHz_TDD_t *)&dci->dci_pdu[0])->rballoc,
              ((DCI1_1_5MHz_TDD_t *)&dci->dci_pdu[0])->mcs,
              ((DCI1_1_5MHz_TDD_t *)&dci->dci_pdu[0])->harq_pid,
              ((DCI1_1_5MHz_TDD_t *)&dci->dci_pdu[0])->ndi,
              ((DCI1_1_5MHz_TDD_t *)&dci->dci_pdu[0])->rv,
              ((DCI1_1_5MHz_TDD_t *)&dci->dci_pdu[0])->TPC,
              ((DCI1_1_5MHz_TDD_t *)&dci->dci_pdu[0])->dai);
        break;

      case 25:
        LOG_D(PHY,"DCI format1 (TDD 5 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d, harq_pid %d, ndi %d, RV %d, TPC %d, dai %d\n",
              dci->rnti,
              ((uint32_t*)&dci->dci_pdu)[0],
              ((DCI1_5MHz_TDD_t *)&dci->dci_pdu[0])->rah,
              ((DCI1_5MHz_TDD_t *)&dci->dci_pdu[0])->rballoc,
              ((DCI1_5MHz_TDD_t *)&dci->dci_pdu[0])->mcs,
              ((DCI1_5MHz_TDD_t *)&dci->dci_pdu[0])->harq_pid,
              ((DCI1_5MHz_TDD_t *)&dci->dci_pdu[0])->ndi,
              ((DCI1_5MHz_TDD_t *)&dci->dci_pdu[0])->rv,
              ((DCI1_5MHz_TDD_t *)&dci->dci_pdu[0])->TPC,
              ((DCI1_5MHz_TDD_t *)&dci->dci_pdu[0])->dai);
        break;

      case 50:
        LOG_D(PHY,"DCI format1 (TDD 10 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d, harq_pid %d, ndi %d, RV %d, TPC %d, dai %d\n",
              dci->rnti,
              ((uint32_t*)&dci->dci_pdu)[0],
              ((DCI1_10MHz_TDD_t *)&dci->dci_pdu[0])->rah,
              ((DCI1_10MHz_TDD_t *)&dci->dci_pdu[0])->rballoc,
              ((DCI1_10MHz_TDD_t *)&dci->dci_pdu[0])->mcs,
              ((DCI1_10MHz_TDD_t *)&dci->dci_pdu[0])->harq_pid,
              ((DCI1_10MHz_TDD_t *)&dci->dci_pdu[0])->ndi,
              ((DCI1_10MHz_TDD_t *)&dci->dci_pdu[0])->rv,
              ((DCI1_10MHz_TDD_t *)&dci->dci_pdu[0])->TPC,
              ((DCI1_10MHz_TDD_t *)&dci->dci_pdu[0])->dai);
        break;

      case 100:
        LOG_D(PHY,"DCI format1 (TDD 20 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d, harq_pid %d, ndi %d, RV %d, TPC %d, dai %d\n",
              dci->rnti,
              ((uint32_t*)&dci->dci_pdu)[0],
              ((DCI1_20MHz_TDD_t *)&dci->dci_pdu[0])->rah,
              ((DCI1_20MHz_TDD_t *)&dci->dci_pdu[0])->rballoc,
              ((DCI1_20MHz_TDD_t *)&dci->dci_pdu[0])->mcs,
              ((DCI1_20MHz_TDD_t *)&dci->dci_pdu[0])->harq_pid,
              ((DCI1_20MHz_TDD_t *)&dci->dci_pdu[0])->ndi,
              ((DCI1_20MHz_TDD_t *)&dci->dci_pdu[0])->rv,
              ((DCI1_20MHz_TDD_t *)&dci->dci_pdu[0])->TPC,
              ((DCI1_20MHz_TDD_t *)&dci->dci_pdu[0])->dai);
        break;

      default:
        LOG_E(PHY,"Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
        DevParam (frame_parms->N_RB_DL, 0, 0);
        break;
      }
    else if (frame_parms->frame_type == FDD) {
      switch(frame_parms->N_RB_DL) {
      case 6:
        LOG_D(PHY,"DCI format1 (FDD, 1.5 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d, harq_pid %d, ndi %d, RV %d, TPC %d\n",
              dci->rnti,
              ((uint32_t*)&dci->dci_pdu)[0],
              ((DCI1_1_5MHz_FDD_t *)&dci->dci_pdu[0])->rah,
              ((DCI1_1_5MHz_FDD_t *)&dci->dci_pdu[0])->rballoc,
              ((DCI1_1_5MHz_FDD_t *)&dci->dci_pdu[0])->mcs,
              ((DCI1_1_5MHz_FDD_t *)&dci->dci_pdu[0])->harq_pid,
              ((DCI1_1_5MHz_FDD_t *)&dci->dci_pdu[0])->ndi,
              ((DCI1_1_5MHz_FDD_t *)&dci->dci_pdu[0])->rv,
              ((DCI1_1_5MHz_FDD_t *)&dci->dci_pdu[0])->TPC);
        break;

      case 25:
        LOG_D(PHY,"DCI format1 (FDD, 5 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d, harq_pid %d, ndi %d, RV %d, TPC %d\n",
              dci->rnti,
              ((uint32_t*)&dci->dci_pdu)[0],
              ((DCI1_5MHz_FDD_t *)&dci->dci_pdu[0])->rah,
              ((DCI1_5MHz_FDD_t *)&dci->dci_pdu[0])->rballoc,
              ((DCI1_5MHz_FDD_t *)&dci->dci_pdu[0])->mcs,
              ((DCI1_5MHz_FDD_t *)&dci->dci_pdu[0])->harq_pid,
              ((DCI1_5MHz_FDD_t *)&dci->dci_pdu[0])->ndi,
              ((DCI1_5MHz_FDD_t *)&dci->dci_pdu[0])->rv,
              ((DCI1_5MHz_FDD_t *)&dci->dci_pdu[0])->TPC);
        break;

      case 50:
        LOG_D(PHY,"DCI format1 (FDD, 10 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d, harq_pid %d, ndi %d, RV %d, TPC %d\n",
              dci->rnti,
              ((uint32_t*)&dci->dci_pdu)[0],
              ((DCI1_10MHz_FDD_t *)&dci->dci_pdu[0])->rah,
              ((DCI1_10MHz_FDD_t *)&dci->dci_pdu[0])->rballoc,
              ((DCI1_10MHz_FDD_t *)&dci->dci_pdu[0])->mcs,
              ((DCI1_10MHz_FDD_t *)&dci->dci_pdu[0])->harq_pid,
              ((DCI1_10MHz_FDD_t *)&dci->dci_pdu[0])->ndi,
              ((DCI1_10MHz_FDD_t *)&dci->dci_pdu[0])->rv,
              ((DCI1_10MHz_FDD_t *)&dci->dci_pdu[0])->TPC);
        break;

      case 100:
        LOG_D(PHY,"DCI format1 (FDD, 20 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d, harq_pid %d, ndi %d, RV %d, TPC %d\n",
              dci->rnti,
              ((uint32_t*)&dci->dci_pdu)[0],
              ((DCI1_20MHz_FDD_t *)&dci->dci_pdu[0])->rah,
              ((DCI1_20MHz_FDD_t *)&dci->dci_pdu[0])->rballoc,
              ((DCI1_20MHz_FDD_t *)&dci->dci_pdu[0])->mcs,
              ((DCI1_20MHz_FDD_t *)&dci->dci_pdu[0])->harq_pid,
              ((DCI1_20MHz_FDD_t *)&dci->dci_pdu[0])->ndi,
              ((DCI1_20MHz_FDD_t *)&dci->dci_pdu[0])->rv,
              ((DCI1_20MHz_FDD_t *)&dci->dci_pdu[0])->TPC);
        break;

      default:
        LOG_E(PHY,"Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
        DevParam (frame_parms->N_RB_DL, 0, 0);
        break;
      }
    }

    else
      LOG_E(PHY,"Don't know how to handle TDD format 0 yet\n");

    break;

  case format1A:  // This is DLSCH allocation for control traffic
    if ((frame_parms->frame_type == TDD) &&
        (frame_parms->tdd_config>0)) {
      switch (frame_parms->N_RB_DL) {
      case 6:
        LOG_D(PHY,"DCI format1A (TDD1-6, 1_5MHz), rnti %x (%x)\n",dci->rnti,((uint32_t*)&dci->dci_pdu[0])[0]);
        LOG_D(PHY,"VRB_TYPE %d\n",((DCI1A_1_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->vrb_type);
        LOG_D(PHY,"RB_ALLOC %x (NB_RB %d)\n",((DCI1A_1_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->rballoc,RIV2nb_rb_LUT25[((DCI1A_1_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->rballoc]);
        LOG_D(PHY,"MCS %d\n",((DCI1A_1_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->mcs);
        LOG_D(PHY,"HARQ_PID %d\n",((DCI1A_1_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->harq_pid);
        LOG_D(PHY,"NDI %d\n",((DCI1A_1_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->ndi);
        LOG_D(PHY,"RV %d\n",((DCI1A_1_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->rv);
        LOG_D(PHY,"TPC %d\n",((DCI1A_1_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->TPC);
        LOG_D(PHY,"DAI %d\n",((DCI1A_1_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->dai);
        break;

      case 25:
        LOG_D(PHY,"DCI format1A (TDD1-6, 5MHz), rnti %x (%x)\n",dci->rnti,((uint32_t*)&dci->dci_pdu[0])[0]);
        LOG_D(PHY,"VRB_TYPE %d\n",((DCI1A_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->vrb_type);
        LOG_D(PHY,"RB_ALLOC %d (NB_RB %d)\n",((DCI1A_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->rballoc,RIV2nb_rb_LUT25[((DCI1A_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->rballoc]);
        LOG_D(PHY,"MCS %d\n",((DCI1A_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->mcs);
        LOG_D(PHY,"HARQ_PID %d\n",((DCI1A_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->harq_pid);
        LOG_D(PHY,"NDI %d\n",((DCI1A_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->ndi);
        LOG_D(PHY,"RV %d\n",((DCI1A_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->rv);
        LOG_D(PHY,"TPC %d\n",((DCI1A_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->TPC);
        LOG_D(PHY,"DAI %d\n",((DCI1A_5MHz_TDD_1_6_t *)&dci->dci_pdu[0])->dai);
        break;

      case 50:
        LOG_D(PHY,"DCI format1A (TDD1-6, 10MHz), rnti %x (%x)\n",dci->rnti,((uint32_t*)&dci->dci_pdu[0])[0]);
        LOG_D(PHY,"VRB_TYPE %d\n",((DCI1A_10MHz_TDD_1_6_t *)&dci->dci_pdu[0])->vrb_type);
        LOG_D(PHY,"RB_ALLOC %x (NB_RB %d)\n",((DCI1A_10MHz_TDD_1_6_t *)&dci->dci_pdu[0])->rballoc,RIV2nb_rb_LUT50[((DCI1A_10MHz_TDD_1_6_t *)&dci->dci_pdu[0])->rballoc]);
        LOG_D(PHY,"MCS %d\n",((DCI1A_10MHz_TDD_1_6_t *)&dci->dci_pdu[0])->mcs);
        LOG_D(PHY,"HARQ_PID %d\n",((DCI1A_10MHz_TDD_1_6_t *)&dci->dci_pdu[0])->harq_pid);
        LOG_D(PHY,"NDI %d\n",((DCI1A_10MHz_TDD_1_6_t *)&dci->dci_pdu[0])->ndi);
        LOG_D(PHY,"RV %d\n",((DCI1A_10MHz_TDD_1_6_t *)&dci->dci_pdu[0])->rv);
        LOG_D(PHY,"TPC %d\n",((DCI1A_10MHz_TDD_1_6_t *)&dci->dci_pdu[0])->TPC);
        LOG_D(PHY,"DAI %d\n",((DCI1A_10MHz_TDD_1_6_t *)&dci->dci_pdu[0])->dai);
        break;

      case 100:
        LOG_D(PHY,"DCI format1A (TDD1-6, 20MHz), rnti %x (%x)\n",dci->rnti,((uint32_t*)&dci->dci_pdu[0])[0]);
        LOG_D(PHY,"VRB_TYPE %d\n",((DCI1A_20MHz_TDD_1_6_t *)&dci->dci_pdu[0])->vrb_type);
        LOG_D(PHY,"RB_ALLOC %x (NB_RB %d)\n",((DCI1A_20MHz_TDD_1_6_t *)&dci->dci_pdu[0])->rballoc,RIV2nb_rb_LUT100[((DCI1A_20MHz_TDD_1_6_t *)&dci->dci_pdu[0])->rballoc]);
        LOG_D(PHY,"MCS %d\n",((DCI1A_20MHz_TDD_1_6_t *)&dci->dci_pdu[0])->mcs);
        LOG_D(PHY,"HARQ_PID %d\n",((DCI1A_20MHz_TDD_1_6_t *)&dci->dci_pdu[0])->harq_pid);
        LOG_D(PHY,"NDI %d\n",((DCI1A_20MHz_TDD_1_6_t *)&dci->dci_pdu[0])->ndi);
        LOG_D(PHY,"RV %d\n",((DCI1A_20MHz_TDD_1_6_t *)&dci->dci_pdu[0])->rv);
        LOG_D(PHY,"TPC %d\n",((DCI1A_20MHz_TDD_1_6_t *)&dci->dci_pdu[0])->TPC);
        LOG_D(PHY,"DAI %d\n",((DCI1A_20MHz_TDD_1_6_t *)&dci->dci_pdu[0])->dai);
        break;

      default:
        LOG_E(PHY,"Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
        DevParam (frame_parms->N_RB_DL, 0, 0);
        break;
      }

    } else if (frame_parms->frame_type == FDD) {
      switch (frame_parms->N_RB_DL) {
      case 6:
        LOG_D(PHY,"DCI format1A(FDD, 1.5MHz), rnti %x (%x)\n",dci->rnti,((uint32_t*)&dci->dci_pdu[0])[0]);
        LOG_D(PHY,"VRB_TYPE %d\n",((DCI1A_1_5MHz_FDD_t *)&dci->dci_pdu[0])->vrb_type);
        LOG_D(PHY,"RB_ALLOC %x (NB_RB %d)\n",((DCI1A_1_5MHz_FDD_t *)&dci->dci_pdu[0])->rballoc,RIV2nb_rb_LUT25[((DCI1A_1_5MHz_FDD_t *)&dci->dci_pdu[0])->rballoc]);
        LOG_D(PHY,"MCS %d\n",((DCI1A_1_5MHz_FDD_t *)&dci->dci_pdu[0])->mcs);
        LOG_D(PHY,"HARQ_PID %d\n",((DCI1A_1_5MHz_FDD_t *)&dci->dci_pdu[0])->harq_pid);
        LOG_D(PHY,"NDI %d\n",((DCI1A_1_5MHz_FDD_t *)&dci->dci_pdu[0])->ndi);
        LOG_D(PHY,"RV %d\n",((DCI1A_1_5MHz_FDD_t *)&dci->dci_pdu[0])->rv);
        LOG_D(PHY,"TPC %d\n",((DCI1A_1_5MHz_FDD_t *)&dci->dci_pdu[0])->TPC);
        break;

      case 25:
        LOG_D(PHY,"DCI format1A(FDD, 5MHz), rnti %x (%x)\n",dci->rnti,((uint32_t*)&dci->dci_pdu[0])[0]);
        LOG_D(PHY,"VRB_TYPE %d\n",((DCI1A_5MHz_FDD_t *)&dci->dci_pdu[0])->vrb_type);
        LOG_D(PHY,"RB_ALLOC %x (NB_RB %d)\n",((DCI1A_5MHz_FDD_t *)&dci->dci_pdu[0])->rballoc,RIV2nb_rb_LUT25[((DCI1A_5MHz_FDD_t *)&dci->dci_pdu[0])->rballoc]);
        LOG_D(PHY,"MCS %d\n",((DCI1A_5MHz_FDD_t *)&dci->dci_pdu[0])->mcs);
        LOG_D(PHY,"HARQ_PID %d\n",((DCI1A_5MHz_FDD_t *)&dci->dci_pdu[0])->harq_pid);
        LOG_D(PHY,"NDI %d\n",((DCI1A_5MHz_FDD_t *)&dci->dci_pdu[0])->ndi);
        LOG_D(PHY,"RV %d\n",((DCI1A_5MHz_FDD_t *)&dci->dci_pdu[0])->rv);
        LOG_D(PHY,"TPC %d\n",((DCI1A_5MHz_FDD_t *)&dci->dci_pdu[0])->TPC);
        break;

      case 50:
        LOG_D(PHY,"DCI format1A(FDD, 10MHz), rnti %x (%x)\n",dci->rnti,((uint32_t*)&dci->dci_pdu[0])[0]);
        LOG_D(PHY,"VRB_TYPE %d\n",((DCI1A_10MHz_FDD_t *)&dci->dci_pdu[0])->vrb_type);
        LOG_D(PHY,"RB_ALLOC %x (NB_RB %d)\n",((DCI1A_10MHz_FDD_t *)&dci->dci_pdu[0])->rballoc,RIV2nb_rb_LUT50[((DCI1A_10MHz_FDD_t *)&dci->dci_pdu[0])->rballoc]);
        LOG_D(PHY,"MCS %d\n",((DCI1A_10MHz_FDD_t *)&dci->dci_pdu[0])->mcs);
        LOG_D(PHY,"HARQ_PID %d\n",((DCI1A_10MHz_FDD_t *)&dci->dci_pdu[0])->harq_pid);
        LOG_D(PHY,"NDI %d\n",((DCI1A_10MHz_FDD_t *)&dci->dci_pdu[0])->ndi);
        LOG_D(PHY,"RV %d\n",((DCI1A_10MHz_FDD_t *)&dci->dci_pdu[0])->rv);
        LOG_D(PHY,"TPC %d\n",((DCI1A_10MHz_FDD_t *)&dci->dci_pdu[0])->TPC);
        break;

      case 100:
        LOG_D(PHY,"DCI format1A(FDD, 20MHz), rnti %x (%x)\n",dci->rnti,((uint32_t*)&dci->dci_pdu[0])[0]);
        LOG_D(PHY,"VRB_TYPE %d\n",((DCI1A_20MHz_FDD_t *)&dci->dci_pdu[0])->vrb_type);
        LOG_D(PHY,"RB_ALLOC %x (NB_RB %d)\n",((DCI1A_20MHz_FDD_t *)&dci->dci_pdu[0])->rballoc,RIV2nb_rb_LUT100[((DCI1A_20MHz_FDD_t *)&dci->dci_pdu[0])->rballoc]);
        LOG_D(PHY,"MCS %d\n",((DCI1A_20MHz_FDD_t *)&dci->dci_pdu[0])->mcs);
        LOG_D(PHY,"HARQ_PID %d\n",((DCI1A_20MHz_FDD_t *)&dci->dci_pdu[0])->harq_pid);
        LOG_D(PHY,"NDI %d\n",((DCI1A_20MHz_FDD_t *)&dci->dci_pdu[0])->ndi);
        LOG_D(PHY,"RV %d\n",((DCI1A_20MHz_FDD_t *)&dci->dci_pdu[0])->rv);
        LOG_D(PHY,"TPC %d\n",((DCI1A_20MHz_FDD_t *)&dci->dci_pdu[0])->TPC);
        break;

      default:
        LOG_E(PHY,"Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
        DevParam (frame_parms->N_RB_DL, 0, 0);
        break;
      }
    }

    break;

  case format1C:  // This is DLSCH allocation for control traffic
    switch (frame_parms->N_RB_DL) {
    case 6:
      LOG_D(PHY,"DCI format1C (1.5MHz), rnti %x (%x)\n",dci->rnti,((uint32_t*)&dci->dci_pdu[0])[0]);
      LOG_D(PHY,"RB_ALLOC %x (NB_RB %d)\n",
      ((DCI1C_1_5MHz_t *)&dci->dci_pdu[0])->rballoc,RIV2nb_rb_LUT6[conv_1C_RIV(((DCI1C_1_5MHz_t *)&dci->dci_pdu[0])->rballoc,6)]);
      LOG_D(PHY,"MCS %d\n",((DCI1C_1_5MHz_t *)&dci->dci_pdu[0])->mcs);
      break;

    case 25:
      LOG_D(PHY,"DCI format1C (5MHz), rnti %x (%x)\n",dci->rnti,((uint32_t*)&dci->dci_pdu[0])[0]);
      LOG_D(PHY,"RB_ALLOC %x (NB_RB %d)\n",((DCI1C_5MHz_t *)&dci->dci_pdu[0])->rballoc,RIV2nb_rb_LUT25[conv_1C_RIV(((DCI1C_5MHz_t *)&dci->dci_pdu[0])->rballoc,25)]);
      LOG_D(PHY,"MCS %d\n",((DCI1C_5MHz_t *)&dci->dci_pdu[0])->mcs);
      break;

    case 50:
      LOG_D(PHY,"DCI format1C (10MHz), rnti %x (%x)\n",dci->rnti,((uint32_t*)&dci->dci_pdu[0])[0]);
      LOG_D(PHY,"Ngap %d\n",((DCI1C_10MHz_t *)&dci->dci_pdu[0])->Ngap);
      LOG_D(PHY,"RB_ALLOC %x (NB_RB %d)\n",((DCI1C_10MHz_t *)&dci->dci_pdu[0])->rballoc,RIV2nb_rb_LUT50[conv_1C_RIV(((DCI1C_10MHz_t *)&dci->dci_pdu[0])->rballoc,50)]);
      LOG_D(PHY,"MCS %d\n",((DCI1C_10MHz_t *)&dci->dci_pdu[0])->mcs);
      break;

    case 100:
      LOG_D(PHY,"DCI format1C (20MHz), rnti %x (%x)\n",dci->rnti,((uint32_t*)&dci->dci_pdu[0])[0]);
      LOG_D(PHY,"Ngap %d\n",((DCI1C_20MHz_t *)&dci->dci_pdu[0])->Ngap);
      LOG_D(PHY,"RB_ALLOC %x (NB_RB %d)\n",((DCI1C_20MHz_t *)&dci->dci_pdu[0])->rballoc,RIV2nb_rb_LUT50[conv_1C_RIV(((DCI1C_20MHz_t *)&dci->dci_pdu[0])->rballoc,100)]);
      LOG_D(PHY,"MCS %d\n",((DCI1C_20MHz_t *)&dci->dci_pdu[0])->mcs);
      break;


    default:
      LOG_E(PHY,"Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
        DevParam (frame_parms->N_RB_DL, 0, 0);
        break;
    }


    break;

  case format2:

    if ((frame_parms->frame_type == TDD) &&
        (frame_parms->tdd_config>0)) {
      if (frame_parms->nb_antenna_ports_eNB == 2) {
        switch(frame_parms->N_RB_DL) {
        case 6:
          LOG_D(PHY,"DCI format2 2 antennas (TDD 1.5 MHz), rnti %x (%x): rb_alloc %x, mcs %d|%d, harq_pid %d, ndi %d|%d, RV %d|%d, TPC %d, dai %d, tbswap %d, tpmi %d\n",
                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->TPC,
                ((DCI2_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->dai,
                ((DCI2_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->tpmi
               );
          break;

        case 25:
          LOG_D(PHY,"DCI format2 2 antennas (TDD 5 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d|%d, harq_pid %d, ndi %d|%d, RV %d|%d, TPC %d, dai %d, tb_swap %d, tpmi %d\n",
                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->TPC,
                ((DCI2_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->dai,
                ((DCI2_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->tpmi);
          break;

        case 50:
          LOG_D(PHY,"DCI format2 2 antennas (TDD 10 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d|%d, harq_pid %d, ndi %d|%d, RV %d|%d, TPC %d, dai %d, tb_swap %d, tpmi %d\n",
                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->TPC,
                ((DCI2_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->dai,
                ((DCI2_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->tpmi);
          break;

        case 100:
          LOG_D(PHY,"DCI format2 2 antennas (TDD 20 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d|%d, harq_pid %d, ndi %d|%d, RV %d|%d, TPC %d, dai %d, tb_swap %d, tpmi %d\n",
                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->TPC,
                ((DCI2_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->dai,
                ((DCI2_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->tpmi);
          break;

        default:
          LOG_E(PHY,"Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
          DevParam (frame_parms->N_RB_DL, 0, 0);
          break;
        }
      } else if (frame_parms->nb_antenna_ports_eNB == 4) {
        switch(frame_parms->N_RB_DL) {
        case 6:
          LOG_D(PHY,"DCI format2 2 antennas (TDD 1.5 MHz), rnti %x (%x): rb_alloc %x, mcs %d|%d, harq_pid %d, ndi %d|%d, RV %d|%d, TPC %d, dai %d, tbswap %d, tpmi %d\n",
                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->TPC,
                ((DCI2_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->dai,
                ((DCI2_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->tpmi
               );
          break;

        case 25:
          LOG_D(PHY,"DCI format2 2 antennas (TDD 5 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d|%d, harq_pid %d, ndi %d|%d, RV %d|%d, TPC %d, dai %d, tb_swap %d, tpmi %d\n",
                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->TPC,
                ((DCI2_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->dai,
                ((DCI2_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->tpmi);
          break;

        case 50:
          LOG_D(PHY,"DCI format2 2 antennas (TDD 10 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d|%d, harq_pid %d, ndi %d|%d, RV %d|%d, TPC %d, dai %d, tb_swap %d, tpmi %d\n",
                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->TPC,
                ((DCI2_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->dai,
                ((DCI2_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->tpmi);
          break;

        case 100:
          LOG_D(PHY,"DCI format2 2 antennas (TDD 20 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d|%d, harq_pid %d, ndi %d|%d, RV %d|%d, TPC %d, dai %d, tb_swap %d, tpmi %d\n",
                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->TPC,
                ((DCI2_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->dai,
                ((DCI2_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->tpmi);
          break;

        default:
          LOG_E(PHY,"Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
          DevParam (frame_parms->N_RB_DL, 0, 0);
          break;
        }
      }
    } else if (frame_parms->frame_type == FDD) {
      if (frame_parms->nb_antenna_ports_eNB == 2) {
        switch(frame_parms->N_RB_DL) {
        case 6:
          LOG_D(PHY,"DCI format2 2 antennas (FDD, 1.5 MHz), rnti %x (%x):  rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, tb_swap %d, tpmi %d, TPC %d\n",
                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->tpmi,
                ((DCI2_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->TPC);
          break;

        case 25:

          LOG_D(PHY,"DCI format2 2 antennas (FDD, 5 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, swap %d, TPMI %d, TPC %d\n",

                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->tpmi,
                ((DCI2_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->TPC);
          break;

        case 50:
          LOG_D(PHY,"DCI format2 2 antennas (FDD, 10 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d|%d, harq_pid %d, ndi %d|%d, RV %d|%d, tb_swap %d, tpmi %d, TPC %d\n",
                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->tpmi,
                ((DCI2_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->TPC);
          break;

        case 100:
          LOG_D(PHY,"DCI format2 2 antennas (FDD, 20 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d|%d, harq_pid %d, ndi %d|%d, RV %d|%d, tb_swap %d, tpmi %d, TPC %d\n",
                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->tpmi,
                ((DCI2_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->TPC);
          break;

        default:
          LOG_E(PHY,"Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
          DevParam (frame_parms->N_RB_DL, 0, 0);
          break;
        }
      } else if (frame_parms->nb_antenna_ports_eNB == 4) {
        switch(frame_parms->N_RB_DL) {

        case 6:
          LOG_D(PHY,"DCI format2 4 antennas (FDD, 1.5 MHz), rnti %x (%x): rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, tb_swap %d, tpmi %d, TPC %d\n",
                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->tpmi,
                ((DCI2_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->TPC);
          break;

        case 25:
          LOG_D(PHY,"DCI format2 4 antennas (FDD, 5 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d|%d, harq_pid %d, ndi %d|%d, RV %d|%d, tb_swap %d, tpmi %d, TPC %d\n",
                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->tpmi,
                ((DCI2_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->TPC);
          break;

        case 50:
          LOG_D(PHY,"DCI format2 4 antennas (FDD, 10 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d|%d, harq_pid %d, ndi %d|%d, RV %d|%d, tb_swap %d, tpmi %d, TPC %d\n",
                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->tpmi,
                ((DCI2_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->TPC);
          break;

        case 100:
          LOG_D(PHY,"DCI format2 4 antennas (FDD, 20 MHz), rnti %x (%x): rah %d, rb_alloc %x, mcs %d|%d, harq_pid %d, ndi %d|%d, RV %d|%d, tb_swap %d, tpmi %d, TPC %d\n",
                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->tpmi,
                ((DCI2_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->TPC);
          break;

        default:
          LOG_E(PHY,"Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
          DevParam (frame_parms->N_RB_DL, 0, 0);
          break;
        }
      }
    }

    else
      LOG_E(PHY,"Don't know how to handle TDD format 0 yet\n");

    break;

  case format2A:

    if ((frame_parms->frame_type == TDD) &&
        (frame_parms->tdd_config>0)) {
      if (frame_parms->nb_antenna_ports_eNB == 2) {
        switch(frame_parms->N_RB_DL) {
        case 6:
          LOG_D(PHY,"DCI format2A 2 antennas (FDD 1.5 MHz), rnti %x (%x): rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, TPC %d, dai %d, tbswap %d\n",
                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2A_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2A_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2A_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2A_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2A_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2A_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2A_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2A_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2A_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->TPC,
                ((DCI2A_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->dai,
                ((DCI2A_1_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->tb_swap
               );
          break;

        case 25:
          LOG_D(PHY,"DCI format2A 2 antennas (FDD 5 MHz), rnti %x (%"PRIu64"): rah %d, rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, TPC %d, dai %d, tbswap %d\n",
                dci->rnti,
                ((uint64_t*)&dci->dci_pdu)[0],
                ((DCI2A_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2A_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2A_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2A_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2A_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2A_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2A_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2A_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2A_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2A_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->TPC,
                ((DCI2A_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->dai,
                ((DCI2A_5MHz_2A_TDD_t *)&dci->dci_pdu[0])->tb_swap);
          break;

        case 50:
          LOG_D(PHY,"DCI format2A 2 antennas (FDD 10 MHz), rnti %x (%"PRIu64"): rah %d, rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, TPC %d, dai %d, tbswap %d\n",
                dci->rnti,
                ((uint64_t*)&dci->dci_pdu)[0],
                ((DCI2A_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2A_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2A_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2A_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2A_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2A_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2A_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2A_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2A_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2A_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->TPC,
                ((DCI2A_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->dai,
                ((DCI2A_10MHz_2A_TDD_t *)&dci->dci_pdu[0])->tb_swap);
          break;

        case 100:
          LOG_D(PHY,"DCI format2A 2 antennas (FDD 20 MHz), rnti %x (%"PRIu64"): rah %d, rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, TPC %d, dai %d, tbswap %d\n",
                dci->rnti,
                ((uint64_t*)&dci->dci_pdu)[0],
                ((DCI2A_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2A_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2A_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2A_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2A_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2A_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2A_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2A_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2A_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2A_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->TPC,
                ((DCI2A_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->dai,
                ((DCI2A_20MHz_2A_TDD_t *)&dci->dci_pdu[0])->tb_swap);
          break;

        default:
          LOG_E(PHY,"Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
          DevParam (frame_parms->N_RB_DL, 0, 0);
          break;
        }
      } else if (frame_parms->nb_antenna_ports_eNB == 4) {
        switch(frame_parms->N_RB_DL) {
        case 6:
          LOG_D(PHY,"DCI format2A 4 antennas (TDD 1.5 MHz), rnti %x (%"PRIu64"): rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, TPC %d, dai %d, tbswap %d, tpmi %d\n",
                dci->rnti,
                ((uint64_t*)&dci->dci_pdu)[0],
                ((DCI2A_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2A_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2A_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2A_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2A_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2A_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2A_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2A_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2A_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->TPC,
                ((DCI2A_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->dai,
                ((DCI2A_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2A_1_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->tpmi
               );
          break;

        case 25:
          LOG_D(PHY,"DCI format2A 4 antennas (TDD 5 MHz), rnti %x (%"PRIu64"): rah %d, rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, TPC %d, dai %d, tbswap %d, tpmi %d\n",
                dci->rnti,
                ((uint64_t*)&dci->dci_pdu)[0],
                ((DCI2A_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2A_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2A_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2A_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2A_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2A_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2A_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2A_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2A_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2A_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->TPC,
                ((DCI2A_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->dai,
                ((DCI2A_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2A_5MHz_4A_TDD_t *)&dci->dci_pdu[0])->tpmi);
          break;

        case 50:
          LOG_D(PHY,"DCI format2A 4 antennas (TDD 10 MHz), rnti %x (%"PRIu64"): rah %d, rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, TPC %d, dai %d, tbswap %d, tpmi %d\n",
                dci->rnti,
                ((uint64_t*)&dci->dci_pdu)[0],
                ((DCI2A_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2A_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2A_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2A_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2A_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2A_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2A_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2A_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2A_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2A_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->TPC,
                ((DCI2A_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->dai,
                ((DCI2A_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2A_10MHz_4A_TDD_t *)&dci->dci_pdu[0])->tpmi);
          break;

        case 100:
          LOG_D(PHY,"DCI format2A 4 antennas (TDD 20 MHz), rnti %x (%"PRIu64"): rah %d, rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, TPC %d, dai %d, tbswap %d, tpmi %d\n",
                dci->rnti,
                ((uint64_t*)&dci->dci_pdu)[0],
                ((DCI2A_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2A_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2A_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2A_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2A_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2A_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2A_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2A_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2A_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2A_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->TPC,
                ((DCI2A_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->dai,
                ((DCI2A_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2A_20MHz_4A_TDD_t *)&dci->dci_pdu[0])->tpmi);
          break;

        default:
          LOG_E(PHY,"Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
          DevParam (frame_parms->N_RB_DL, 0, 0);
          break;
        }
      }
    } else if (frame_parms->frame_type == FDD) {
      if (frame_parms->nb_antenna_ports_eNB == 2) {
        switch(frame_parms->N_RB_DL) {
        case 6:
          LOG_D(PHY,"DCI format2A 2 antennas (FDD, 1.5 MHz), rnti %x (%x):  rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, tb_swap %d, TPC %d\n",
                dci->rnti,
                ((uint32_t*)&dci->dci_pdu)[0],
                ((DCI2A_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2A_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2A_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2A_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2A_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2A_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2A_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2A_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2A_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2A_1_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->TPC);
          break;

        case 25:
          LOG_D(PHY,"DCI format2A 2 antennas (FDD, 5 MHz), rnti %x (%"PRIu64"): rah %d, rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, tb_swap %d, TPC %d\n",
                dci->rnti,
                ((uint64_t*)&dci->dci_pdu)[0],
                ((DCI2A_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2A_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2A_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2A_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2A_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2A_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2A_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2A_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2A_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2A_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2A_5MHz_2A_FDD_t *)&dci->dci_pdu[0])->TPC);
          break;

        case 50:
          LOG_D(PHY,"DCI format2A 2 antennas (FDD, 10 MHz), rnti %x (%"PRIu64"): rah %d, rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, tb_swap %d, TPC %d\n",
                dci->rnti,
                ((uint64_t*)&dci->dci_pdu)[0],
                ((DCI2A_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2A_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2A_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2A_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2A_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2A_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2A_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2A_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2A_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2A_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2A_10MHz_2A_FDD_t *)&dci->dci_pdu[0])->TPC);
          break;

        case 100:
          LOG_D(PHY,"DCI format2A 2 antennas (FDD, 20 MHz), rnti %x (%"PRIu64"): rah %d, rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, tb_swap %d, TPC %d\n",
                dci->rnti,
                ((uint64_t*)&dci->dci_pdu)[0],
                ((DCI2A_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2A_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2A_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2A_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2A_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2A_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2A_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2A_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2A_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2A_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2A_20MHz_2A_FDD_t *)&dci->dci_pdu[0])->TPC);
          break;

        default:
          LOG_E(PHY,"Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
          DevParam (frame_parms->N_RB_DL, 0, 0);
          break;
        }
      } else if (frame_parms->nb_antenna_ports_eNB == 4) {
        switch(frame_parms->N_RB_DL) {

        case 6:
          LOG_D(PHY,"DCI format2A 4 antennas (FDD, 1.5 MHz), rnti %x (%"PRIu64"): rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, tb_swap %d, tpmi %d, TPC %d\n",
                dci->rnti,
                ((uint64_t*)&dci->dci_pdu)[0],
                ((DCI2A_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2A_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2A_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2A_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2A_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2A_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2A_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2A_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2A_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2A_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->tpmi,
                ((DCI2A_1_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->TPC);
          break;

        case 25:
          LOG_D(PHY,"DCI format2A 4 antennas (FDD, 5 MHz), rnti %x (%"PRIu64"): rah %d, rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, tb_swap %d, tpmi %d, TPC %d\n",
                dci->rnti,
                ((uint64_t*)&dci->dci_pdu)[0],
                ((DCI2A_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2A_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2A_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2A_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2A_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2A_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2A_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2A_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2A_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2A_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2A_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->tpmi,
                ((DCI2A_5MHz_4A_FDD_t *)&dci->dci_pdu[0])->TPC);
          break;

        case 50:
          LOG_D(PHY,"DCI format2A 4 antennas (FDD, 5 MHz), rnti %x (%"PRIu64"): rah %d, rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, tb_swap %d, tpmi %d, TPC %d\n",
                dci->rnti,
                ((uint64_t*)&dci->dci_pdu)[0],
                ((DCI2A_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2A_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2A_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2A_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2A_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2A_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2A_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2A_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2A_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2A_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2A_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->tpmi,
                ((DCI2A_10MHz_4A_FDD_t *)&dci->dci_pdu[0])->TPC);
          break;

        case 100:
          LOG_D(PHY,"DCI format2A 4 antennas (FDD, 5 MHz), rnti %x (%"PRIu64"): rah %d, rb_alloc %x, mcs1 %d, mcs2 %d, harq_pid %d, ndi1 %d, ndi2 %d, RV1 %d, RV2 %d, tb_swap %d, tpmi %d, TPC %d\n",
                dci->rnti,
                ((uint64_t*)&dci->dci_pdu)[0],
                ((DCI2A_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->rah,
                ((DCI2A_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->rballoc,
                ((DCI2A_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->mcs1,
                ((DCI2A_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->mcs2,
                ((DCI2A_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->harq_pid,
                ((DCI2A_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->ndi1,
                ((DCI2A_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->ndi2,
                ((DCI2A_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->rv1,
                ((DCI2A_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->rv2,
                ((DCI2A_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->tb_swap,
                ((DCI2A_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->tpmi,
                ((DCI2A_20MHz_4A_FDD_t *)&dci->dci_pdu[0])->TPC);
          break;

        default:
          LOG_E(PHY,"Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
          DevParam (frame_parms->N_RB_DL, 0, 0);
          break;
        }
      }
    }

    else
      LOG_E(PHY,"Don't know how to handle TDD format 0 yet\n");

    break;

  case format1E_2A_M10PRB:

    LOG_D(PHY,"DCI format1E_2A_M10PRB, rnti %x (%8x): harq_pid %d, rah %d, rb_alloc %x, mcs %d, rv %d, tpmi %d, ndi %d, dl_power_offset %d\n",
          dci->rnti,
          ((uint32_t *)&dci->dci_pdu)[0],
          ((DCI1E_5MHz_2A_M10PRB_TDD_t *)&dci->dci_pdu[0])->harq_pid,
          //((DCI1E_5MHz_2A_M10PRB_TDD_t *)&dci->dci_pdu[0])->tb_swap,
          ((DCI1E_5MHz_2A_M10PRB_TDD_t *)&dci->dci_pdu[0])->rah,
          ((DCI1E_5MHz_2A_M10PRB_TDD_t *)&dci->dci_pdu[0])->rballoc,
          ((DCI1E_5MHz_2A_M10PRB_TDD_t *)&dci->dci_pdu[0])->mcs,
          ((DCI1E_5MHz_2A_M10PRB_TDD_t *)&dci->dci_pdu[0])->rv,
          ((DCI1E_5MHz_2A_M10PRB_TDD_t *)&dci->dci_pdu[0])->tpmi,
          ((DCI1E_5MHz_2A_M10PRB_TDD_t *)&dci->dci_pdu[0])->ndi,
          ((DCI1E_5MHz_2A_M10PRB_TDD_t *)&dci->dci_pdu[0])->dl_power_off
         );

    break;

  default:
    LOG_E(PHY,"dci_tools.c: dump_dci, unknown format %d\n",dci->format);
    return(-1);
  }

  return(0);
}
