

#include "lte_refsig.h"




//unsigned int lte_gold_table[3][20][2][14];  // need 55 bytes for sequence
// slot index x pilot within slot x sequence

void lte_gold(LTE_DL_FRAME_PARMS *frame_parms,uint32_t lte_gold_table[20][2][14],uint16_t Nid_cell)
{
  unsigned char ns,l,Ncp=1-frame_parms->Ncp;
  unsigned int n,x1,x2;//,x1tmp,x2tmp;

  for (ns=0; ns<20; ns++) {

    for (l=0; l<2; l++) {

      x2 = Ncp +
           (Nid_cell<<1) +
       (((1+(Nid_cell<<1))*(1 + (((frame_parms->Ncp==0)?4:3)*l) + (7*(1+ns))))<<10); //cinit
      //x2 = frame_parms->Ncp + (Nid_cell<<1) + (1+(Nid_cell<<1))*(1 + (3*l) + (7*(1+ns))); //cinit
      //n = 0
      x1 = 1+ (1U<<31);
      x2=x2 ^ ((x2 ^ (x2>>1) ^ (x2>>2) ^ (x2>>3))<<31);

      // skip first 50 double words (1600 bits)
      for (n=1; n<50; n++) {
        x1 = (x1>>1) ^ (x1>>4);
        x1 = x1 ^ (x1<<31) ^ (x1<<28);
        x2 = (x2>>1) ^ (x2>>2) ^ (x2>>3) ^ (x2>>4);
        x2 = x2 ^ (x2<<31) ^ (x2<<30) ^ (x2<<29) ^ (x2<<28);
      }

      for (n=0; n<14; n++) {
        x1 = (x1>>1) ^ (x1>>4);
        x1 = x1 ^ (x1<<31) ^ (x1<<28);
        x2 = (x2>>1) ^ (x2>>2) ^ (x2>>3) ^ (x2>>4);
        x2 = x2 ^ (x2<<31) ^ (x2<<30) ^ (x2<<29) ^ (x2<<28);
        lte_gold_table[ns][l][n] = x1^x2;
      }

    }

  }
}

void lte_gold_ue_spec(uint32_t lte_gold_uespec_table[2][20][2][21],uint16_t Nid_cell, uint16_t *n_idDMRS)
{

  unsigned char ns,l;
  unsigned int n,x1,x2;//,x1tmp,x2tmp;
  int nscid;
  int nid;

  for (nscid=0; nscid<2; nscid++) {
    if (n_idDMRS)
      nid = n_idDMRS[nscid];
    else
      nid = Nid_cell;

    for (ns=0; ns<20; ns++) {

      for (l=0; l<2; l++) {


        x2 = ((((ns>>1)+1)*((nid<<1)+1))<<16) + nscid;
        //x2 = frame_parms->Ncp + (Nid_cell<<1) + (1+(Nid_cell<<1))*(1 + (3*l) + (7*(1+ns))); //cinit
        //n = 0
        //      printf("cinit (ns %d, l %d) => %d\n",ns,l,x2);
        x1 = 1+ (1U<<31);
        x2=x2 ^ ((x2 ^ (x2>>1) ^ (x2>>2) ^ (x2>>3))<<31);

        // skip first 50 double words (1600 bits)
        //printf("n=0 : x1 %x, x2 %x\n",x1,x2);
        for (n=1; n<50; n++) {
          x1 = (x1>>1) ^ (x1>>4);
          x1 = x1 ^ (x1<<31) ^ (x1<<28);
          x2 = (x2>>1) ^ (x2>>2) ^ (x2>>3) ^ (x2>>4);
          x2 = x2 ^ (x2<<31) ^ (x2<<30) ^ (x2<<29) ^ (x2<<28);
          //  printf("x1 : %x, x2 : %x\n",x1,x2);
        }

        for (n=0; n<14; n++) {
          x1 = (x1>>1) ^ (x1>>4);
          x1 = x1 ^ (x1<<31) ^ (x1<<28);
          x2 = (x2>>1) ^ (x2>>2) ^ (x2>>3) ^ (x2>>4);
          x2 = x2 ^ (x2<<31) ^ (x2<<30) ^ (x2<<29) ^ (x2<<28);
          lte_gold_uespec_table[nscid][ns][l][n] = x1^x2;
          //  printf("n=%d : c %x\n",n,x1^x2);
        }

      }
    }
  }
}

void lte_gold_ue_spec_port5(uint32_t lte_gold_uespec_port5_table[20][38],uint16_t Nid_cell, uint16_t n_rnti)
{

  unsigned char ns;
  unsigned int n,x1,x2;


  for (ns=0; ns<20; ns++) {

    x2 = ((((ns>>1)+1)*((Nid_cell<<1)+1))<<16) + n_rnti;
    //x2 = frame_parms->Ncp + (Nid_cell<<1) + (1+(Nid_cell<<1))*(1 + (3*l) + (7*(1+ns))); //cinit
    //n = 0
    //printf("cinit (ns %d, l %d) => %d\n",ns,l,x2);
    x1 = 1+ (1U<<31);
    x2=x2 ^ ((x2 ^ (x2>>1) ^ (x2>>2) ^ (x2>>3))<<31);

    //skip first 50 double words (1600 bits)
    //printf("n=0 : x1 %x, x2 %x\n",x1,x2);
    for (n=1; n<50; n++) {
      x1 = (x1>>1) ^ (x1>>4);
      x1 = x1 ^ (x1<<31) ^ (x1<<28);
      x2 = (x2>>1) ^ (x2>>2) ^ (x2>>3) ^ (x2>>4);
      x2 = x2 ^ (x2<<31) ^ (x2<<30) ^ (x2<<29) ^ (x2<<28);
      //printf("x1 : %x, x2 : %x\n",x1,x2);
    }

    for (n=0; n<38; n++) {
      x1 = (x1>>1) ^ (x1>>4);
      x1 = x1 ^ (x1<<31) ^ (x1<<28);
      x2 = (x2>>1) ^ (x2>>2) ^ (x2>>3) ^ (x2>>4);
      x2 = x2 ^ (x2<<31) ^ (x2<<30) ^ (x2<<29) ^ (x2<<28);
      lte_gold_uespec_port5_table[ns][n] = x1^x2;
      //printf("n=%d : c %x\n",n,x1^x2);
    }

  }
}

#ifdef LTE_GOLD_MAIN
main()
{

  lte_gold(423,0);

}
#endif

