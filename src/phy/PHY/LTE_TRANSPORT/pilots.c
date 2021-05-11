


#include "PHY/defs_eNB.h"
#include "PHY/LTE_REFSIG/lte_refsig.h"

void generate_pilots(PHY_VARS_eNB *eNB,
                     int32_t **txdataF,
                     int16_t amp,
                     uint16_t Ntti)
{

  LTE_DL_FRAME_PARMS *frame_parms = &eNB->frame_parms;

  uint32_t tti,tti_offset,slot_offset,Nsymb,samples_per_symbol;
  uint8_t second_pilot;



  Nsymb = (frame_parms->Ncp==0)?14:12;
  second_pilot = (frame_parms->Ncp==0)?4:3;

  //  printf("Doing TX pilots Nsymb %d, second_pilot %d\n",Nsymb,second_pilot);

  for (tti=0; tti<Ntti; tti++) {




    tti_offset = tti*frame_parms->ofdm_symbol_size*Nsymb;
    samples_per_symbol = frame_parms->ofdm_symbol_size;
    slot_offset = (tti*2)%20;

    //    printf("tti %d : offset %d (slot %d)\n",tti,tti_offset,slot_offset);
    //Generate Pilots

    //antenna port 0 symbol 0 slot 0
    lte_dl_cell_spec(eNB,&txdataF[0][tti_offset],
                     amp,
                     slot_offset,
                     0,
                     0);


    //    printf("tti %d : second_pilot offset %d \n",tti,tti_offset+(second_pilot*samples_per_symbol));
    //antenna port 0 symbol 3/4 slot 0
    lte_dl_cell_spec(eNB,&txdataF[0][tti_offset+(second_pilot*samples_per_symbol)],
                     amp,
                     slot_offset,
                     1,
                     0);

    //    printf("tti %d : third_pilot offset %d \n",tti,tti_offset+((Nsymb>>1)*samples_per_symbol));
    //antenna port 0 symbol 0 slot 1
    lte_dl_cell_spec(eNB,&txdataF[0][tti_offset+((Nsymb>>1)*samples_per_symbol)],
                     amp,
                     1+slot_offset,
                     0,
                     0);

    //    printf("tti %d : third_pilot offset %d \n",tti,tti_offset+(((Nsymb>>1)+second_pilot)*samples_per_symbol));
    //antenna port 0 symbol 3/4 slot 1
    lte_dl_cell_spec(eNB,&txdataF[0][tti_offset+(((Nsymb>>1)+second_pilot)*samples_per_symbol)],
                     amp,
                     1+slot_offset,
                     1,
                     0);


    if (frame_parms->nb_antenna_ports_eNB > 1) {

        // antenna port 1 symbol 0 slot 0
        lte_dl_cell_spec(eNB,&txdataF[1][tti_offset],
                         amp,
                         slot_offset,
                         0,
                         1);

        // antenna port 1 symbol 3 slot 0
        lte_dl_cell_spec(eNB,&txdataF[1][tti_offset+(second_pilot*samples_per_symbol)],
                         amp,
                         slot_offset,
                         1,
                         1);

        //antenna port 1 symbol 0 slot 1
        lte_dl_cell_spec(eNB,&txdataF[1][tti_offset+(Nsymb>>1)*samples_per_symbol],
                         amp,
                         1+slot_offset,
                         0,
                         1);

        // antenna port 1 symbol 3 slot 1
        lte_dl_cell_spec(eNB,&txdataF[1][tti_offset+(((Nsymb>>1)+second_pilot)*samples_per_symbol)],
                         amp,
                         1+slot_offset,
                         1,
                         1);
    }
  }
}

int generate_pilots_slot(PHY_VARS_eNB *eNB,
                         int32_t **txdataF,
                         int16_t amp,
                         uint16_t slot,
                         int first_pilot_only)
{

  LTE_DL_FRAME_PARMS *frame_parms = &eNB->frame_parms;
  uint32_t slot_offset,Nsymb,samples_per_symbol;
  uint8_t second_pilot;

  if (slot<0 || slot>= 20) {
    LOG_E(PHY,"generate_pilots_slot: slot not in range (%d)\n",slot);
    return(-1);
  }

  Nsymb = (frame_parms->Ncp==0)?7:6;
  second_pilot = (frame_parms->Ncp==0)?4:3;


  slot_offset = slot*frame_parms->ofdm_symbol_size*Nsymb;
  samples_per_symbol = frame_parms->ofdm_symbol_size;

  //    printf("tti %d : offset %d (slot %d)\n",tti,tti_offset,slot_offset);
  //Generate Pilots

  //antenna port 0 symbol 0 slot 0
  lte_dl_cell_spec(eNB,
                   &txdataF[0][slot_offset],
                   amp,
                   slot,
                   0,
                   0);


  if (first_pilot_only==0) {
    //antenna 0 symbol 3 slot 0
    lte_dl_cell_spec(eNB,
                     &txdataF[0][slot_offset+(second_pilot*samples_per_symbol)],
                     amp,
                     slot,
                     1,
                     0);
  }

  if (frame_parms->nb_antenna_ports_eNB > 1) {

    // antenna port 1 symbol 0 slot 0
    lte_dl_cell_spec(eNB,
                     &txdataF[1][slot_offset],
                     amp,
                     slot,
                     0,
                     1);

    if (first_pilot_only == 0) {
      // antenna port 1 symbol 3 slot 0
      lte_dl_cell_spec(eNB,
                       &txdataF[1][slot_offset+(second_pilot*samples_per_symbol)],
                       amp,
                       slot,
                       1,
                       1);
    }
  }

  return(0);
}

