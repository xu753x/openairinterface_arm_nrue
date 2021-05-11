

#include <time.h>
#include <stdlib.h>
//#include "SIMULATION/TOOLS/sim.h"


unsigned int s0, s1, s2, b;

//----------------------------------------------
//

//

unsigned int taus(void)
{
  b = (((s0 << 13) ^ s0) >> 19);
  s0 = (((s0 & 0xFFFFFFFE) << 12)^  b);
  b = (((s1 << 2) ^ s1) >> 25);
  s1 = (((s1 & 0xFFFFFFF8) << 4)^  b);
  b = (((s2 << 3) ^ s2) >> 11);
  s2 = (((s2 & 0xFFFFFFF0) << 17)^  b);
  return s0 ^ s1 ^ s2;
}

void set_taus_seed(unsigned int seed_init)
{

  struct drand48_data buffer;
  unsigned long result = 0;

  if (seed_init == 0) {
    s0 = (unsigned int)time(NULL);
    s1 = (unsigned int)time(NULL);
    s2 = (unsigned int)time(NULL);
  } else {
    /* Use reentrant version of rand48 to ensure that no conflicts with other generators occur */
    srand48_r((long int)seed_init, &buffer);
    mrand48_r(&buffer, (long int *)&result);
    s0 = result;
    mrand48_r(&buffer, (long int *)&result);
    s1 = result;
    mrand48_r(&buffer, (long int *)&result);
    s2 = result;
  }
}

#ifdef MAIN

main()
{
  unsigned int i,rand;

  set_taus_seed();

  for (i=0; i<10; i++) {
    rand = taus();
    printf("%u\n",rand);
  }
}
#endif //MAIN
