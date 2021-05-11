

openair0_device openair0;
volatile int oai_exit=0;

void exit_function(const char* file, const char* function, const int line,const char *s) {
   const char * msg= s==NULL ? "no comment": s;
   printf("Exiting at: %s:%d %s(), %s\n", file, line, function, msg);
   exit(-1);
}

extern unsigned int dlsch_tbs25[27][25],TBStable[27][110];
extern unsigned char offset_mumimo_llr_drange_fix;

extern unsigned short dftsizes[34];
extern short *ul_ref_sigs[30][2][34];
