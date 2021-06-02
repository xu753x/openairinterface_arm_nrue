#ifndef __DEV2_DATA_FILE__
#define __DEV2_DATA_FILE__

//
int getBinDataFromFile(unsigned char *pData,int *len, char * fname);
//
void saveBinDataTofile(unsigned char *pData,int len, char *fname);
//
void saveBinData(unsigned char *pData, int type);
//
int compare_data(unsigned char *pDst1, unsigned char *pDst2, int len);
//
void load_expect_data(unsigned char *pData, char *fname);



#endif  //__DEV2_DATA_FILE__
