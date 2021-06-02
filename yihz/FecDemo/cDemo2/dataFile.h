#ifndef __DEV2_DATA_FILE__
#define __DEV2_DATA_FILE__
#if 1
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
#endif
int add(int x, int y);
#endif  //__DEV2_DATA_FILE__
