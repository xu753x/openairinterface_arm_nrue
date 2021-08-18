#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>

#if 1
#include <dev2.0/logger_wrapper.h>
#include "ff_define.h"


int getBinDataFromFile(unsigned char *pData, int *len,char *fname)
{
    int fileSize,ret;
    FILE *fp;
    int i,j,count;
    int tempData;

    fp=fopen(fname,"rb");
    if(fp==NULL){
        printf("This %s file is open failed.\n",fname);
    }

    fseek(fp,0,SEEK_END);
    fileSize=ftell(fp);
    rewind(fp);
    printf("fileSize=0x%x\n",fileSize);
    ret=fread(pData,1,fileSize,fp);
    *len=fileSize;
    //printf("ret=0x%x\n",ret);
    fclose(fp);

    return 1;
}

//
//
//
////////////////////////////////
void saveBinDataTofile(unsigned char *pData,int len,char * fname)
{
    int i,fileSize,ret;
    FILE *fp;

    fp=fopen(fname,"wb+");
    if(fp==NULL){
        printf("This %s file is open failed.\n",fname);
    }

    ret=fwrite(pData,1,len,fp);
    //printf("ret=0x%x\n",ret);

    fclose(fp);
}

char tempData[512];
char saveDataDirDe[]="/tmp/saveData/de";
char saveDataDirEn[]="/tmp/saveData/en";
static int count1=0,count2=0,loop_count_de=0,loop_count_en=0;
void saveBinData(unsigned char *pData, int type)
{
    int i;
    DataHeadInfo *pDataHeadInfo;
    DataHeadInfo TempDataHead;
    unsigned char *pDataHead;
    char fname[64];

    pDataHeadInfo=(DataHeadInfo *)pData;
    if(type==FEC_DECODE_DATA){
        loop_count_de++;
        if(loop_count_de>=10000){
            sprintf(fname,"%s/decode_%08d.bin",saveDataDirDe,count2++);
            saveBinDataTofile(pData, pDataHeadInfo->pktLen+16, fname);
            loop_count_de=0;
        }
    }else if(type==FEC_ENCODE_DATA){
        loop_count_en++;
        if(loop_count_en>=100000){
            sprintf(fname,"%s/encode_%08d.bin",saveDataDirEn,count1++);
            saveBinDataTofile(pData, pDataHeadInfo->pktLen+16, fname);
            loop_count_en=0;
        }
    }
    //zLog(PHY_LOG_INFO,"rec header: len=%x---[DRV_LOG]",pDataHeadInfo->pktLen);
}



//
//
//
//
////////////////////////////////////////////
void load_expect_data(unsigned char *pData, char *fname)
{
    int len;

    getBinDataFromFile(pData, &len, fname);
}

//
//
//
//
////////////////////////////////////////////
int compare_data(unsigned char *pDst1, unsigned char *pDst2, int len)
{
    int ret=0;

    //ret=memcmp(pDst1,pDst2,len);
    return ret;
}

#endif
#include "dataFile.h"
int add(int x, int y)
{
    printf("dataFile x+y = %d\n", x+y);
    return(x+y);
}

