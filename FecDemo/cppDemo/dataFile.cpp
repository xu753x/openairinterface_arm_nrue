#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>


#include <cstdio>
#include <ctime>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdint>

#include "logger_wrapper.h"
#include "ff_define.h"


//using namespace std;


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
/*
    count=ret/32;
    for(j=0;j<count;j++){
        for(i=0;i<16;i++){
            tempData=pData[i+j*32];
            pData[i+j*32]=pData[j*32+31-i];
            pData[j*32+31-i]=tempData;
        }
    }
*/

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
char saveDataDirDe[]="./saveData/de";
char saveDataDirEn[]="./saveData/en";
static int count1=0,count2=0;
void saveBinData(unsigned char *pData, int type)
{
    int i;
    DataHeadInfo *pDataHeadInfo;
    DataHeadInfo TempDataHead;
    unsigned char *pDataHead;
    char fname[64];

/*
    for(i=0;i<32;i++){
        //printf("%02x_",*(pData+i));
        sprintf(tempData+(i*3),"%02x_",*(pData+i));
    }
    *(tempData+32*3)='\0';
    //sprintf(tempData+(6*3),"%02x_", 0x17);
    zLog(PHY_LOG_INFO,"rec header:=%s---[DRV_LOG]",tempData);
*/
    pDataHeadInfo=(DataHeadInfo *)pData;
    //zLog(PHY_LOG_INFO,"rec header: len=%x---[DRV_LOG]",pDataHeadInfo->pktLen);
    if(type==FEC_DECODE_DATA){
       sprintf(fname,"%s/decode_%08d.bin",saveDataDirDe,count2++);
    }else if(type==FEC_ENCODE_DATA){
       sprintf(fname,"%s/encode_%08d.bin",saveDataDirEn,count1++);
    }
    saveBinDataTofile(pData, pDataHeadInfo->pktLen+32, fname);
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



