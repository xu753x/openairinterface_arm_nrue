#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <cuda_runtime.h>

#define SIZE  1024*1024

char *gpu;
// int16_t *cuda_temp;
void init_cuda()
{
    cudaMalloc((void**)&gpu, SIZE * sizeof(char));
}

FILE *fp;
FILE *fp1;
FILE *fp2;
void cudamemcpy(char *x,char *y,int z)
{
    cudaEvent_t start, stop;
    float time;
    cudaMemcpy(gpu, x, z * sizeof(char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();//wait to be done
    cudaEventRecord( start, 0 );
    cudaMemcpy(y, gpu, z * sizeof(char), cudaMemcpyDeviceToHost);
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize(start);
    cudaEventSynchronize( stop );//注意函数所处位置
    cudaEventElapsedTime( &time, start, stop );
    fprintf(fp2,"%.8f\n",time*1000);
}


int main()
{
    if((fp=fopen("time.txt","w"))==NULL)
        printf("Cannot open .\n");
    if((fp1=fopen("time1.txt","w"))==NULL)
        printf("Cannot open .\n");
    if((fp2=fopen("time2.txt","w"))==NULL)
        printf("Cannot open .\n");
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    init_cuda();
    char *cpu1,*cpu2;
    int a;
    cudaHostAlloc((void **)&cpu1, SIZE * sizeof(char), cudaHostAllocDefault);
    cudaHostAlloc((void **)&cpu2, SIZE * sizeof(char), cudaHostAllocDefault);
    for (int i = 0; i < SIZE; i++)
    {
        *cpu1 =rand();
        cpu1++;
    }
    
    a=1;
    for (int i = 1; i < 1024; i++)
    {
        a =i;
        printf("%d\t\t",a);
        cudaEventRecord( start, 0 );
        cudamemcpy(cpu1,cpu2,a);
        cudaEventRecord( stop, 0 );
        cudaEventSynchronize(start);
        cudaEventSynchronize( stop );//注意函数所处位置
        cudaEventElapsedTime( &time, start, stop );
        printf("cudft2048执行时间：%f(us)\n",time*1000);
        fprintf(fp,"%.8f\n",time*1000);
    }
    for (int i = 1; i < 1024; i++)
    {
        a = i;
        fprintf(fp1,"%d\n",a);
    }*/
    for (int i = 1; i < 1024; i++)
    {
        a =i*1024;
        cudamemcpy(cpu1,cpu2,a);
    }
}