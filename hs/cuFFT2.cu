#include <stdio.h>
#include <cufft.h>
#include <cuda_runtime.h>

#define LEN 2048
#define SQRT2048_real 45.2876
#define SQRT2048_imag 45.3065
#define SYMBOLS_PER_SLOT 14

__global__ void int_cufftComplex(int16_t *a, cufftComplex *b, int length)
{
    int id = (blockIdx.x * blockDim.x + threadIdx.x);
    if(id >=length)
    {
        return;
    }
    b[id].x = a[id*2];
    b[id].y = a[id*2+1];
}

__global__ void cufftComplex_int(cufftComplex *a, int16_t *b, int length)
{
    int id = (blockIdx.x * blockDim.x + threadIdx.x);
    if(id >=length)
    {
        return;
    }
    b[id*2] = a[id].x/SQRT2048_real;
    b[id*2+1] = a[id].y/SQRT2048_imag;
}

// cufftComplex *fftData;
cufftComplex *d_fftData;
cufftHandle plan1;
// cufftComplex *CompData;

int16_t *temp;
int16_t *cuda_temp1;
// int16_t *cuda_temp;
void initcudft()
{
    // CompData = (cufftComplex*)malloc(SYMBOLS_PER_SLOT*LEN * sizeof(cufftComplex));
	int rank=1;
	int n[1]; n[0]=LEN;
    int nembed[2]; nembed[0]=LEN; nembed[1]=SYMBOLS_PER_SLOT;
    int stride=1;
    int dist = LEN;
    int batch=SYMBOLS_PER_SLOT;
    cufftPlanMany(&plan1,rank,n,nembed, stride ,dist , nembed, stride,dist, CUFFT_C2C, batch);
    // cufftPlan1d(&plan1, LEN, CUFFT_C2C, SYMBOLS_PER_SLOT);
    // cudaMallocHost((void **)&fftData, SYMBOLS_PER_SLOT*LEN * sizeof(cufftComplex), cudaHostAllocMapped);
    // cudaHostGetDevicePointer ((void**)&d_fftData, (void*)fftData, 0 );
    cudaMalloc((void**)&d_fftData, SYMBOLS_PER_SLOT*LEN  * sizeof(cufftComplex));
    cudaHostAlloc((void **)&temp, SYMBOLS_PER_SLOT*LEN * sizeof(int32_t), cudaHostAllocPortable);
    cudaMalloc((void **)&cuda_temp1, SYMBOLS_PER_SLOT*LEN * sizeof(int32_t));
    // cudaHostGetDevicePointer ((void**)&cuda_temp, (void*)temp, 0 );
    
}

void cudft2048(int16_t *x,int16_t *y,unsigned char scale)
{
    // cudaEvent_t start, stop;
    // float time;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord( start, 0 );
    
    // for (int i = 0; i < SYMBOLS_PER_SLOT*LEN; i++)
    // {
    //     fftData[i].x = x[i*2];
    //     fftData[i].y = x[i*2+1];
    // }
    memcpy(temp,x,SYMBOLS_PER_SLOT*LEN * sizeof(int32_t));
    int threadNum = 512;
    int blockNum = (SYMBOLS_PER_SLOT * LEN - 1) / threadNum + 1;
    

    
    int_cufftComplex<<<blockNum, threadNum>>>(temp, d_fftData, SYMBOLS_PER_SLOT*LEN);
    
   

    // memcpy(fftData,CompData,SYMBOLS_PER_SLOT*LEN * sizeof(cufftComplex));
    // cudaMemcpy(d_fftData, CompData, SYMBOLS_PER_SLOT*LEN * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    
     
    cufftExecC2C(plan1, (cufftComplex*)d_fftData, (cufftComplex*)d_fftData, CUFFT_FORWARD);//execute
    // cudaDeviceSynchronize();//wait to be done
     
    
    // memcpy(CompData,fftData,SYMBOLS_PER_SLOT*LEN * sizeof(cufftComplex));
    // cudaMemcpy(CompData, d_fftData, SYMBOLS_PER_SLOT*LEN * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < SYMBOLS_PER_SLOT*LEN; i++)
    // {
    //     y[i*2] = fftData[i].x/SQRT2048_real;
    //     y[i*2+1] = fftData[i].y/SQRT2048_imag;
    // }
    
    cufftComplex_int<<<blockNum, threadNum>>>(d_fftData, cuda_temp1, SYMBOLS_PER_SLOT*LEN);
    
    cudaMemcpy(temp, cuda_temp1, SYMBOLS_PER_SLOT*LEN * sizeof(int32_t), cudaMemcpyDeviceToHost);
    memcpy(y,temp,SYMBOLS_PER_SLOT*LEN * sizeof(int32_t));
    // cudaEventRecord( stop, 0 );
    // cudaEventSynchronize(start);
    // cudaEventSynchronize( stop );//注意函数所处位置
    // cudaEventElapsedTime( &time, start, stop );
    // printf("cudft2048执行时间：%f(us)\n",time*1000);
    // printf("----------------------------------\n");
}


void load_cuFFT(void) 
{
    initcudft();
    int16_t *a = (int16_t *)malloc(SYMBOLS_PER_SLOT*LEN * sizeof(int32_t));
    int16_t *b = (int16_t *)malloc(SYMBOLS_PER_SLOT*LEN * sizeof(int32_t));
    cudft2048(a,b,1);
}



int main()
{
    load_cuFFT();
    int16_t *a = (int16_t *)malloc(SYMBOLS_PER_SLOT*LEN * sizeof(int32_t));
    int16_t *b = (int16_t *)malloc(SYMBOLS_PER_SLOT*LEN * sizeof(int32_t));
    // int16_t *a;
    // int16_t *b;
    // cudaHostAlloc((void **)&a, SYMBOLS_PER_SLOT*LEN * sizeof(int32_t), cudaHostAllocDefault);
    // cudaHostAlloc((void **)&b, SYMBOLS_PER_SLOT*LEN * sizeof(int32_t), cudaHostAllocDefault);

    for (int j = 0; j < 10; j++)
    {
        for (int i = 0; i < SYMBOLS_PER_SLOT*LEN; i++)
        {
            *(a+2*i) = rand()%LEN;
            *(a+2*i+1) = rand()%LEN;
        }
        cudaEvent_t start, stop;
        float time;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord( start, 0 );
        cudft2048((int16_t *)a,(int16_t *)b,0);
        cudaEventRecord( stop, 0 );
        cudaEventSynchronize(start);
        cudaEventSynchronize( stop );//注意函数所处位置
        cudaEventElapsedTime( &time, start, stop );
        printf("cudft2048执行时间：%f(us)\n",time*1000);
        // printf("hs1111111111111111:\n");
        // for (int j = 0; j < SYMBOLS_PER_SLOT*LEN; j++)
        // {
        //     printf("a=%d + %dj\tb=%d + %dj\n", a[j*2],a[j*2+1],b[j*2],b[j*2+1]);
        // }
    }
}

