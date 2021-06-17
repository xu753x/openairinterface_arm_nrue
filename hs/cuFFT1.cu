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

int16_t *cuda_x;
int16_t *cuda_y;
int16_t *x11;
cufftComplex *CompData1;
cufftHandle plan1;
void initcudft()
{
    cudaMalloc((void**)&x11, SYMBOLS_PER_SLOT*LEN * sizeof(int32_t));
    cudaMalloc((void**)&CompData1, SYMBOLS_PER_SLOT*LEN * sizeof(cufftComplex));
    // cufftPlan1d(&plan1, LEN, CUFFT_C2C, SYMBOLS_PER_SLOT);
	int rank=1;
	int n[1]; n[0]=LEN;
    int nembed[2]; nembed[0]=LEN; nembed[1]=SYMBOLS_PER_SLOT;
    int stride=1;
    int dist = LEN;
    int batch=SYMBOLS_PER_SLOT;
    cufftPlanMany(&plan1,rank,n,nembed, stride ,dist , nembed, stride,dist, CUFFT_C2C, batch);
    cudaHostAlloc((void **)&cuda_x, SYMBOLS_PER_SLOT*LEN * sizeof(int32_t), cudaHostAllocDefault);
    cudaHostAlloc((void **)&cuda_y, SYMBOLS_PER_SLOT*LEN * sizeof(int32_t), cudaHostAllocDefault);
}

void cudft2048(int16_t *x,int16_t *y,unsigned char scale)
{
    // cudaStream_t stream;
	// cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );

    memcpy(cuda_x,x,SYMBOLS_PER_SLOT*LEN * sizeof(int32_t));
    cudaMemcpy(x11, cuda_x, SYMBOLS_PER_SLOT*LEN * sizeof(int32_t), cudaMemcpyHostToDevice);

    int threadNum = 512;
    int blockNum = (SYMBOLS_PER_SLOT * LEN - 1) / threadNum + 1;
    
    int_cufftComplex<<<blockNum, threadNum>>>(x11, CompData1, SYMBOLS_PER_SLOT*LEN);

    cufftExecC2C(plan1, (cufftComplex*)CompData1, (cufftComplex*)CompData1, CUFFT_FORWARD);//execute
    cudaDeviceSynchronize();//wait to be done

    cufftComplex_int<<<blockNum, threadNum>>>(CompData1, x11, SYMBOLS_PER_SLOT*LEN);
    cudaMemcpy(cuda_y, x11, SYMBOLS_PER_SLOT*LEN * sizeof(int32_t), cudaMemcpyDeviceToHost);// copy the result from device to host
    memcpy(y,cuda_y,SYMBOLS_PER_SLOT*LEN * sizeof(int32_t));

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize(start);
    cudaEventSynchronize( stop );//注意函数所处位置
    cudaEventElapsedTime( &time, start, stop );
    printf("cudft2048执行时间：%f(us)\n",time*1000);

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
    for (int i = 0; i < SYMBOLS_PER_SLOT*LEN; i++)
    {
        *(a+2*i) = rand()%LEN;
        *(a+2*i+1) = rand()%LEN;
    }
    for (int i = 0; i < 10; i++)
    {
        cudft2048((int16_t *)a,(int16_t *)b,0);
        // printf("hs1111111111111111:\n");
        // for (int j = 0; j < SYMBOLS_PER_SLOT*LEN; j++)
        // {
        //     printf("a=%d + %dj\tb=%d + %dj\n", a[j*2],a[j*2+1],b[j*2],b[j*2+1]);
        // }
    }
}
