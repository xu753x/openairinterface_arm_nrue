#include <stdio.h>
#include <cufft.h>
#include<cuda_runtime.h>
#define LEN 2048

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
    b[id*2] = a[id].x/45.2;
    b[id*2+1] = a[id].y/45.2;
}

void initcudft()
{
}

void cudft2048(int16_t *x,int16_t *y,unsigned char scale)
{

    int16_t *x1;
    cudaMalloc((void**)&x1, LEN * sizeof(int32_t));
    cudaMemcpy(x1, x, LEN * sizeof(int32_t), cudaMemcpyHostToDevice);

    int threadNum = 512;
    int blockNum = 4;
    cufftComplex *CompData;
    cudaMalloc((void**)&CompData, LEN * sizeof(cufftComplex));
    int_cufftComplex<<<blockNum, threadNum>>>(x1, CompData, LEN);

    cufftHandle plan;// cuda library function handle
    cufftPlan1d(&plan, LEN, CUFFT_C2C, 1);//declaration,这句要warm-up
    cufftExecC2C(plan, (cufftComplex*)CompData, (cufftComplex*)CompData, CUFFT_FORWARD);//execute
    cudaDeviceSynchronize();//wait to be done

    
    cufftComplex_int<<<blockNum, threadNum>>>(CompData, x1, LEN);
    cudaMemcpy(y, x1, LEN * sizeof(int32_t), cudaMemcpyDeviceToHost);// copy the result from device to host

    // printf("hs1111111111111111:\n");
    // for (int i = 0; i < LEN; i++)
    // {
    //     printf("a=%d + %dj\tb=%d + %dj\n", x[i*2],x[i*2+1],y[i*2],y[i*2+1]);
    // }

    cufftDestroy(plan);
    cudaFree(CompData);
    cudaFree(x1);
}

int load_cuFFT(void) 
{
    initcudft();
    cudft2048(0,0,1);
    return 0;
}

int main()
{
    load_cuFFT();
    int16_t *a = (int16_t *)malloc(LEN * sizeof(int32_t));
    int16_t *b = (int16_t *)malloc(LEN * sizeof(int32_t));
    int i;
    for (i = 0; i < LEN; i++)
    {
        *(a+2*i) = i;
        *(a+2*i+1) = LEN-i;
    }
    for (i = 0; i < 3; i++)
    {
        cudft2048((int16_t *)a,(int16_t *)b,0);
    }

        

}