// add rotation

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
__global__ void rotate_cpx_vector(int16_t *x,
                      int16_t *alpha,
                      int16_t *y,
                      unsigned int N,
                      unsigned short output_shift)
{
  int id = (blockIdx.x * blockDim.x + threadIdx.x);
  if(id >=N)
    return;
  int temp,temp1;
  //x=a+bi,alpha=c+di
  temp = (*(x+2*id)) * (*(alpha)) - (*(x+1+2*id)) * (*(alpha+1));
  temp1 = (*(x+2*id)) * (*(alpha+1)) + (*(x+1+2*id)) * (*(alpha));
  *(y+2*id) = temp>>output_shift;//这四句不能颠倒，防止y=x的情况
  *(y+2*id+1) = temp1>>output_shift;
}
__global__ void rotate_cpx_vector1(int16_t *x,
                      int16_t *alpha,
                      int16_t *y,
                      unsigned int N,
                      unsigned short output_shift)
{
  int32_t temp,temp1;
  int id = (blockIdx.x * blockDim.x + threadIdx.x);
  if(id >=N)
    return;
  //x=a+bi,alpha=c+di
  int id_temp=id/LEN;
  temp = (*(x+2*id)) * (*(alpha+2*id_temp)) - (*(x+1+2*id)) * (*(alpha+2*id_temp+1));
  temp1 = (*(x+2*id)) * (*(alpha+2*id_temp+1)) + (*(x+1+2*id)) * (*(alpha+2*id_temp));
  *(y+2*id) = temp>>output_shift;//这四句不能颠倒，防止y=x的情况
  *(y+2*id+1) = temp1>>output_shift;
}

// cufftComplex *fftData;
cufftComplex *d_fftData;
cufftHandle plan1;
// cufftComplex *CompData;

int16_t *temp;
int16_t *cuda_temp1;
int16_t *cuda_temp2;
int16_t *cuda_temp3;
int16_t *cuda_alpha;
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
    cudaMalloc((void **)&cuda_temp2, SYMBOLS_PER_SLOT*LEN * sizeof(int32_t));
    cudaMalloc((void **)&cuda_temp3, SYMBOLS_PER_SLOT*LEN * sizeof(int32_t));
    cudaMalloc((void **)&cuda_alpha, SYMBOLS_PER_SLOT*LEN * sizeof(int32_t));
    // cudaHostGetDevicePointer ((void**)&cuda_temp, (void*)temp, 0 );
    
}

void cudft2048(int16_t *x,int16_t *y,int16_t *alpha)
{
    memcpy(temp,x,SYMBOLS_PER_SLOT*LEN * sizeof(int32_t));
    int threadNum = 512;
    int blockNum = (SYMBOLS_PER_SLOT * LEN - 1) / threadNum + 1;

    int_cufftComplex<<<blockNum, threadNum>>>(temp, d_fftData, SYMBOLS_PER_SLOT*LEN);
     
    cufftExecC2C(plan1, (cufftComplex*)d_fftData, (cufftComplex*)d_fftData, CUFFT_FORWARD);//execute
    cudaDeviceSynchronize();//wait to be done
    
    cufftComplex_int<<<blockNum, threadNum>>>(d_fftData, cuda_temp1, SYMBOLS_PER_SLOT*LEN);
    cudaMemcpy(cuda_alpha, alpha, SYMBOLS_PER_SLOT*sizeof(int32_t), cudaMemcpyHostToDevice);
    rotate_cpx_vector1<<<blockNum, threadNum>>>(cuda_temp1, cuda_alpha, cuda_temp2, SYMBOLS_PER_SLOT*LEN,15);
    cudaMemcpy(temp, cuda_temp2, SYMBOLS_PER_SLOT*LEN * sizeof(int32_t), cudaMemcpyDeviceToHost);
    memcpy(y,temp,SYMBOLS_PER_SLOT*LEN * sizeof(int32_t));
}

void cuda_rotate_cpx_vector(int16_t *x,
                      int16_t *alpha,
                      int16_t *y,
                      unsigned int N,
                      unsigned short output_shift)
{
  // int temp,temp1;
  // for(int i=0; i<N; i++) {
  //   temp = (*(x+2*i)) * (*(alpha)) - (*(x+2*i+1)) * (*(alpha + 1)); //R(y) = (a*c - b*d)/|alpha|
  //   temp1 = (*(x+2*i)) * (*(alpha + 1)) + (*(x+2*i+1)) * (*(alpha)); //I(y) = (a*d + b*c)/|alpha|
  //   *(y+2*i) = temp >> output_shift;//这四句不能颠倒，防止y=x的情况
  //   *(y+2*i+1) = temp1 >> output_shift;
  // }

  cudaMemcpy(cuda_temp2, x, SYMBOLS_PER_SLOT*LEN * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_alpha, alpha, SYMBOLS_PER_SLOT*sizeof(int32_t), cudaMemcpyHostToDevice);
  int threadNum = 512;
  int blockNum = (N - 1) / threadNum + 1;
  rotate_cpx_vector1<<<blockNum, threadNum>>>(cuda_temp2, cuda_alpha, cuda_temp3, N,output_shift);
  cudaMemcpy(y, cuda_temp3, SYMBOLS_PER_SLOT*LEN * sizeof(int32_t), cudaMemcpyDeviceToHost);
}

void load_cuFFT(void) 
{
    initcudft();
    int16_t *a = (int16_t *)malloc(SYMBOLS_PER_SLOT*LEN * sizeof(int32_t));
    int16_t *b = (int16_t *)malloc(SYMBOLS_PER_SLOT*LEN * sizeof(int32_t));
    int16_t *c = (int16_t *)malloc(224 * 2 * sizeof(int16_t));
    cudft2048(a,b,c);
}



int main()
{
    load_cuFFT();//= [-1 + j*(-32767)
    int16_t *a = (int16_t *)malloc(LEN * sizeof(int32_t));
    int16_t *b = (int16_t *)malloc(LEN * sizeof(int32_t));
    for (int j = 0; j < 10; j++)
    {
        for (int i = 0; i < LEN; i++)
        {
            *(a+2*i) = rand()%LEN;
            *(a+2*i+1) = rand()%LEN;
        }
    }
    int16_t alpha[] = {-1,-32767};
    cuda_rotate_cpx_vector(a,alpha,b,LEN,15);
    for (int j = 0; j < LEN; j++)
    {
        printf("a=%d + %dj\tb=%d + %dj\n", a[j*2],a[j*2+1],b[j*2],b[j*2+1]);
    }
}

