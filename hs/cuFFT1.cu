#include <stdio.h>
#include <cufft.h>

//gdb debug
// void testtest(int16_t *x,int16_t *y,unsigned char scale)
// {
//     printf("testtest \n");
// }
#define LEN 2048 //signal sampling points
void cudft2048(int16_t *x,int16_t *y,unsigned char scale)
{
    // testtest(0,0,0);
    // printf("hs222222：\n");
    cufftComplex *CompData = (cufftComplex*)malloc(LEN * sizeof(cufftComplex));//allocate memory for the data in host
    cufftComplex *CompData1 = (cufftComplex*)malloc(LEN * sizeof(cufftComplex));
    for (int i = 0; i < LEN; i++)
    {
        // printf("%d\n",i);
        CompData[i].x = x[i*2];
        CompData[i].y = x[i*2+1];
    }

    cufftComplex *d_fftData;
    cudaMalloc((void**)&d_fftData, LEN * sizeof(cufftComplex));// allocate memory for the data in device
    cudaMemcpy(d_fftData, CompData, LEN * sizeof(cufftComplex), cudaMemcpyHostToDevice);// copy data from host to device
    
    cufftHandle plan;// cuda library function handle
    cufftPlan1d(&plan, LEN, CUFFT_C2C, 1);//declaration
    cufftExecC2C(plan, (cufftComplex*)d_fftData, (cufftComplex*)d_fftData, CUFFT_FORWARD);//execute
    cudaDeviceSynchronize();//wait to be done
    cudaMemcpy(CompData1, d_fftData, LEN * sizeof(cufftComplex), cudaMemcpyDeviceToHost);// copy the result from device to host

    for (int i = 0; i < LEN; i++)
    {
        y[i*2] = CompData1[i].x/45.2;
        y[i*2+1] = CompData1[i].y/45.2;
    }
    // printf("hs1111111111111111:\n");
    // for (int i = 0; i < LEN; i++)
    // {
    //     printf("a=%d + %dj\tb=%d + %dj\n", x[i*2],x[i*2+1],y[i*2],y[i*2+1]);
    // }
    cufftDestroy(plan);
    free(CompData);
    cudaFree(d_fftData);

}
void initcudft()
{
}
int main()
{
    int16_t *a = (int16_t *)malloc(LEN * sizeof(int32_t));
    int i;
    for (i = 0; i < LEN; i++)
    {
        *(a+2*i) = i;
        *(a+2*i+1) = LEN-i;
    }
    for (i = 0; i < 3; i++)
    {
        int32_t *b = (int32_t *)malloc(LEN * sizeof(int32_t));
        cudft2048((int16_t *)a,(int16_t *)b,1);
        free(b);
    }
    
}