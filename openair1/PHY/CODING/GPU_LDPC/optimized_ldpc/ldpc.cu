#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "bg1_i1_index_array.h"

void print_arr_cpu(const char *file, int *arr, int size)
{
	FILE *fp = fopen(file, "w");
	for(int i = 0; i < size; i++){
		fprintf(fp, "%s[%d]: %d\n", file, i, arr[i]);
	}
	fclose(fp);
}

void print_arr(const char *file, int *arr, int size)
{
	int *tmp = (int*)malloc(sizeof(int)*size);
	FILE *fp = fopen(file, "w");
	cudaMemcpy((void*)tmp, (const void*)arr, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for(int i = 0; i < size; i++){
		fprintf(fp, "%s[%d]: %d\n", file, i, tmp[i]);
	}
	free(tmp);
	fclose(fp);
}

__global__ void llr2CN(int *llr, int *cnbuf, int *l2c_idx)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	cnbuf[tid] = llr[l2c_idx[tid]];
	__syncthreads();
}

__global__ void llr2BN(int *llr, int *const_llr, int *l2b_idx)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	const_llr[tid] = llr[l2b_idx[tid]];
	__syncthreads();
}

__global__ void CNProcess(int *cnbuf, int *bnbuf, int *b2c_idx, int *cnproc_idx)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	int start = cnproc_idx[tid*2];
	int end = cnproc_idx[tid*2+1];
	
	
	int sgn = 1, val = INT32_MAX;
	for(int i = start; i < end; i++){
		if(i == tid)	continue;

		int tmp = cnbuf[i];
		if(tmp < 0){
			tmp = -tmp;
			sgn = -sgn;
		}
		if(val > tmp){
			val = tmp;
		}
	}
	bnbuf[b2c_idx[tid]] = sgn*val;// + const_llr[tid];
	__syncthreads();
}

__global__ void BNProcess(int *const_llr, int *bnbuf, int *cnbuf, int *c2b_idx, int *bnproc_idx)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	
	int start = bnproc_idx[tid*2];
	int end = bnproc_idx[tid*2+1];
	
	int val = 0;
	for(int i = start; i < end; i++){
		if(i == tid)	continue;
		val += bnbuf[i];
	}
	cnbuf[c2b_idx[tid]] = val + const_llr[tid];
	__syncthreads();
}

__global__ void BN2llr(int *bnbuf, int *llrbuf, int *llr_idx)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	int start = llr_idx[tid];
	int end = llr_idx[tid+1];

	int res = 0;
	for(int i = start; i < end; i++){
		res += bnbuf[i];
	}
	llrbuf[tid] = res;
	__syncthreads();
}

__global__ void BitDetermination(int *BN, unsigned int *decode_d)
{
	__shared__ int tmp[256];
	int tid = blockIdx.x*256 + threadIdx.x;
	int bid = threadIdx.x;
	tmp[bid] = 0;
	
	
	if(BN[tid] < 0)
	{
		tmp[bid] = 1 << (bid&7);
	}

	__syncthreads();
	
	if(threadIdx.x < 32)
	{
		decode_d[blockIdx.x*32 + threadIdx.x] = 0;
		for(int i = 0; i < 8; i++)
		{
			decode_d[blockIdx.x*32 + threadIdx.x] += tmp[threadIdx.x*8+i];
		}
	}
}

void Read_Data(char *filename, int *data_sent, int *data_received)
{
	FILE *fp = fopen(filename, "r");
	fscanf(fp, "%*s");
	for(int i = 0; i < 1056; i++){
		fscanf(fp, "%d", &data_sent[i]);
	}
	fscanf(fp, "%*s");
	fscanf(fp, "%*s");
	fscanf(fp, "%*s");
	for(int i = 0; i < 26112; i++){
		fscanf(fp, "%d", &data_received[i]);
	}
	fclose(fp);
}

int main(int argc, char **argv)
{
	int *input = (int*)malloc(1056*sizeof(int));
	int *llr = (int*)malloc(26112*sizeof(int));

	int *llr_d, *llrbuf_d, *const_llr_d, *cnbuf_d, *bnbuf_d;
	unsigned int *decode_output_h, *decode_output_d;

	int *l2c_idx_d, *cnproc_idx_d, *c2b_idx_d, *bnproc_idx_d, *b2c_idx_d, *llr_idx_d, *l2b_idx_d;

	char *file = argv[1];
	
	
	int blockNum = 237, threadNum = 512;
	//int blockNum = 33, threadNum = 256;
	//int blockNum = 17, threadNum = 512;

	int rounds = 5, Zc = 384;

	Read_Data(file, input, llr);



	size_t p_llr;
	cudaHostAlloc((void**)&decode_output_h, 1056*sizeof(unsigned int), cudaHostAllocMapped);

	cudaMallocPitch((void**)&llr_d, &p_llr, 26112*sizeof(int), 1);
	cudaMallocPitch((void**)&llrbuf_d, &p_llr, 26112*sizeof(int), 1);
	cudaMallocPitch((void**)&const_llr_d, &p_llr, 316*384*sizeof(int), 1);
	cudaMallocPitch((void**)&cnbuf_d, &p_llr, 316*384*sizeof(int), 1);
	cudaMallocPitch((void**)&bnbuf_d, &p_llr, 316*384*sizeof(int), 1);
	cudaMallocPitch((void**)&l2c_idx_d, &p_llr, 316*384*sizeof(int), 1);
	cudaMallocPitch((void**)&l2b_idx_d, &p_llr, 316*384*sizeof(int), 1);
	cudaMallocPitch((void**)&cnproc_idx_d, &p_llr, 316*384*2*sizeof(int), 1);
	cudaMallocPitch((void**)&c2b_idx_d, &p_llr, 316*384*sizeof(int), 1);
	cudaMallocPitch((void**)&bnproc_idx_d, &p_llr, 316*384*2*sizeof(int), 1);
	cudaMallocPitch((void**)&b2c_idx_d, &p_llr, 316*384*sizeof(int), 1);
	cudaMallocPitch((void**)&llr_idx_d, &p_llr, 26113*sizeof(int), 1);

	cudaMemcpyAsync((void*)llr_d, (const void*)llr, 68*384*sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemcpyAsync((void*)l2c_idx_d, (const void*)l2c_idx, 316*384*sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemcpyAsync((void*)l2b_idx_d, (const void*)l2b_idx, 316*384*sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemcpyAsync((void*)cnproc_idx_d, (const void*)cnproc_idx, 316*384*2*sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemcpyAsync((void*)c2b_idx_d, (const void*)c2b_idx, 316*384*sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemcpyAsync((void*)bnproc_idx_d, (const void*)bnproc_idx, 316*384*2*sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemcpyAsync((void*)b2c_idx_d, (const void*)b2c_idx, 316*384*sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemcpyAsync((void*)llr_idx_d, (const void*)llr_idx, 26113*sizeof(int), cudaMemcpyHostToDevice);	

	cudaHostGetDevicePointer((void**)&decode_output_d, (void*)decode_output_h, 0);
	cudaDeviceSynchronize();


/*
	cudaEvent_t start, end;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);
*/

	llr2CN<<<blockNum, threadNum>>>(llr_d, cnbuf_d, l2c_idx_d);
	llr2BN<<<blockNum, threadNum>>>(llr_d, const_llr_d, l2b_idx_d);

/*
	print_arr("debug/const_llr_d", const_llr_d, 26112);
	print_arr("debug/cnbuf_d", cnbuf_d, 316*384);
	print_arr("debug/const_llrbuf_d", const_llrbuf_d, 316*384);
*/

	char debug[] = "debug/";
	char cn[] = "cnbuf";
	char bn[] = "bnbuf";
	char llrstr[] = "llrbuf_d";
	char str[100] = {};
	for(int i = 0; i < rounds; i++){
		CNProcess<<<blockNum, threadNum>>>(cnbuf_d, bnbuf_d, b2c_idx_d, cnproc_idx_d);
	//	snprintf(str, 20, "%s%s_%d", debug, bn, i+1);
	//	print_arr(str, bnbuf_d, 316*384);

		BNProcess<<<blockNum, threadNum>>>(const_llr_d, bnbuf_d, cnbuf_d, c2b_idx_d, bnproc_idx_d);
	//	snprintf(str, 20, "%s%s_%d", debug, cn, i+1);
	//	print_arr(str, cnbuf_d, 316*384);

		BN2llr<<<51, 512>>>(bnbuf_d, llrbuf_d, llr_idx_d);
	//	snprintf(str, 20, "%s%s_%d", debug, llrstr, i+1);
	//	print_arr(str, llrbuf_d, 26112);
	}

	BitDetermination<<<33, 256>>>(llrbuf_d, decode_output_d);
	cudaDeviceSynchronize();

/*
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	printf("time: %.6f ms\n", time);
*/

	int err = 0;
	for(int i = 0; i < 8448/8; i++){
		if(input[i] != decode_output_h[i]){
//			printf("input[%d] :%d, decode_output[%d]: %d\n", i, input[i], i, decode_output_h[i]);
			err++;
		}
	}
	printf("err: %d\n", err);

	free(input);
	free(llr);
	cudaFree(llr_d);
	cudaFree(llrbuf_d);
	cudaFree(bnbuf_d);
	cudaFree(cnbuf_d);
	cudaFree(l2c_idx_d);
	cudaFree(cnproc_idx_d);
	cudaFree(c2b_idx_d);
	cudaFree(bnproc_idx_d);
	cudaFree(b2c_idx_d);
	cudaFree(const_llr_d);
	cudaFree(llr_idx_d);

	cudaFreeHost(decode_output_h);
	return 0;
}
