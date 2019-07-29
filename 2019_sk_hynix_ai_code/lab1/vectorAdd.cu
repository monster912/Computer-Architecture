#include<stdio.h>
#include<cuda_runtime.h>

/**
 * CUDA kernel code
 */
__global__
void vectorAdd(float *A,  float *B, float *C, int numElemnets)
{
	int i = threadIdx.x+blockDim.x*blockIdx.x;
	//vector addition
	if(i<numElemnets){
		C[i] = A[i] + B[i];
	}
}
/**
 * Host main routine
 */
int main(void)
{
	cudaError_t err = cudaSuccess;

	int n = 50000;
	size_t size = n * sizeof(float);
	// alloc host side memory
	float *h_A = (float*)malloc(size);
	float *h_B = (float*)malloc(size);
	float *h_C = (float*)malloc(size);

	//alloc device vetors
	float *d_A = NULL;
	float *d_B = NULL;
	float *d_C = NULL;
  //TODO: cudaMalloc for d_A, d_B, d_C
	cudaMalloc(&d_A,size);
	cudaMalloc(&d_B,size);
	cudaMalloc(&d_C,size);

	//init vector A and vector B
	for(int j=0;j < n; j++){
	 h_A[j] = rand()%2;
	 h_B[j] = rand()%2;
	}

	// copy host data to device
	printf("Copy input vectors to device\n");
  //TODO: cudaMemcpy h_A -> d_A, h_B -> d_B
	cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);


	//Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads \n", blocksPerGrid, threadsPerBlock);
	vectorAdd<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,d_C,n);
	err = cudaGetLastError();
	//error check
	if(err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//Copy device output data to host
	printf("Copy output data to host\n");
  //TODO: cudaMemcpy d_C -> h_C
	cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);

	//Verifiy output
	int pass = 0;

	pass = 1;
	for (int i=0;i<n;i++)
	{
		if(fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
		{
			pass = 0;
			fprintf(stderr, "Result is invalid at element %d!\n",i);
			exit(EXIT_FAILURE);
		}
	}

	if (pass)
		printf("Test PASSED\n");
	else
		printf("Test FAILED\n");

	//free device memory
  //TODO: cudaFree for d_A, d_B, d_C
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	//free host memory
	free(h_A);
	free(h_B);
	free(h_C);

	printf("Done\n");
	return 0;
}
