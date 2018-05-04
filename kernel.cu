
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdio.h>
#include <assert.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "GpuTimer.h"

#define BLOCK_WIDTH 16

cudaError_t testWithCuda(uint8_t *inputImage, uint8_t *outputImage, unsigned int imageWidth, unsigned int imageHeight, unsigned int imageChannels);

__global__ void testKernel(uint8_t *inputImage, uint8_t *outputImage, unsigned int imageWidth, unsigned int imageHeight, unsigned int imageChannels)
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int pixelIdx = (imageWidth * y + x) * imageChannels;

	if (x < imageWidth && y < imageHeight) {
		unsigned int subpixelIdx;
		for (unsigned int i = 0; i < imageChannels; i++) {
			subpixelIdx = pixelIdx + i;
			outputImage[subpixelIdx] = inputImage[subpixelIdx];
		}
	}
}

int main(int argc, char *argv[])
{
	if (argc != 2) {
		fprintf(stderr, "Invalid arguments.");
		return 1;
	}
	char *imageName = argv[1];
	cv::Mat inputImage = cv::imread(imageName, CV_LOAD_IMAGE_COLOR);
	if (inputImage.empty()) {
		fprintf(stderr, "Cannot read image file %s.", imageName);
		return 1;
	}
	unsigned int imageSize = inputImage.cols * inputImage.rows * inputImage.channels();
	uint8_t *outputImageData = (uint8_t*)malloc(imageSize * sizeof(uint8_t));

    cudaError_t cudaStatus = testWithCuda(inputImage.data, outputImageData, inputImage.cols, inputImage.rows, inputImage.channels());
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "testWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC3, outputImageData);
	cv::namedWindow("Input Image", cv::WINDOW_NORMAL);
	cv::namedWindow("Output Image", cv::WINDOW_NORMAL);
	cv::imshow("Input Image", inputImage);
	cv::imshow("Output Image", outputImage);

	cv::waitKey(0);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t testWithCuda(uint8_t *inputImage, uint8_t *outputImage, unsigned int imageWidth, unsigned int imageHeight, unsigned int imageChannels)
{
	GpuTimer cudaMallocInputTimer;
	GpuTimer cudaMallocOutputTimer;
	GpuTimer cudaMemcpyInputTimer;
	GpuTimer cudaKernelTimer;
	GpuTimer cudaMemcpyOutputTimer;
	unsigned int imageSize = imageWidth * imageHeight * imageChannels;
	cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	// Allocate GPU buffers for two vectors (one input, one output).
	uint8_t *d_InputImage;
	cudaMallocInputTimer.Start();
    cudaStatus = cudaMalloc((void**)&d_InputImage, imageSize * sizeof(uint8_t));
	cudaMallocInputTimer.Stop();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc (input image) failed!");
        goto Error;
    }
	printf("Time to allocate input:\t\t\t\t%f ms\n", cudaMallocInputTimer.Elapsed());

	
	uint8_t *d_OutputImage;
	cudaMallocOutputTimer.Start();
    cudaStatus = cudaMalloc((void**)&d_OutputImage, imageSize * sizeof(uint8_t));
	cudaMallocOutputTimer.Stop();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc (output image) failed!");
        goto Error;
    }
	printf("Time to allocate output:\t\t\t%f ms\n", cudaMallocOutputTimer.Elapsed());

    // Copy input vectors from host memory to GPU buffers.
	cudaMemcpyInputTimer.Start();
    cudaStatus = cudaMemcpy(d_InputImage, inputImage, imageSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpyInputTimer.Stop();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (host -> device) failed!");
        goto Error;
    }
	printf("Time to copy input from host to device:\t\t%f ms\n", cudaMemcpyInputTimer.Elapsed());

    // Launch a kernel on the GPU with one thread for each element.
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	dim3 dimGrid((imageWidth - 1) / BLOCK_WIDTH + 1, (imageHeight - 1) / BLOCK_WIDTH + 1, 1);
	cudaKernelTimer.Start();
    testKernel<<<dimGrid, dimBlock>>>(d_InputImage, d_OutputImage, imageWidth, imageHeight, imageChannels);
	cudaKernelTimer.Stop();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "testKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
	printf("Time to execute kernel:\t\t\t\t%f ms\n", cudaKernelTimer.Elapsed());

    // Copy output vector from GPU buffer to host memory.
	cudaMemcpyOutputTimer.Start();
    cudaStatus = cudaMemcpy(outputImage, d_OutputImage, imageSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaMemcpyOutputTimer.Stop();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (device -> host) failed!");
        goto Error;
    }

	{
		printf("Time to copy output from device to host:\t%f ms\n", cudaMemcpyOutputTimer.Elapsed());
	}

Error:
	cudaFree(d_InputImage);
    cudaFree(d_OutputImage);
    
    return cudaStatus;
}
