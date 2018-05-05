
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdio.h>
#include <assert.h>
#include <string>
#include <iostream>
#include <fstream>
#include <random>

#define _USE_MATH_DEFINES
#include <math.h>

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

void computeRandomSprays_CPU(short int ***spraysX, short int ***spraysY, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfSprays) {
	const unsigned int width = 2 * radius + 1;
	const unsigned int area = width * width;								// compute area of neighborhood
	bool *neighborhood = (bool*)malloc(area * sizeof(bool));				// allocate boolean neighborhood array of size area
	short int *sprayX;														// short integer spray point abscissas array
	short int *sprayY;														// short integer spray point ordinates array
	unsigned int pointIdx;													// sample point index
	float randomRadius;														// random radius
	float randomTheta;														//random theta
	short int randomPointX;													// random point abscissa
	short int randomPointY;													// random point ordinate
	unsigned int randomPointNeighborhoodIdx;								// random point neighborhood index
	std::default_random_engine generator;									// random number generator engine
	std::uniform_real_distribution<float> radiusDistribution(0, radius);	// uniform real distribution for radius in the range (0, radius)
	std::uniform_real_distribution<float> thetaDistribution(0, 2 * M_PI);	// uniform real distribution for theta in the range (0, 2*pi)

	*spraysX = (short int**)malloc(numOfSprays * sizeof(short int*));	// sprays abscissas array
	*spraysY = (short int**)malloc(numOfSprays * sizeof(short int*));	// sprays ordinates array
	
	// initialize neighborbood as empty
	for (unsigned int neighborIdx; neighborIdx < area; neighborIdx++) {
		neighborhood[neighborIdx] = false;
	}

	const unsigned int centerPointNeighborhoodIdx = (width + 1) * radius; // calculate center point neighborhood index
	neighborhood[centerPointNeighborhoodIdx] = true;	// block out upcoming random points from coinciding with the center point

	// spray generation loop
	for (unsigned int sprayIdx = 0; sprayIdx < numOfSprays; sprayIdx++) {
		sprayX = (short int*)malloc(numOfSamplePoints * sizeof(short int));	// allocate spray point abscissas array of size numOfSamplePoints
		sprayY = (short int*)malloc(numOfSamplePoints * sizeof(short int));	// allocate spray point abscissas array of size numOfSamplePoints
		pointIdx = 0;											// reset sample point index to 0
		while(pointIdx < numOfSamplePoints) {					// sample point loop
			randomRadius = radiusDistribution(generator);	// get a random distance from the uniform real distribution for distance
			randomTheta = thetaDistribution(generator);		// get a random theta from the uniform real distribution for theta
			randomPointX = randomRadius * cos(randomTheta);	// compute random point abscissa
			randomPointY = randomRadius * sin(randomTheta);	// compute random point ordinate
			randomPointNeighborhoodIdx = width * (randomPointY + radius) + randomPointX + radius;	//compute random point neighborhood index

			if (!neighborhood[randomPointNeighborhoodIdx]) {		// if the random point is not already a sample point
				neighborhood[randomPointNeighborhoodIdx] = true;	// random point is now in the neighborhood
				sprayX[pointIdx] = randomPointX;					// random point is now a sample point (abscissa)
				sprayY[pointIdx] = randomPointY;					// random point is now a sample point (ordinate)
				pointIdx++;											// advance point index
			}
		}
		(*spraysX)[sprayIdx] = sprayX;				// add resultant spray abscissas to sprays abscissas
		(*spraysY)[sprayIdx] = sprayY;				// add resultant spray ordinates to sprays ordinates
		
		// set neighborhood back to empty
		for (pointIdx = 0; pointIdx < numOfSamplePoints; pointIdx++) {
			randomPointNeighborhoodIdx = width * (sprayY[pointIdx] + radius) + sprayX[pointIdx] + radius;
			neighborhood[randomPointNeighborhoodIdx] = false;	// remove each sample point from neighborhood
		}
	}

	free(neighborhood);	// release allocated memory for neighborhood array
}

cv::Mat generateRandomSprayImage(short int *sprayX, short int *sprayY, const unsigned short int radius, const unsigned int numOfSamplePoints) {
	const unsigned int width = radius * 2 + 1;
	const unsigned int area = width * width;
	uint8_t *neighborhood = (uint8_t*)malloc(area * sizeof(uint8_t));
	for (unsigned int pointIdx = 0; pointIdx < area; pointIdx++) {
		neighborhood[pointIdx] = 0;	// black image
	}
	
	unsigned int pointNeighborhoodIdx;
	for (unsigned int pointIdx = 0; pointIdx < numOfSamplePoints; pointIdx++) {
		pointNeighborhoodIdx = width * (sprayY[pointIdx] + radius) + sprayX[pointIdx] + radius;
		neighborhood[pointNeighborhoodIdx] = 255;	// white pixel where a sample point is present
	}
	
	cv::Mat sprayImage(width, width, CV_8UC1, neighborhood);	//create OpenCV grayscale image from data
	return sprayImage;
}

int main(int argc, char *argv[])
{
	srand(time(NULL));
	const unsigned short int radius = 200;
	const unsigned int numOfSamplePoints = 200;
	const unsigned int numOfSprays = 100;
	short int **spraysX;
	short int **spraysY;
	GpuTimer computeRandomSpraysCPUTimer;
	computeRandomSpraysCPUTimer.Start();
	computeRandomSprays_CPU(&spraysX, &spraysY, radius, numOfSamplePoints, numOfSprays);
	computeRandomSpraysCPUTimer.Stop();

	printf("Time to compute random sprays (CPU): %f ms\n", computeRandomSpraysCPUTimer.Elapsed());

	printf("Writing random sprays (%i) to disk ...\n", numOfSprays);
	char sprayImageName[20];
	for (unsigned int sprayIdx = 0; sprayIdx < numOfSprays; sprayIdx++) {
		cv::Mat sprayImage = generateRandomSprayImage(spraysX[sprayIdx], spraysY[sprayIdx], radius, numOfSamplePoints);
		sprintf(sprayImageName, "spray%i.png", sprayIdx);
		cv::imwrite(sprayImageName, sprayImage);
	}

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
	printf("%s\n", "Writing output image to disk ...");
	cv::imwrite("output.png", outputImage);

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
