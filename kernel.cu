
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <assert.h>
#include <ctime>
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
#include <curand_kernel.h>

#include "GpuTimer.h"
#include "config.h"

#if defined(ENABLE_STRESS_C2G_GPU_1)
cudaError_t STRESSColorToGrayscaleKernelHelper(uint8_t *outputImage, uint8_t *inputImage, short int **spraysX, short int **spraysY, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfSprays, const unsigned int numOfIterations, const unsigned long long seed);
#elif defined(ENABLE_STRESS_C2G_GPU_1B)
cudaError_t STRESSColorToGrayscaleKernelHelper(uint8_t *outputImage, uint8_t *inputImage, short int **spraysX, short int **spraysY, const unsigned int numOfSharedSprays, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfSprays, const unsigned int numOfIterations, const unsigned long long seed);
#elif defined(ENABLE_STRESS_C2G_GPU_2)
cudaError_t STRESSColorToGrayscaleKernelHelper(uint8_t *grayscaleOutputImage, uint8_t *colorInputImage, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfIterations, const unsigned long long seed);
#elif defined(ENABLE_STRESS_C2G_GPU_2B) || defined(ENABLE_STRESS_C2G_GPU_2D)
cudaError_t STRESSColorToGrayscaleKernelHelper(uint8_t *grayscaleOutputImage, uint8_t *colorInputImage, float *sinLUT, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned int sinLUTLength, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfIterations, const unsigned long long seed);
#elif defined(ENABLE_STRESS_C2G_GPU_2C)
cudaError_t STRESSColorToGrayscaleKernelHelper(uint8_t *grayscaleOutputImage, uint8_t *colorInputImage, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned int sinLUTLength, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfIterations, const unsigned long long seed);
#elif defined(ENABLE_STRESS_C2G_GPU_3)
cudaError_t STRESSColorToGrayscaleKernelHelper(uint8_t *grayscaleOutputImage, uint8_t *colorInputImage, short int **spraysX, short int **spraysY, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfSprays, const unsigned int numOfIterations, const unsigned long long seed);
#endif

void computeRandomSpraysCPU(short int ***spraysX, short int ***spraysY, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfSprays) {
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
	for (unsigned int neighborIdx = 0; neighborIdx < area; neighborIdx++) {
		neighborhood[neighborIdx] = false;
	}

	const unsigned int centerPointNeighborhoodIdx = (width + 1) * radius; // calculate center point neighborhood index
	neighborhood[centerPointNeighborhoodIdx] = true;	// block out upcoming random points from coinciding with the center point

														// spray generation loop
	for (unsigned int sprayIdx = 0; sprayIdx < numOfSprays; sprayIdx++) {
		sprayX = (short int*)malloc(numOfSamplePoints * sizeof(short int));	// allocate spray point abscissas array of size numOfSamplePoints
		sprayY = (short int*)malloc(numOfSamplePoints * sizeof(short int));	// allocate spray point abscissas array of size numOfSamplePoints
		pointIdx = 0;											// reset sample point index to 0
		while (pointIdx < numOfSamplePoints) {					// sample point loop
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

#if defined(ENABLE_STRESS_G2G_CPU_1)
// This version of the function uses pre-computed random sprays. It chooses, in each iteration, for each pixel, a random spray at random out of the available sprays.
// This introduces an issue whereby pixels closer to the edge of the image in particular face reduced sampling due to many sample points lying outside the image and
// thus not being factored into calculating the envelope. The issue manifests itself particularly when more iterations are used.
void STRESSGrayscaleToGrayscaleCPU1(uint8_t *outputImage, uint8_t *inputImage, const unsigned short int imageWidth, const unsigned short int imageHeight, short int **spraysX, short int **spraysY, const unsigned int numOfSamplePoints, const unsigned int numOfSprays, const unsigned int numOfIterations) {
	unsigned int targetPixelIdx; // target pixel (p) absolute index
	int samplePointX; // spray sample point abscissa
	int samplePointY; // spray sample point ordinate
	unsigned int samplePointPixelIdx; // spray sample point pixel index
	uint8_t Emin;
	uint8_t Emax;

	unsigned int randomSprayIdx;  // random spray index
	short int *randomSprayX;    // abscissas for spray chosen at random
	short int *randomSprayY;    // ordinates for spray chosen at random

	// allocate temporary output image array for storing sum of all iteration results
	unsigned int imageSize = imageWidth * imageHeight;
	float *tempOutputImage = (float*)malloc(imageSize * sizeof(float));

	// initial temporary output image as empty
	for (unsigned int pixelIdx = 0; pixelIdx < imageSize; pixelIdx++) {
		tempOutputImage[pixelIdx] = 0.0f;
	}

	// iteration loop
	for (unsigned int iterationIdx = 0; iterationIdx < numOfIterations; iterationIdx++) {
		targetPixelIdx = 0; // reset target pixel absolute index to 0
		for (unsigned short int targetPixelY = 0; targetPixelY < imageHeight; targetPixelY++) {
			for (unsigned short int targetPixelX = 0; targetPixelX < imageWidth; targetPixelX++) {
				//set Emin and Emax equal to target pixel value
				Emin = Emax = inputImage[targetPixelIdx];

				// choose spray at random
				randomSprayIdx = rand() % numOfSprays;
				randomSprayX = spraysX[randomSprayIdx];
				randomSprayY = spraysY[randomSprayIdx];

				// calculate envelope
				for (unsigned int sampleIdx = 0; sampleIdx < numOfSamplePoints; sampleIdx++) {
					samplePointX = targetPixelX + randomSprayX[sampleIdx];  // get sample point abscissa in input image
					samplePointY = targetPixelY + randomSprayY[sampleIdx];  // get sample point ordinate in input image
																			//printf("%i %i\n", samplePointX, samplePointY);
					if (samplePointX >= 0 && samplePointX < imageWidth && samplePointY >= 0 && samplePointY < imageHeight) {  // only proceed if sample point is within the input image
						samplePointPixelIdx = imageWidth * samplePointY + samplePointX; // get sample point index in input image
						if (inputImage[samplePointPixelIdx] < Emin) // if sample point color channel is less than Emin at that channel
							Emin = inputImage[samplePointPixelIdx]; // it is the new Emin at that channel
						else if (inputImage[samplePointPixelIdx] > Emax)
							Emax = inputImage[samplePointPixelIdx];
					}
				}

				// calculate (p - Emin) / (Emax - Emin)
				tempOutputImage[targetPixelIdx] += (inputImage[targetPixelIdx] - Emin) * 255.0 / (Emax - Emin);

				targetPixelIdx++;
			}
		}
	}

	// divide each accumulated pixel value by the number of iterations to obtain the average pixel value across iterations.
	// place the average value in the output image array.
	for (unsigned int pixelIdx = 0; pixelIdx < imageSize; pixelIdx++) {
		outputImage[pixelIdx] = tempOutputImage[pixelIdx] / numOfIterations;
	}
}
#endif

#if defined(ENABLE_STRESS_G2G_CPU_2)
// This version of the function does not use pre-computed random sprays. Instead, it generates, in each iteration, for each pixel in the image, a random spray for that pixel.
// This solves the issue of reduced sampling seen in the first version of the function. However, this approach is much slower than using pre-computed sprays
void STRESSGrayscaleToGrayscaleCPU2(uint8_t *outputImage, uint8_t *inputImage, const unsigned short int imageWidth, const unsigned short int imageHeight, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfIterations) {
	unsigned int randomSamplePixelIdx;									// random sample pixel index
	unsigned int randomSampleImagePixelIdx;								// random sample pixel absolute index in image
	float randomRadius;													// random radius
	float randomTheta;													// random theta
	int randomSamplePixelX;												// random sample pixel abscissa
	int randomSamplePixelY;												// random sample pixel ordinate
	std::default_random_engine generator;								// random number generator engine
	std::uniform_real_distribution<float> radiusDistribution(0, radius);	// uniform real distribution for radius in the range (0, radius)
	std::uniform_real_distribution<float> thetaDistribution(0, 2 * M_PI);	// uniform real distribution for theta in the range (0, 2*pi)

	unsigned int targetPixelIdx; // target pixel (p) absolute index
	uint8_t Emin;
	uint8_t Emax;

	// allocate temporary output image array for storing sum of all iteration results
	unsigned int imageSize = imageWidth * imageHeight;
	float *tempOutputImage = (float*)malloc(imageSize * sizeof(float));

	// initial temporary output image as empty
	for (unsigned int pixelIdx = 0; pixelIdx < imageSize; pixelIdx++) {
		tempOutputImage[pixelIdx] = 0.0f;
	}

	// iteration loop
	for (unsigned int iterationIdx = 0; iterationIdx < numOfIterations; iterationIdx++) {
		targetPixelIdx = 0;	// reset target pixel absolute index to 0
		for (unsigned short int targetPixelY = 0; targetPixelY < imageHeight; targetPixelY++) {
			for (unsigned short int targetPixelX = 0; targetPixelX < imageWidth; targetPixelX++) {
				//set Emin and Emax equal to target pixel value
				Emin = Emax = inputImage[targetPixelIdx];

				// generate random sample points and calculate envelope
				randomSamplePixelIdx = 0;
				while (randomSamplePixelIdx < numOfSamplePoints) {		// random sample pixel point loop
					randomRadius = radiusDistribution(generator);	// get a random distance from the uniform real distribution for distance
					randomTheta = thetaDistribution(generator);		// get a random theta from the uniform real distribution for theta
					randomSamplePixelX = targetPixelX + randomRadius * cos(randomTheta);	// compute random pixel abscissa
					if (randomSamplePixelX >= 0 && randomSamplePixelX < imageWidth) {		// if random pixel abscissa is within image
						randomSamplePixelY = targetPixelY + randomRadius * sin(randomTheta);		// compute random pixel ordinate
						if (randomSamplePixelY >= 0 && randomSamplePixelY < imageHeight) {	// if random pixel ordinate is within image
							randomSampleImagePixelIdx = imageWidth * randomSamplePixelY + randomSamplePixelX; // get random sample pixel index in image
							if (inputImage[randomSampleImagePixelIdx] < Emin)		// if sample pixel value is less than Emin
								Emin = inputImage[randomSampleImagePixelIdx];		// it is the new Emin
							else if (inputImage[randomSampleImagePixelIdx] > Emax)	// if sample pixel value is greater than Emax 
								Emax = inputImage[randomSampleImagePixelIdx];	// it is the new Emax
							randomSamplePixelIdx++;	// advance random sample pixel index
						}
					}
				}

				// calculate (p - Emin) / (Emax - Emin)
				tempOutputImage[targetPixelIdx] += (inputImage[targetPixelIdx] - Emin) * 255.0 / (Emax - Emin);

				targetPixelIdx++; // advance target pixel index
			}
		}
	}

	// divide each accumulated pixel value by the number of iterations to obtain the average pixel value across iterations.
	// place the average value in the output image array.
	for (unsigned int pixelIdx = 0; pixelIdx < imageSize; pixelIdx++) {
		outputImage[pixelIdx] = tempOutputImage[pixelIdx] / numOfIterations;
	}
}
#endif

#if defined(ENABLE_STRESS_G2G_CPU_3)
// This version of the function is a hybrid between the first two approaches. It uses pre-computed sprays similarly to the first approach.
// However, for any pixel, if any sample point in its chosen pre-computed spray is found to be lying outside the image, it is replaced with
// randomly chosen sample points lying within the image. This should solve the issue of the first approach while not being as slow as the second approach,
// particularly for pixels not close to the edges of the image, since the likelihood of a sample point not lying within the image for those diminishes greatly.
void STRESSGrayscaleToGrayscaleCPU3(uint8_t *outputImage, uint8_t *inputImage, const unsigned short int imageWidth, const unsigned short int imageHeight, short int **spraysX, short int **spraysY, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfSprays, const unsigned int numOfIterations) {
	float randomRadius;													// random radius
	float randomTheta;													// random theta
	int samplePixelX;													// (pre-computed / random) sample pixel abscissa
	int samplePixelY;													// (pre-computed / random) sample pixel ordinate
	unsigned int sampleIdx;												// sample index
	unsigned int samplePixelIdx;										// sample pixel index
	unsigned int numOfValidSamplePoints;								// number of valid sample points in envelope
	std::default_random_engine generator;								// random number generator engine
	std::uniform_real_distribution<float> radiusDistribution(0, radius);	// uniform real distribution for radius in the range (0, radius)
	std::uniform_real_distribution<float> thetaDistribution(0, 2 * M_PI);	// uniform real distribution for theta in the range (0, 2*pi)
	
	unsigned int targetPixelIdx; // target pixel (p) absolute index
	uint8_t Emin;
	uint8_t Emax;

	unsigned int randomSprayIdx;  // random spray index
	short int *randomSprayX;    // abscissas for spray chosen at random
	short int *randomSprayY;    // ordinates for spray chosen at random

								// allocate temporary output image array for storing sum of all iteration results
	unsigned int imageSize = imageWidth * imageHeight;
	float *tempOutputImage = (float*)malloc(imageSize * sizeof(float));

	// initial temporary output image as empty
	for (unsigned int pixelIdx = 0; pixelIdx < imageSize; pixelIdx++) {
		tempOutputImage[pixelIdx] = 0.0f;
	}

	// iteration loop
	for (unsigned int iterationIdx = 0; iterationIdx < numOfIterations; iterationIdx++) {
		targetPixelIdx = 0; // reset target pixel absolute index to 0
		for (unsigned short int targetPixelY = 0; targetPixelY < imageHeight; targetPixelY++) {
			for (unsigned short int targetPixelX = 0; targetPixelX < imageWidth; targetPixelX++) {
				//set Emin and Emax equal to target pixel value
				Emin = Emax = inputImage[targetPixelIdx];

				// choose spray at random
				randomSprayIdx = rand() % numOfSprays;
				randomSprayX = spraysX[randomSprayIdx];
				randomSprayY = spraysY[randomSprayIdx];

				// calculate envelope
				numOfValidSamplePoints = 0;	// reset number of valid sample points to 0
				for (sampleIdx = 0; sampleIdx < numOfSamplePoints; sampleIdx++) {
					samplePixelX = targetPixelX + randomSprayX[sampleIdx];  // get sample pixel abscissa in input image
					samplePixelY = targetPixelY + randomSprayY[sampleIdx];  // get sample pixel ordinate in input image
					if (samplePixelX >= 0 && samplePixelX < imageWidth && samplePixelY >= 0 && samplePixelY < imageHeight) {  // only proceed if sample pixel is within the input image
						samplePixelIdx = imageWidth * samplePixelY + samplePixelX; // get sample pixel index in input image
						if (inputImage[samplePixelIdx] < Emin) // if sample pixel value is less than Emin
							Emin = inputImage[samplePixelIdx]; // it is the new Emin
						else if (inputImage[samplePixelIdx] > Emax)	// if sample pixel value is greater than Emax
							Emax = inputImage[samplePixelIdx];			// it is the new Emax
						numOfValidSamplePoints++;	// increment number of valid sample points
					}
				}

				// generate sample points to compensate for invalid sample points
				sampleIdx = numOfValidSamplePoints;
				while (sampleIdx < numOfSamplePoints) {
					randomRadius = radiusDistribution(generator);	// get a random distance from the uniform real distribution for distance
					randomTheta = thetaDistribution(generator);		// get a random theta from the uniform real distribution for theta
					samplePixelX = targetPixelX + randomRadius * cos(randomTheta);	// compute random pixel abscissa
					if (samplePixelX >= 0 && samplePixelX < imageWidth) {		// if random pixel abscissa is within image
						samplePixelY = targetPixelY + randomRadius * sin(randomTheta);		// compute random pixel ordinate
						if (samplePixelY >= 0 && samplePixelY < imageHeight) {	// if random pixel ordinate is within image
							samplePixelIdx = imageWidth * samplePixelY + samplePixelX; // get random sample pixel index in image
							if (inputImage[samplePixelIdx] < Emin)		// if sample pixel value is less than Emin
								Emin = inputImage[samplePixelIdx];		// it is the new Emin
							else if (inputImage[samplePixelIdx] > Emax)	// if sample pixel value is greater than Emax 
								Emax = inputImage[samplePixelIdx];	// it is the new Emax
							sampleIdx++;	// advance random sample pixel index
						}
					}
				}

				// calculate (p - Emin) / (Emax - Emin)
				tempOutputImage[targetPixelIdx] += (inputImage[targetPixelIdx] - Emin) * 255.0 / (Emax - Emin);

				targetPixelIdx++;
			}
		}
	}

	// divide each accumulated pixel value by the number of iterations to obtain the average pixel value across iterations.
	// place the average value in the output image array.
	for (unsigned int pixelIdx = 0; pixelIdx < imageSize; pixelIdx++) {
		outputImage[pixelIdx] = tempOutputImage[pixelIdx] / numOfIterations;
	}
}
#endif

#if defined(ENABLE_STRESS_C2G_CPU_3)
// This version of the function is a hybrid between the first two approaches. It uses pre-computed sprays similarly to the first approach.
// However, for any pixel, if any sample point in its chosen pre-computed spray is found to be lying outside the image, it is replaced with
// randomly chosen sample points lying within the image. This should solve the issue of the first approach while not being as slow as the second approach,
// particularly for pixels not close to the edges of the image, since the likelihood of a sample point not lying within the image for those diminishes greatly.
void STRESSColorToGrayscaleCPU3(uint8_t *outputImage, uint8_t *inputImage, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, short int **spraysX, short int **spraysY, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfSprays, const unsigned int numOfIterations) {
	float randomRadius;													// random radius
	float randomTheta;													// random theta
	int samplePixelX;													// (pre-computed / random) sample pixel abscissa
	int samplePixelY;													// (pre-computed / random) sample pixel ordinate
	unsigned int sampleIdx;												// sample index
	uint8_t channelIdx;													// channel index
	unsigned int samplePixelIdx;										// sample pixel index
	unsigned int samplePixelChannelIdx;									// sample pixel channel index
	unsigned int numOfValidSamplePoints;								// number of valid sample points in envelope
	std::default_random_engine generator;								// random number generator engine
	std::uniform_real_distribution<float> radiusDistribution(0, radius);	// uniform real distribution for radius in the range (0, radius)
	std::uniform_real_distribution<float> thetaDistribution(0, 2 * M_PI);	// uniform real distribution for theta in the range (0, 2*pi)

	unsigned int targetInputPixelIdx; // target input pixel (p) index
	unsigned int targetOutputPixelIdx; // target output pixel index
	uint8_t *Emin = (uint8_t*)malloc(imageChannels * sizeof(uint8_t));	// Emin array of size imageChannels
	uint8_t *Emax = (uint8_t*)malloc(imageChannels * sizeof(uint8_t));	// Emax array of size imageChannels
	
	// for calculating (p - Emin).(Emax - Emin) / |Emax - Emin|^2
	uint8_t Edelta;
	unsigned int dotProd, ElenSq;

	unsigned int randomSprayIdx;  // random spray index
	short int *randomSprayX;    // abscissas for spray chosen at random
	short int *randomSprayY;    // ordinates for spray chosen at random

	// allocate temporary output image array for storing sum of all iteration results
	unsigned int outputImageSize = imageWidth * imageHeight;
	float *tempOutputImage = (float*)malloc(outputImageSize * sizeof(float));

	// initial temporary output image as empty
	for (unsigned int pixelIdx = 0; pixelIdx < outputImageSize; pixelIdx++) {
		tempOutputImage[pixelIdx] = 0.0f;
	}

	// iteration loop
	for (unsigned int iterationIdx = 0; iterationIdx < numOfIterations; iterationIdx++) {
		targetInputPixelIdx = 0; // reset target input pixel index to 0
		targetOutputPixelIdx = 0; // reset target output pixel index to 0
		for (unsigned short int targetPixelY = 0; targetPixelY < imageHeight; targetPixelY++) {
			for (unsigned short int targetPixelX = 0; targetPixelX < imageWidth; targetPixelX++) {
				//set Emin and Emax equal to target pixel across all color channels
				for (channelIdx = 0; channelIdx < imageChannels; channelIdx++)
					Emin[channelIdx] = Emax[channelIdx] = inputImage[targetInputPixelIdx + channelIdx];

				// choose spray at random
				randomSprayIdx = rand() % numOfSprays;
				randomSprayX = spraysX[randomSprayIdx];
				randomSprayY = spraysY[randomSprayIdx];

				// calculate envelope
				sampleIdx = 0;	// reset sample index to 0
				numOfValidSamplePoints = 0;	// reset number of valid sample points to 0
				for (sampleIdx = 0; sampleIdx < numOfSamplePoints; sampleIdx++) {
					samplePixelX = targetPixelX + randomSprayX[sampleIdx];  // get sample pixel abscissa in input image
					samplePixelY = targetPixelY + randomSprayY[sampleIdx];  // get sample pixel ordinate in input image
					if (samplePixelX >= 0 && samplePixelX < imageWidth && samplePixelY >= 0 && samplePixelY < imageHeight) {  // only proceed if sample pixel is within the input image
						samplePixelIdx = (imageWidth * samplePixelY + samplePixelX) * imageChannels; // get sample pixel index in input image
						for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
							samplePixelChannelIdx = samplePixelIdx + channelIdx;
							if (inputImage[samplePixelChannelIdx] < Emin[channelIdx]) // if sample pixel value is less than Emin
								Emin[channelIdx] = inputImage[samplePixelChannelIdx];		// it is the new Emin
							else if (inputImage[samplePixelChannelIdx] > Emax[channelIdx])	// if sample pixel value is greater than Emax
								Emax[channelIdx] = inputImage[samplePixelChannelIdx];		// it is the new Emax
						}
						numOfValidSamplePoints++;	// increment number of valid sample points
					}
				}

				// generate sample points to compensate for invalid sample points
				sampleIdx = numOfValidSamplePoints;
				while (sampleIdx < numOfSamplePoints) {
					randomRadius = radiusDistribution(generator);	// get a random distance from the uniform real distribution for distance
					randomTheta = thetaDistribution(generator);		// get a random theta from the uniform real distribution for theta
					samplePixelX = targetPixelX + randomRadius * cos(randomTheta);	// compute random pixel abscissa
					if (samplePixelX >= 0 && samplePixelX < imageWidth) {		// if random pixel abscissa is within image
						samplePixelY = targetPixelY + randomRadius * sin(randomTheta);		// compute random pixel ordinate
						if (samplePixelY >= 0 && samplePixelY < imageHeight) {	// if random pixel ordinate is within image
							samplePixelIdx = (imageWidth * samplePixelY + samplePixelX) * imageChannels; // get random sample pixel index in image
							for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
								samplePixelChannelIdx = samplePixelIdx + channelIdx;
								if (inputImage[samplePixelChannelIdx] < Emin[channelIdx])		// if sample pixel value is less than Emin
									Emin[channelIdx] = inputImage[samplePixelChannelIdx];		// it is the new Emin
								else if (inputImage[samplePixelChannelIdx] > Emax[channelIdx])	// if sample pixel value is greater than Emax 
									Emax[channelIdx] = inputImage[samplePixelChannelIdx];		// it is the new Emax
							}
							sampleIdx++;	// advance random sample pixel index
						}
					}
				}

				// calculate (p - Emin).(Emax - Emin), |Emax - Emin|^2
				dotProd = 0;
				ElenSq = 0;
				for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
					Edelta = Emax[channelIdx] - Emin[channelIdx];
					dotProd += Edelta * (inputImage[targetInputPixelIdx + channelIdx] - Emin[channelIdx]);
					ElenSq += Edelta * Edelta;
				}

				// calculate g = (p - Emin).(Emax - Emin) / |Emax - Emin|^2
				tempOutputImage[targetOutputPixelIdx] += dotProd * 255.0 / ElenSq;

				targetInputPixelIdx += imageChannels;
				targetOutputPixelIdx++;
			}
		}
	}

	// divide each accumulated pixel value by the number of iterations to obtain the average pixel value across iterations.
	// place the average value in the output image array.
	for (unsigned int pixelIdx1 = 0; pixelIdx1 < outputImageSize; pixelIdx1++) {
		outputImage[pixelIdx1] = tempOutputImage[pixelIdx1] / numOfIterations;
	}
}
#endif

#if defined(ENABLE_STRESS_C2C_CPU_3)
void STRESSColorToColorCPU3(uint8_t *outputImage, uint8_t *inputImage, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, short int **spraysX, short int **spraysY, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfSprays, const unsigned int numOfIterations) {
	float randomRadius;													// random radius
	float randomTheta;													// random theta
	int samplePixelX;													// (pre-computed / random) sample pixel abscissa
	int samplePixelY;													// (pre-computed / random) sample pixel ordinate
	unsigned int sampleIdx;												// sample index
	uint8_t channelIdx;													// channel index
	unsigned int samplePixelIdx;										// sample pixel index
	unsigned int pixelChannelIdx;										// pixel channel index
	unsigned int numOfValidSamplePoints;								// number of valid sample points in envelope
	std::default_random_engine generator;								// random number generator engine
	std::uniform_real_distribution<float> radiusDistribution(0, radius);	// uniform real distribution for radius in the range (0, radius)
	std::uniform_real_distribution<float> thetaDistribution(0, 2 * M_PI);	// uniform real distribution for theta in the range (0, 2*pi)

	unsigned int targetPixelIdx; // target pixel (p) index
	uint8_t *Emin = (uint8_t*)malloc(imageChannels * sizeof(uint8_t));	// Emin array of size imageChannels
	uint8_t *Emax = (uint8_t*)malloc(imageChannels * sizeof(uint8_t));	// Emax array of size imageChannels

	unsigned int randomSprayIdx;  // random spray index
	short int *randomSprayX;    // abscissas for spray chosen at random
	short int *randomSprayY;    // ordinates for spray chosen at random

	// allocate temporary output image array for storing sum of all iteration results
	unsigned int imageSize = imageWidth * imageHeight * imageChannels;
	float *tempOutputImage = (float*)malloc(imageSize * sizeof(float));

	// initial temporary output image as empty
	for (unsigned int pixelIdx = 0; pixelIdx < imageSize; pixelIdx++) {
		tempOutputImage[pixelIdx] = 0.0f;
	}

	// iteration loop
	for (unsigned int iterationIdx = 0; iterationIdx < numOfIterations; iterationIdx++) {
		targetPixelIdx = 0; // reset target pixel index to 0
		for (unsigned short int targetPixelY = 0; targetPixelY < imageHeight; targetPixelY++) {
			for (unsigned short int targetPixelX = 0; targetPixelX < imageWidth; targetPixelX++) {
				//set Emin and Emax equal to target pixel across all color channels
				for (channelIdx = 0; channelIdx < imageChannels; channelIdx++)
					Emin[channelIdx] = Emax[channelIdx] = inputImage[targetPixelIdx + channelIdx];

				// choose spray at random
				randomSprayIdx = rand() % numOfSprays;
				randomSprayX = spraysX[randomSprayIdx];
				randomSprayY = spraysY[randomSprayIdx];

				// calculate envelope
				sampleIdx = 0;	// reset sample index to 0
				numOfValidSamplePoints = 0;	// reset number of valid sample points to 0
				for (sampleIdx = 0; sampleIdx < numOfSamplePoints; sampleIdx++) {
					samplePixelX = targetPixelX + randomSprayX[sampleIdx];  // get sample pixel abscissa in input image
					samplePixelY = targetPixelY + randomSprayY[sampleIdx];  // get sample pixel ordinate in input image
					if (samplePixelX >= 0 && samplePixelX < imageWidth && samplePixelY >= 0 && samplePixelY < imageHeight) {  // only proceed if sample pixel is within the input image
						samplePixelIdx = (imageWidth * samplePixelY + samplePixelX) * imageChannels; // get sample pixel index in input image
						for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
							pixelChannelIdx = samplePixelIdx + channelIdx;
							if (inputImage[pixelChannelIdx] < Emin[channelIdx]) // if sample pixel value is less than Emin
								Emin[channelIdx] = inputImage[pixelChannelIdx];		// it is the new Emin
							else if (inputImage[pixelChannelIdx] > Emax[channelIdx])	// if sample pixel value is greater than Emax
								Emax[channelIdx] = inputImage[pixelChannelIdx];		// it is the new Emax
						}
						numOfValidSamplePoints++;	// increment number of valid sample points
					}
				}

				// generate sample points to compensate for invalid sample points
				sampleIdx = numOfValidSamplePoints;
				while (sampleIdx < numOfSamplePoints) {
					randomRadius = radiusDistribution(generator);	// get a random distance from the uniform real distribution for distance
					randomTheta = thetaDistribution(generator);		// get a random theta from the uniform real distribution for theta
					samplePixelX = targetPixelX + randomRadius * cos(randomTheta);	// compute random pixel abscissa
					if (samplePixelX >= 0 && samplePixelX < imageWidth) {		// if random pixel abscissa is within image
						samplePixelY = targetPixelY + randomRadius * sin(randomTheta);		// compute random pixel ordinate
						if (samplePixelY >= 0 && samplePixelY < imageHeight) {	// if random pixel ordinate is within image
							samplePixelIdx = imageWidth * samplePixelY + samplePixelX; // get random sample pixel index in image
							for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
								pixelChannelIdx = samplePixelIdx + channelIdx;
								if (inputImage[pixelChannelIdx] < Emin[channelIdx])		// if sample pixel value is less than Emin
									Emin[channelIdx] = inputImage[pixelChannelIdx];		// it is the new Emin
								else if (inputImage[pixelChannelIdx] > Emax[channelIdx])	// if sample pixel value is greater than Emax 
									Emax[channelIdx] = inputImage[pixelChannelIdx];		// it is the new Emax
							}
							sampleIdx++;	// advance random sample pixel index
						}
					}
				}

				// calculate (p - Emin) / (Emax - Emin) for each color channel
				for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
					pixelChannelIdx = targetPixelIdx + channelIdx;
					tempOutputImage[pixelChannelIdx] += (inputImage[pixelChannelIdx] - Emin[channelIdx]) * 255.0 / (Emax[channelIdx] - Emin[channelIdx]);
				}

				targetPixelIdx += imageChannels;
			}
		}
	}

	// divide each accumulated pixel value by the number of iterations to obtain the average pixel value across iterations.
	// place the average value in the output image array.
	for (unsigned int pixelIdx = 0; pixelIdx < imageSize; pixelIdx++) {
		outputImage[pixelIdx] = tempOutputImage[pixelIdx] / numOfIterations;
	}
}
#endif

#if defined(ENABLE_STRESS_C2G_GPU_1) || defined(ENABLE_STRESS_C2G_GPU_1B) || defined(ENABLE_STRESS_C2G_GPU_2) || defined(ENABLE_STRESS_C2G_GPU_2B) || defined(ENABLE_STRESS_C2G_GPU_2C) || defined(ENABLE_STRESS_C2G_GPU_2D) || defined(ENABLE_STRESS_C2G_GPU_3)
// thanks to http://aresio.blogspot.com/2011/05/cuda-random-numbers-inside-kernels.html
// and to https://hpc.oit.uci.edu/nvidia-doc/sdk-cuda-doc/CUDALibraries/doc/CURAND_Library.pdf
__global__ void setupRandomKernel(curandState *state, const unsigned long long seed, const unsigned short int imageWidth, const unsigned short int imageHeight) {
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
	if (row < imageHeight && col < imageWidth) {
		unsigned int idx = imageWidth * row + col; // absolute thread index
		//curand_init(seed, idx, 0, &state[idx]);
		//curand_init((unsigned long long)clock(), 0, 0, &state[idx]);
		//curand_init(seed, 0, 0, &state[idx]);
		//curand_init(seed, idx % 2048, 0, &state[idx]);
		curand_init(seed + idx, 0, 0, &state[idx]);	// initialize random number generator state in global memory
	}
}
#endif

#if defined(ENABLE_STRESS_C2G_GPU_1)
// uses pre-computed random sprays loaded per each thread from global memory
__global__ void STRESSColorToGrayscaleKernel1(curandState *state, uint8_t *outputImage, uint8_t *inputImage, short int *spraysX, short int *spraysY, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned int numOfSprays, const unsigned int radius, const unsigned int numOfSamplePoints, const unsigned int numOfIterations) {
	unsigned int targetPixelX = blockDim.x * blockIdx.x + threadIdx.x; // target pixel abscissa
	unsigned int targetPixelY = blockDim.y * blockIdx.y + threadIdx.y; // target pixel ordinate

	if (targetPixelX < imageWidth && targetPixelY < imageHeight) {	// if target pixel is within image
		unsigned int idx = imageWidth * targetPixelY + targetPixelX;	// thread / output pixel absolute index
		curandState localState = state[idx];	// load random number generator state from global memory

		int samplePixelX;	// sample pixel abscissa
		int samplePixelY;	// sample pixel ordinate
		unsigned int sampleIdx;	// sample index from spray
		unsigned int sampleImagePixelIdx;	// sample pixel absolute index in image
		unsigned int sampleImagePixelChannelIdx;	// sample pixel channel index

		unsigned int targetPixelIdx = idx * imageChannels;	// target pixel (p) absolute index
		unsigned int targetPixelChannelIdx;		// target pixel channel absolute index
		uint8_t targetPixel[3]; // target pixel array for channels
		uint8_t samplePixel[3];	// sample pixel array for channels
		double outputPixel;		// output pixel values accumulator across iterations
		uint8_t channelIdx;		// channel index

		uint8_t Emin[3];	// Emin array of size imageChannels
		uint8_t Emax[3];	// Emax array of size imageChannels

		// for calculating (p - Emin).(Emax - Emin) / |Emax - Emin|^2
		uint8_t Edelta;
		unsigned int dotProd, ElenSq;

		unsigned int sprayIdx; // spray index
		unsigned int spraySampleStartIdx; // spray sample start index
		unsigned int spraySampleIdx; // spray sample absolute index

		// initialize output pixel values accumulator to 0
		outputPixel = 0.0f;

		// iteration loop
		for (unsigned int iterationIdx = 0; iterationIdx < numOfIterations; iterationIdx++) {
			// load target pixel and set Emin and Emax equal to target pixel value at each channel
			for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
				targetPixelChannelIdx = targetPixelIdx + channelIdx;
				Emin[channelIdx] = Emax[channelIdx] = targetPixel[channelIdx] = inputImage[targetPixelChannelIdx];
			}

			// get random sample points from spray and calculate envelope
			sprayIdx = curand_uniform(&localState) * numOfSprays;	// choose spray at random from pre-computed sprays
			spraySampleStartIdx = numOfSamplePoints * sprayIdx;	// calculate spray sample start index
			for (sampleIdx = 0; sampleIdx < numOfSamplePoints; sampleIdx++) {
				spraySampleIdx = spraySampleStartIdx + sampleIdx;
				samplePixelX = targetPixelX + spraysX[spraySampleIdx];	// compute sample pixel abscissa
				samplePixelY = targetPixelY + spraysY[spraySampleIdx];	// compute sample pixel ordinate
				if (samplePixelX >= 0 && samplePixelX < imageWidth && samplePixelY >= 0 && samplePixelY < imageHeight) {	// if sample pixel is within image
					sampleImagePixelIdx = (imageWidth * samplePixelY + samplePixelX) * imageChannels;	// get sample pixel index in image
					for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
						sampleImagePixelChannelIdx = sampleImagePixelIdx + channelIdx;	// get sample pixel channel index
						samplePixel[channelIdx] = inputImage[sampleImagePixelChannelIdx];
						if (samplePixel[channelIdx] < Emin[channelIdx])			// if sample pixel channel value is less than Emin at that channel
							Emin[channelIdx] = samplePixel[channelIdx];			// it is the new Emin
						else if (samplePixel[channelIdx] > Emax[channelIdx])	// if sample pixel channel value is greater than Emax at that channel
							Emax[channelIdx] = samplePixel[channelIdx];			// it is the new Emax
					}
				}
			}

			dotProd = 0;
			ElenSq = 0;
			for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
				Edelta = Emax[channelIdx] - Emin[channelIdx];
				dotProd += Edelta * (targetPixel[channelIdx] - Emin[channelIdx]);
				ElenSq += Edelta * Edelta;
			}

			// calculate g = (p - Emin).(Emax - Emin) / |Emax - Emin|^2
			outputPixel += dotProd * 255.0 / ElenSq;
		}
		outputImage[idx] = outputPixel / numOfIterations;
		state[idx] = localState;	// store updated random number generator state back into global memory
	}
}
#endif

#if defined(ENABLE_STRESS_C2G_GPU_1B)
// uses pre-computed random sprays loaded from global memory into shared memory as far 
// as shared memory size allows. Loading the sprays into shared memory would allow threads
// in a thread block to share and reuse sprays, reducing global memory bandwidth requirements.
// The threads-to-sprays ratio in a thread block determine whether this approach is useful.
// For ratios closer to 1, global memory bandwidth requirements are similar to the previous approach
// and reuse is minimal. However, for ratios closer to 0, global memory bandwidth requirements
// are much lower at the expense of higher spray reuse in a single thread block.
// thanks to https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared
__global__ void STRESSColorToGrayscaleKernel1B(curandState *state, uint8_t *outputImage, uint8_t *inputImage, short int *spraysX, short int *spraysY, const unsigned int numOfSharedSprays, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned int numOfSprays, const unsigned int radius, const unsigned int numOfSamplePoints, const unsigned int numOfIterations) {
	extern __shared__ short int sharedSprays[];
	
	unsigned int targetPixelX = blockDim.x * blockIdx.x + threadIdx.x; // target pixel abscissa
	unsigned int targetPixelY = blockDim.y * blockIdx.y + threadIdx.y; // target pixel ordinate

	if (targetPixelX < imageWidth && targetPixelY < imageHeight) {	// if target pixel is within image
		unsigned int idx = imageWidth * targetPixelY + targetPixelX;	// thread / output pixel absolute index
		curandState localState = state[idx];	// load random number generator state from global memory

		int samplePixelX;	// sample pixel abscissa
		int samplePixelY;	// sample pixel ordinate
		unsigned int sampleIdx;	// sample index from spray
		unsigned int sampleImagePixelIdx;	// sample pixel absolute index in image
		unsigned int sampleImagePixelChannelIdx;	// sample pixel channel index

		unsigned int targetPixelIdx = idx * imageChannels;	// target pixel (p) absolute index
		unsigned int targetPixelChannelIdx;		// target pixel channel absolute index
		uint8_t targetPixel[3]; // target pixel array for channels
		uint8_t samplePixel[3];	// sample pixel array for channels
		double outputPixel;		// output pixel values accumulator across iterations
		uint8_t channelIdx;		// channel index

		uint8_t Emin[3];	// Emin array of size imageChannels
		uint8_t Emax[3];	// Emax array of size imageChannels

		// for calculating (p - Emin).(Emax - Emin) / |Emax - Emin|^2
		uint8_t Edelta;
		unsigned int dotProd, ElenSq;

		unsigned int sprayIdx; // spray index
		unsigned int spraySampleStartIdx; // spray sample start index
		unsigned int spraySampleIdx; // spray sample absolute index

		unsigned int threadBlockIdx = blockDim.x * threadIdx.y + threadIdx.x;	// thread absolute index in thread block
		unsigned int sharedSprayIdx;	// shared spray index
		unsigned int sharedSpraySampleXStartIdx = numOfSamplePoints * 2 * threadBlockIdx; // shared spray sample abscissa start index
		unsigned int sharedSpraySampleYStartIdx = sharedSpraySampleXStartIdx + numOfSamplePoints; // shared spray sample ordinate start index

		unsigned int sharedSpraySampleXIdx;	// shared spray sample abscissa index
		unsigned int sharedSpraySampleYIdx;	// shared spray sample ordinate index

		// initialize output pixel values accumulator to 0
		outputPixel = 0.0f;

		// iteration loop
		for (unsigned int iterationIdx = 0; iterationIdx < numOfIterations; iterationIdx++) {
			// load target pixel and set Emin and Emax equal to target pixel value at each channel
			for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
				targetPixelChannelIdx = targetPixelIdx + channelIdx;
				Emin[channelIdx] = Emax[channelIdx] = targetPixel[channelIdx] = inputImage[targetPixelChannelIdx];
			}

			__syncthreads();
			if (threadBlockIdx < numOfSharedSprays) {
				sprayIdx = curand_uniform(&localState) * numOfSprays;	// choose spray at random from pre-computed sprays in global memory
				spraySampleStartIdx = numOfSamplePoints * sprayIdx;

				for (sampleIdx = 0; sampleIdx < numOfSamplePoints; sampleIdx++) {
					sharedSpraySampleXIdx = sharedSpraySampleXStartIdx + sampleIdx;
					sharedSpraySampleYIdx = sharedSpraySampleYStartIdx + sampleIdx;
					spraySampleIdx = spraySampleStartIdx + sampleIdx;
					sharedSprays[sharedSpraySampleXIdx] = spraysX[spraySampleIdx];	// load spray sample point abscissa into shared memory
					sharedSprays[sharedSpraySampleYIdx] = spraysY[spraySampleIdx];	// load spray sample point ordinate into shared memory
				}
			}
			__syncthreads();

			sharedSprayIdx = curand_uniform(&localState) * numOfSharedSprays;	// choose spray at random from pre-computed sprays in shared memory
			sharedSpraySampleXStartIdx = numOfSamplePoints * 2 * sharedSprayIdx; // shared spray sample abscissa start index
			sharedSpraySampleYStartIdx = sharedSpraySampleXStartIdx + numOfSamplePoints; // shared spray sample ordinate start index

			// get random sample points from spray and calculate envelope
			for (sampleIdx = 0; sampleIdx < numOfSamplePoints; sampleIdx++) {
				sharedSpraySampleXIdx = sharedSpraySampleXStartIdx + sampleIdx;
				sharedSpraySampleYIdx = sharedSpraySampleYStartIdx + sampleIdx;
				samplePixelX = targetPixelX + sharedSprays[sharedSpraySampleXIdx];	// compute sample pixel abscissa
				samplePixelY = targetPixelY + sharedSprays[sharedSpraySampleYIdx];	// compute sample pixel ordinate
				if (samplePixelX >= 0 && samplePixelX < imageWidth && samplePixelY >= 0 && samplePixelY < imageHeight) {	// if sample pixel is within image
					sampleImagePixelIdx = (imageWidth * samplePixelY + samplePixelX) * imageChannels;	// get sample pixel index in image
					for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
						sampleImagePixelChannelIdx = sampleImagePixelIdx + channelIdx;	// get sample pixel channel index
						samplePixel[channelIdx] = inputImage[sampleImagePixelChannelIdx];
						if (samplePixel[channelIdx] < Emin[channelIdx])			// if sample pixel channel value is less than Emin at that channel
							Emin[channelIdx] = samplePixel[channelIdx];			// it is the new Emin
						else if (samplePixel[channelIdx] > Emax[channelIdx])	// if sample pixel channel value is greater than Emax at that channel
							Emax[channelIdx] = samplePixel[channelIdx];			// it is the new Emax
					}
				}
			}

			dotProd = 0;
			ElenSq = 0;
			for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
				Edelta = Emax[channelIdx] - Emin[channelIdx];
				dotProd += Edelta * (targetPixel[channelIdx] - Emin[channelIdx]);
				ElenSq += Edelta * Edelta;
			}

			// calculate g = (p - Emin).(Emax - Emin) / |Emax - Emin|^2
			outputPixel += dotProd * 255.0 / ElenSq;
		}
		outputImage[idx] = outputPixel / numOfIterations;
		state[idx] = localState;	// store updated random number generator state back into global memory
	}
}
#endif

#if defined(ENABLE_STRESS_C2G_GPU_2)
// Does not use pre-computed random sprays. Instead, it generates, in each iteration, for each pixel in the image, a random spray for that pixel.
// This solves the issue of reduced sampling seen in the first version of the function. However, this approach is much slower than using pre-computed sprays
__global__ void STRESSColorToGrayscaleKernel2(curandState *state, uint8_t *outputImage, uint8_t *inputImage, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned int radius, const unsigned int numOfSamplePoints, const unsigned int numOfIterations) {
	unsigned int targetPixelX = blockDim.x * blockIdx.x + threadIdx.x; // target pixel abscissa
	unsigned int targetPixelY = blockDim.y * blockIdx.y + threadIdx.y; // target pixel ordinate

	if (targetPixelX < imageWidth && targetPixelY < imageHeight) {	// if target pixel is within image
		unsigned int idx = imageWidth * targetPixelY + targetPixelX;	// thread / output pixel absolute index
		curandState localState = state[idx];	// load random number generator state from global memory
		
		float randomRadius;		// random radius
		float randomTheta;		// random theta
		const float circle = 2 * M_PI; // 2Pi
		int randomSamplePixelX;	// random sample pixel abscissa
		int randomSamplePixelY;	// random sample pixel ordinate
		unsigned int randomSamplePixelIdx;	// random sample pixel index
		unsigned int randomSampleImagePixelIdx;	// random sample pixel absolute index in image
		unsigned int randomSampleImagePixelChannelIdx;	// random sample pixel channel index

		float sinVal;	// sine value
		float cosVal;	// cosine value

		unsigned int targetPixelIdx = idx * imageChannels;	// target pixel (p) absolute index
		unsigned int targetPixelChannelIdx;		// target pixel channel absolute index
		uint8_t targetPixel[3]; // target pixel array for channels
		uint8_t samplePixel[3];	// sample pixel array for channels
		double outputPixel;		// output pixel values accumulator across iterations
		uint8_t channelIdx;		// channel index
		
		uint8_t Emin[3];	// Emin array of size imageChannels
		uint8_t Emax[3];	// Emax array of size imageChannels
		
		// for calculating (p - Emin).(Emax - Emin) / |Emax - Emin|^2
		uint8_t Edelta;
		unsigned int dotProd, ElenSq;
		
		// initialize output pixel values accumulator to 0
		outputPixel = 0.0f;

		// iteration loop
		for (unsigned int iterationIdx = 0; iterationIdx < numOfIterations; iterationIdx++) {
			// load target pixel and set Emin and Emax equal to target pixel value at each channel
			for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
				targetPixelChannelIdx = targetPixelIdx + channelIdx;
				Emin[channelIdx] = Emax[channelIdx] = targetPixel[channelIdx] = inputImage[targetPixelChannelIdx];
			}

			// generate random sample points and calculate envelope
			randomSamplePixelIdx = 0;
			while (randomSamplePixelIdx < numOfSamplePoints) {
				randomRadius = curand_uniform(&localState) * radius; // get a random distance from the uniform real distribution
				randomTheta = curand_uniform(&localState) * circle; // get a random angle from the uniform real distribution
				sincosf(randomTheta, &sinVal, &cosVal);
				randomSamplePixelX = targetPixelX + randomRadius * cosVal;	// compute random pixel abscissa
				randomSamplePixelY = targetPixelY + randomRadius * sinVal;	// compute random pixel ordinate
				if (randomSamplePixelX >= 0 && randomSamplePixelX < imageWidth && randomSamplePixelY >= 0 && randomSamplePixelY < imageHeight) {	// if random pixel is within image
					randomSampleImagePixelIdx = (imageWidth * randomSamplePixelY + randomSamplePixelX) * imageChannels;	// get random sample pixel index in image
					for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
						randomSampleImagePixelChannelIdx = randomSampleImagePixelIdx + channelIdx;	// get random sample pixel channel index
						samplePixel[channelIdx] = inputImage[randomSampleImagePixelChannelIdx];
						if (samplePixel[channelIdx] < Emin[channelIdx])			// if random sample pixel channel value is less than Emin at that channel
							Emin[channelIdx] = samplePixel[channelIdx];			// it is the new Emin
						else if (samplePixel[channelIdx] > Emax[channelIdx])	// if random sample pixel channel value is greater than Emax at that channel
							Emax[channelIdx] = samplePixel[channelIdx];			// it is the new Emax
					}
					randomSamplePixelIdx++;	// advance random sample pixel index
				}
			}

			dotProd = 0;
			ElenSq = 0;
			for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
				Edelta = Emax[channelIdx] - Emin[channelIdx];
				dotProd += Edelta * (targetPixel[channelIdx] - Emin[channelIdx]);
				ElenSq += Edelta * Edelta;
			}

			// calculate g = (p - Emin).(Emax - Emin) / |Emax - Emin|^2
			outputPixel += dotProd * 255.0 / ElenSq;
		}
		outputImage[idx] = outputPixel / numOfIterations;
		state[idx] = localState;	// store updated random number generator state back into global memory
	}
}
#endif

#if defined(ENABLE_STRESS_C2G_GPU_2B)
// similar to previous approach but replaces sin and cos computations with a LUT. This LUT contains
// float values of sin from 0 to 2Pi in lambda increments and is of size sinLUTLength. Linear interpolation
// is performed since random theta usually falls in-between the angles of a pre-computed sin value pair.
// Make sure to always load a Sine LUT which size is a multiple of 4 so that Sine and Cosine are aligned at exactly Pi / 4 radians.
__global__ void STRESSColorToGrayscaleKernel2B(curandState *state, uint8_t *outputImage, uint8_t *inputImage, float *sinLUT, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned int sinLUTLength, const unsigned int radius, const unsigned int numOfSamplePoints, const unsigned int numOfIterations) {
	extern __shared__ float sharedSinLUT[];

	// copy Sine LUT from global memory to shared memory in coalesced form
	unsigned int threadBlockIdx = blockDim.x * threadIdx.y + threadIdx.x;	// thread absolute index in thread block
	unsigned int stride = blockDim.x * blockDim.y; // stride = thread block size
	for (unsigned int sinLUTIdx = threadBlockIdx; sinLUTIdx < sinLUTLength; sinLUTIdx += stride) {
		sharedSinLUT[sinLUTIdx] = sinLUT[sinLUTIdx];
	}
	__syncthreads();
	
	unsigned int targetPixelX = blockDim.x * blockIdx.x + threadIdx.x; // target pixel abscissa
	unsigned int targetPixelY = blockDim.y * blockIdx.y + threadIdx.y; // target pixel ordinate

	if (targetPixelX < imageWidth && targetPixelY < imageHeight) {	// if target pixel is within image
		unsigned int idx = imageWidth * targetPixelY + targetPixelX;	// thread / output pixel absolute index
		curandState localState = state[idx];	// load random number generator state from global memory
		float randomRadius;		// random radius
		float randomTheta;		// random theta (normalized)

		float sinLUTFloatIdx;	// Sine LUT float index (for linear interpolation)
		float sinLUTFloatOffset;	// Sine LUT float offset from floor index (for linear interpolation)
		unsigned int sinLUTFloorSinValIdx;		// Sine LUT floor value index (Sine)
		unsigned int sinLUTFloorCosValIdx;		// Sine LUT floor value index (Cosine)
		float sinLUTFloorVal;	// Sine LUT floor value for particular theta (for linear interpolation)
		float sinLUTCeilVal;	// Sine LUT ceil value for particular theta (for linear interpolation)
		float sinLUTInterpolatedVal;	// Sine LUT interpolated value
		const unsigned int sinLUTCosShift = sinLUTLength / 4; // Sine LUT Cosine offset (for alignment with Sine)

		int randomSamplePixelX;	// random sample pixel abscissa
		int randomSamplePixelY;	// random sample pixel ordinate
		unsigned int randomSamplePixelIdx;	// random sample pixel index
		unsigned int randomSampleImagePixelIdx;	// random sample pixel absolute index in image
		unsigned int randomSampleImagePixelChannelIdx;	// random sample pixel channel index

		unsigned int targetPixelIdx = idx * imageChannels;	// target pixel (p) absolute index
		unsigned int targetPixelChannelIdx;		// target pixel channel absolute index
		uint8_t targetPixel[3]; // target pixel array for channels
		uint8_t samplePixel[3];	// sample pixel array for channels
		double outputPixel;		// output pixel values accumulator across iterations
		uint8_t channelIdx;		// channel index

		uint8_t Emin[3];	// Emin array of size imageChannels
		uint8_t Emax[3];	// Emax array of size imageChannels

		// for calculating (p - Emin).(Emax - Emin) / |Emax - Emin|^2
		uint8_t Edelta;
		unsigned int dotProd, ElenSq;

		// initialize output pixel values accumulator to 0
		outputPixel = 0.0f;

		// iteration loop
		for (unsigned int iterationIdx = 0; iterationIdx < numOfIterations; iterationIdx++) {
			// load target pixel and set Emin and Emax equal to target pixel value at each channel
			for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
				targetPixelChannelIdx = targetPixelIdx + channelIdx;
				Emin[channelIdx] = Emax[channelIdx] = targetPixel[channelIdx] = inputImage[targetPixelChannelIdx];
			}

			// generate random sample points and calculate envelope
			randomSamplePixelIdx = 0;
			while (randomSamplePixelIdx < numOfSamplePoints) {
				randomRadius = curand_uniform(&localState) * radius; // get a random distance from the uniform real distribution
				randomTheta = curand_uniform(&localState);	// get a random theta from the uniform real distribution
				
				sinLUTFloatIdx = randomTheta * (sinLUTLength - 1);	// calculate Sine LUT float index (Sine)
				sinLUTFloorSinValIdx = floor(sinLUTFloatIdx);	// calculate Sine LUT floor value index (Sine)
				sinLUTFloatOffset = sinLUTFloatIdx - sinLUTFloorSinValIdx;	// calculate Sine LUT float offset from floor index (Sine - same for Cosine due to alignment)
				
				sinLUTFloorCosValIdx = (sinLUTFloorSinValIdx + sinLUTCosShift) % sinLUTLength;	// calculate Sine LUT floor value index (Cosine) - loopback if random theta with shift >= 1.0 (angle >= 2Pi radians)
				
				sinLUTFloorVal = sharedSinLUT[sinLUTFloorCosValIdx];	// get Sine LUT floor value (Cosine)
				sinLUTCeilVal = sharedSinLUT[(sinLUTFloorCosValIdx + 1) % sinLUTLength];	// get Sine LUT ceil value (Cosine)
				sinLUTInterpolatedVal = (sinLUTFloatOffset * sinLUTCeilVal + (1 - sinLUTFloatOffset) * sinLUTFloorVal) / 2;	// calculate Sine LUT interpolated value (Cosine)
				randomSamplePixelX = targetPixelX + randomRadius * sinLUTInterpolatedVal;	// compute random pixel abscissa

				sinLUTFloorVal = sharedSinLUT[sinLUTFloorSinValIdx % sinLUTLength];
				sinLUTCeilVal = sharedSinLUT[(sinLUTFloorSinValIdx + 1) % sinLUTLength];
				sinLUTInterpolatedVal = (sinLUTFloatOffset * sinLUTCeilVal + (1 - sinLUTFloatOffset) * sinLUTFloorVal) / 2;	// calculate Sine LUT interpolated value (Sine)
				randomSamplePixelY = targetPixelY + randomRadius * sinLUTInterpolatedVal;	// compute random pixel ordinate

				if (randomSamplePixelX >= 0 && randomSamplePixelX < imageWidth && randomSamplePixelY >= 0 && randomSamplePixelY < imageHeight) {	// if random pixel is within image
					randomSampleImagePixelIdx = (imageWidth * randomSamplePixelY + randomSamplePixelX) * imageChannels;	// get random sample pixel index in image
					for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
						randomSampleImagePixelChannelIdx = randomSampleImagePixelIdx + channelIdx;	// get random sample pixel channel index
						samplePixel[channelIdx] = inputImage[randomSampleImagePixelChannelIdx];
						if (samplePixel[channelIdx] < Emin[channelIdx])			// if random sample pixel channel value is less than Emin at that channel
							Emin[channelIdx] = samplePixel[channelIdx];			// it is the new Emin
						else if (samplePixel[channelIdx] > Emax[channelIdx])	// if random sample pixel channel value is greater than Emax at that channel
							Emax[channelIdx] = samplePixel[channelIdx];			// it is the new Emax
					}
					randomSamplePixelIdx++;	// advance random sample pixel index
				}
			}

			dotProd = 0;
			ElenSq = 0;
			for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
				Edelta = Emax[channelIdx] - Emin[channelIdx];
				dotProd += Edelta * (targetPixel[channelIdx] - Emin[channelIdx]);
				ElenSq += Edelta * Edelta;
			}

			// calculate g = (p - Emin).(Emax - Emin) / |Emax - Emin|^2
			outputPixel += dotProd * 255.0 / ElenSq;
		}
		outputImage[idx] = outputPixel / numOfIterations;
		state[idx] = localState;	// store updated random number generator state back into global memory
	}
}
#endif

#if defined(ENABLE_STRESS_C2G_GPU_2C)
// similar to previous approach but calculates Sine LUT in thread block instead of loading from global memory
__global__ void STRESSColorToGrayscaleKernel2C(curandState *state, uint8_t *outputImage, uint8_t *inputImage, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned int sinLUTLength, const unsigned int radius, const unsigned int numOfSamplePoints, const unsigned int numOfIterations) {
	extern __shared__ float sharedSinLUT[];

	// copy Sine LUT from global memory to shared memory in coalesced form
	unsigned int threadBlockIdx = blockDim.x * threadIdx.y + threadIdx.x;	// thread absolute index in thread block
	unsigned int stride = blockDim.x * blockDim.y; // stride = thread block size
	float lambda = M_PI / (sinLUTLength / 2);
	float theta;
	for (unsigned int sinLUTIdx = threadBlockIdx; sinLUTIdx < sinLUTLength; sinLUTIdx += stride) {
		theta = lambda * sinLUTIdx;
		sharedSinLUT[sinLUTIdx] = sinf(theta);
	}
	__syncthreads();

	unsigned int targetPixelX = blockDim.x * blockIdx.x + threadIdx.x; // target pixel abscissa
	unsigned int targetPixelY = blockDim.y * blockIdx.y + threadIdx.y; // target pixel ordinate

	if (targetPixelX < imageWidth && targetPixelY < imageHeight) {	// if target pixel is within image
		unsigned int idx = imageWidth * targetPixelY + targetPixelX;	// thread / output pixel absolute index
		curandState localState = state[idx];	// load random number generator state from global memory

		float randomRadius;		// random radius
		float randomTheta;		// random theta (normalized)

		float sinLUTFloatIdx;	// Sine LUT float index (for linear interpolation)
		float sinLUTFloatOffset;	// Sine LUT float offset from floor index (for linear interpolation)
		unsigned int sinLUTFloorSinValIdx;		// Sine LUT floor value index (Sine)
		unsigned int sinLUTFloorCosValIdx;		// Sine LUT floor value index (Cosine)
		float sinLUTFloorVal;	// Sine LUT floor value for particular theta (for linear interpolation)
		float sinLUTCeilVal;	// Sine LUT ceil value for particular theta (for linear interpolation)
		float sinLUTInterpolatedVal;	// Sine LUT interpolated value
		const unsigned int sinLUTCosShift = sinLUTLength / 4; // Sine LUT Cosine offset (for alignment with Sine)

		int randomSamplePixelX;	// random sample pixel abscissa
		int randomSamplePixelY;	// random sample pixel ordinate
		unsigned int randomSamplePixelIdx;	// random sample pixel index
		unsigned int randomSampleImagePixelIdx;	// random sample pixel absolute index in image
		unsigned int randomSampleImagePixelChannelIdx;	// random sample pixel channel index

		unsigned int targetPixelIdx = idx * imageChannels;	// target pixel (p) absolute index
		unsigned int targetPixelChannelIdx;		// target pixel channel absolute index
		uint8_t targetPixel[3]; // target pixel array for channels
		uint8_t samplePixel[3];	// sample pixel array for channels
		double outputPixel;		// output pixel values accumulator across iterations
		uint8_t channelIdx;		// channel index

		uint8_t Emin[3];	// Emin array of size imageChannels
		uint8_t Emax[3];	// Emax array of size imageChannels

		// for calculating (p - Emin).(Emax - Emin) / |Emax - Emin|^2
		uint8_t Edelta;
		unsigned int dotProd, ElenSq;

		// initialize output pixel values accumulator to 0
		outputPixel = 0.0f;

		// iteration loop
		for (unsigned int iterationIdx = 0; iterationIdx < numOfIterations; iterationIdx++) {
			// load target pixel and set Emin and Emax equal to target pixel value at each channel
			for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
				targetPixelChannelIdx = targetPixelIdx + channelIdx;
				Emin[channelIdx] = Emax[channelIdx] = targetPixel[channelIdx] = inputImage[targetPixelChannelIdx];
			}

			// generate random sample points and calculate envelope
			randomSamplePixelIdx = 0;
			while (randomSamplePixelIdx < numOfSamplePoints) {
				randomRadius = curand_uniform(&localState) * radius; // get a random distance from the uniform real distribution
				randomTheta = curand_uniform(&localState);	// get a random theta from the uniform real distribution

				sinLUTFloatIdx = randomTheta * sinLUTLength;	// calculate Sine LUT float index (Sine)
				sinLUTFloorSinValIdx = floor(sinLUTFloatIdx);	// calculate Sine LUT floor value index (Sine)
				sinLUTFloatOffset = sinLUTFloatIdx - sinLUTFloorSinValIdx;	// calculate Sine LUT float offset from floor index (Sine - same for Cosine due to alignment)

				sinLUTFloorCosValIdx = (sinLUTFloorSinValIdx + sinLUTCosShift) % sinLUTLength;	// calculate Sine LUT floor value index (Cosine) - loopback if random theta with shift >= 1.0 (angle >= 2Pi radians)

				sinLUTFloorVal = sharedSinLUT[sinLUTFloorCosValIdx];	// get Sine LUT floor value (Cosine)
				sinLUTCeilVal = sharedSinLUT[(sinLUTFloorCosValIdx + 1) % sinLUTLength];	// get Sine LUT ceil value (Cosine)
				sinLUTInterpolatedVal = (sinLUTFloatOffset * sinLUTCeilVal + (1 - sinLUTFloatOffset) * sinLUTFloorVal) / 2;	// calculate Sine LUT interpolated value (Cosine)
				randomSamplePixelX = targetPixelX + randomRadius * sinLUTInterpolatedVal;	// compute random pixel abscissa

				sinLUTFloorVal = sharedSinLUT[sinLUTFloorSinValIdx % sinLUTLength];
				sinLUTCeilVal = sharedSinLUT[(sinLUTFloorSinValIdx + 1) % sinLUTLength];
				sinLUTInterpolatedVal = (sinLUTFloatOffset * sinLUTCeilVal + (1 - sinLUTFloatOffset) * sinLUTFloorVal) / 2;	// calculate Sine LUT interpolated value (Sine)
				randomSamplePixelY = targetPixelY + randomRadius * sinLUTInterpolatedVal;	// compute random pixel ordinate

				if (randomSamplePixelX >= 0 && randomSamplePixelX < imageWidth && randomSamplePixelY >= 0 && randomSamplePixelY < imageHeight) {	// if random pixel is within image
					randomSampleImagePixelIdx = (imageWidth * randomSamplePixelY + randomSamplePixelX) * imageChannels;	// get random sample pixel index in image
					for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
						randomSampleImagePixelChannelIdx = randomSampleImagePixelIdx + channelIdx;	// get random sample pixel channel index
						samplePixel[channelIdx] = inputImage[randomSampleImagePixelChannelIdx];
						if (samplePixel[channelIdx] < Emin[channelIdx])			// if random sample pixel channel value is less than Emin at that channel
							Emin[channelIdx] = samplePixel[channelIdx];			// it is the new Emin
						else if (samplePixel[channelIdx] > Emax[channelIdx])	// if random sample pixel channel value is greater than Emax at that channel
							Emax[channelIdx] = samplePixel[channelIdx];			// it is the new Emax
					}
					randomSamplePixelIdx++;	// advance random sample pixel index
				}
			}

			dotProd = 0;
			ElenSq = 0;
			for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
				Edelta = Emax[channelIdx] - Emin[channelIdx];
				dotProd += Edelta * (targetPixel[channelIdx] - Emin[channelIdx]);
				ElenSq += Edelta * Edelta;
			}

			// calculate g = (p - Emin).(Emax - Emin) / |Emax - Emin|^2
			outputPixel += dotProd * 255.0 / ElenSq;
		}
		outputImage[idx] = outputPixel / numOfIterations;
		state[idx] = localState;	// store updated random number generator state back into global memory
	}
}
#endif

#if defined(ENABLE_STRESS_C2G_GPU_2D)
// similar to B but loads a Sine LUT that is 4 times smaller for the same precision or
// 4 times more precise for the same size since it only includes angles <= Pi/2 radians.
// This approach thus maps all angles to <= Pi/2 radians.
__global__ void STRESSColorToGrayscaleKernel2D(curandState *state, uint8_t *outputImage, uint8_t *inputImage, float *sinLUT, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned int sinLUTLength, const unsigned int radius, const unsigned int numOfSamplePoints, const unsigned int numOfIterations) {
	extern __shared__ float sharedSinLUT[];

	// copy Sine LUT from global memory to shared memory in coalesced form
	unsigned int threadBlockIdx = blockDim.x * threadIdx.y + threadIdx.x;	// thread absolute index in thread block
	unsigned int stride = blockDim.x * blockDim.y; // stride = thread block size
	for (unsigned int sinLUTIdx = threadBlockIdx; sinLUTIdx < sinLUTLength; sinLUTIdx += stride) {
		sharedSinLUT[sinLUTIdx] = sinLUT[sinLUTIdx];
	}
	__syncthreads();

	unsigned int targetPixelX = blockDim.x * blockIdx.x + threadIdx.x; // target pixel abscissa
	unsigned int targetPixelY = blockDim.y * blockIdx.y + threadIdx.y; // target pixel ordinate

	if (targetPixelX < imageWidth && targetPixelY < imageHeight) {	// if target pixel is within image
		unsigned int idx = imageWidth * targetPixelY + targetPixelX;	// thread / output pixel absolute index
		curandState localState = state[idx];	// load random number generator state from global memory

		float randomRadius;		// random radius
		float randomTheta;		// random angle (normalized)
		float alpha;			// angle restricted to <= Pi/2 radians (normalized)

		float sinLUTFloatSinIdx;	// Sine LUT float index (Sine, for linear interpolation)
		float sinLUTFloatCosIdx;	// Sine LUT float index (Cosine, for linear interpolation)
		float sinLUTFloatSinOffset;	// Sine LUT float offset from floor index (Sine, for linear interpolation)
		float sinLUTFloatCosOffset;	// Sine LUT float offset from floor index (Cosine, for linear interpolation)
		unsigned int sinLUTFloorSinValIdx;		// Sine LUT floor value index (Sine)
		unsigned int sinLUTFloorCosValIdx;		// Sine LUT floor value index (Cosine)
		float sinLUTFloorSinVal;	// Sine LUT floor value for particular theta (Sine, for linear interpolation)
		float sinLUTFloorCosVal;	// Sine LUT floor value for particular theta (Cosine, for linear interpolation)
		float sinLUTCeilSinVal;	// Sine LUT ceil value for particular theta (Sine, for linear interpolation)
		float sinLUTCeilCosVal;	// Sine LUT ceil value for particular theta (Sine, for linear interpolation)
		float sinLUTInterpolatedSinVal;	// Sine LUT interpolated value (Sine)
		float sinLUTInterpolatedCosVal;	// Sine LUT interpolated value (Cosine)
		bool sinIsNegative;		// Sine is negative flag
		bool cosIsNegative;		// Cosine is negative flag
		uint8_t sinNegator;		// Sine negator
		uint8_t cosNegator;		// Cosine negator

		int randomSamplePixelX;	// random sample pixel abscissa
		int randomSamplePixelY;	// random sample pixel ordinate
		unsigned int randomSamplePixelIdx;	// random sample pixel index
		unsigned int randomSampleImagePixelIdx;	// random sample pixel absolute index in image
		unsigned int randomSampleImagePixelChannelIdx;	// random sample pixel channel index

		unsigned int targetPixelIdx = idx * imageChannels;	// target pixel (p) absolute index
		unsigned int targetPixelChannelIdx;		// target pixel channel absolute index
		uint8_t targetPixel[3]; // target pixel array for channels
		uint8_t samplePixel[3];	// sample pixel array for channels
		double outputPixel;		// output pixel values accumulator across iterations
		uint8_t channelIdx;		// channel index

		uint8_t Emin[3];	// Emin array of size imageChannels
		uint8_t Emax[3];	// Emax array of size imageChannels

		// for calculating (p - Emin).(Emax - Emin) / |Emax - Emin|^2
		uint8_t Edelta;
		unsigned int dotProd, ElenSq;

		// initialize output pixel values accumulator to 0
		outputPixel = 0.0f;

		// iteration loop
		for (unsigned int iterationIdx = 0; iterationIdx < numOfIterations; iterationIdx++) {
			// load target pixel and set Emin and Emax equal to target pixel value at each channel
			for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
				targetPixelChannelIdx = targetPixelIdx + channelIdx;
				Emin[channelIdx] = Emax[channelIdx] = targetPixel[channelIdx] = inputImage[targetPixelChannelIdx];
			}

			// generate random sample points and calculate envelope
			randomSamplePixelIdx = 0;
			while (randomSamplePixelIdx < numOfSamplePoints) {
				randomRadius = curand_uniform(&localState) * radius; // get a random distance from the uniform real distribution
				randomTheta = curand_uniform(&localState);	// get a random theta from the uniform real distribution

				alpha = (uint8_t)(randomTheta <= 0.25f) * randomTheta + (uint8_t)(randomTheta > 0.25f && randomTheta <= 0.5f) * (0.5f - randomTheta) + (uint8_t)(randomTheta > 0.5f && randomTheta <= 0.75f) * (randomTheta - 0.5f) + (uint8_t)(randomTheta > 0.75f) * (1.0f - randomTheta); // trigonometric circle - avoids control divergence

				sinLUTFloatSinIdx = alpha * 4 * (sinLUTLength - 1);	// calculate Sine LUT float index (Sine)
				sinLUTFloorSinValIdx = floor(sinLUTFloatSinIdx);	// calculate Sine LUT floor value index (Sine)
				sinLUTFloatSinOffset = sinLUTFloatSinIdx - sinLUTFloorSinValIdx;	// calculate Sine LUT float offset from floor index (Sine)
				sinLUTFloorSinVal = sharedSinLUT[sinLUTFloorSinValIdx % sinLUTLength];	// get Sine LUT floor value (Sine)
				sinLUTCeilSinVal = sharedSinLUT[(sinLUTFloorSinValIdx + 1) % sinLUTLength];	// get Sine LUT ceil value (Sine)
				sinIsNegative = (randomTheta > 0.5f); // value becomes negative if random theta >= 0.5 (angles >= Pi radians) - avoids control divergence
				sinNegator = (uint8_t)!sinIsNegative - (uint8_t)sinIsNegative;
				sinLUTInterpolatedSinVal = sinNegator * (sinLUTFloatSinOffset * sinLUTCeilSinVal + (1.0f - sinLUTFloatSinOffset) * sinLUTFloorSinVal) / 2;	// calculate Sine LUT interpolated value (Sine)
				randomSamplePixelY = targetPixelY + randomRadius * sinLUTInterpolatedSinVal;	// compute random pixel ordinate
				
				sinLUTFloatCosIdx = (0.25f - alpha) * 4 * (sinLUTLength - 1);
				sinLUTFloorCosValIdx = floor(sinLUTFloatCosIdx);	// calculate Sine LUT floor value index (Cosine, complement of Sine, aligned)
				sinLUTFloatCosOffset = sinLUTFloatCosIdx - sinLUTFloorCosValIdx;	// calculate Sine LUT float offset from floor index (Cosine, complement of Sine)
				sinLUTFloorCosVal = sharedSinLUT[sinLUTFloorCosValIdx % sinLUTLength];	// get Sine LUT floor value (Cosine)
				sinLUTCeilCosVal = sharedSinLUT[(sinLUTFloorCosValIdx + 1) % sinLUTLength];	// get Sine LUT ceil value (Cosine)
				cosIsNegative = (randomTheta > 0.25f) && (randomTheta < 0.75f); // value becomes negative if 0.25 < random theta < 0.75 (angles >= Pi radians) - avoids control divergence
				cosNegator = (uint8_t)!cosIsNegative - (uint8_t)cosIsNegative;
				sinLUTInterpolatedCosVal = cosNegator * (sinLUTFloatCosOffset * sinLUTCeilCosVal + (1.0f - sinLUTFloatCosOffset) * sinLUTFloorCosVal) / 2;	// calculate Sine LUT interpolated value (Cosine)
				randomSamplePixelX = targetPixelX + randomRadius * sinLUTInterpolatedCosVal;	// compute random pixel abscissa

				if (randomSamplePixelX >= 0 && randomSamplePixelX < imageWidth && randomSamplePixelY >= 0 && randomSamplePixelY < imageHeight) {	// if random pixel is within image
					randomSampleImagePixelIdx = (imageWidth * randomSamplePixelY + randomSamplePixelX) * imageChannels;	// get random sample pixel index in image
					for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
						randomSampleImagePixelChannelIdx = randomSampleImagePixelIdx + channelIdx;	// get random sample pixel channel index
						samplePixel[channelIdx] = inputImage[randomSampleImagePixelChannelIdx];
						if (samplePixel[channelIdx] < Emin[channelIdx])			// if random sample pixel channel value is less than Emin at that channel
							Emin[channelIdx] = samplePixel[channelIdx];			// it is the new Emin
						else if (samplePixel[channelIdx] > Emax[channelIdx])	// if random sample pixel channel value is greater than Emax at that channel
							Emax[channelIdx] = samplePixel[channelIdx];			// it is the new Emax
					}
					randomSamplePixelIdx++;	// advance random sample pixel index
				}
			}

			dotProd = 0;
			ElenSq = 0;
			for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
				Edelta = Emax[channelIdx] - Emin[channelIdx];
				dotProd += Edelta * (targetPixel[channelIdx] - Emin[channelIdx]);
				ElenSq += Edelta * Edelta;
			}

			// calculate g = (p - Emin).(Emax - Emin) / |Emax - Emin|^2
			outputPixel += dotProd * 255.0 / ElenSq;
		}
		outputImage[idx] = outputPixel / numOfIterations;
		state[idx] = localState;	// store updated random number generator state back into global memory
	}
}
#endif

#if defined(ENABLE_STRESS_C2G_GPU_3)
// This version of the function is a hybrid between the first two approaches. It uses pre-computed sprays similarly to the first approach.
// However, for any pixel, if any sample point in its chosen pre-computed spray is found to be lying outside the image, it is replaced with
// randomly chosen sample points lying within the image. This should solve the issue of the first approach while not being as slow as the second approach,
// particularly for pixels not close to the edges of the image, since the likelihood of a sample point not lying within the image for those diminishes greatly.
__global__ void STRESSColorToGrayscaleKernel3(curandState *state, uint8_t *outputImage, uint8_t *inputImage, short int *spraysX, short int *spraysY, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned int numOfSprays, const unsigned int radius, const unsigned int numOfSamplePoints, const unsigned int numOfIterations) {
	unsigned int targetPixelX = blockDim.x * blockIdx.x + threadIdx.x; // target pixel abscissa
	unsigned int targetPixelY = blockDim.y * blockIdx.y + threadIdx.y; // target pixel ordinate

	if (targetPixelX < imageWidth && targetPixelY < imageHeight) {	// if target pixel is within image
		unsigned int idx = imageWidth * targetPixelY + targetPixelX;	// thread / output pixel absolute index
		curandState localState = state[idx];	// load random number generator state from global memory

		int samplePixelX;	// sample pixel abscissa
		int samplePixelY;	// sample pixel ordinate
		unsigned int sampleIdx;	// sample index from spray
		unsigned int sampleImagePixelIdx;	// sample pixel absolute index in image
		unsigned int sampleImagePixelChannelIdx;	// sample pixel channel index
		unsigned int numOfValidSamplePoints;	// number of valid sample points in envelope

		float randomRadius;		// random radius
		float randomTheta;		// random angle (normalized)
		const float circle = 2 * M_PI; // 2Pi
		/*float alpha;			// angle restricted to <= Pi/2 radians (normalized)

		float sinLUTFloatSinIdx;	// Sine LUT float index (Sine, for linear interpolation)
		float sinLUTFloatCosIdx;	// Sine LUT float index (Cosine, for linear interpolation)
		float sinLUTFloatSinOffset;	// Sine LUT float offset from floor index (Sine, for linear interpolation)
		float sinLUTFloatCosOffset;	// Sine LUT float offset from floor index (Cosine, for linear interpolation)
		unsigned int sinLUTFloorSinValIdx;		// Sine LUT floor value index (Sine)
		unsigned int sinLUTFloorCosValIdx;		// Sine LUT floor value index (Cosine)
		float sinLUTFloorSinVal;	// Sine LUT floor value for particular theta (Sine, for linear interpolation)
		float sinLUTFloorCosVal;	// Sine LUT floor value for particular theta (Cosine, for linear interpolation)
		float sinLUTCeilSinVal;	// Sine LUT ceil value for particular theta (Sine, for linear interpolation)
		float sinLUTCeilCosVal;	// Sine LUT ceil value for particular theta (Sine, for linear interpolation)
		float sinLUTInterpolatedSinVal;	// Sine LUT interpolated value (Sine)
		float sinLUTInterpolatedCosVal;	// Sine LUT interpolated value (Cosine)
		bool sinIsNegative;		// Sine is negative flag
		bool cosIsNegative;		// Cosine is negative flag
		uint8_t sinNegator;		// Sine negator
		uint8_t cosNegator;		// Cosine negator*/

		int randomSamplePixelX;	// random sample pixel abscissa
		int randomSamplePixelY;	// random sample pixel ordinate
		unsigned int randomSamplePixelIdx;	// random sample pixel index
		unsigned int randomSampleImagePixelIdx;	// random sample pixel absolute index in image
		unsigned int randomSampleImagePixelChannelIdx;	// random sample pixel channel index

		float sinVal;	// sine value
		float cosVal;	// cosine value

		unsigned int targetPixelIdx = idx * imageChannels;	// target pixel (p) absolute index
		unsigned int targetPixelChannelIdx;		// target pixel channel absolute index
		uint8_t targetPixel[3]; // target pixel array for channels
		uint8_t samplePixel[3];	// sample pixel array for channels
		double outputPixel;		// output pixel values accumulator across iterations
		uint8_t channelIdx;		// channel index

		uint8_t Emin[3];	// Emin array of size imageChannels
		uint8_t Emax[3];	// Emax array of size imageChannels

		// for calculating (p - Emin).(Emax - Emin) / |Emax - Emin|^2
		uint8_t Edelta;
		unsigned int dotProd, ElenSq;

		unsigned int sprayIdx; // spray index
		unsigned int spraySampleStartIdx; // spray sample start index
		unsigned int spraySampleIdx; // spray sample absolute index

		// initialize output pixel values accumulator to 0
		outputPixel = 0.0f;

		// iteration loop
		for (unsigned int iterationIdx = 0; iterationIdx < numOfIterations; iterationIdx++) {
			// load target pixel and set Emin and Emax equal to target pixel value at each channel
			for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
				targetPixelChannelIdx = targetPixelIdx + channelIdx;
				Emin[channelIdx] = Emax[channelIdx] = targetPixel[channelIdx] = inputImage[targetPixelChannelIdx];
			}

			// get random sample points from spray and calculate envelope
			numOfValidSamplePoints = 0;	// reset number of valid sample points to 0
			sprayIdx = curand_uniform(&localState) * numOfSprays;	// choose spray at random from pre-computed sprays
			spraySampleStartIdx = numOfSamplePoints * sprayIdx;	// calculate spray sample start index
			for (sampleIdx = 0; sampleIdx < numOfSamplePoints; sampleIdx++) {
				spraySampleIdx = spraySampleStartIdx + sampleIdx;
				samplePixelX = targetPixelX + spraysX[spraySampleIdx];	// compute sample pixel abscissa
				samplePixelY = targetPixelY + spraysY[spraySampleIdx];	// compute sample pixel ordinate
				if (samplePixelX >= 0 && samplePixelX < imageWidth && samplePixelY >= 0 && samplePixelY < imageHeight) {	// if sample pixel is within image
					sampleImagePixelIdx = (imageWidth * samplePixelY + samplePixelX) * imageChannels;	// get sample pixel index in image
					for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
						sampleImagePixelChannelIdx = sampleImagePixelIdx + channelIdx;	// get sample pixel channel index
						samplePixel[channelIdx] = inputImage[sampleImagePixelChannelIdx];
						if (samplePixel[channelIdx] < Emin[channelIdx])			// if sample pixel channel value is less than Emin at that channel
							Emin[channelIdx] = samplePixel[channelIdx];			// it is the new Emin
						else if (samplePixel[channelIdx] > Emax[channelIdx])	// if sample pixel channel value is greater than Emax at that channel
							Emax[channelIdx] = samplePixel[channelIdx];			// it is the new Emax
					}
					numOfValidSamplePoints++;
				}
			}

			// generate sample points to compensate for invalid sample points
			sampleIdx = numOfValidSamplePoints;
			while (sampleIdx < numOfSamplePoints) {
				randomRadius = curand_uniform(&localState) * radius; // get a random distance from the uniform real distribution
				randomTheta = curand_uniform(&localState) * circle; // get a random angle from the uniform real distribution
				sincosf(randomTheta, &sinVal, &cosVal);
				randomSamplePixelX = targetPixelX + randomRadius * cosVal;	// compute random pixel abscissa
				randomSamplePixelY = targetPixelY + randomRadius * sinVal;	// compute random pixel ordinate
				if (randomSamplePixelX >= 0 && randomSamplePixelX < imageWidth && randomSamplePixelY >= 0 && randomSamplePixelY < imageHeight) {	// if random pixel is within image
					randomSampleImagePixelIdx = (imageWidth * randomSamplePixelY + randomSamplePixelX) * imageChannels;	// get random sample pixel index in image
					for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
						randomSampleImagePixelChannelIdx = randomSampleImagePixelIdx + channelIdx;	// get random sample pixel channel index
						samplePixel[channelIdx] = inputImage[randomSampleImagePixelChannelIdx];
						if (samplePixel[channelIdx] < Emin[channelIdx])			// if random sample pixel channel value is less than Emin at that channel
							Emin[channelIdx] = samplePixel[channelIdx];			// it is the new Emin
						else if (samplePixel[channelIdx] > Emax[channelIdx])	// if random sample pixel channel value is greater than Emax at that channel
							Emax[channelIdx] = samplePixel[channelIdx];			// it is the new Emax
					}
					sampleIdx++;	// advance random sample pixel index
				}
			}

			dotProd = 0;
			ElenSq = 0;
			for (channelIdx = 0; channelIdx < imageChannels; channelIdx++) {
				Edelta = Emax[channelIdx] - Emin[channelIdx];
				dotProd += Edelta * (targetPixel[channelIdx] - Emin[channelIdx]);
				ElenSq += Edelta * Edelta;
			}

			// calculate g = (p - Emin).(Emax - Emin) / |Emax - Emin|^2
			outputPixel += dotProd * 255.0 / ElenSq;
		}
		outputImage[idx] = outputPixel / numOfIterations;
		state[idx] = localState;	// store updated random number generator state back into global memory
	}
}
#endif

int main(int argc, char *argv[])
{
	if (argc != 8) {
		fprintf(stderr, "Invalid number of arguments.");
		return 1;
	}

	srand(time(NULL));
	const unsigned short int radius = atoi(argv[2]);
	const unsigned int numOfSamplePoints = atoi(argv[3]);
	const unsigned int numOfIterations = atoi(argv[4]);
	const unsigned int numOfSprays = atoi(argv[5]);
	const unsigned int numOfSharedSprays = atoi(argv[6]);
	const unsigned int sinLUTLength = atoi(argv[7]);
	char *imageName = argv[1];
	char outputImageName[50];
	#if defined(ENABLE_STRESS_G2G_CPU_1) || defined(ENABLE_STRESS_G2G_CPU_3) || defined(ENABLE_STRESS_C2G_CPU_3) || defined(ENABLE_STRESS_C2C_CPU_3) || defined(ENABLE_STRESS_C2G_GPU_1) || defined(ENABLE_STRESS_C2G_GPU_1B) || defined(ENABLE_STRESS_C2G_GPU_3)
	short int **spraysX;
	short int **spraysY;
	clock_t computeRandomSpraysCPUClock = clock();
	computeRandomSpraysCPU(&spraysX, &spraysY, radius, numOfSamplePoints, numOfSprays);
	double computeRandomSpraysCPUDuration = (clock() - computeRandomSpraysCPUClock) / (double)CLOCKS_PER_SEC;

	printf("Time to compute random sprays (CPU): %fs\n", computeRandomSpraysCPUDuration);
	
	#if defined(WRITE_RANDOM_SPRAYS_TO_DISK)
	printf("Writing random sprays (%i) to disk ...\n", numOfSprays);
	char sprayImageName[20];
	for (unsigned int sprayIdx = 0; sprayIdx < numOfSprays; sprayIdx++) {
		cv::Mat sprayImage = generateRandomSprayImage(spraysX[sprayIdx], spraysY[sprayIdx], radius, numOfSamplePoints);
		sprintf(sprayImageName, "spray%i.png", sprayIdx);
		cv::imwrite(sprayImageName, sprayImage);
	}
	#endif
	#elif defined(ENABLE_STRESS_C2G_GPU_2B)
	float *sinLUT = (float*)malloc(sinLUTLength * sizeof(float));
	const float lambda = M_PI / (sinLUTLength / 2); // angle moves from 0 to 2Pi radians - calculate lambda
	float theta;
	for (unsigned int sinLUTIdx; sinLUTIdx < sinLUTLength; sinLUTIdx++) {
		theta = lambda * sinLUTIdx;
		sinLUT[sinLUTIdx] = sin(theta);
	}
	#elif defined(ENABLE_STRESS_C2G_GPU_2D)
	float *sinLUT = (float*)malloc(sinLUTLength * sizeof(float));
	const float lambda = M_PI / (sinLUTLength * 2); // angle moves from 0 to Pi/2 radians - calculate lambda
	float theta;
	for (unsigned int sinLUTIdx; sinLUTIdx < sinLUTLength; sinLUTIdx++) {
		theta = lambda * sinLUTIdx;
		sinLUT[sinLUTIdx] = sin(theta);
	}
	#endif

	#if defined(ENABLE_STRESS_G2G_CPU_1) || defined(ENABLE_STRESS_G2G_CPU_2) || defined(ENABLE_STRESS_G2G_CPU_3)
	cv::Mat grayscaleInputImage = cv::imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
	if (grayscaleInputImage.empty()) {
		fprintf(stderr, "Cannot read grayscale image file %s.", imageName);
		return 1;
	}
	uint8_t *grayscaleOutputImageData = (uint8_t*)malloc(grayscaleInputImage.cols * grayscaleInputImage.rows * sizeof(uint8_t));
	#endif
	#if defined(ENABLE_STRESS_C2C_CPU_3) || defined(ENABLE_STRESS_C2G_CPU_3) || defined(ENABLE_STRESS_C2G_GPU_1) || defined(ENABLE_STRESS_C2G_GPU_1B) || defined(ENABLE_STRESS_C2G_GPU_2) || defined(ENABLE_STRESS_C2G_GPU_2B) || defined(ENABLE_STRESS_C2G_GPU_2C) || defined(ENABLE_STRESS_C2G_GPU_2D) || defined(ENABLE_STRESS_C2G_GPU_3)
	cv::Mat colorInputImage = cv::imread(imageName, CV_LOAD_IMAGE_COLOR);
	#endif
	#if defined(ENABLE_STRESS_C2C_CPU_3)
	unsigned int colorImageSize = colorInputImage.cols * colorInputImage.rows * colorInputImage.channels();
	uint8_t *colorOutputImageData = (uint8_t*)malloc(colorImageSize * sizeof(uint8_t));
	#elif defined(ENABLE_STRESS_C2G_CPU_3) || defined(ENABLE_STRESS_C2G_GPU_1) || defined(ENABLE_STRESS_C2G_GPU_1B) || defined(ENABLE_STRESS_C2G_GPU_2) || defined(ENABLE_STRESS_C2G_GPU_2B) || defined(ENABLE_STRESS_C2G_GPU_2C) || defined(ENABLE_STRESS_C2G_GPU_2D) || defined(ENABLE_STRESS_C2G_GPU_3)
	uint8_t *grayscaleOutputImageData = (uint8_t*)malloc(colorInputImage.cols * colorInputImage.rows * sizeof(uint8_t));
	#endif

	#if defined(ENABLE_STRESS_G2G_CPU_1)
	printf("Running STRESSGrayscaleToGrayscaleCPU1 (R=%i, M=%i, N=%i, S=%i) ...\n", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	clock_t STRESSG2GCPU1Clock = clock();
	STRESSGrayscaleToGrayscaleCPU1(grayscaleOutputImageData, grayscaleInputImage.data, grayscaleInputImage.cols, grayscaleInputImage.rows, spraysX, spraysY, numOfSamplePoints, numOfSprays, numOfIterations);
	double STRESSG2GCPU1Duration = (clock() - STRESSG2GCPU1Clock) / (double) CLOCKS_PER_SEC;
	printf("Finished STRESSGrayscaleToGrayscaleCPU1 in %fs, dumping to disk ...\n", STRESSG2GCPU1Duration);
	cv::Mat G2GOutputImageCPU1(grayscaleInputImage.rows, grayscaleInputImage.cols, CV_8UC1, grayscaleOutputImageData);
	sprintf(imageName, "outG2GCPU1_R%i_M%i_N%i_S%i.png", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	cv::imwrite(imageName, G2GOutputImageCPU1);
	#endif

	#if defined(ENABLE_STRESS_G2G_CPU_2)
	printf("Running STRESSGrayscaleToGrayscaleCPU2 (R=%i, M=%i, N=%i) ...\n", radius, numOfSamplePoints, numOfIterations);
	clock_t STRESSG2GCPU2Clock = clock();
	STRESSGrayscaleToGrayscaleCPU2(grayscaleOutputImageData, grayscaleInputImage.data, grayscaleInputImage.cols, grayscaleInputImage.rows, radius, numOfSamplePoints, numOfIterations);
	double STRESSG2GCPU2Duration = (clock() - STRESSG2GCPU2Clock) / (double)CLOCKS_PER_SEC;
	printf("Finished STRESSGrayscaleToGrayscaleCPU2 in %fs, dumping to disk ...\n", STRESSG2GCPU2Duration);
	cv::Mat G2GOutputImageCPU2(grayscaleInputImage.rows, grayscaleInputImage.cols, CV_8UC1, grayscaleOutputImageData);
	sprintf(imageName, "outG2GCPU2_R%i_M%i_N%i_S%i.png", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	cv::imwrite(imageName, G2GOutputImageCPU2);
	#endif

	#if defined(ENABLE_STRESS_G2G_CPU_3)
	printf("Running STRESSGrayscaleToGrayscaleCPU3 (R=%i, M=%i, N=%i, S=%i) ...\n", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	clock_t STRESSG2GCPU3Clock = clock();
	STRESSGrayscaleToGrayscaleCPU3(grayscaleOutputImageData, grayscaleInputImage.data, grayscaleInputImage.cols, grayscaleInputImage.rows, spraysX, spraysY, radius, numOfSamplePoints, numOfSprays, numOfIterations);
	double STRESSG2GCPU3Duration = (clock() - STRESSG2GCPU3Clock) / (double)CLOCKS_PER_SEC;
	printf("Finished STRESSGrayscaleToGrayscaleCPU3 in %fs, dumping to disk ...\n", STRESSG2GCPU3Duration);
	cv::Mat G2GOutputImageCPU3(grayscaleInputImage.rows, grayscaleInputImage.cols, CV_8UC1, grayscaleOutputImageData);
	sprintf(imageName, "outG2GCPU3_R%i_M%i_N%i_S%i.png", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	cv::imwrite(imageName, G2GOutputImageCPU3);
	#endif

	#if defined(ENABLE_STRESS_C2G_CPU_3)
	printf("Running STRESSColorToGrayscaleCPU3 (R=%i, M=%i, N=%i, S=%i) ...\n", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	clock_t STRESSC2GCPU3Clock = clock();
	STRESSColorToGrayscaleCPU3(grayscaleOutputImageData, colorInputImage.data, colorInputImage.cols, colorInputImage.rows, colorInputImage.channels(), spraysX, spraysY, radius, numOfSamplePoints, numOfSprays, numOfIterations);
	double STRESSC2GCPU3Duration = (clock() - STRESSC2GCPU3Clock) / (double)CLOCKS_PER_SEC;
	printf("Finished STRESSColorToGrayscaleCPU3 in %fs, dumping to disk ...\n", STRESSC2GCPU3Duration);
	cv::Mat C2GOutputImageCPU3(colorInputImage.rows, colorInputImage.cols, CV_8UC1, grayscaleOutputImageData);
	sprintf(imageName, "outC2GCPU3_R%i_M%i_N%i_S%i.png", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	cv::imwrite(imageName, C2GOutputImageCPU3);
	#endif

	#if defined(ENABLE_STRESS_C2C_CPU_3)
	printf("Running STRESSColorToColorCPU3 (R=%i, M=%i, N=%i, S=%i) ...\n", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	clock_t STRESSC2CCPU3Clock = clock();
	STRESSColorToColorCPU3(colorOutputImageData, colorInputImage.data, colorInputImage.cols, colorInputImage.rows, colorInputImage.channels(), spraysX, spraysY, radius, numOfSamplePoints, numOfSprays, numOfIterations);
	double STRESSC2CCPU3Duration = (clock() - STRESSC2CCPU3Clock) / (double)CLOCKS_PER_SEC;
	printf("Finished STRESSColorToColorCPU3 in %fs, dumping to disk ...\n", STRESSC2CCPU3Duration);
	cv::Mat C2COutputImageCPU3(colorInputImage.rows, colorInputImage.cols, CV_8UC3, colorOutputImageData);
	sprintf(imageName, "outC2CCPU3_R%i_M%i_N%i_S%i.png", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	cv::imwrite(imageName, C2COutputImageCPU3);
	#endif
	
	#if defined(ENABLE_STRESS_C2G_GPU_1) || defined(ENABLE_STRESS_C2G_GPU_1B) || defined(ENABLE_STRESS_C2G_GPU_2) || defined(ENABLE_STRESS_C2G_GPU_2B) || defined(ENABLE_STRESS_C2G_GPU_2C) || defined(ENABLE_STRESS_C2G_GPU_2D) || defined(ENABLE_STRESS_C2G_GPU_3)
	printf("Running STRESSC2GKernel (R=%i, M=%i, N=%i, S=%i, SS=%i, SL=%i) ...\n", radius, numOfSamplePoints, numOfIterations, numOfSprays, numOfSharedSprays, sinLUTLength);
	unsigned long seed = time(NULL);
	#if defined(ENABLE_STRESS_C2G_GPU_1)
	cudaError_t cudaStatus = STRESSColorToGrayscaleKernelHelper(grayscaleOutputImageData, colorInputImage.data, spraysX, spraysY, colorInputImage.cols, colorInputImage.rows, colorInputImage.channels(), radius, numOfSamplePoints, numOfSprays, numOfIterations, seed);
	#elif defined(ENABLE_STRESS_C2G_GPU_1B)
	cudaError_t cudaStatus = STRESSColorToGrayscaleKernelHelper(grayscaleOutputImageData, colorInputImage.data, spraysX, spraysY, numOfSharedSprays, colorInputImage.cols, colorInputImage.rows, colorInputImage.channels(), radius, numOfSamplePoints, numOfSprays, numOfIterations, seed);
	#elif defined(ENABLE_STRESS_C2G_GPU_2)
	cudaError_t cudaStatus = STRESSColorToGrayscaleKernelHelper(grayscaleOutputImageData, colorInputImage.data, colorInputImage.cols, colorInputImage.rows, colorInputImage.channels(), radius, numOfSamplePoints, numOfIterations, seed);
	#elif defined(ENABLE_STRESS_C2G_GPU_2B) || defined(ENABLE_STRESS_C2G_GPU_2D)
	cudaError_t cudaStatus = STRESSColorToGrayscaleKernelHelper(grayscaleOutputImageData, colorInputImage.data, sinLUT, colorInputImage.cols, colorInputImage.rows, colorInputImage.channels(), sinLUTLength, radius, numOfSamplePoints, numOfIterations, seed);
	#elif defined(ENABLE_STRESS_C2G_GPU_2C)
	cudaError_t cudaStatus = STRESSColorToGrayscaleKernelHelper(grayscaleOutputImageData, colorInputImage.data, colorInputImage.cols, colorInputImage.rows, colorInputImage.channels(), sinLUTLength, radius, numOfSamplePoints, numOfIterations, seed);
	#elif defined(ENABLE_STRESS_C2G_GPU_3)
	cudaError_t cudaStatus = STRESSColorToGrayscaleKernelHelper(grayscaleOutputImageData, colorInputImage.data, spraysX, spraysY, colorInputImage.cols, colorInputImage.rows, colorInputImage.channels(), radius, numOfSamplePoints, numOfSprays, numOfIterations, seed);
	#endif
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "STRESSC2GKernelHelper failed!");
        return 1;
    }
	printf("Finished STRESSColorToGrayscaleKernel, dumping to disk ...\n");
	cv::Mat grayscaleOutputImageGPU(colorInputImage.rows, colorInputImage.cols, CV_8UC1, grayscaleOutputImageData);
	sprintf(outputImageName, "outC2GGPU_R%i_M%i_N%i_S%i_SS%i_SL%i.png", radius, numOfSamplePoints, numOfIterations, numOfSprays, numOfSharedSprays, sinLUTLength);
	cv::imwrite(outputImageName, grayscaleOutputImageGPU);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	#endif
	//#endif

	system("PAUSE");
    return 0;
}

#if defined(ENABLE_STRESS_C2G_GPU_1) || defined(ENABLE_STRESS_C2G_GPU_1B) || defined (ENABLE_STRESS_C2G_GPU_2) || defined (ENABLE_STRESS_C2G_GPU_2B) || defined (ENABLE_STRESS_C2G_GPU_2C) || defined(ENABLE_STRESS_C2G_GPU_2D) || defined (ENABLE_STRESS_C2G_GPU_3)
// CUDA helper function
#if defined(ENABLE_STRESS_C2G_GPU_1)
cudaError_t STRESSColorToGrayscaleKernelHelper(uint8_t *outputImage, uint8_t *inputImage, short int **spraysX, short int **spraysY, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfSprays, const unsigned int numOfIterations, const unsigned long long seed)
#elif defined(ENABLE_STRESS_C2G_GPU_1B)
cudaError_t STRESSColorToGrayscaleKernelHelper(uint8_t *outputImage, uint8_t *inputImage, short int **spraysX, short int **spraysY, const unsigned int numOfSharedSprays, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfSprays, const unsigned int numOfIterations, const unsigned long long seed)
#elif defined(ENABLE_STRESS_C2G_GPU_2)
cudaError_t STRESSColorToGrayscaleKernelHelper(uint8_t *outputImage, uint8_t *inputImage, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfIterations, const unsigned long long seed)
#elif defined(ENABLE_STRESS_C2G_GPU_2B) || defined(ENABLE_STRESS_C2G_GPU_2D)
cudaError_t STRESSColorToGrayscaleKernelHelper(uint8_t *outputImage, uint8_t *inputImage, float *sinLUT, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned int sinLUTLength, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfIterations, const unsigned long long seed)
#elif defined(ENABLE_STRESS_C2G_GPU_2C)
cudaError_t STRESSColorToGrayscaleKernelHelper(uint8_t *outputImage, uint8_t *inputImage, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned int sinLUTLength, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfIterations, const unsigned long long seed)
#elif defined(ENABLE_STRESS_C2G_GPU_3)
cudaError_t STRESSColorToGrayscaleKernelHelper(uint8_t *outputImage, uint8_t *inputImage, short int **spraysX, short int **spraysY, const unsigned short int imageWidth, const unsigned short int imageHeight, const uint8_t imageChannels, const unsigned short int radius, const unsigned int numOfSamplePoints, const unsigned int numOfSprays, const unsigned int numOfIterations, const unsigned long long seed)
#endif
{
	GpuTimer cudaMallocInputTimer;
	GpuTimer cudaMallocOutputTimer;
	GpuTimer cudaMemcpyInputTimer;
	#if defined(ENABLE_STRESS_C2G_GPU_1) || defined(ENABLE_STRESS_C2G_GPU_1B) || defined(ENABLE_STRESS_C2G_GPU_3)
	GpuTimer cudaMallocSpraysTimer;
	GpuTimer cudaMemcpySpraysTimer;
	#endif
	#if defined(ENABLE_STRESS_C2G_GPU_1) || defined(ENABLE_STRESS_C2G_GPU_1B) || defined(ENABLE_STRESS_C2G_GPU_2) || defined (ENABLE_STRESS_C2G_GPU_2B) || defined(ENABLE_STRESS_C2G_GPU_2C) || defined(ENABLE_STRESS_C2G_GPU_2D) || defined(ENABLE_STRESS_C2G_GPU_3)
	GpuTimer cudaMallocCurandStatesTimer;
	GpuTimer cudaSetupRandomKernelTimer;
	#endif
	#if defined(ENABLE_STRESS_C2G_GPU_2B) || defined(ENABLE_STRESS_C2G_GPU_2D)
	GpuTimer cudaMallocSinLUTTimer;
	GpuTimer cudaMemcpySinLUTTimer;
	#endif
	GpuTimer cudaSTRESSColorToGrayscaleKernelTimer;
	GpuTimer cudaMemcpyOutputTimer;
	unsigned int outputImageSize = imageWidth * imageHeight;
	unsigned int inputImageSize = outputImageSize * imageChannels;
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
    cudaStatus = cudaMalloc((void**)&d_InputImage, inputImageSize * sizeof(uint8_t));
	cudaMallocInputTimer.Stop();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc (input image) failed!");
        goto Error;
    }
	printf("Time to allocate input:\t\t\t\t\t%f ms\n", cudaMallocInputTimer.Elapsed());

	uint8_t *d_OutputImage;
	cudaMallocOutputTimer.Start();
    cudaStatus = cudaMalloc((void**)&d_OutputImage, outputImageSize * sizeof(uint8_t));
	cudaMallocOutputTimer.Stop();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc (output image) failed!");
        goto Error;
    }
	printf("Time to allocate output:\t\t\t\t%f ms\n", cudaMallocOutputTimer.Elapsed());

	// Declare block and grid dimensions
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	unsigned int gridDimX = (imageWidth - 1) / BLOCK_WIDTH + 1;
	unsigned int gridDimY = (imageHeight - 1) / BLOCK_WIDTH + 1;
	dim3 dimGrid(gridDimX, gridDimY, 1);

	#if defined(ENABLE_STRESS_C2G_GPU_1) || defined(ENABLE_STRESS_C2G_GPU_1B) || defined(ENABLE_STRESS_C2G_GPU_2) || defined(ENABLE_STRESS_C2G_GPU_2B) || defined(ENABLE_STRESS_C2G_GPU_2C) || defined(ENABLE_STRESS_C2G_GPU_2D) || defined(ENABLE_STRESS_C2G_GPU_3)
	// Allocate random number generator states
	//unsigned int numOfThreads = gridDimX * gridDimY * BLOCK_WIDTH * BLOCK_WIDTH;
	curandState *d_CURANDStates;
	cudaMallocCurandStatesTimer.Start();
	cudaStatus = cudaMalloc((void**)&d_CURANDStates, outputImageSize * sizeof(curandState));
	cudaMallocCurandStatesTimer.Stop();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (CURAND states) failed!");
		goto Error;
	}
	printf("Time to allocate CURAND states:\t\t\t\t%f ms\n", cudaMallocCurandStatesTimer.Elapsed());

	// Launch the setup random number generator kernel on the GPU with one thread for each element.
	cudaSetupRandomKernelTimer.Start();
	setupRandomKernel <<<dimGrid, dimBlock>>>(d_CURANDStates, seed, imageWidth, imageHeight);
	cudaSetupRandomKernelTimer.Stop();

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "setupRandomKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching setupRandomKernel!\n", cudaStatus);
		goto Error;
	}
	printf("Time to execute setupRandomKernel kernel:\t\t%f ms\n", cudaSetupRandomKernelTimer.Elapsed());
	#endif

	#if defined(ENABLE_STRESS_C2G_GPU_1) || defined(ENABLE_STRESS_C2G_GPU_1B) || defined(ENABLE_STRESS_C2G_GPU_3)
	short int *d_spraysX;
	short int *d_spraysY;
	cudaMallocSpraysTimer.Start();
	unsigned int spraySampleStartIdx;
	cudaMalloc((void**)&d_spraysX, numOfSprays * numOfSamplePoints * sizeof(short int));
	cudaMalloc((void**)&d_spraysY, numOfSprays * numOfSamplePoints * sizeof(short int));
	cudaMallocSpraysTimer.Stop();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMalloc (pre-computed random sprays) failed!");
	goto Error;
	}
	printf("Time to allocate pre-computed random sprays:\t\t\t%f ms\n", cudaMallocSpraysTimer.Elapsed());

	// Copy pre-computed random sprays to GPU buffers.
	short int *allSpraysX = (short int*)malloc(numOfSprays * numOfSamplePoints * sizeof(short int));
	short int *allSpraysY = (short int*)malloc(numOfSprays * numOfSamplePoints * sizeof(short int));
	for (unsigned int sprayIdx = 0; sprayIdx < numOfSprays; sprayIdx++) {
		for (unsigned int sampleIdx = 0; sampleIdx < numOfSamplePoints; sampleIdx++) {
			allSpraysX[numOfSamplePoints * sprayIdx + sampleIdx] = spraysX[sprayIdx][sampleIdx];
			allSpraysY[numOfSamplePoints * sprayIdx + sampleIdx] = spraysY[sprayIdx][sampleIdx];
		}
	}
	cudaMemcpySpraysTimer.Start();
	cudaMemcpy(d_spraysX, allSpraysX, numOfSprays * numOfSamplePoints * sizeof(short int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_spraysY, allSpraysY, numOfSprays * numOfSamplePoints * sizeof(short int), cudaMemcpyHostToDevice);
	cudaMemcpySpraysTimer.Stop();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy (pre-computed random sprays, host -> device) failed!");
	goto Error;
	}
	printf("Time to copy pre-computed random sprays from host to device:\t\t\t%f ms\n", cudaMemcpySpraysTimer.Elapsed());
	#endif

	#if defined(ENABLE_STRESS_C2G_GPU_2B) || defined(ENABLE_STRESS_C2G_GPU_2D)
	// Allocate Sine LUT
	float *d_SinLUT;
	cudaMallocSinLUTTimer.Start();
	cudaStatus = cudaMalloc((void**)&d_SinLUT, sinLUTLength * sizeof(float));
	cudaMallocSinLUTTimer.Stop();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (Sine LUT) failed!");
		goto Error;
	}
	printf("Time to allocate Sine LUT:\t\t\t\t%f ms\n", cudaMallocSinLUTTimer.Elapsed());

	// Copy sin LUT from host memory to GPU buffers.
	cudaMemcpySinLUTTimer.Start();
	cudaStatus = cudaMemcpy(d_SinLUT, sinLUT, sinLUTLength * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpySinLUTTimer.Stop();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (Sine LUT, host -> device) failed!");
		goto Error;
	}
	printf("Time to copy Sine LUT from host to device:\t\t\t%f ms\n", cudaMemcpySinLUTTimer.Elapsed());
	#endif

	// Copy input vectors from host memory to GPU buffers.
	cudaMemcpyInputTimer.Start();
	cudaStatus = cudaMemcpy(d_InputImage, inputImage, inputImageSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpyInputTimer.Stop();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (host -> device) failed!");
		goto Error;
	}
	printf("Time to copy input from host to device:\t\t\t%f ms\n", cudaMemcpyInputTimer.Elapsed());

	// Launch the STRESS color to grayscale kernel on the GPU with one thread for each element.
	#if defined(ENABLE_STRESS_C2G_GPU_1)
	cudaSTRESSColorToGrayscaleKernelTimer.Start();
	STRESSColorToGrayscaleKernel1 << <dimGrid, dimBlock>> >(d_CURANDStates, d_OutputImage, d_InputImage, d_spraysX, d_spraysY, imageWidth, imageHeight, imageChannels, numOfSprays, radius, numOfSamplePoints, numOfIterations);
	#elif defined(ENABLE_STRESS_C2G_GPU_1B)
	// thanks to https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration
	size_t numOfBytesDynamicSharedMemory = numOfSharedSprays * numOfSamplePoints * 2 * sizeof(short int);
	printf("Dynamic: %i\n", numOfBytesDynamicSharedMemory);
	cudaSTRESSColorToGrayscaleKernelTimer.Start();
	STRESSColorToGrayscaleKernel1B << <dimGrid, dimBlock, numOfBytesDynamicSharedMemory>> >(d_CURANDStates, d_OutputImage, d_InputImage, d_spraysX, d_spraysY, numOfSharedSprays, imageWidth, imageHeight, imageChannels, numOfSprays, radius, numOfSamplePoints, numOfIterations);
	#elif defined(ENABLE_STRESS_C2G_GPU_2)
	cudaSTRESSColorToGrayscaleKernelTimer.Start();
    STRESSColorToGrayscaleKernel2<<<dimGrid, dimBlock>>>(d_CURANDStates, d_OutputImage, d_InputImage, imageWidth, imageHeight, imageChannels, radius, numOfSamplePoints, numOfIterations);
	#elif defined(ENABLE_STRESS_C2G_GPU_2B) || defined(ENABLE_STRESS_C2G_GPU_2C) || defined(ENABLE_STRESS_C2G_GPU_2D)
	size_t numOfBytesDynamicSharedMemory = sinLUTLength * sizeof(float);
	printf("Dynamic: %i\n", numOfBytesDynamicSharedMemory);
	cudaSTRESSColorToGrayscaleKernelTimer.Start();
	#if defined(ENABLE_STRESS_C2G_GPU_2B)
	STRESSColorToGrayscaleKernel2B << <dimGrid, dimBlock, numOfBytesDynamicSharedMemory >> >(d_CURANDStates, d_OutputImage, d_InputImage, d_SinLUT, imageWidth, imageHeight, imageChannels, sinLUTLength, radius, numOfSamplePoints, numOfIterations);
	#elif defined(ENABLE_STRESS_C2G_GPU_2C)
	STRESSColorToGrayscaleKernel2C << <dimGrid, dimBlock, numOfBytesDynamicSharedMemory >> >(d_CURANDStates, d_OutputImage, d_InputImage, imageWidth, imageHeight, imageChannels, sinLUTLength, radius, numOfSamplePoints, numOfIterations);
	#elif defined(ENABLE_STRESS_C2G_GPU_2D)
	STRESSColorToGrayscaleKernel2D << <dimGrid, dimBlock, numOfBytesDynamicSharedMemory >> >(d_CURANDStates, d_OutputImage, d_InputImage, d_SinLUT, imageWidth, imageHeight, imageChannels, sinLUTLength, radius, numOfSamplePoints, numOfIterations);
	#endif
	#elif defined(ENABLE_STRESS_C2G_GPU_3)
	cudaSTRESSColorToGrayscaleKernelTimer.Start();
	STRESSColorToGrayscaleKernel3 << <dimGrid, dimBlock >> >(d_CURANDStates, d_OutputImage, d_InputImage, d_spraysX, d_spraysY, imageWidth, imageHeight, imageChannels, numOfSprays, radius, numOfSamplePoints, numOfIterations);
	#endif
	cudaSTRESSColorToGrayscaleKernelTimer.Stop();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "STRESSColorToGrayscaleKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching STRESSColorToGrayscaleKernel!\n", cudaStatus);
        goto Error;
    }
	printf("Time to execute STRESSColorToGrayscaleKernel kernel:\t%f ms\n", cudaSTRESSColorToGrayscaleKernelTimer.Elapsed());

    // Copy output vector from GPU buffer to host memory.
	cudaMemcpyOutputTimer.Start();
    cudaStatus = cudaMemcpy(outputImage, d_OutputImage, outputImageSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaMemcpyOutputTimer.Stop();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (device -> host) failed!");
        goto Error;
    }

	{
		printf("Time to copy output from device to host:\t\t%f ms\n", cudaMemcpyOutputTimer.Elapsed());
	}


Error:
	cudaFree(d_InputImage);
	cudaFree(d_OutputImage);
	#if defined(ENABLE_STRESS_C2G_GPU_2) || defined(ENABLE_STRESS_C2G_GPU_3)
	cudaFree(d_CURANDStates);
	#endif
	#if defined(ENABLE_STRESS_C2G_GPU_1) || defined(ENABLE_STRESS_C2G_GPU_3)
	for (unsigned int sprayIdx = 0; sprayIdx < numOfSprays; sprayIdx++) {
		cudaFree(spraysX[sprayIdx]);
		cudaFree(spraysY[sprayIdx]);
	}
	cudaFree(spraysX);
	cudaFree(spraysY);
	#endif

	return cudaStatus;
}
#endif