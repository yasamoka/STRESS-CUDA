
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

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

				// calculate envelope
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

// This version of the function is a hybrid between the first two approaches. It uses pre-computed sprays similarly to the first approach.
// However, for any pixel, if any sample point in its chosen pre-computed spray is found to be lying outside the image, it is replaced with
// a randomly chosen sample points lying within the image. This should solve the issue of the first approach while not being as slow as the second approach,
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
				sampleIdx = 0;	// reset sample index to 0
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
							samplePixelIdx = imageWidth * samplePixelY + samplePixelX; // get random sample pixel index in image
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
				//printf("%f %f\n", dotProd, ElenSq);
				tempOutputImage[targetOutputPixelIdx] += dotProd * 255.0 / ElenSq;

				/*for (channelIdx = 0; channelIdx < imageChannels; channelIdx++)
					dotProd += (Emax[channelIdx] - Emin[channelIdx]) * (inputImage[targetInputPixelIdx + channelIdx] - Emin[channelIdx]);
				ElenSq = 0;
				for (channelIdx = 0; channelIdx < imageChannels; channelIdx++)
					ElenSq += (Emax[channelIdx] - Emin[channelIdx]) * (Emax[channelIdx] - Emin[channelIdx])*/

				targetInputPixelIdx += imageChannels;
				targetOutputPixelIdx++;
			}
		}
	}

	// divide each accumulated pixel value by the number of iterations to obtain the average pixel value across iterations.
	// place the average value in the output image array.
	for (unsigned int pixelIdx = 0; pixelIdx < outputImageSize; pixelIdx++) {
		outputImage[pixelIdx] = tempOutputImage[pixelIdx] / numOfIterations;
	}
}

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

int main(int argc, char *argv[])
{
	if (argc != 7) {
		fprintf(stderr, "Invalid number of arguments.");
		return 1;
	}

	srand(time(NULL));
	const unsigned short int radius = atoi(argv[3]);
	const unsigned int numOfSamplePoints = atoi(argv[4]);
	const unsigned int numOfIterations = atoi(argv[5]);
	const unsigned int numOfSprays = atoi(argv[6]);
	short int **spraysX;
	short int **spraysY;
	clock_t computeRandomSpraysCPUClock = clock();
	computeRandomSpraysCPU(&spraysX, &spraysY, radius, numOfSamplePoints, numOfSprays);
	double computeRandomSpraysCPUDuration = (clock() - computeRandomSpraysCPUClock) / (double)CLOCKS_PER_SEC;

	printf("Time to compute random sprays (CPU): %fs\n", computeRandomSpraysCPUDuration);

	/*printf("Writing random sprays (%i) to disk ...\n", numOfSprays);
	char sprayImageName[20];
	for (unsigned int sprayIdx = 0; sprayIdx < numOfSprays; sprayIdx++) {
		cv::Mat sprayImage = generateRandomSprayImage(spraysX[sprayIdx], spraysY[sprayIdx], radius, numOfSamplePoints);
		sprintf(sprayImageName, "spray%i.png", sprayIdx);
		cv::imwrite(sprayImageName, sprayImage);
	}*/

	char imageName[50];

	char *grayscaleImageName = argv[1];
	cv::Mat grayscaleInputImage = cv::imread(grayscaleImageName, CV_LOAD_IMAGE_GRAYSCALE);
	if (grayscaleInputImage.empty()) {
		fprintf(stderr, "Cannot read grayscale image file %s.", grayscaleImageName);
		return 1;
	}
	uint8_t *grayscaleOutputImageData = (uint8_t*)malloc(grayscaleInputImage.cols * grayscaleInputImage.rows * sizeof(uint8_t));

	printf("Running STRESSGrayscaleToGrayscaleCPU1 (R=%i, M=%i, N=%i, S=%i) ...\n", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	clock_t STRESSG2GCPU1Clock = clock();
	STRESSGrayscaleToGrayscaleCPU1(grayscaleOutputImageData, grayscaleInputImage.data, grayscaleInputImage.cols, grayscaleInputImage.rows, spraysX, spraysY, numOfSamplePoints, numOfSprays, numOfIterations);
	double STRESSG2GCPU1Duration = (clock() - STRESSG2GCPU1Clock) / (double) CLOCKS_PER_SEC;
	printf("Finished STRESSGrayscaleToGrayscaleCPU1 in %fs, dumping to disk ...\n", STRESSG2GCPU1Duration);
	cv::Mat G2GOutputImageCPU1(grayscaleInputImage.rows, grayscaleInputImage.cols, CV_8UC1, grayscaleOutputImageData);
	sprintf(imageName, "outG2GCPU1_R%i_M%i_N%i_S%i.png", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	cv::imwrite(imageName, G2GOutputImageCPU1);

	printf("Running STRESSGrayscaleToGrayscaleCPU2 (R=%i, M=%i, N=%i) ...\n", radius, numOfSamplePoints, numOfIterations);
	clock_t STRESSG2GCPU2Clock = clock();
	STRESSGrayscaleToGrayscaleCPU2(grayscaleOutputImageData, grayscaleInputImage.data, grayscaleInputImage.cols, grayscaleInputImage.rows, radius, numOfSamplePoints, numOfIterations);
	double STRESSG2GCPU2Duration = (clock() - STRESSG2GCPU2Clock) / (double)CLOCKS_PER_SEC;
	printf("Finished STRESSGrayscaleToGrayscaleCPU2 in %fs, dumping to disk ...\n", STRESSG2GCPU2Duration);
	cv::Mat G2GOutputImageCPU2(grayscaleInputImage.rows, grayscaleInputImage.cols, CV_8UC1, grayscaleOutputImageData);
	sprintf(imageName, "outG2GCPU2_R%i_M%i_N%i_S%i.png", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	cv::imwrite(imageName, G2GOutputImageCPU2);

	printf("Running STRESSGrayscaleToGrayscaleCPU3 (R=%i, M=%i, N=%i, S=%i) ...\n", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	clock_t STRESSG2GCPU3Clock = clock();
	STRESSGrayscaleToGrayscaleCPU3(grayscaleOutputImageData, grayscaleInputImage.data, grayscaleInputImage.cols, grayscaleInputImage.rows, spraysX, spraysY, radius, numOfSamplePoints, numOfSprays, numOfIterations);
	double STRESSG2GCPU3Duration = (clock() - STRESSG2GCPU3Clock) / (double)CLOCKS_PER_SEC;
	printf("Finished STRESSGrayscaleToGrayscaleCPU3 in %fs, dumping to disk ...\n", STRESSG2GCPU3Duration);
	cv::Mat G2GOutputImageCPU3(grayscaleInputImage.rows, grayscaleInputImage.cols, CV_8UC1, grayscaleOutputImageData);
	sprintf(imageName, "outG2GCPU3_R%i_M%i_N%i_S%i.png", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	cv::imwrite(imageName, G2GOutputImageCPU3);
	
	char *colorimageName = argv[2];
	cv::Mat colorInputImage = cv::imread(colorimageName, CV_LOAD_IMAGE_COLOR);

	printf("Running STRESSColorToGrayscaleCPU3 (R=%i, M=%i, N=%i, S=%i) ...\n", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	clock_t STRESSC2GCPU3Clock = clock();
	STRESSColorToGrayscaleCPU3(grayscaleOutputImageData, colorInputImage.data, colorInputImage.cols, colorInputImage.rows, colorInputImage.channels(), spraysX, spraysY, radius, numOfSamplePoints, numOfSprays, numOfIterations);
	double STRESSC2GCPU3Duration = (clock() - STRESSC2GCPU3Clock) / (double)CLOCKS_PER_SEC;
	printf("Finished STRESSColorToGrayscaleCPU3 in %fs, dumping to disk ...\n", STRESSC2GCPU3Duration);
	cv::Mat C2GOutputImageCPU3(colorInputImage.rows, colorInputImage.cols, CV_8UC1, grayscaleOutputImageData);
	sprintf(imageName, "outC2GCPU3_R%i_M%i_N%i_S%i.png", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	cv::imwrite(imageName, C2GOutputImageCPU3);

	unsigned int colorImageSize = colorInputImage.cols * colorInputImage.rows * colorInputImage.channels();
	uint8_t *colorOutputImageData = (uint8_t*)malloc(colorImageSize * sizeof(uint8_t));

	printf("Running STRESSColorToColorCPU3 (R=%i, M=%i, N=%i, S=%i) ...\n", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	clock_t STRESSC2CCPU3Clock = clock();
	STRESSColorToColorCPU3(colorOutputImageData, colorInputImage.data, colorInputImage.cols, colorInputImage.rows, colorInputImage.channels(), spraysX, spraysY, radius, numOfSamplePoints, numOfSprays, numOfIterations);
	double STRESSC2CCPU3Duration = (clock() - STRESSC2CCPU3Clock) / (double)CLOCKS_PER_SEC;
	printf("Finished STRESSColorToColorCPU3 in %fs, dumping to disk ...\n", STRESSC2CCPU3Duration);
	cv::Mat C2COutputImageCPU3(colorInputImage.rows, colorInputImage.cols, CV_8UC3, colorOutputImageData);
	sprintf(imageName, "outC2CCPU3_R%i_M%i_N%i_S%i.png", radius, numOfSamplePoints, numOfIterations, numOfSprays);
	cv::imwrite(imageName, C2COutputImageCPU3);
	
	system("PAUSE");
	return 0;

    /*cudaError_t cudaStatus = testWithCuda(inputImage.data, outputImageData, inputImage.cols, inputImage.rows, inputImage.channels());
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

    return 0;*/
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
