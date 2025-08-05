/**
 * @file LineMandelCalculator.cc
 * @author Ján Maťufka <xmatuf00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date November 2023
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <string.h>

#include "LineMandelCalculator.h"


LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	data = (int *)(_mm_malloc(height * width * sizeof(int), 64));
	zReal = (float *)(_mm_malloc((width) * sizeof(float), 64));
	zImag = (float *)(_mm_malloc((width) * sizeof(float), 64));
	processed = (bool *)(_mm_malloc((width) * sizeof(bool), 64));
	hReal = (float *)(_mm_malloc((width) * sizeof(float), 64));
	hImag = (float *)(_mm_malloc((height / 2) * sizeof(float), 64));

	if (data == NULL || zReal == NULL || zImag == NULL || processed == NULL || hReal == NULL || hImag == NULL) {
		fprintf(stderr, "malloc failed\n");
		exit(1);
	}

	for (int i = 0; i < width * height; ++i) {
		*(data + i) = limit;
	}
	for (int i = 0; i < width; ++i) {
		*(hReal + i) = x_start + i * dx;
	}
	for (int i = 0; i < height / 2; ++i) {
		*(hImag + i) = y_start + i * dy;
	}
}

LineMandelCalculator::~LineMandelCalculator() {
	_mm_free(data);
	_mm_free(zReal);
	_mm_free(zImag);
	_mm_free(processed);
	_mm_free(hReal);
	_mm_free(hImag);
	data = NULL;
	zReal = NULL;
	zImag = NULL;
	processed = NULL;
	hReal = NULL;
	hImag = NULL;
}


int * LineMandelCalculator::calculateMandelbrot () {
	float *zReal = this->zReal;
	float *zImag = this->zImag;
	bool *processed = this->processed;

	int *direct = data;
	int *mirror = data + (height - 1) * width;

	for (int i = 0; i < height / 2; ++i) {
		#pragma omp simd simdlen(32) \
			aligned(zImag, zReal, processed: 64)
		for (int j = 0; j < width; j++) {
			zReal[j] = hReal[j];
			zImag[j] = hImag[i];
			processed[j] = false;
		}

		int processed_count = 0;
		for (int k = 0; k < limit && processed_count < width; ++k) {
			#pragma omp simd \
				aligned(zImag, zReal, processed: 64) \
				simdlen(32)
			for (int j = 0; j < width; ++j) {
				if (processed[j]) {
					continue;
				}
				float r2 = zReal[j] * zReal[j];
				float i2 = zImag[j] * zImag[j];

				if (r2 + i2 > 4.0f)  {
					processed[j] = true;
					direct[j] = mirror[j] = k;
					processed_count++;
				}

				zImag[j] = 2.0f * zReal[j] * zImag[j] + hImag[i];
				zReal[j] = r2 - i2 + hReal[j];
			}
		}
		direct += width;
		mirror -= width;
	}
	return data;
}
