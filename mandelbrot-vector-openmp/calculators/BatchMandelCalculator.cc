/**
 * @file BatchMandelCalculator.cc
 * @author Ján Maťufka <xmatuf00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date November 2023
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <string.h>
#include <stdexcept>

#include "BatchMandelCalculator.h"

#define BLOCK_SIZE 128


BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{    
	data = (int *)(_mm_malloc(height * width * sizeof(int), 64));
	data_block = (int *)(_mm_malloc(BLOCK_SIZE * sizeof(int), 64));
	zReal = (float *)(_mm_malloc(BLOCK_SIZE * sizeof(float), 64));
	zImag = (float *)(_mm_malloc(BLOCK_SIZE * sizeof(float), 64));
	hReal = (float *)(_mm_malloc(BLOCK_SIZE * sizeof(float), 64));

	if (data == NULL || data_block == NULL || zReal == NULL || zImag == NULL  || hReal == NULL) {
		fprintf(stderr, "malloc failed\n");
		exit(1);
	}
	for (int i = 0; i < width * height; ++i) {
		data[i] = 0;
	}
}

BatchMandelCalculator::~BatchMandelCalculator() {
	_mm_free(data);
	_mm_free(data_block);
	_mm_free(zReal);
	_mm_free(zImag);
	_mm_free(hReal);
	data = NULL;
	data_block = NULL;
	zReal = NULL;
	zImag = NULL;
	hReal = NULL;
}


int * BatchMandelCalculator::calculateMandelbrot () {
    const int halfsize = height * width / 2;
	float r2, i2;

    // these pointers are needed for aligned clause (otherwise pragma wont compile)
	float *real_buffer_ptr = zReal;
	float *imag_buffer_ptr = zImag;
	int *block_ptr = data_block;

    // casting doubles to float for speedup
    const float x_start = this->x_start;
    const float y_start = this->y_start;
    const float dx = this->dx;
    const float dy = this->dy;

    // helping pointers - starting points of symmetrical buffer arrays in data
    int *direct = data;
    int *mirror = data + (height - 1) * width;
        for (int i = 0; i < halfsize; i += BLOCK_SIZE, direct += BLOCK_SIZE, mirror += BLOCK_SIZE) {
            const float y = y_start + (i / width) * dy;
            float x = x_start + (i % width) * dx;
            #pragma omp simd simdlen(32) reduction(+: x)
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                block_ptr[j] = 0;
                real_buffer_ptr[j] = x;
                hReal[j] = x;
                imag_buffer_ptr[j] = y;
                x += dx;
            }
            int processed_count = 0; // for skipping unnecessary iterations
            int processed = 0; // for branch removal and
            for (int k = 0; k < limit && processed_count < BLOCK_SIZE; ++k) {
                processed_count = 0;
                #pragma omp simd simdlen(BLOCK_SIZE) \
                    aligned(imag_buffer_ptr, real_buffer_ptr, block_ptr: 64) \
                    reduction(+: processed_count)
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    r2 = real_buffer_ptr[j] * real_buffer_ptr[j];
                    i2 = imag_buffer_ptr[j] * imag_buffer_ptr[j];
                    imag_buffer_ptr[j] = 2.0f * real_buffer_ptr[j] * imag_buffer_ptr[j] + y;
                    real_buffer_ptr[j] = r2 - i2 + hReal[j];
                    processed = r2 + i2 <= 4.0f;
                    block_ptr[j] += processed;
                    processed_count += !processed;
                }
            }
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                direct[j] = mirror[j] = block_ptr[j];
            }
            // if next step would get to end of row
            // move back two rows (so we essentialy get to the previous line)
            mirror -= (i + BLOCK_SIZE) % width == 0 ? 2 * width : 0;
        }
	return data;
}
