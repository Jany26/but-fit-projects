/**
 * @file LineMandelCalculator.h
 * @author Ján Maťufka <xmatuf00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date November 2023
 */

#include <BaseMandelCalculator.h>

class LineMandelCalculator : public BaseMandelCalculator
{
public:
    LineMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~LineMandelCalculator();
    int *calculateMandelbrot();

private:
    int *data;
    float *zReal;
    float *zImag;
    bool *processed;

    float *hReal;
    float *hImag;
};
