/**
 * @file      nbody.cuh
 *
 * @author    Ján Maťufka \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            xmatuf00@stud.fit.vutbr.cz
 *
 * @brief     PCG Assignment 1
 *
 * @version   2024
 *
 * @date      04 October   2023, 09:00 (created) \n
 */

#ifndef NBODY_CUH
#define NBODY_CUH

#include <cuda_runtime.h>

#include "h5Helper.h"

/**
 * Particles data structure
 */
struct Particles
{
  /********************************************************************************************************************/
  /*                             TODO: Particle data structure optimized for use on GPU                               */
  /********************************************************************************************************************/

  float* posx;
  float* posy;
  float* posz;
  float* w;
  float* velx;
  float* vely;
  float* velz;
};

/**
 * CUDA kernel to calculate gravitation velocity
 * @param pIn  - particles input
 * @param pOut - particles output
 * @param N    - Number of particles
 * @param dt   - Size of the time step
 */
__global__ void calculateVelocity(Particles      pIn,
                                  Particles      pOut,
                                  const unsigned N,
                                  float          dt);

/**
 * CUDA kernel to calculate particles center of mass
 * @param p    - particles
 * @param com  - pointer to a center of mass
 * @param lock - pointer to a user-implemented lock
 * @param N    - Number of particles
 */
__global__ void centerOfMass(Particles      p,
                             float4*        com,
                             int*           lock,
                             const unsigned N);

/**
 * CPU implementation of the Center of Mass calculation
 * @param memDesc - Memory descriptor of particle data on CPU side
 */
float4 centerOfMassRef(MemDesc& memDesc);

#endif /* NBODY_H */
