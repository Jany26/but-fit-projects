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

  // At first I thought that using 2 arrays of float4 (one for position and weight, one for velocity info)
  // would be more optimal -- less global memory accesses per thread, but since we have 7 floats per particle,
  // one float (4B of memory) will be unused.
  // And there would be 8 128-Byte memory transactions per warp to get all data.
  // 4 bytes x 8 floats x 32 threads = 1024 B = 8 x 128B
  // This is in theory slower than using just 7 simple float arrays -- one memory transaction for each array per warp.

  // Not only in theory it is more efficient, but progress benchmarks show the same,
  // 7 x float is faster than 2 x float4.

  // I also thought about using one float4, one float2, and one float array, but:
  //  - a) it offers no improvement (moreover, there is a small constant slowdown)
  //  - b) it is less readable
  // -- so I stuck with 7 float arrays.

  float* posx;
  float* posy;
  float* posz;
  float* w;
  float* velx;
  float* vely;
  float* velz;
};

/**
/* Velocities data structure (to be used as buffer for partial results)
 */
struct Velocities
{
  /********************************************************************************************************************/
  /*                             TODO: Velocities data structure optimized for use on GPU                             */
  /********************************************************************************************************************/

  // similar thought process here as with Particles

  float* velx;
  float* vely;
  float* velz;
};


/**
 * CUDA kernel to calculate gravitation velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void calculateGravitationVelocity(Particles      p,
                                             Velocities     tmpVel,
                                             const unsigned N,
                                             float          dt);

/**
 * CUDA kernel to calculate collision velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void calculateCollisionVelocity(Particles      p,
                                           Velocities     tmpVel,
                                           const unsigned N,
                                           float          dt);

/**
 * CUDA kernel to update particles
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void updateParticles(Particles      p,
                                Velocities     tmpVel,
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
