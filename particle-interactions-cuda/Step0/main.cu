/**
 * @file      main.cu
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

#include <cmath>
#include <cstdio>
#include <chrono>
#include <string>

#include "nbody.cuh"
#include "h5Helper.h"

/**
 * @brief CUDA error checking macro
 * @param call CUDA API call
 */
#define CUDA_CALL(call) \
  do { \
    const cudaError_t _error = (call); \
    if (_error != cudaSuccess) \
    { \
      std::fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(_error)); \
      std::exit(EXIT_FAILURE); \
    } \
  } while(0)

/**
 * Main rotine
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
  if (argc != 10)
  {
    std::printf("Usage: nbody <N> <dt> <steps> <threads/block> <write intesity> <reduction threads> <reduction threads/block> <input> <output>\n");
    std::exit(1);
  }

  // Number of particles
  const unsigned N                   = static_cast<unsigned>(std::stoul(argv[1]));
  // Length of time step
  const float    dt                  = std::stof(argv[2]);
  // Number of steps
  const unsigned steps               = static_cast<unsigned>(std::stoul(argv[3]));
  // Number of thread blocks
  const unsigned simBlockDim         = static_cast<unsigned>(std::stoul(argv[4]));
  // Write frequency
  const unsigned writeFreq           = static_cast<unsigned>(std::stoul(argv[5]));
  // number of reduction threads
  const unsigned redTotalThreadCount = static_cast<unsigned>(std::stoul(argv[6]));
  // Number of reduction threads/blocks
  const unsigned redBlockDim         = static_cast<unsigned>(std::stoul(argv[7]));

  // Size of the simulation CUDA grid - number of blocks
  const unsigned simGridDim = (N + simBlockDim - 1) / simBlockDim;
  // Size of the reduction CUDA grid - number of blocks
  const unsigned redGridDim = (redTotalThreadCount + redBlockDim - 1) / redBlockDim;

  // NOTE: since simGridDim = ceil(N / simBlockDim), this means all kernels concerned about velocity
  // are run on 1 grid of blocks -- thus we don't need to do a for-loop over all grids


  // Log benchmark setup
  std::printf("       NBODY GPU simulation\n"
              "N:                       %u\n"
              "dt:                      %f\n"
              "steps:                   %u\n"
              "threads/block:           %u\n"
              "blocks/grid:             %u\n"
              "reduction threads/block: %u\n"
              "reduction blocks/grid:   %u\n",
              N, dt, steps, simBlockDim, simGridDim, redBlockDim, redGridDim);

  const std::size_t recordsCount = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;

  Particles hParticles{};

  /********************************************************************************************************************/
  /*                              TODO: CPU side memory allocation (pinned)                                           */
  /********************************************************************************************************************/

  hParticles.posx = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.posy = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.posz = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.w    = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.velx = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.vely = static_cast<float*>(operator new[](N * sizeof(float)));
  hParticles.velz = static_cast<float*>(operator new[](N * sizeof(float)));

  /********************************************************************************************************************/
  /*                              TODO: Fill memory descriptor layout                                                 */
  /********************************************************************************************************************/
  /*
   * Caution! Create only after CPU side allocation
   * parameters:
   *                            Stride of two            Offset of the first
   *       Data pointer       consecutive elements        element in FLOATS,
   *                          in FLOATS, not bytes            not bytes
  */
  MemDesc md((float *) hParticles.posx,  1,                       0,  // pos_x
             (float *) hParticles.posy,  1,                       0,  // pos_y
             (float *) hParticles.posz,  1,                       0,  // pos_z
             (float *) hParticles.velx,  1,                       0,  // vel_x
             (float *) hParticles.vely,  1,                       0,  // vel_y
             (float *) hParticles.velz,  1,                       0,  // vel_z
             (float *) hParticles.w,     1,                       0,  // weight
             N,
             recordsCount);

  // Initialisation of helper class and loading of input data
  H5Helper h5Helper(argv[8], argv[9], md);

  try
  {
    h5Helper.init();
    h5Helper.readParticleData();
  }
  catch (const std::exception& e)
  {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return EXIT_FAILURE;
  }

  Particles  dParticles{};
  Velocities dTmpVelocities{};
  
  const dim3 blockDim{simBlockDim};
  const dim3 gridDim{simGridDim};

  /********************************************************************************************************************/
  /*                                     TODO: GPU side memory allocation                                             */
  /********************************************************************************************************************/

  CUDA_CALL(cudaMalloc(&(dParticles.posx), sizeof(float) * N));
  CUDA_CALL(cudaMalloc(&(dParticles.posy), sizeof(float) * N));
  CUDA_CALL(cudaMalloc(&(dParticles.posz), sizeof(float) * N));
  CUDA_CALL(cudaMalloc(&(dParticles.w),    sizeof(float) * N));
  CUDA_CALL(cudaMalloc(&(dParticles.velx), sizeof(float) * N));
  CUDA_CALL(cudaMalloc(&(dParticles.vely), sizeof(float) * N));
  CUDA_CALL(cudaMalloc(&(dParticles.velz), sizeof(float) * N));

  CUDA_CALL(cudaMalloc(&(dTmpVelocities.velx), sizeof(float) * N));
  CUDA_CALL(cudaMalloc(&(dTmpVelocities.vely), sizeof(float) * N));
  CUDA_CALL(cudaMalloc(&(dTmpVelocities.velz), sizeof(float) * N));

  /********************************************************************************************************************/
  /*                                     TODO: Memory transfer CPU -> GPU                                             */
  /********************************************************************************************************************/

  CUDA_CALL(cudaMemcpy(dParticles.posx, hParticles.posx, sizeof(float) * N, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles.posy, hParticles.posy, sizeof(float) * N, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles.posz, hParticles.posz, sizeof(float) * N, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles.w,    hParticles.w,    sizeof(float) * N, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles.velx, hParticles.velx, sizeof(float) * N, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles.vely, hParticles.vely, sizeof(float) * N, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles.velz, hParticles.velz, sizeof(float) * N, cudaMemcpyHostToDevice));

  // Start measurement
  const auto start = std::chrono::steady_clock::now();

  for (unsigned s = 0u; s < steps; ++s)
  {
    /******************************************************************************************************************/
    /*                                     TODO: GPU kernels invocation                                               */
    /******************************************************************************************************************/

    calculateGravitationVelocity<<<gridDim, blockDim>>>(dParticles, dTmpVelocities, N, dt);
    calculateCollisionVelocity<<<gridDim, blockDim>>>(dParticles, dTmpVelocities, N, dt);
    updateParticles<<<gridDim, blockDim>>>(dParticles, dTmpVelocities, N, dt);
  }

  // Wait for all CUDA kernels to finish
  CUDA_CALL(cudaDeviceSynchronize());

  // End measurement
  const auto end = std::chrono::steady_clock::now();

  // Approximate simulation wall time
  const float elapsedTime = std::chrono::duration<float>(end - start).count();
  std::printf("Time: %f s\n", elapsedTime);


  /********************************************************************************************************************/
  /*                                     TODO: Memory transfer GPU -> CPU                                             */
  /********************************************************************************************************************/

  CUDA_CALL(cudaMemcpy(hParticles.posx, dParticles.posx, sizeof(float) * N, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.posy, dParticles.posy, sizeof(float) * N, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.posz, dParticles.posz, sizeof(float) * N, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.w,    dParticles.w,    sizeof(float) * N, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velx, dParticles.velx, sizeof(float) * N, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.vely, dParticles.vely, sizeof(float) * N, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velz, dParticles.velz, sizeof(float) * N, cudaMemcpyDeviceToHost));

  // Compute reference center of mass on CPU
  const float4 refCenterOfMass = centerOfMassRef(md);

  std::printf("Reference center of mass: %f, %f, %f, %f\n",
              refCenterOfMass.x,
              refCenterOfMass.y,
              refCenterOfMass.z,
              refCenterOfMass.w);

  std::printf("Center of mass on GPU: %f, %f, %f, %f\n", 0.f, 0.f, 0.f, 0.f);

  // Writing final values to the file
  h5Helper.writeComFinal(refCenterOfMass);
  h5Helper.writeParticleDataFinal();

  /********************************************************************************************************************/
  /*                                     TODO: GPU side memory deallocation                                           */
  /********************************************************************************************************************/

  CUDA_CALL(cudaFree(dParticles.posx));
  CUDA_CALL(cudaFree(dParticles.posy));
  CUDA_CALL(cudaFree(dParticles.posz));
  CUDA_CALL(cudaFree(dParticles.w));
  CUDA_CALL(cudaFree(dParticles.velx));
  CUDA_CALL(cudaFree(dParticles.vely));
  CUDA_CALL(cudaFree(dParticles.velz));

  CUDA_CALL(cudaFree(dTmpVelocities.velx));
  CUDA_CALL(cudaFree(dTmpVelocities.vely));
  CUDA_CALL(cudaFree(dTmpVelocities.velz)); 

  /********************************************************************************************************************/
  /*                                     TODO: CPU side memory deallocation                                           */
  /********************************************************************************************************************/

  operator delete[](hParticles.posx);
  operator delete[](hParticles.posy);
  operator delete[](hParticles.posz);
  operator delete[](hParticles.w);
  operator delete[](hParticles.velx);
  operator delete[](hParticles.vely);
  operator delete[](hParticles.velz);

}// end of main
//----------------------------------------------------------------------------------------------------------------------
