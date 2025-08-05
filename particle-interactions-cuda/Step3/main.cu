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
  float4*   hCenterOfMass{};

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

  hCenterOfMass = static_cast<float4*>(operator new(sizeof(float4)));

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

  Particles dParticles[2]{};
  float4*   dCenterOfMass{};
  int*      dLock{};

  /********************************************************************************************************************/
  /*                                     TODO: GPU side memory allocation                                             */
  /********************************************************************************************************************/

  for (unsigned i = 0; i < 2; i++) {
    CUDA_CALL(cudaMalloc(&(dParticles[i].posx), sizeof(float) * N));
    CUDA_CALL(cudaMalloc(&(dParticles[i].posy), sizeof(float) * N));
    CUDA_CALL(cudaMalloc(&(dParticles[i].posz), sizeof(float) * N));
    CUDA_CALL(cudaMalloc(&(dParticles[i].w),    sizeof(float) * N));
    CUDA_CALL(cudaMalloc(&(dParticles[i].velx), sizeof(float) * N));
    CUDA_CALL(cudaMalloc(&(dParticles[i].vely), sizeof(float) * N));
    CUDA_CALL(cudaMalloc(&(dParticles[i].velz), sizeof(float) * N));
  }

  CUDA_CALL(cudaMalloc(&dCenterOfMass, sizeof(float4)));
  CUDA_CALL(cudaMalloc(&dLock, sizeof(int)));

  /********************************************************************************************************************/
  /*                                     TODO: Memory transfer CPU -> GPU                                             */
  /********************************************************************************************************************/

  for (unsigned i = 0; i < 2; i++) {
    CUDA_CALL(cudaMemcpy(dParticles[i].posx, hParticles.posx, sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dParticles[i].posy, hParticles.posy, sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dParticles[i].posz, hParticles.posz, sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dParticles[i].w,    hParticles.w,    sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dParticles[i].velx, hParticles.velx, sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dParticles[i].vely, hParticles.vely, sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dParticles[i].velz, hParticles.velz, sizeof(float) * N, cudaMemcpyHostToDevice));
  }

  /********************************************************************************************************************/
  /*                                     TODO: Clear GPU center of mass                                               */
  /********************************************************************************************************************/

  CUDA_CALL(cudaMemset(dCenterOfMass, 0, sizeof(float4)));
  CUDA_CALL(cudaMemset(dLock, 0, sizeof(int))); // 0 = unlocked, 1 = locked

  // Get CUDA device warp size
  int device;
  int warpSize;

  CUDA_CALL(cudaGetDevice(&device));
  CUDA_CALL(cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device));

  /********************************************************************************************************************/
  /*                                  TODO: Set dynamic shared memory computation                                     */
  /********************************************************************************************************************/
  const std::size_t sharedMemSize    = simBlockDim * sizeof(float) * 7;
  const std::size_t redSharedMemSize = ((redBlockDim + warpSize - 1) / warpSize) * sizeof(float4);   // you can use warpSize variable

  // Start measurement
  const auto start = std::chrono::steady_clock::now();

  for (unsigned s = 0u; s < steps; ++s)
  {
    const unsigned srcIdx = s % 2;        // source particles index
    const unsigned dstIdx = (s + 1) % 2;  // destination particles index

    /******************************************************************************************************************/
    /*                   TODO: GPU kernel invocation with correctly set dynamic memory size                           */
    /******************************************************************************************************************/
    calculateVelocity<<<simGridDim, simBlockDim, sharedMemSize>>>(dParticles[srcIdx], dParticles[dstIdx], N, dt);
  }

  const unsigned resIdx = steps % 2;    // result particles index

  /********************************************************************************************************************/
  /*                                 TODO: Invocation of center of mass kernel                                        */
  /********************************************************************************************************************/
  centerOfMass<<<redGridDim, redBlockDim, redSharedMemSize>>>(dParticles[resIdx], dCenterOfMass, dLock, N);

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

  CUDA_CALL(cudaMemcpy(hParticles.posx, dParticles[resIdx].posx, sizeof(float) * N, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.posy, dParticles[resIdx].posy, sizeof(float) * N, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.posz, dParticles[resIdx].posz, sizeof(float) * N, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.w,    dParticles[resIdx].w,    sizeof(float) * N, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velx, dParticles[resIdx].velx, sizeof(float) * N, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.vely, dParticles[resIdx].vely, sizeof(float) * N, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velz, dParticles[resIdx].velz, sizeof(float) * N, cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaMemcpy(hCenterOfMass, dCenterOfMass, sizeof(float4), cudaMemcpyDeviceToHost));

  // Compute reference center of mass on CPU
  const float4 refCenterOfMass = centerOfMassRef(md);

  std::printf("Reference center of mass: %f, %f, %f, %f\n",
              refCenterOfMass.x,
              refCenterOfMass.y,
              refCenterOfMass.z,
              refCenterOfMass.w);

  std::printf("Center of mass on GPU: %f, %f, %f, %f\n",
              hCenterOfMass->x,
              hCenterOfMass->y,
              hCenterOfMass->z,
              hCenterOfMass->w);

  // Writing final values to the file
  h5Helper.writeComFinal(*hCenterOfMass);
  h5Helper.writeParticleDataFinal();

  /********************************************************************************************************************/
  /*                                     TODO: GPU side memory deallocation                                           */
  /********************************************************************************************************************/

  for (unsigned i = 0; i < 2; i++) {
    CUDA_CALL(cudaFree(dParticles[i].posx));
    CUDA_CALL(cudaFree(dParticles[i].posy));
    CUDA_CALL(cudaFree(dParticles[i].posz));
    CUDA_CALL(cudaFree(dParticles[i].w));
    CUDA_CALL(cudaFree(dParticles[i].velx));
    CUDA_CALL(cudaFree(dParticles[i].vely));
    CUDA_CALL(cudaFree(dParticles[i].velz));
  }

  CUDA_CALL(cudaFree(dCenterOfMass));
  CUDA_CALL(cudaFree(dLock));

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

  operator delete(hCenterOfMass);

}// end of main
//----------------------------------------------------------------------------------------------------------------------
