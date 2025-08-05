/**
 * @file      main.cpp
 *
 * @author    Ján Maťufka \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            xmatuf0000@fit.vutbr.cz
 *
 * @brief     PCG Assignment 2
 *
 * @version   2023
 *
 * @date      04 October   2023, 09:00 (created) \n
 */

#include <cmath>
#include <cstdio>
#include <chrono>
#include <string>

#include "nbody.h"
#include "h5Helper.h"

/**
 * Main rotine
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
  if (argc != 7)
  {
    std::printf("Usage: %s <N> <dt> <steps> <write intesity> <input> <output>\n", argv[0]);
    std::exit(1);
  }

  // Number of particles
  const unsigned N         = static_cast<unsigned>(std::stoul(argv[1]));
  // Length of time step
  const float    dt        = std::stof(argv[2]);
  // Number of steps
  const unsigned steps     = static_cast<unsigned>(std::stoul(argv[3]));
  // Write frequency
  const unsigned writeFreq = static_cast<unsigned>(std::stoul(argv[4]));

  // Log benchmark setup
  std::printf("       NBODY GPU simulation\n"
              "N:                       %u\n"
              "dt:                      %f\n"
              "steps:                   %u\n",
              N, dt, steps);

  const std::size_t recordsCount = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;

  Particles particles[2]{Particles{N}, Particles{N}};

  /********************************************************************************************************************/
  /*                                     TODO: Fill memory descriptor parameters                                      */
  /********************************************************************************************************************/

  /*
   * Caution! Create only after CPU side allocation
   * parameters:
   *                            Stride of two            Offset of the first
   *       Data pointer       consecutive elements        element in FLOATS,
   *                          in FLOATS, not bytes            not bytes
  */
  MemDesc md(&particles[0].pos[0].x, 4,                         0, // pos x
             &particles[0].pos[0].y, 4,                         0, // pos y
             &particles[0].pos[0].z, 4,                         0, // pos z
             &particles[0].vel[0].x, 3,                         0, // vel x
             &particles[0].vel[0].y, 3,                         0, // vel y
             &particles[0].vel[0].z, 3,                         0, // vel z
             &particles[0].pos[0].w, 4,                         0, // weight
             N,
             recordsCount);

  // Initialisation of helper class and loading of input data
  H5Helper h5Helper(argv[5], argv[6], md);

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

  /********************************************************************************************************************/
  /*                   TODO: Allocate memory for center of mass buffer. Remember to clear it.                         */
  /********************************************************************************************************************/
  float4* comBuffer = new float4[1];
  comBuffer[0] = {0.0f, 0.0f, 0.0f, 0.0f};
  #pragma acc enter data create(comBuffer)
  #pragma acc enter data copyin(comBuffer[0:1])

  /********************************************************************************************************************/
  /*                                     TODO: Memory transfer CPU -> GPU                                             */
  /********************************************************************************************************************/

  particles[0].copyToDevice();
  particles[1].copyToDevice();
  
  // Start measurement
  const auto start = std::chrono::steady_clock::now();

  for (unsigned s = 0u; s < steps; ++s)
  {
    const unsigned srcIdx = s % 2;        // source particles index
    const unsigned dstIdx = (s + 1) % 2;  // destination particles index

    /******************************************************************************************************************/
    /*                                        TODO: GPU computation                                                   */
    /******************************************************************************************************************/
    calculateVelocity(particles[srcIdx], particles[dstIdx], N, dt);
  }

  const unsigned resIdx = steps % 2;    // result particles index

  /********************************************************************************************************************/
  /*                                 TODO: Invocation of center of mass kernel                                        */
  /********************************************************************************************************************/

  centerOfMass(particles[resIdx], comBuffer, N);

  // End measurement
  const auto end = std::chrono::steady_clock::now();

  // Approximate simulation wall time
  const float elapsedTime = std::chrono::duration<float>(end - start).count();
  std::printf("Time: %f s\n", elapsedTime);

  /********************************************************************************************************************/
  /*                                     TODO: Memory transfer GPU -> CPU                                             */
  /********************************************************************************************************************/

  particles[resIdx].copyToHost();
  #pragma acc exit data copyout(comBuffer[0:1])
  float4 comFinal = comBuffer[0];

  // Compute reference center of mass on CPU
  const float4 refCenterOfMass = centerOfMassRef(md);

  std::printf("Reference center of mass: %f, %f, %f, %f\n",
              refCenterOfMass.x,
              refCenterOfMass.y,
              refCenterOfMass.z,
              refCenterOfMass.w);

  std::printf("Center of mass on GPU: %f, %f, %f, %f\n",
              comFinal.x,
              comFinal.y,
              comFinal.z,
              comFinal.w);

  // Writing final values to the file
  h5Helper.writeComFinal(comFinal);
  h5Helper.writeParticleDataFinal();

  /********************************************************************************************************************/
  /*                                TODO: Free center of mass buffer memory                                           */
  /********************************************************************************************************************/

  # pragma acc exit data delete(comBuffer[0:1])
  delete [] comBuffer;

}// end of main
//----------------------------------------------------------------------------------------------------------------------
