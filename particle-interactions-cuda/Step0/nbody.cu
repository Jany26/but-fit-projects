/**
 * @file      nbody.cu
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

#include <device_launch_parameters.h>

#include "nbody.cuh"

/* Constants */
constexpr float G                  = 6.67384e-11f;
constexpr float COLLISION_DISTANCE = 0.01f;
constexpr float FLOAT_MIN          = 1.1754944e-38f;

/**
 * CUDA kernel to calculate gravitation velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void calculateGravitationVelocity(Particles p, Velocities tmpVel, const unsigned N, float dt)
{
  /********************************************************************************************************************/
  /*              TODO: CUDA kernel to calculate gravitation velocity, see reference CPU version                      */
  /********************************************************************************************************************/

  // for velocity and position calculations/updates, we assume that all particles fit on one grid,
  // based on how simGridDim is initialized -- see main.cu (around line 70)

  unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
  const float posx  = p.posx[i];
  const float posy  = p.posy[i];
  const float posz  = p.posz[i];
  const float w     = p.w[i];
  float newVelx     = 0.0f;
  float newVely     = 0.0f;
  float newVelz     = 0.0f;

  for (unsigned j = 0u; j < N; ++j) {
    const float otherPosx = p.posx[j];
    const float otherPosy = p.posy[j];
    const float otherPosz = p.posz[j];
    const float otherW    = p.w[j];

    const float dx = otherPosx - posx;
    const float dy = otherPosy - posy;
    const float dz = otherPosz - posz;

    const float r2 = dx * dx + dy * dy + dz * dz;
    const float r = std::sqrt(r2) + FLOAT_MIN;

    const float f = G * w * otherW / r2 + FLOAT_MIN;

    newVelx += (r > COLLISION_DISTANCE) ? dx / r * f : 0.f;
    newVely += (r > COLLISION_DISTANCE) ? dy / r * f : 0.f;
    newVelz += (r > COLLISION_DISTANCE) ? dz / r * f : 0.f;
  }

  newVelx *= dt / w;
  newVely *= dt / w;
  newVelz *= dt / w;

  tmpVel.velx[i] = newVelx;
  tmpVel.vely[i] = newVely;
  tmpVel.velz[i] = newVelz;
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate collision velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void calculateCollisionVelocity(Particles p, Velocities tmpVel, const unsigned N, float dt)
{
  /********************************************************************************************************************/
  /*              TODO: CUDA kernel to calculate collision velocity, see reference CPU version                        */
  /********************************************************************************************************************/

  unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
  const float posx = p.posx[i];
  const float posy = p.posy[i];
  const float posz = p.posz[i];
  const float w = p.w[i];
  const float velx = p.velx[i];
  const float vely = p.vely[i];
  const float velz = p.velz[i];
  float newVelx = 0.0f;
  float newVely = 0.0f;
  float newVelz = 0.0f;

  for (unsigned j = 0u; j < N; ++j) {
    const float otherPosx = p.posx[j];
    const float otherPosy = p.posy[j];
    const float otherPosz = p.posz[j];
    const float otherW = p.w[j];
    const float otherVelx = p.velx[j];
    const float otherVely = p.vely[j];
    const float otherVelz = p.velz[j];
    const float dx = otherPosx - posx;
    const float dy = otherPosy - posy;
    const float dz = otherPosz - posz;

    const float r2 = dx * dx + dy * dy + dz * dz;
    const float r = std::sqrt(r2);

    newVelx += (r > 0.f && r < COLLISION_DISTANCE)
                ? (((w * velx - otherW * velx + 2.f * otherW * otherVelx) / (w + otherW)) - velx)
                : 0.f;
    newVely += (r > 0.f && r < COLLISION_DISTANCE)
                ? (((w * vely - otherW * vely + 2.f * otherW * otherVely) / (w + otherW)) - vely)
                : 0.f;
    newVelz += (r > 0.f && r < COLLISION_DISTANCE)
                ? (((w * velz - otherW * velz + 2.f * otherW * otherVelz) / (w + otherW)) - velz)
                : 0.f;
  }
  tmpVel.velx[i] += newVelx;
  tmpVel.vely[i] += newVely;
  tmpVel.velz[i] += newVelz;
}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void updateParticles(Particles p, Velocities tmpVel, const unsigned N, float dt)
{
  /********************************************************************************************************************/
  /*             TODO: CUDA kernel to update particles velocities and positions, see reference CPU version            */
  /********************************************************************************************************************/

  unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
  float posx = p.posx[i];
  float posy = p.posy[i];
  float posz = p.posz[i];
  float velx = p.velx[i];
  float vely = p.vely[i];
  float velz = p.velz[i];
  float newVelx = tmpVel.velx[i];
  float newVely = tmpVel.vely[i];
  float newVelz = tmpVel.velz[i];

  velx += newVelx;
  vely += newVely;
  velz += newVelz;

  posx += velx * dt;
  posy += vely * dt;
  posz += velz * dt;

  p.posx[i] = posx;
  p.posy[i] = posy;
  p.posz[i] = posz;
  p.velx[i] = velx;
  p.vely[i] = vely;
  p.velz[i] = velz;
}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate particles center of mass
 * @param p    - particles
 * @param com  - pointer to a center of mass
 * @param lock - pointer to a user-implemented lock
 * @param N    - Number of particles
 */
__global__ void centerOfMass(Particles p, float4* com, int* lock, const unsigned N)
{

}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
__host__ float4 centerOfMassRef(MemDesc& memDesc)
{
  float4 com{};

  for (std::size_t i{}; i < memDesc.getDataSize(); i++)
  {
    const float3 pos = {memDesc.getPosX(i), memDesc.getPosY(i), memDesc.getPosZ(i)};
    const float  w   = memDesc.getWeight(i);

    // Calculate the vector on the line connecting current body and most recent position of center-of-mass
    // Calculate weight ratio only if at least one particle isn't massless
    const float4 d = {pos.x - com.x,
                      pos.y - com.y,
                      pos.z - com.z,
                      ((memDesc.getWeight(i) + com.w) > 0.0f)
                        ? ( memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w))
                        : 0.0f};

    // Update position and weight of the center-of-mass according to the weight ration and vector
    com.x += d.x * d.w;
    com.y += d.y * d.w;
    com.z += d.z * d.w;
    com.w += w;
  }

  return com;
}// enf of centerOfMassRef
//----------------------------------------------------------------------------------------------------------------------
