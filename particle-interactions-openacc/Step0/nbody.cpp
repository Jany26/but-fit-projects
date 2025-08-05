/**
 * @file      nbody.cpp
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

#include <cfloat>
#include <cmath>

#include "nbody.h"

/* Constants */
constexpr float G                  = 6.67384e-11f;
constexpr float COLLISION_DISTANCE = 0.01f;
constexpr float FLOAT_MIN          = 1.1754944E-38f;

/*********************************************************************************************************************/
/*                TODO: Fullfill Partile's and Velocitie's constructors, destructors and methods                     */
/*                                    for data copies between host and device                                        */
/*********************************************************************************************************************/

/**
 * @brief Constructor
 * @param N - Number of particles
 */
Particles::Particles(const unsigned N)
{
  count = N;
  pos = new float4[count];
  vel = new float3[count];
  #pragma acc enter data copyin(this)
  #pragma acc enter data create(pos[0:count])
  #pragma acc enter data create(vel[0:count])
}

/// @brief Destructor
Particles::~Particles()
{
  #pragma acc exit data delete(pos[0:count])
  #pragma acc exit data delete(vel[0:count])
  #pragma acc exit data delete(this)
  delete [] pos;
  delete [] vel;
}
/**
 * @brief Copy particles from host to device
 */
void Particles::copyToDevice()
{
  #pragma acc update device(pos[0:count])
  #pragma acc update device(vel[0:count])
}

/**
 * @brief Copy particles from device to host
 */
void Particles::copyToHost()
{
  #pragma acc update host(pos[0:count])
  #pragma acc update host(vel[0:count])
}

/**
 * @brief Constructor
 * @param N - Number of particles
 */
Velocities::Velocities(const unsigned N)
{
  count = N;
  vel = new float3[count];

  #pragma acc enter data copyin(this)
  #pragma acc enter data create(vel[0:count])
}

/// @brief Destructor
Velocities::~Velocities()
{
  #pragma acc exit data delete(vel[0:count])
  #pragma acc exit data delete(this)
  delete [] vel;
}

/**
 * @brief Copy velocities from host to device
 */
void Velocities::copyToDevice()
{
  #pragma acc update device(vel[0:count])
}

/**
 * @brief Copy velocities from device to host
 */
void Velocities::copyToHost()
{
  #pragma acc update host(vel[0:count])
}

/*********************************************************************************************************************/

/**
 * Calculate gravitation velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
void calculateGravitationVelocity(Particles& p, Velocities& tmpVel, const unsigned N, float dt)
{
  /*******************************************************************************************************************/
  /*                    TODO: Calculate gravitation velocity, see reference CPU version,                             */
  /*                            you can use overloaded operators defined in Vec.h                                    */
  /*******************************************************************************************************************/

  #pragma acc parallel loop present(p, tmpVel)
  for (unsigned i = 0u; i < N; ++i)
  {
    float3 newVel{};
    const float3 posi = {p.pos[i].x, p.pos[i].y, p.pos[i].z};
    const float posiw = p.pos[i].w;

    #pragma acc loop seq
    for (unsigned j = 0u; j < N; ++j)
    {
      const float3 posj = {p.pos[j].x, p.pos[j].y, p.pos[j].z};
      const float posjw = p.pos[j].w;
      const float3 d = posj - posi;
      const float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
      const float r = std::sqrt(r2) + FLOAT_MIN;

      const float f = G * posiw * posjw / r2 + FLOAT_MIN;

      newVel.x += (r > COLLISION_DISTANCE) ? d.x / r * f : 0.f;
      newVel.y += (r > COLLISION_DISTANCE) ? d.y / r * f : 0.f;
      newVel.z += (r > COLLISION_DISTANCE) ? d.z / r * f : 0.f;
      // newVel += (r > COLLISION_DISTANCE) ? d / r * f : 0.f;
    }
    newVel.x *= dt / posiw;
    newVel.y *= dt / posiw;
    newVel.z *= dt / posiw;
    // newVel *= dt / posiw;

    tmpVel.vel[i] = newVel;
  }
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate collision velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
void calculateCollisionVelocity(Particles& p, Velocities& tmpVel, const unsigned N, float dt)
{
  /*******************************************************************************************************************/
  /*                    TODO: Calculate collision velocity, see reference CPU version,                               */
  /*                            you can use overloaded operators defined in Vec.h                                    */
  /*******************************************************************************************************************/

  #pragma acc parallel loop present(p, tmpVel)
  for (unsigned i = 0u; i < N; ++i)
  {
    float3 newVel{};
    const float3 posi = {p.pos[i].x, p.pos[i].y, p.pos[i].z};
    const float posiw = p.pos[i].w;
    const float3 veli = p.vel[i];

    #pragma acc loop seq
    for (unsigned j = 0u; j < N; ++j)
    {
      const float3 posj = {p.pos[j].x, p.pos[j].y, p.pos[j].z};
      const float posjw = p.pos[j].w;
      const float3 velj = p.vel[j];
      const float3 d = posj - posi;
      const float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
      const float r = std::sqrt(r2);

      newVel.x += (r > 0.f && r < COLLISION_DISTANCE)
                 ? (((posiw * veli.x - posjw * veli.x + 2.f * posjw * velj.x) / (posiw + posjw)) - veli.x)
                 : 0.f;
      newVel.y += (r > 0.f && r < COLLISION_DISTANCE)
                 ? (((posiw * veli.y - posjw * veli.y + 2.f * posjw * velj.y) / (posiw + posjw)) - veli.y)
                 : 0.f;
      newVel.z += (r > 0.f && r < COLLISION_DISTANCE)
                 ? (((posiw * veli.z - posjw * veli.z + 2.f * posjw * velj.z) / (posiw + posjw)) - veli.z)
                 : 0.f;
    }

    tmpVel.vel[i] += newVel;
  }
}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Update particles
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
void updateParticles(Particles& p, Velocities& tmpVel, const unsigned N, float dt)
{
  /*******************************************************************************************************************/
  /*                    TODO: Update particles position and velocity, see reference CPU version,                     */
  /*                            you can use overloaded operators defined in Vec.h                                    */
  /*******************************************************************************************************************/

  #pragma acc parallel loop present(p, tmpVel)
  for (unsigned i = 0u; i < N; ++i)
  {
    float4 posi = p.pos[i];
    float3 veli = p.vel[i];
    const float3 newVel = tmpVel.vel[i];

    veli += newVel;
    posi.x += veli.x * dt;
    posi.y += veli.y * dt;
    posi.z += veli.z * dt;

    p.pos[i] = posi;
    p.vel[i] = veli;
  }
}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate particles center of mass
 * @param p    - particles
 * @param com  - pointer to a center of mass
 * @param lock - pointer to a user-implemented lock
 * @param N    - Number of particles
 */
void centerOfMass(Particles& p, float4& com, int* lock, const unsigned N)
{

}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
float4 centerOfMassRef(MemDesc& memDesc)
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
