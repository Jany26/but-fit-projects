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
#include "Vec.h"

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
  #pragma acc update host(pos[0:count]) async(2) // 2 = DATA_STREAM
  #pragma acc update host(vel[0:count]) async(2) // 2 = DATA_STREAM
}

/*********************************************************************************************************************/

/**
 * Calculate velocity
 * @param pIn  - particles input
 * @param pOut - particles output
 * @param N    - Number of particles
 * @param dt   - Size of the time step
 */
void calculateVelocity(Particles& pIn, Particles& pOut, const unsigned N, float dt)
{
  /*******************************************************************************************************************/
  /*                    TODO: Calculate gravitation velocity, see reference CPU version,                             */
  /*                            you can use overloaded operators defined in Vec.h                                    */
  /*******************************************************************************************************************/
    #pragma acc parallel loop present(pIn, pOut) async(1) // 1 = VEL_STREAM
    for (unsigned i = 0u; i < N; ++i)
    {
      float3 gVel{};
      float3 cVel{};
      const float3 posi = {pIn.pos[i].x, pIn.pos[i].y, pIn.pos[i].z};
      const float3 veli = pIn.vel[i];
      const float wi = pIn.pos[i].w;

      #pragma acc loop seq
      for (unsigned j = 0u; j < N; ++j)
      {
        const float3 posj = {pIn.pos[j].x, pIn.pos[j].y, pIn.pos[j].z};
        const float3 velj = pIn.vel[j];
        const float wj = pIn.pos[j].w;
        const float3 d = posj - posi;
        const float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
        const float r = std::sqrt(r2) + FLOAT_MIN;

        const float f = G * wi * wj / r2 + FLOAT_MIN;

        gVel += (r > COLLISION_DISTANCE) ? d / r * f : 0.f;
        cVel += (r > 0.f && r < COLLISION_DISTANCE)
                  ? (((wi * veli - wj * veli + 2.f * wj * velj) / (wi + wj)) - veli)
                  : 0.f;
      }
      gVel *= dt / wi;
      pOut.vel[i] = veli + gVel + cVel;
      pOut.pos[i].x = posi.x + pOut.vel[i].x * dt;
      pOut.pos[i].y = posi.y + pOut.vel[i].y * dt;
      pOut.pos[i].z = posi.z + pOut.vel[i].z * dt;
      pOut.pos[i].w = wi;  // this is needed so that during first iteration, the weights are properly set
    }
  // }
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate particles center of mass
 * @param p         - particles
 * @param comBuffer - pointer to a center of mass buffer
 * @param N         - Number of particles
 */
void centerOfMass(Particles& p, float4* comBuffer, const unsigned N)
{
  /********************************************************************************************************************/
  /*                 TODO: Calculate partiles center of mass inside center of mass buffer                             */
  /********************************************************************************************************************/

  // we can compute the coordinates of center of mass as a weighted average
  // this approach is much less prone to error
  // using the original (cpu) implementation will result in cumulative approximation errors
  // connected with associativity issues stemming from out-of-order computation

  // reduction clause did not want to accept float4/float3 (not even element-wise)
  // so the position info is split into 4 separate floats
  // comBuffer[0] = {0.0f, 0.0f, 0.0f, 0.0f};
  float comx = 0.0f, comy = 0.0f, comz = 0.0f, comw = 0.0f;
  #pragma acc parallel loop present(p, comBuffer) reduction(+: comx, comy, comz, comw) async(3)
  for (unsigned i = 0u; i < N; ++i)
  {
    const float3 pos = {p.pos[i].x, p.pos[i].y, p.pos[i].z};
    const float w = p.pos[i].w;
    comx += pos.x * w;
    comy += pos.y * w;
    comz += pos.z * w;
    comw += w;
  }
  // #pragma acc update host(comBuffer[0:1]) async
  comBuffer[0] = {comx / comw, comy / comw, comz / comw, comw};
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
