/**
 * @file      nbody.cu
 *
 * @author    Ján Maťukfa \n
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
 * CUDA kernel to calculate new particles velocity and position
 * @param pIn  - particles in
 * @param pOut - particles out
 * @param N    - Number of particles
 * @param dt   - Size of the time step
 */
__global__ void calculateVelocity(Particles pIn, Particles pOut, const unsigned N, float dt)
{
  /********************************************************************************************************************/
  /*  TODO: CUDA kernel to calculate new particles velocity and position, use shared memory to minimize memory access */
  /********************************************************************************************************************/

  unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
  const float posx  = pIn.posx[i];
  const float posy  = pIn.posy[i];
  const float posz  = pIn.posz[i];
  const float w     = pIn.w[i];
  const float velx  = pIn.velx[i];
  const float vely  = pIn.vely[i];
  const float velz  = pIn.velz[i];

  float gVelx     = 0.0f;
  float gVely     = 0.0f;
  float gVelz     = 0.0f;

  float cVelx     = 0.0f;
  float cVely     = 0.0f;
  float cVelz     = 0.0f;

  // filled shared memory will look like this:
  // [ posx1 .. posxB posy1 .. posyB posz1 .. poszB w1 .. wB velx1 .. velxB vely1 .. velyB velz1 .. velzB ]
  // where B = blockSize (which is assumed to be divisible by warpSize)
  extern __shared__ float shared[];

  for (unsigned b = 0u; b < N; b += blockDim.x) {
    shared[threadIdx.x                 ] = pIn.posx[b + threadIdx.x];
    shared[threadIdx.x +     blockDim.x] = pIn.posy[b + threadIdx.x];
    shared[threadIdx.x + 2 * blockDim.x] = pIn.posz[b + threadIdx.x];
    shared[threadIdx.x + 3 * blockDim.x] = pIn.w   [b + threadIdx.x];
    shared[threadIdx.x + 4 * blockDim.x] = pIn.velx[b + threadIdx.x];
    shared[threadIdx.x + 5 * blockDim.x] = pIn.vely[b + threadIdx.x];
    shared[threadIdx.x + 6 * blockDim.x] = pIn.velz[b + threadIdx.x];

    __syncthreads();

    for (unsigned j = 0u; j < blockDim.x; j++) {
      const float otherPosx = shared[j                 ];
      const float otherPosy = shared[j +     blockDim.x];
      const float otherPosz = shared[j + 2 * blockDim.x];
      const float otherW    = shared[j + 3 * blockDim.x];
      const float otherVelx = shared[j + 4 * blockDim.x];
      const float otherVely = shared[j + 5 * blockDim.x];
      const float otherVelz = shared[j + 6 * blockDim.x];

      const float dx = otherPosx - posx;
      const float dy = otherPosy - posy;
      const float dz = otherPosz - posz;

      const float r2 = dx * dx + dy * dy + dz * dz;
      const float r = sqrtf(r2) + FLOAT_MIN;
      const float f = G * w * otherW / r2 + FLOAT_MIN;

      // gravitational velocity computation

      gVelx += (r > COLLISION_DISTANCE) ? dx / r * f : 0.f;
      gVely += (r > COLLISION_DISTANCE) ? dy / r * f : 0.f;
      gVelz += (r > COLLISION_DISTANCE) ? dz / r * f : 0.f;

      // collision velocity computation

      cVelx += (r > 0.f && r < COLLISION_DISTANCE)
                  ? ((((w - otherW) * velx + 2.f * otherW * otherVelx) / (w + otherW)) - velx)
                  : 0.f;
      cVely += (r > 0.f && r < COLLISION_DISTANCE)
                  ? ((((w - otherW) * vely + 2.f * otherW * otherVely) / (w + otherW)) - vely)
                  : 0.f;
      cVelz += (r > 0.f && r < COLLISION_DISTANCE)
                  ? ((((w - otherW) * velz + 2.f * otherW * otherVelz) / (w + otherW)) - velz)
                  : 0.f;
    }

    __syncthreads();
  }
  gVelx *= dt / w;
  gVely *= dt / w;
  gVelz *= dt / w;

  // merging the velocity results and updating particle positions
  // note that we cannot simply perform pOut.pos += ... or pOut.vel += ...
  // since the pIn and pOut data differ after the first iteration

  const float outVelx = velx + gVelx + cVelx;
  const float outVely = vely + gVely + cVely;
  const float outVelz = velz + gVelz + cVelz;

  pOut.posx[i] = posx + outVelx * dt;
  pOut.posy[i] = posy + outVely * dt;
  pOut.posz[i] = posz + outVelz * dt;
  pOut.velx[i] = outVelx;
  pOut.vely[i] = outVely;
  pOut.velz[i] = outVelz;

}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------


__device__ static inline void reduceCOM(float4 *res, float4 *val)
{
  // helper function for reducing second parameter COM into the first
  // res += val, where + is the center of mass merging operation
  const float4 d = {
    val->x - res->x,
    val->y - res->y,
    val->z - res->z,
    ((val->w + res->w) > 0.0f) ? ( val->w / (val->w + res->w)) : 0.0f
  };
  res->x += d.x * d.w;
  res->y += d.y * d.w;
  res->z += d.z * d.w;
  res->w += val->w;
}


/**
 * CUDA kernel to calculate particles center of mass
 * @param p    - particles
 * @param com  - pointer to a center of mass
 * @param lock - pointer to a user-implemented lock
 * @param N    - Number of particles
 */
__global__ void centerOfMass(Particles p, float4* com, int* lock, const unsigned N)
{
  /********************************************************************************************************************/
  /*           TODO: CUDA kernel to calculate particles center of mass, see reference CPU implementation,             */
  /*                                 use CUDA predefined warpSize variable                                            */
  /********************************************************************************************************************/

  // the reduction design closely follows that presented in the 4th lab exercise - Example05
  // it assumes that blocksize is limited to max 1024 threads - 32 warps

  extern __shared__ float4 redShared[]; // stores warp-level reduced COMs -- ceil(blockDim / warpSize) elements

  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned warpInBlock = threadIdx.x / warpSize;
  const unsigned threadInWarp = threadIdx.x % warpSize;

  const unsigned gridStride = gridDim.x * blockDim.x; // constant for all threads
  const unsigned warpCount = (blockDim.x + warpSize - 1) / warpSize;   // constant for all threads

  // since redGridDim * redBlockDim can be smaller than simGridDim * simBlockDim,
  // we need to merge all particles into COMs that fit into one "reduction" grid
  // if all particles initially fit into the reduction grid, the following loop does 1 iteration
  float4 threadCOM = make_float4(0.f, 0.f, 0.f, 0.f);
  for (unsigned j = i; j < N; j += gridStride) {
    float4 otherCOM = make_float4(p.posx[j], p.posy[j], p.posz[j], p.w[j]);
    reduceCOM(&threadCOM, &otherCOM);
  }
  __syncthreads();
  // now all particles are merged in one grid, each thread holding aggregate COM in the threadCOM variable
  // we now perform warp-wise (independent) reductions using intrinsic shuffle instructions
  // each warp will store its COM in the shared memory
  // note that during warp-wise reduction, no additional global memory accesses are needed

  for (unsigned stride = warpSize / 2; stride > 0; stride >>= 1) {
    float4 otherCOM = {
      __shfl_down_sync(0xffffffff, threadCOM.x, stride),
      __shfl_down_sync(0xffffffff, threadCOM.y, stride),
      __shfl_down_sync(0xffffffff, threadCOM.z, stride),
      __shfl_down_sync(0xffffffff, threadCOM.w, stride),
    };
    reduceCOM(&threadCOM, &otherCOM);
  }

  if (threadInWarp == 0) {
    // if the number of particles is not divisible by blocksize, there will be some warps
    // that have no data to process, in that case, their corresponding threadCOM will be <0,0,0,0>,
    // as was initialized at the beginning -> warp-wise reduction, too, will produce <0,0,0,0>
    redShared[warpInBlock] = threadCOM;
  }
  __syncthreads();

  // theoretically, there will be no more than 32 warps within a block,
  // since max blockDim.x is 1024 = 32 warps of 32 threads.
  // so one warp-wise reduction is sufficient for obtaining the block-wise COM.

  float4 partialCOM = (threadIdx.x < warpCount) ? redShared[threadIdx.x] : make_float4(0.f, 0.f, 0.f, 0.f);
  for (unsigned stride = warpSize / 2; stride > 0; stride >>= 1) {
    float4 nextThreadCOM = make_float4(
      __shfl_down_sync(0xffffffff, partialCOM.x, stride),
      __shfl_down_sync(0xffffffff, partialCOM.y, stride),
      __shfl_down_sync(0xffffffff, partialCOM.z, stride),
      __shfl_down_sync(0xffffffff, partialCOM.w, stride)
    );
    reduceCOM(&partialCOM, &nextThreadCOM);
  }

  if (threadIdx.x == 0) {
    while (atomicCAS(lock, 0, 1)); // while compare-and-swap returns non-zero => while lock contains 1 == LOCKED
    reduceCOM(com, &partialCOM);
    atomicExch(lock, 0); // unlock => set lock = 0
  }
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
