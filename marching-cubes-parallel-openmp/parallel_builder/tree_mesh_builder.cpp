/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Ján Maťufka <xmatuf00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    December 15th, 2023
 **/

#include <iostream>
#include <math.h>
#include <limits>
#include <omp.h>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::ocTreeDecompose(const ParametricScalarField &field, Vec3_t<float> &pos, unsigned gridSize)
{
    if (gridSize <= 1) {
        return buildCube(pos, field);
    }

    constexpr size_t seqThreshold = 2;
    unsigned newGridSize = gridSize / 2;
    unsigned totalTriangles = 0;

    Vec3_t<float> halfOffset(
        (pos.x + newGridSize) * mGridResolution,
        (pos.y + newGridSize) * mGridResolution,
        (pos.z + newGridSize) * mGridResolution
    );
    for (int i = 0; i < 8; i++) {
        Vec3_t<float> newOffset(
            pos.x + (bool) (i & 1) * newGridSize,
            pos.y + (bool) (i & 2) * newGridSize,
            pos.z + (bool) (i & 4) * newGridSize
        );

        #pragma omp task shared(totalTriangles) if(seqThreshold < newGridSize)
        {
        if (evaluateFieldAt(halfOffset, field) <= mIsoLevel + newGridSize * mGridResolution * sqrt(3)) {
            #pragma omp atomic update
            totalTriangles += ocTreeDecompose(field, newOffset, newGridSize);
        }
        }
    }
    #pragma omp taskwait
    return totalTriangles;
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.

    Vec3_t<float> offset(0.0, 0.0, 0.0);
    unsigned totalTriangles = 0;
    #pragma omp parallel
    {
        #pragma omp single
        totalTriangles = ocTreeDecompose(field, offset, mGridSize);
    }
    return totalTriangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());
    float value = std::numeric_limits<float>::max();
    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);
        value = std::min(value, distanceSquared);
    }

    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    #pragma omp critical
    mTriangles.push_back(triangle);
}
