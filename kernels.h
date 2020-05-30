#ifndef KERNELS_H
#define KERNELS_H

#include "cuda.h"

// Algunas variables globales
static const float lambda = 50.0f;
static const float a_param = 134.552662342f;
static const float b_param = 1.0204081632f;
static const float temp = 1.4737f;

__global__ void rdf_force(float4 *positions, float4 *forces, int num_part, float box_l, 
float *ener, float *vir);

__global__ void position(float4 *positions, float4 *forces, float dtt,
float box_l, int num_part, int pbc, float *randx, float *randy, float *randz);

#endif