#ifndef FUNCTIONSL_H
#define FUNCTIONSL_H
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "functionsl.h"
#include "cuda.h"
#include "curand.h"

// Algunas variables globales
static const float lambda = 50.0f;
static const float a_param = 134.552662342f;
static const float b_param = 1.0204081632f;
static const float temp = 1.4737f;
static const int mt_n = 200000;
static const int nm = 2048;
static const float PI = 3.141592653f;

// Funciones generales del código
void iniconf(float3 *positions, float rho, float t_caja, int num_part);

__global__ void rdf_force(float3 *positions, float3 *forces, int num_part, float box_l, 
float *ener, float *vir);

__global__ void position(float3 *positions, float3 *forces, float dtt,
float box_l, int num_part, int pbc, float *randx, float *randy, float *randz);

void gr(float3 *positions, float *g, int num_part, float box_l);

__global__ void difusion(const int n_part, double *cfx, double *cfy, double *cfz, float *dif, size_t i, size_t j);

#endif
