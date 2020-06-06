#ifndef FUNCTIONSL_H
#define FUNCTIONSL_H

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "cuda.h"
#include "curand.h"

// Algunas variables globales
static const int mt_n = 200000;
static const int nm = 4096;
static const float PI = 3.141592653f;

// Funciones generales del código
void iniconf(float4 *positions, float rho, float t_caja, int num_part);

void gr(float4 *positions, float *g, int num_part, float box_l);

void difusion( const int nprom, const int n_part, 
float *cfx, float *cfy, float *cfz, float *wt);

#endif
