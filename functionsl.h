#ifndef FUNCTIONSL_H
#define FUNCTIONSL_H
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "functionsl.h"
#include "cuda.h"
#include "curand.h"

// Algunas variables globales
static const double lambda = 50.0;
static const double a_param = 134.5526623421209;
static const double b_param = 1.0204081632653061;
static const double temp = 1.4737;
static const int mt_n = 200000;
static const int nm = 256;
static const double PI = 3.141592653589793;

// Funciones generales del código
void iniconf(double *x, double *y, double *z, double rho, double t_caja, int num_part);

__global__ void rdf_force(double *x, double *y, double *z, double *fx, double *fy, double *fz,
int num_part, double box_l, double *ener);

__global__ void position(double* x, double* y, double* z, double* fx, double* fy, double* fz, double dtt,
double box_l, int num_part, int pbc, double *randvec);

void gr(double *x, double *y, double *z, double *g, int num_part, double box_l);

void difusion(const int nprom, const int n_part, double *cfx, double *cfy, double *cfz, double *wt);

#endif
