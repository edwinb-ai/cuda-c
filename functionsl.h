#ifndef FUNCTIONSL_H
#define FUNCTIONSL_H

// Algunas variables globales
static const float lambda = 50.0f;
static const float a_param = 134.5526623421209f;
static const float b_param = 1.0204081632653061f;
static const float temp = 1.4737f;
static const int mt_n = 30000;
static const int mp = 3000;
static const int nm = 3000;
static const float pi = 3.141592653589793f;

// Funciones generales del código
void iniconf(float *x, float *y, float *z, float rho, float t_caja, int num_part);

__global__ void rdf_force(float *x, float *y, float *z, float *fx, float *fy, float *fz,
int num_part, float box_l, float ener);

float hardsphere(float rij);

void position(float *x, float *y, float *z, float *fx, float *fy, float *fz, float dtt,
              float box_l, int num_part, int pbc);

void gr(float *x, float *y, float *z, float *g, int num_part, float box_l);

void difusion(const int nprom, const int n_part, float *cfx, float *cfy, float *cfz, float *wt);

#endif
