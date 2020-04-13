#include "functionsl.h"

void iniconf(float *x, float *y, float *z, float rho, float rc, int num_part)
{
    // Definir la distancia según la densidad
    float dist = powf(1.0 / rho, 1.0 / 3.0);

    // Inicializar las primeras posiciones
    x[0] = -rc + (dist / 2.0);
    y[0] = -rc + (dist / 2.0);
    z[0] = -rc + (dist / 2.0);

    for (int i = 1; i < num_part - 1; i++)
    {
        x[i] = x[i - 1] + dist;
        y[i] = y[i - 1];
        z[i] = z[i - 1];

        if (x[i] > rc)
        {
            x[i] = x[0];
            y[i] = y[i - 1] + dist;

            if (y[i] > rc)
            {
                x[i] = x[0];
                y[i] = y[0];
                z[i] = z[i - 1] + dist;
            }
        }
    }
}

__device__ void hardsphere(float r_pos, float uij)
{
    uij = (a_param / temp) * (powf(1.0f / r_pos, lambda) - powf(1.0f / r_pos, lambda - 1.0f));

    uij += 1.0f / temp;
}

__global__ void rdf_force(float *x, float *y, float *z, float *fx, float *fy, float *fz,
                          int num_part, float box_l, float ener)
{
    // Parámetros
    float rc = box_l / 2.0f;
    // float d_r = rc / nm;

    // Inicializar algunas variables de la posicion
    float xij = 0.0f, yij = 0.0f, zij = 0.0f, rij = 0.0f;
    float fij = 0.0f;
    float uij = 0.0f;
    int i = 0, j = 0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Inicializar arreglos para la fuerza
    for (i = idx; i < num_part; i += stride)
    {
        fx[i] = 0.0f;
        fy[i] = 0.0f;
        fz[i] = 0.0f;
    }

    for (i = idx; i < num_part; i += stride)
    {
        for (j = 0; j < num_part; j++)
        {
            if (i == j)
                continue;
            // Siempre inicializar en cero
            uij = 0.0f;
            fij = 0.0f;

            // Contribucion de pares
            xij = x[i] - x[j];
            yij = y[i] - y[j];
            zij = z[i] - z[j];

            // Condiciones de frontera
            xij -= (box_l * roundf(xij / box_l));
            yij -= (box_l * roundf(yij / box_l));
            zij -= (box_l * roundf(zij / box_l));

            rij = sqrtf(xij * xij + yij * yij + zij * zij);

            if (rij < rc)
            {
                // Siempre se calcula la fuerza
                if (rij < b_param)
                {
                    hardsphere(rij, uij);
                    fij = lambda * powf(1.0f / rij, lambda + 1.0f) - (lambda - 1.0f) * powf(1.0f / rij, lambda);
                    fij *= (a_param / temp);
                }
                else
                {
                    uij = 0.0f;
                    fij = 0.0f;
                }

                // Actualizar los valores de las fuerzas
                atomicAdd(&fx[j], (fij * xij) / rij);
                atomicAdd(&fy[j], (fij * yij) / rij);
                atomicAdd(&fz[j], (fij * zij) / rij);

                atomicAdd(&fx[i], -(fij * xij) / rij);
                atomicAdd(&fy[i], -(fij * yij) / rij);
                atomicAdd(&fz[i], -(fij * zij) / rij);
                ener = ener + uij;
                // printf("%f\n", ener);
            }
        }
    }
}

// void gr(float* x, float* y, float* z, float* g, int num_part, float box_l)
// {
//     // Parámetros
//     float rc = box_l/2.0;
//     float d_r = rc / nm;

//     int nbin = 0;
//     int i = 0, j = 0;
//     float xij = 0.0f, yij = 0.0f, zij = 0.0f, rij = 0.0f;

//     // #pragma omp parallel for num_threads(30) default(shared) private(xij,yij,zij,i,j,rij)
//     for (i = 0; i < num_part; i++)
//     {
//         for (j = i+1; j < num_part-1; j++)
//         {

//             // Contribucion de pares
//             xij = x[j] - x[i];
//             yij = y[j] - y[i];
//             zij = z[j] - z[i];

//             // Condiciones de frontera
//             xij -= (box_l * round(xij/box_l));
//             yij -= (box_l * round(yij/box_l));
//             zij -= (box_l * round(zij/box_l));

//             rij = sqrtf(xij*xij + yij*yij + zij*zij);

//             if (rij < rc)
//             {
//                 nbin = (int)(rij/d_r) + 1;
//                 if (nbin <= nm)
//                 {
//                     g[nbin] += 2.0;
//                 }
//             }
//         }
//     }
// }

__global__ void position(float *x, float *y, float *z, float *fx, float *fy, float *fz, float dtt,
                         float box_l, int num_part, int pbc, float *randvec)
{
    // Inicializar algunas variables
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    float sigma = sqrtf(2.0 * dtt);
    int i = 0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (i = idx; i < num_part; i += stride)
    {
        dx = sigma * randvec[3*i];
        dy = sigma * randvec[(3*i)+1];
        dz = sigma * randvec[(3*i)+2];

        x[i] += fx[i] * dtt + dx;
        y[i] += fy[i] * dtt + dy;
        z[i] += fz[i] * dtt + dz;

        if (pbc == 1)
        {
            x[i] -= (box_l * round(x[i] / box_l));
            y[i] -= (box_l * round(y[i] / box_l));
            z[i] -= (box_l * round(z[i] / box_l));
        }
    }
}

// // void difusion(int nprom, int n_part, float cfx[mt_n][mp], float cfy[mt_n][mp], float cfz[mt_n][mp], float* wt)

// void difusion(const int nprom, const int n_part, float* cfx, float* cfy, float* cfz, float* wt)
// {
//     float dif = 0.0f;
//     int i = 0, j = 0, k = 0;
//     float dx = 0.0f, dy = 0.0f, dz = 0.0f, aux = 0.0f;

//     // #pragma omp parallel for
//     // Mean-squared displacement
//     for (i = 0; i < nprom; i++)
//     {
//         dif = 0.0f;
//         // printf("%d\n", nprom-i);
//         for (j = 0; j < nprom-i; j++)
//         {
//             for (k = 0; k < n_part; k++)
//             {
//                 dx = cfx[(j+i)*mp + k] - cfx[j*mp + k];
//                 dy = cfy[(j+i)*mp + k] - cfy[j*mp + k];
//                 dz = cfz[(j+i)*mp + k] - cfz[j*mp + k];
//                 dif += dx*dx + dy*dy + dz*dz;
//             }
//         }
//         aux = (n_part*(nprom-i));
//         wt[i] = (dif/aux);
//     }
// }
