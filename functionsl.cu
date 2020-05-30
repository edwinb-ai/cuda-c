#include "functionsl.h"

void iniconf(float3 *positions, float rho, float rc, int num_part)
{
    // Definir la distancia según la densidad
    float dist = powf(1.0f / rho, 1.0f / 3.0f);

    // Inicializar las primeras posiciones
    positions[0].x = -rc + (dist / 2.0f);
    positions[0].y = -rc + (dist / 2.0f);
    positions[0].z = -rc + (dist / 2.0f);

    for (int i = 1; i < (num_part - 1); i++)
    {
        positions[i].x = positions[i - 1].x + dist;
        positions[i].y = positions[i - 1].y;
        positions[i].z = positions[i - 1].z;

        if (positions[i].x > rc)
        {
            positions[i].x = positions[0].x;
            positions[i].y = positions[i- 1].y + dist;

            if (positions[i].y > rc)
            {
                positions[i].x = positions[0].x;
                positions[i].y = positions[0].y;
                positions[i].z = positions[i - 1].z + dist;
            }
        }
    }
}

__global__ void rdf_force(float3 *positions, float3 *forces, int num_part, float box_l, 
    float *ener, float *vir)
{
    // Parámetros
    float rc = box_l * 0.5f;
    // float d_r = rc / nm;
    float virial_sum = 0.0f;

    // Inicializar algunas variables de la posicion
    float xij = 0.0f, yij = 0.0f, zij = 0.0f, rij = 0.0f;
    float fij = 0.0f;
    float uij = 0.0f;
    float potential = 0.0f;
    int i = 0, j = 0;

    // Índices para GPU
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (i = idx; i < num_part; i += stride)
    {
        // Inicializar valores
        forces[i].x = 0.0f;
        forces[i].y = 0.0f;
        forces[i].z = 0.0f;
    }

    for (i = idx; i < num_part; i += stride)
    {
        // Inicializar valores
        potential = 0.0f;
        virial_sum = 0.0f;

        for (j = 0; j < num_part; j++)
        {
            if (i == j)
                continue;
            // Siempre inicializar en cero
            uij = 0.0f;
            fij = 0.0f;

            // Contribucion de pares
            xij = positions[i].x - positions[j].x;
            yij = positions[i].y - positions[j].y;
            zij = positions[i].z - positions[j].z;

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
                    uij = (a_param / temp) * (powf(1.0f / rij, lambda) - powf(1.0f / rij, lambda - 1.0f));
                    uij += (1.0f / temp);
                    
                    fij = lambda * powf(1.0f / rij, lambda + 1.0f) - (lambda - 1.0f) * powf(1.0f / rij, lambda);
                    fij *= a_param / temp;
                }
                else
                {
                    uij = 0.0f;
                    fij = 0.0f;
                }

                // Actualizar los valores de las fuerzas
                // atomicAdd(&fx[i], (fij * xij) / rij);
                // atomicAdd(&fy[i], (fij * yij) / rij);
                // atomicAdd(&fz[i], (fij * zij) / rij);
                forces[i].x += fij * xij / rij;
                forces[i].y += fij * yij / rij;
                forces[i].z += fij * zij / rij;

                // atomicAdd(&fx[j], -(fij * xij) / rij);
                // atomicAdd(&fy[j], -(fij * yij) / rij);
                // atomicAdd(&fz[j], -(fij * zij) / rij);
                forces[i].x -= fij * xij / rij;
                forces[i].y -= fij * yij / rij;
                forces[i].z -= fij * zij / rij;

                // Actualizar los valores de la energía
                potential += uij;

                // Calcular el valor del virial
                virial_sum += (fij * xij * xij / rij) + (fij * yij * yij / rij) + (fij * zij * zij / rij);
            }
        }
        ener[i] = potential;
        vir[i] = virial_sum;
    }
}

void gr(float3 *positions, float *g, int num_part, float box_l)
{
    // Parámetros
    float rc = box_l * 0.5f;
    float d_r = rc / nm;

    int nbin = 0;
    int i = 0, j = 0;
    float xij = 0.0f, yij = 0.0f, zij = 0.0f, rij = 0.0f;

    for (i = 0; i < (num_part-1); i++)
    {
        for (j = i + 1; j < num_part; j++)
        {

            // Contribucion de pares
            xij = positions[i].x - positions[j].x;
            yij = positions[i].y - positions[j].y;
            zij = positions[i].z - positions[j].z;

            // Condiciones de frontera
            xij -= (box_l * roundf(xij / box_l));
            yij -= (box_l * roundf(yij / box_l));
            zij -= (box_l * roundf(zij / box_l));

            rij = sqrtf(xij * xij + yij * yij + zij * zij);

            if (rij < rc)
            {
                nbin = (int)(rij / d_r) + 1;
                if (nbin <= nm)
                {
                    g[nbin] += 2.0f;
                }
            }
        }
    }
}

__global__
void position(float3 *positions, float3 *forces, float dtt,
float box_l, int num_part, int pbc, float *randx, float *randy, float *randz)
{
    // Inicializar algunas variables
    int i = 0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (i = idx; i < num_part; i += stride)
    {
        positions[i].x += forces[i].x * dtt + randx[i];
        positions[i].y += forces[i].y * dtt + randy[i];
        positions[i].z += forces[i].z * dtt + randz[i];

        if (pbc == 1)
        {
            positions[i].x -= (box_l * roundf(positions[i].x / box_l));
            positions[i].y -= (box_l * roundf(positions[i].y / box_l));
            positions[i].z -= (box_l * roundf(positions[i].z / box_l));
        }
    }
}

__global__
void difusion(const int n_part, double *cfx, double *cfy, double *cfz, float *dif, size_t i, size_t j)
{
    size_t k = 0;
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Mean-squared displacement
    for (k = idx; k < n_part; k += stride)
    {
        dx = cfx[(j + i) * n_part + k] - cfx[j * n_part + k];
        dy = cfy[(j + i) * n_part + k] - cfy[j * n_part + k];
        dz = cfz[(j + i) * n_part + k] - cfz[j * n_part + k];
        atomicAdd(&dif[1], dx * dx + dy * dy + dz * dz);
    }
}
