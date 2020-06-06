#include "kernels.h"

__global__
void rdf_force(float4 *positions, float4 *forces, int num_part, float box_l, 
float *ener, float *vir)
{
    // Parámetros
    float rc = box_l * 0.5f;
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
                atomicAdd(&forces[i].x, (fij * xij) / rij);
                atomicAdd(&forces[i].y, (fij * yij) / rij);
                atomicAdd(&forces[i].z, (fij * zij) / rij);

                atomicAdd(&forces[j].x, -(fij * xij) / rij);
                atomicAdd(&forces[j].y, -(fij * yij) / rij);
                atomicAdd(&forces[j].z, -(fij * zij) / rij);

                // Actualizar los valores de la energía
                potential += uij;

                // Calcular el valor del virial
                virial_sum += (fij * xij * xij / rij);
                virial_sum += (fij * yij * yij / rij) + (fij * zij * zij / rij);
            }
        }
        ener[i] = potential;
        vir[i] = virial_sum;
    }
}


__global__
void position(float4 *positions, float4 *forces, float dtt,
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

