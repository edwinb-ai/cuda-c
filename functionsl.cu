#include "functionsl.h"

void iniconf(float4 *positions, float rho, float rc, int num_part)
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


void gr(float4 *positions, float *g, int num_part, float box_l)
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

void difusion( const int nprom, const int n_part,
               float *cfx, float *cfy, float *cfz, float *wt)
{
    double dif = 0.0;
    int i = 0, j = 0, k = 0;
    double dx = 0.0, dy = 0.0, dz = 0.0, aux = 0.0;

    // Mean-squared displacement
    for (i = 0; i < nprom; i++)
    {
        dif = 0.0;
        // printf("%d\n", nprom-i);
        for (j = 0; j < (nprom - i); j++)
        {
            for (k = 0; k < n_part; k++)
            {
                dx = cfx[(j + i) * n_part + k] - cfx[j * n_part + k];
                dy = cfy[(j + i) * n_part + k] - cfy[j * n_part + k];
                dz = cfz[(j + i) * n_part + k] - cfz[j * n_part + k];
                dif += dx * dx + dy * dy + dz * dz;
            }
        }
        aux = (n_part * (nprom - i));
        wt[i] += (dif / aux);
    }
}