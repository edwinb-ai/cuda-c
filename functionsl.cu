#include "functionsl.h"

void iniconf(double *x, double *y, double *z, double rho, double rc, int num_part)
{
    // Definir la distancia según la densidad
    double dist = pow(1.0 / rho, 1.0 / 3.0);

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

__global__ void rdf_force(double *x, double *y, double *z, double *fx, double *fy, double *fz,
                          int num_part, double box_l, double *ener)
{
    // Parámetros
    double rc = box_l * 0.5f;
    // double d_r = rc / nm;

    // Inicializar algunas variables de la posicion
    double xij = 0.0, yij = 0.0, zij = 0.0, rij = 0.0;
    double fij = 0.0;
    double uij = 0.0;
    double potential = 0.0;
    int i = 0, j = 0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (i = idx; i < num_part - 1; i += stride)
    {
        // Inicializar valores
        fx[i] = 0.0;
        fy[i] = 0.0;
        fz[i] = 0.0;
        ener[i] = 0.0;
    }

    for (i = idx; i < num_part - 1; i += stride)
    {
        // Inicializar valores
        potential = 0.0;

        for (j = 0; j < num_part; j++)
        {
            if (i == j)
                continue;
            // Siempre inicializar en cero
            uij = 0.0;
            fij = 0.0;

            // Contribucion de pares
            xij = x[i] - x[j];
            yij = y[i] - y[j];
            zij = z[i] - z[j];

            // Condiciones de frontera
            xij -= (box_l * round(xij / box_l));
            yij -= (box_l * round(yij / box_l));
            zij -= (box_l * round(zij / box_l));

            rij = sqrt(xij * xij + yij * yij + zij * zij);

            if (rij < rc)
            {
                // Siempre se calcula la fuerza
                if (rij < b_param)
                {
                    uij = (a_param / temp) * (pow(1.0 / rij, lambda) - pow(1.0 / rij, lambda - 1.0));
                    fij = lambda * pow(1.0 / rij, lambda + 1.0) - (lambda - 1.0) * pow(1.0 / rij, lambda);
                    fij *= (a_param / temp);
                    uij += (1.0 / temp);
                }
                else
                {
                    uij = 0.0;
                    fij = 0.0;
                }

                // Actualizar los valores de las fuerzas
                fx[i] += (fij * xij) / rij;
                fy[i] += (fij * yij) / rij;
                fz[i] += (fij * zij) / rij;

                fx[j] -= (fij * xij) / rij;
                fy[j] -= (fij * yij) / rij;
                fz[j] -= (fij * zij) / rij;

                // Actualizar los valores de la energía
                potential += uij;
            }
        }
        ener[i] = potential;
    }
}

void gr(double *x, double *y, double *z, double *g, int num_part, double box_l)
{
    // Parámetros
    double rc = box_l * 0.5f;
    double d_r = rc / nm;

    int nbin = 0;
    int i = 0, j = 0;
    double xij = 0.0, yij = 0.0, zij = 0.0, rij = 0.0;

    for (i = 0; i < num_part; i++)
    {
        for (j = i + 1; j < num_part-1; j++)
        {

            // Contribucion de pares
            xij = x[j] - x[i];
            yij = y[j] - y[i];
            zij = z[j] - z[i];

            // Condiciones de frontera
            xij -= box_l * round(xij / box_l);
            yij -= box_l * round(yij / box_l);
            zij -= box_l * round(zij / box_l);

            rij = sqrt(xij * xij + yij * yij + zij * zij);

            if (rij < rc)
            {
                nbin = (int)(rij / d_r) + 1;
                if (nbin <= nm)
                {
                    g[nbin] += 2.0;
                }
            }
        }
    }
}

__global__ void position(double *x, double *y, double *z, double *fx, double *fy, double *fz, double dtt,
                         double box_l, int num_part, int pbc, double *randvec)
{
    // Inicializar algunas variables
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    double sigma = sqrt(2.0 * dtt);
    int i = 0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (i = idx; i < num_part; i += stride)
    {
        dx = sigma * randvec[3 * i];
        dy = sigma * randvec[(3 * i) + 1];
        dz = sigma * randvec[(3 * i) + 2];

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

void difusion(const int nprom, const int n_part, double *cfx, double *cfy, double *cfz, double *wt)
{
    double dif = 0.0;
    size_t i = 0, j = 0, k = 0;
    double dx = 0.0, dy = 0.0, dz = 0.0, aux = 0.0;

    // Mean-squared displacement
    for (i = 0; i < nprom; i++)
    {
        dif = 0.0;
        // printf("%d\n", nprom-i);
        for (j = 0; j < nprom - i; j++)
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
