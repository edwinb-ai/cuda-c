#include <stdlib.h>
#include <math.h>
#include <stdio.h>
//#include <omp.h>
#include "random_s.h"
#include "functionsl.h"

void iniconf(double* x, double* y, double* z, double rho, double rc, int num_part)
{
    // Definir la distancia según la densidad
    double dist = pow(1.0/rho, 1.0/3.0);

    // Inicializar las primeras posiciones
    x[0] = - rc + (dist / 2.0);
    y[0] = - rc + (dist / 2.0);
    z[0] = - rc + (dist / 2.0);

    for (int i = 1; i < num_part-1; i++)
    {
        x[i] = x[i-1] + dist;
        y[i] = y[i-1];
        z[i] = z[i-1];

        if (x[i] > rc)
        {
            x[i] = x[0];
            y[i] = y[i-1] + dist;

            if (y[i] > rc)
            {
                x[i] = x[0];
                y[i] = y[0];
                z[i] = z[i-1] + dist;
            }
        }
    }
}

double hardsphere(double r_pos)
{
    double uij = 0.0;

    uij = (a_param/temp) * (pow(1.0/r_pos, lambda) - pow(1.0/r_pos, lambda-1.0));

    uij += 1.0 / temp;

    return uij;
}

double rdf_force(double* x, double* y, double* z, double* fx, double* fy, double* fz, int num_part, double box_l)
{
    // Parámetros
    double rc = box_l/2.0;
    double d_r = rc / nm;

    // Inicializar algunas variables de la posicion
    double xij = 0.0, yij = 0.0, zij = 0.0, rij = 0.0;
    double fij = 0.0;
    double uij = 0.0, ener = 0.0;
    size_t i = 0, j = 0;

    // Inicializar arreglos para la fuerza
    for (i = 0; i < num_part; i++)
    {
        fx[i] = 0.0;
        fy[i] = 0.0;
        fz[i] = 0.0;
    }

    // #pragma omp parallel for num_threads(30) default(shared) private(xij,yij,zij,i,j,rij) reduction(+:ener) 
    for (i = 0; i < num_part; i++)
    {
        for (j = i+1; j < num_part-1; j++)
        {
            // Siempre inicializar en cero
            uij = 0.0;  
            fij = 0.0;  

            // Contribucion de pares
            xij = x[i] - x[j];
            yij = y[i] - y[j];
            zij = z[i] - z[j];

            // Condiciones de frontera
            xij -= (box_l * round(xij/box_l));
            yij -= (box_l * round(yij/box_l));
            zij -= (box_l * round(zij/box_l));

            rij = sqrt(xij*xij + yij*yij + zij*zij);

            if (rij < rc)
            {
                // Siempre se calcula la fuerza
                if (rij < b_param)
                {
                    uij = hardsphere(rij);
                    fij = lambda*pow(1.0/rij, lambda+1.0) - (lambda-1.0)*pow(1.0/rij, lambda);
                    fij *= (a_param/temp);
                    // printf("%f\n", uij);
                }
                else
                {
                    uij = 0.0;
                    fij = 0.0;
                }
                // Actualizar los valores de las fuerzas
                fx[i] += (fij*xij)/rij;
                fy[i] += (fij*yij)/rij;
                fz[i] += (fij*zij)/rij;
                
                fx[j] -= (fij*xij)/rij;
                fy[j] -= (fij*yij)/rij;
                fz[j] -= (fij*zij)/rij;
                ener = ener + uij;
                // printf("%f\n", ener);
            }
        }
    }

    return ener;
}

void gr(double* x, double* y, double* z, double* g, int num_part, double box_l)
{
    // Parámetros
    double rc = box_l/2.0;
    double d_r = rc / nm;

    int nbin = 0;
    int i = 0, j = 0;
    double xij = 0.0, yij = 0.0, zij = 0.0, rij = 0.0;

    // #pragma omp parallel for num_threads(30) default(shared) private(xij,yij,zij,i,j,rij)
    for (i = 0; i < num_part; i++)
    {
        for (j = i+1; j < num_part-1; j++)
        {

            // Contribucion de pares
            xij = x[j] - x[i];
            yij = y[j] - y[i];
            zij = z[j] - z[i];

            // Condiciones de frontera
            xij -= (box_l * round(xij/box_l));
            yij -= (box_l * round(yij/box_l));
            zij -= (box_l * round(zij/box_l));

            rij = sqrt(xij*xij + yij*yij + zij*zij);

            if (rij < rc)
            {
                nbin = (int)(rij/d_r) + 1;
                if (nbin <= nm)
                {
                    g[nbin] += 2.0;
                }
            }
        }
    }
}

void position(double* x, double* y, double* z, double* fx, double* fy, double* fz, double dtt,
double box_l, int num_part, int pbc)
{
    // Inicializar algunas variables
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    double sigma = sqrt(2.0*dtt);

    for (int i = 0; i < num_part; i++)
    {
        dx = sigma * gasdev();
        dy = sigma * gasdev();
        dz = sigma * gasdev();

        x[i] += fx[i]*dtt + dx;
        y[i] += fy[i]*dtt + dy;
        z[i] += fz[i]*dtt + dz;

        if (pbc == 1)
        {
            x[i] -= (box_l * round(x[i]/box_l));
            y[i] -= (box_l * round(y[i]/box_l));
            z[i] -= (box_l * round(z[i]/box_l));
        }
    }
}

// void difusion(int nprom, int n_part, double cfx[mt_n][mp], double cfy[mt_n][mp], double cfz[mt_n][mp], double* wt)

void difusion(const int nprom, const int n_part, double* cfx, double* cfy, double* cfz, double* wt)
{
    double dif = 0.0;
    int i = 0, j = 0, k = 0;
    double dx = 0.0, dy = 0.0, dz = 0.0, aux = 0.0;
    
    // #pragma omp parallel for 
    // Mean-squared displacement
    for (i = 0; i < nprom; i++)
    {
        dif = 0.0;
        // printf("%d\n", nprom-i);
        for (j = 0; j < nprom-i; j++)
        {
            for (k = 0; k < n_part; k++)
            {
                dx = cfx[(j+i)*mp + k] - cfx[j*mp + k];
                dy = cfy[(j+i)*mp + k] - cfy[j*mp + k];
                dz = cfz[(j+i)*mp + k] - cfz[j*mp + k];
                dif += dx*dx + dy*dy + dz*dz;
            }
        }
        aux = (n_part*(nprom-i));
        wt[i] = (dif/aux);
    }
}
