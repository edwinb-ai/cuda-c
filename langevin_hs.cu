#include "functionsl.h"

int main(int argc, char const *argv[])
{
    // Archivos para trabajar
    FILE *f_iniconf;
    FILE *f_gr;
    FILE *f_final;
    FILE *f_ener;
    FILE *wt_f;

    //  Numero de particulas
    int n_part = 2048;
    //  Fracción de empaquetamiento
    float phi = atof(argv[1]);
    //  Densidad
    float rho = 6.0 * phi / pi;
    //  Configuraciones para termalizar
    int nct = atoi(argv[2]);
    //  Termalización
    int ncp = atoi(argv[3]);
    //  Paso de tiempo
    float d_tiempo = atof(argv[4]);
    unsigned long long seed = (unsigned long long)atoi(argv[5]);
    //  Revisar si ya se tiene una configuración de termalización
    int config_termal = atoi(argv[6]);

    // Tamaño de caja
    float l_caja = powf((float)(n_part) / rho, 1.0 / 3.0);
    float radio_c = l_caja / 2.0;
    float dr = radio_c / nm;

    // Mostrar información del sistema
    printf("El tamaño de la caja es: %f\n", l_caja);
    printf("Distancia media entre partículas: %f\n", powf(rho, -1.0 / 3.0));
    printf("Radio de corte: %f\n", radio_c);

    // ! RNG variables
    curandGenerator_t gen;
    float *rngvec_dev;
    cudaMallocManaged(&rngvec_dev, 3 * n_part * sizeof(float));
    // ! Create pseudo-random number generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    // ! Set seed
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    // Inicializar los arreglos
    float *x;
    cudaMallocManaged(&x, n_part * sizeof(float));
    float *y;
    cudaMallocManaged(&y, n_part * sizeof(float));
    float *z;
    cudaMallocManaged(&z, n_part * sizeof(float));
    float *fx;
    cudaMallocManaged(&fx, n_part * sizeof(float));
    float *fy;
    cudaMallocManaged(&fy, n_part * sizeof(float));
    float *fz;
    cudaMallocManaged(&fz, n_part * sizeof(float));
    // float *g = calloc(nm, sizeof(float));
    // float *t = calloc(mt_n, sizeof(float));
    // float *h = calloc(nm, sizeof(float));
    // float *wt = calloc(mt_n, sizeof(float));

    // float *cfx = calloc(mt_n * n_part, sizeof(float));
    // float *cfy = calloc(mt_n * n_part, sizeof(float));
    // float *cfz = calloc(mt_n * n_part, sizeof(float));
    float ener = 0.0;

    // Asignar hilos y bloques
    int hilos = 256;
    int bloques = 1 + (n_part - 1) / hilos;

    // SI SE INGRESA UNA CONFIGURACION DE TERMALIZACION, SE LEE:
    if (config_termal == 1)
    {
        printf("Se va a leer datos existentes de termalización.\n");
        f_final = fopen("final_conf.dat", "r");
        for (int i = 0; i < n_part; i++)
        {
            fscanf(f_final, "%lf", &x[i]);
            fscanf(f_final, "%lf", &y[i]);
            fscanf(f_final, "%lf", &z[i]);
            fscanf(f_final, "%lf", &fx[i]);
            fscanf(f_final, "%lf", &fy[i]);
            fscanf(f_final, "%lf", &fz[i]);
        }
        fclose(f_final);
    }
    // SI NO SE CUENTA CON UNA, HAY QUE CREARLA:
    else
    {
        // Configuración inicial
        iniconf(x, y, z, rho, radio_c, n_part);
        f_iniconf = fopen("conf_inicial.dat", "w");
        for (int i = 0; i < n_part; i++)
        {
            fprintf(f_iniconf, "%.10f %.10f %.10f\n", x[i], y[i], z[i]);
        }
        fclose(f_iniconf);
    }

    // Verificar que la energía es cero
    rdf_force<<<bloques, hilos>>>(x, y, z, fx, fy, fz, n_part, l_caja, ener);
    cudaDeviceSynchronize();
    printf("E/N: %.10f\n", ener / ((float)(n_part)));

    // Termalizar el sistema
    f_ener = fopen("energia.dat", "w");
    f_final = fopen("final_conf.dat", "w");

    for (size_t i = 0; i < nct; i++)
    {
        // * Crear números aleatorios
        curandGenerateUniform(gen, rngvec_dev, 3 * n_part);
        position<<<bloques, hilos>>>(x, y, z, fx, fy, fz, d_tiempo, l_caja, n_part, 1, rngvec_dev);
        cudaDeviceSynchronize();
        rdf_force<<<bloques, hilos>>>(x, y, z, fx, fy, fz, n_part, l_caja, ener);
        cudaDeviceSynchronize();
        if (i % 1000 == 0)
        {
            printf("%d %.10f Thermal\n", i, ener / ((float)(n_part)));
            for (int i = 0; i < n_part; i++)
            {
                printf("%.10f %.10f %.10f\n", x[i], y[i], z[i]);
            }
        }
        if (i % 100 == 0)
        {
            fprintf(f_ener, "%d %.10f\n", i, ener / ((float)(n_part)));
        }
    }
    fclose(f_ener);

    // // Guardar la configuración final después de termalizar
    // for (int i = 0; i < n_part; i++)
    // {
    //     fprintf(f_final, "%.10f %.10f %.10f %.10f %.10f %.10f\n", x[i], y[i], z[i], fx[i], fy[i], fz[i]);
    // }
    // fclose(f_final);

    // // Calcular la g(r)
    // int nprom = 0;
    // for (int i = 0; i < ncp; i++)
    // {
    //     position(x, y, z, fx, fy, fz, d_tiempo, l_caja, n_part, 0);
    //     ener = rdf_force(x, y, z, fx, fy, fz, n_part, l_caja);
    //     if (i % 1000 == 0)
    //     {
    //         printf("%d %.10f Average\n", i, ener / ((float)(n_part)));
    //     }
    //     if (i % 10 == 0)
    //     // if (i%n_part == 0) // Promediar cada numero total de particulas
    //     {
    //         t[nprom] = d_tiempo * 10.0 * nprom;
    //         for (int j = 0; j < n_part; j++)
    //         {
    //             cfx[nprom * mp + j] = x[j];
    //             cfy[nprom * mp + j] = y[j];
    //             cfz[nprom * mp + j] = z[j];
    //         }
    //         nprom++;
    //         gr(x, y, z, g, n_part, l_caja);
    //     }
    // }

    // printf("%.10f %d\n", dr, nprom);

    // f_gr = fopen(argv[7], "w");
    // float *r = calloc(nm, sizeof(float));
    // float dv = 0.0;
    // float hraux = 0.0, fnorm = 0.0;

    // for (int i = 1; i < nm; i++)
    // {
    //     r[i] = (i - 1) * dr;
    //     dv = 4.0 * pi * r[i] * r[i] * dr;
    //     fnorm = powf(l_caja, 3.0) / (powf(n_part, 2.0) * nprom * dv);
    //     g[i] = g[i] * fnorm;
    //     h[i] = g[i] - 1.0;
    //     fprintf(f_gr, "%.10f %.10f %.10f\n", r[i], g[i], h[i]);
    // }
    // fclose(f_gr);

    // // Mean-square displacement and intermediate scattering function
    // difusion(nprom, n_part, cfx, cfy, cfz, wt);

    // wt_f = fopen("wt.dat", "w");
    // for (int i = 0; i < (ncp / 10); i++)
    // {
    //     fprintf(wt_f, "%.10f %.10f\n", t[i], wt[i]);
    // }
    // fclose(wt_f);

    // ! Cleanup
    curandDestroyGenerator(gen);
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(fx);
    cudaFree(fy);
    cudaFree(fz);
    cudaFree(rngvec_dev);
    // free(r);
    // free(g);
    // free(t);
    // free(cfx);
    // free(cfy);
    // free(cfz);
    // free(wt);
    // free(h);
    return EXIT_SUCCESS;
}
