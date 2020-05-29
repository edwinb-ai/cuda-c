#include "functionsl.h"

int main(int argc, char const *argv[])
{
    // Definir la GPU
    cudaSetDevice(0);

    // Archivos para trabajar
    FILE *f_iniconf;
    FILE *f_gr;
    FILE *f_final;
    FILE *f_ener;
    FILE *wt_f;

    //  Numero de particulas
    int simple_part = 12;
    int n_part = simple_part*simple_part*simple_part;
    //  Fracción de empaquetamiento
    float phi = atof(argv[1]);
    //  Densidad
    float rho = 6.0f * phi / PI;
    //  Configuraciones para termalizar
    int nct = atoi(argv[2]);
    //  Termalización
    int ncp = atoi(argv[3]);
    //  Paso de tiempo
    float d_tiempo = atof(argv[4]);
    unsigned long long int seed = (unsigned long long int)atoi(argv[5]);
    //  Revisar si ya se tiene una configuración de termalización
    int config_termal = atoi(argv[6]);

    // Tamaño de caja
    float l_caja = powf((float)(n_part) / rho, 1.0f / 3.0f);
    float radio_c = l_caja / 2.0f;
    float dr = radio_c / nm;

    // Desviación estándar
    float sigma = sqrtf(2.0f * d_tiempo);

    // Factor de compresibilidad
    float big_z = 0.0f;

    // Mostrar información del sistema
    printf("El tamaño de la caja es: %f\n", l_caja);
    printf("Distancia media entre partículas: %f\n", powf(rho, -1.0f / 3.0f));
    printf("Radio de corte: %f\n", radio_c);

    // ! RNG variables
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    // * Uno por cada dimensión
    float *rngvecx_dev;
    float *rngvecy_dev;
    float *rngvecz_dev;
    cudaMallocManaged(&rngvecx_dev, n_part * sizeof(float));
    cudaMallocManaged(&rngvecy_dev, n_part * sizeof(float));
    cudaMallocManaged(&rngvecz_dev, n_part * sizeof(float));

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
    float *g;
    cudaMallocManaged(&g, nm * sizeof(float));
    float *t;
    cudaMallocManaged(&t, mt_n * sizeof(float));
    float *wt;
    cudaMallocManaged(&wt, mt_n * sizeof(float));

    double *cfx;
    cudaMallocManaged(&cfx, mt_n * n_part * sizeof(double));
    double *cfy;
    cudaMallocManaged(&cfy, mt_n * n_part * sizeof(double));
    double *cfz;
    cudaMallocManaged(&cfz, mt_n * n_part * sizeof(double));
    float *ener;
    cudaMallocManaged(&ener, n_part * sizeof(float));
    float total_ener = 0.0f;
    float *virial;
    cudaMallocManaged(&virial, n_part * sizeof(float));
    float total_virial = 0.0f;

    // Asignar hilos y bloques
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int hilos = 256;
    int bloques = 32 * numSMs;

    // SI SE INGRESA UNA CONFIGURACION DE TERMALIZACION, SE LEE:
    if (config_termal == 1)
    {
        printf("Se va a leer datos existentes de termalización.\n");
        f_final = fopen("final_conf.dat", "r");
        for (int i = 0; i < n_part; i++)
        {
            fscanf(f_final, "%f", &x[i]);
            fscanf(f_final, "%f", &y[i]);
            fscanf(f_final, "%f", &z[i]);
            fscanf(f_final, "%f", &fx[i]);
            fscanf(f_final, "%f", &fy[i]);
            fscanf(f_final, "%f", &fz[i]);
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

        // Verificar que la energía es cero
        rdf_force<<<bloques, hilos>>>(x, y, z, fx, fy, fz, n_part, l_caja, ener, virial);
        cudaDeviceSynchronize();
        total_ener = 0.0f;
        for (int i = 0; i < n_part; i++)
            total_ener += ener[i];
        printf("E/N: %.10f\n", total_ener / ((float)(n_part)));

        // Termalizar el sistema
        f_ener = fopen("energia.dat", "w");
        f_final = fopen("final_conf.dat", "w");

        for (int i = 0; i < nct; i++)
        {
            // * Crear números aleatorios
            curandGenerateNormal(gen, rngvecx_dev, n_part, 0.0f, sigma);
            curandGenerateNormal(gen, rngvecy_dev, n_part, 0.0f, sigma);
            curandGenerateNormal(gen, rngvecz_dev, n_part, 0.0f, sigma);

            // * Mover las partículas
            position<<<bloques, hilos>>>(x, y, z, fx, fy, fz, d_tiempo,
                l_caja, n_part, 1, rngvecx_dev, rngvecy_dev, rngvecz_dev);
            cudaDeviceSynchronize();

            // * Calcular energías, fuerzas y virial
            rdf_force<<<bloques, hilos>>>(x, y, z, fx, fy, fz, n_part, l_caja, ener, virial);
            cudaDeviceSynchronize();

            // ! Calcular la energía total y el virial
            total_ener = 0.0f;
            for (int k = 0; k < n_part; k++)
            {
                total_ener += ener[k];
                total_virial += virial[k];
            }

            if (i % 1000 == 0)
            {
                // for (size_t k = 0; k < n_part; k++)
                // {
                //     printf("%.10f %.10f %.10f\n", x[k], y[k], z[k]);
                //     printf("FORCES\n");
                //     printf("%.10f %.10f %.10f\n", fx[k], fy[k], fz[k]);
                // }
                printf("%d %.10f Thermal\n", i, total_ener / ((float)(n_part)));
            }
            if (i % 100 == 0)
            {
                fprintf(f_ener, "%d %.10f\n", i, total_ener / ((float)(n_part)));
            }
        }
        fclose(f_ener);

        // Guardar la configuración final después de termalizar
        for (int i = 0; i < n_part; i++)
        {
            fprintf(f_final, "%.10lf %.10lf %.10lf %.10lf %.10lf %.10lf\n", x[i], y[i], z[i], fx[i], fy[i], fz[i]);
        }
        fclose(f_final);
    }

    // Calcular la g(r)
    int nprom = 0;
    int ncep = 10;
    f_ener = fopen("promedio.csv", "w");
    for (int i = 0; i < ncp; i++)
    {
        // * Crear números aleatorios
        curandGenerateNormal(gen, rngvecx_dev, n_part, 0.0f, sigma);
        curandGenerateNormal(gen, rngvecy_dev, n_part, 0.0f, sigma);
        curandGenerateNormal(gen, rngvecz_dev, n_part, 0.0f, sigma);

        // * Mover las partículas
        position<<<bloques, hilos>>>(x, y, z, fx, fy, fz, d_tiempo,
            l_caja, n_part, 1, rngvecx_dev, rngvecy_dev, rngvecz_dev);
        cudaDeviceSynchronize();

        // * Calcular energías, fuerzas y virial
        rdf_force<<<bloques, hilos>>>(x, y, z, fx, fy, fz, n_part, l_caja, ener, virial);
        cudaDeviceSynchronize();

        // ! Calcular la energía total y el virial
        total_ener = 0.0f;
        total_virial = 0.0f;
        for (int k = 0; k < n_part; k++)
        {
            total_ener += ener[k];
            total_virial += virial[k];
        }

        if (i % 1000 == 0)
        {
            printf("%d %.10f Average\n", i, total_ener / ((float)(n_part)));
        }
        if (i % ncep == 0)
        {
            t[nprom] = d_tiempo * (float)(ncep * nprom);
            for (int j = 0; j < n_part; j++)
            {
                cfx[nprom * n_part + j] = x[j];
                cfy[nprom * n_part + j] = y[j];
                cfz[nprom * n_part + j] = z[j];
            }
            
            // Actualizar el valor total de promedios
            nprom++;
            
            // * Calcular la g(r)
            gr(x, y, z, g, n_part, l_caja);

            // Normalizar el virial y calcular el factor de compresibilidad
            total_virial /= (float)(3.0f * n_part);
            big_z += 1.0f + total_virial;
            big_z /= (float)(nprom);

            // * Guardar a archivo
            fprintf(f_ener, "%d,%f,%f,%f\n", i, total_ener / ((float)(n_part)), 1.0f + total_virial, big_z);
        }
    }
    
    fclose(f_ener);

    printf("%.10f %d\n", dr, nprom);

    printf("Computing g(r)...\n");
    f_gr = fopen(argv[7], "w");
    float *r;
    cudaMallocManaged(&r, nm * sizeof(float));
    float dv = 0.0f;
    float fnorm = 0.0f;

    for (int i = 1; i < nm; i++)
    {
        r[i] = (i - 1) * dr;
        dv = 4.0f * PI * r[i] * r[i] * dr;
        fnorm = powf(l_caja, 3.0f) / (powf(n_part, 2.0f) * nprom * dv);
        g[i] = g[i] * fnorm;
        fprintf(f_gr, "%.10f %.10f\n", r[i], g[i]);
    }
    fclose(f_gr);
    printf("Done with g(r)...\n");

    // Mean-square displacement and intermediate scattering function
    cudaDeviceSynchronize();
    float aux = 0.0f;
    float *dif;
    cudaMallocManaged(&dif, sizeof(float));
    // Mean-squared displacement
    for (size_t i = 0; i < nprom; i++)
    {
        dif[1] = 0.0f;
        // printf("%d\n", nprom-i);
        for (size_t j = 0; j < (nprom - i); j++)
        {
            difusion<<<bloques, hilos>>>(n_part, cfx, cfy, cfz, dif, i, j);
            // cudaDeviceSynchronize();
        }
        aux = n_part * (nprom - i);
        wt[i] += (dif[1] / (float)(aux));
    }

    wt_f = fopen(argv[8], "w");
    for (int i = 0; i < (ncp / ncep); i++)
    {
        fprintf(wt_f, "%.10f %.10f\n", t[i], wt[i]);
    }
    fclose(wt_f);

    // ! Cleanup
    curandDestroyGenerator(gen);
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(fx);
    cudaFree(fy);
    cudaFree(fz);
    cudaFree(rngvecx_dev);
    cudaFree(rngvecy_dev);
    cudaFree(rngvecz_dev);
    cudaFree(ener);
    cudaFree(virial);
    cudaFree(r);
    cudaFree(g);
    cudaFree(t);
    cudaFree(cfx);
    cudaFree(cfy);
    cudaFree(cfz);
    cudaFree(wt);
    cudaFree(dif);
    // cudaDeviceReset();

    return 0;
}
