#include "functionsl.h"
#include "kernels.h"

int main(int argc, char const *argv[]) {
  // Definir la GPU
  //    cudaSetDevice(0);

  // Archivos para trabajar
  FILE *f_iniconf;
  FILE *f_gr;
  FILE *f_final;
  FILE *f_ener;
  FILE *wt_f;

  //  Numero de particulas
  int simple_part = 16;
  int n_part = simple_part * simple_part * simple_part;
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
  float4 *positions;
  cudaMallocManaged(&positions, n_part * sizeof(float4));
  float4 *rr_ref;
  cudaMallocManaged(&rr_ref, n_part * sizeof(float4));
  float4 *forces;
  cudaMallocManaged(&forces, n_part * sizeof(float4));
  float *g;
  cudaMallocManaged(&g, nm * sizeof(float));
  float t = 0.0f;
  float *wt;
  cudaMallocManaged(&wt, mt_n * sizeof(float));

  float *cfx;
  cudaMallocManaged(&cfx, n_part * sizeof(float));
  float *cfy;
  cudaMallocManaged(&cfy, n_part * sizeof(float));
  float *cfz;
  cudaMallocManaged(&cfz, n_part * sizeof(float));
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
  if (config_termal == 1) {
    printf("Se va a leer datos existentes de termalización.\n");
    f_final = fopen("final_conf.dat", "r");
    for (int i = 0; i < n_part; i++) {
      fscanf(f_final, "%f", &positions[i].x);
      fscanf(f_final, "%f", &positions[i].y);
      fscanf(f_final, "%f", &positions[i].z);
      fscanf(f_final, "%f", &forces[i].x);
      fscanf(f_final, "%f", &forces[i].y);
      fscanf(f_final, "%f", &forces[i].z);
    }
    fclose(f_final);
  }
  // SI NO SE CUENTA CON UNA, HAY QUE CREARLA:
  else {
    // Configuración inicial
    iniconf(positions, rho, radio_c, n_part);
    f_iniconf = fopen("conf_inicial.dat", "w");
    for (int i = 0; i < n_part; i++) {
      fprintf(f_iniconf, "%.10f %.10f %.10f\n", positions[i].x, positions[i].y,
              positions[i].y);
    }
    fclose(f_iniconf);

    // Verificar que la energía es cero, o cercana a cero
    rdf_force<<<bloques, hilos>>>(positions, forces, n_part, l_caja, ener,
                                  virial);
    cudaDeviceSynchronize();
    total_ener = 0.0f;
    for (int i = 0; i < n_part; i++)
      total_ener += ener[i];
    printf("E/N: %.10f\n", total_ener / ((float)(n_part)));

    // Termalizar el sistema
    f_ener = fopen("energia.dat", "w");
    f_final = fopen("final_conf.dat", "w");

    for (int i = 0; i < nct; i++) {
      // * Crear números aleatorios
      curandGenerateNormal(gen, rngvecx_dev, n_part, 0.0f, sigma);
      curandGenerateNormal(gen, rngvecy_dev, n_part, 0.0f, sigma);
      curandGenerateNormal(gen, rngvecz_dev, n_part, 0.0f, sigma);

      // * Mover las partículas
      position<<<bloques, hilos>>>(positions, forces, d_tiempo, l_caja, n_part,
                                   1, rngvecx_dev, rngvecy_dev, rngvecz_dev);
      cudaDeviceSynchronize();

      // * Calcular energías, fuerzas y virial
      rdf_force<<<bloques, hilos>>>(positions, forces, n_part, l_caja, ener,
                                    virial);
      cudaDeviceSynchronize();

      // ! Calcular la energía total y el virial
      total_ener = 0.0f;
      for (int k = 0; k < n_part; k++) {
        total_ener += ener[k];
        total_virial += virial[k];
      }

      if (i % 1000 == 0) {
        printf("%d %.10f Thermal\n", i, total_ener / ((float)(n_part)));
      }
      if (i % 100 == 0) {
        fprintf(f_ener, "%d %.10f\n", i, total_ener / ((float)(n_part)));
      }
    }
    fclose(f_ener);

    // Guardar la configuración final después de termalizar
    for (int i = 0; i < n_part; i++) {
      fprintf(f_final, "%f %f %f %f %f %f\n",
              positions[i].x, positions[i].y, positions[i].z, forces[i].x,
              forces[i].y, forces[i].z);
    }
    fclose(f_final);
  }

  // Calcular la g(r)
  int nprom = 0;
  int ncep = 1;
  float z_ave = 0.0f;
  float msd = 0.0f;
  float dxx, dyy, dzz;

  f_ener = fopen("promedio.csv", "w");
  wt_f = fopen(argv[8], "w");

  for (int i = 0; i < ncp; i++) {
    // * Crear números aleatorios
    curandGenerateNormal(gen, rngvecx_dev, n_part, 0.0f, sigma);
    curandGenerateNormal(gen, rngvecy_dev, n_part, 0.0f, sigma);
    curandGenerateNormal(gen, rngvecz_dev, n_part, 0.0f, sigma);

    // * Mover las partículas
    position<<<bloques, hilos>>>(positions, forces, d_tiempo, l_caja, n_part, 0,
                                 rngvecx_dev, rngvecy_dev, rngvecz_dev);
    cudaDeviceSynchronize();

    // * Calcular energías, fuerzas y virial
    rdf_force<<<bloques, hilos>>>(positions, forces, n_part, l_caja, ener,
                                  virial);
    cudaDeviceSynchronize();

    // ! Calcular la energía total y el virial
    total_ener = 0.0f;
    total_virial = 0.0f;
    for (int k = 0; k < n_part; k++) {
      total_ener += ener[k];
      total_virial += virial[k];
    }

    if (i % 1000 == 0) {
      printf("%d %.10f Average\n", i, total_ener / ((float)(n_part)));
    }
    if (i % ncep == 0) {
      if (i == 0) {
        // La referencia inicial del vector posición
        for (int j = 0; j < n_part; j++) {
          cfx[j] = positions[j].x;
          cfy[j] = positions[j].y;
          cfz[j] = positions[j].z;
        }
      }
      msd = 0.0f;
      for (int j = 0; j < n_part; j++) {
        dxx = positions[j].x - cfx[j];
        dyy = positions[j].y - cfy[j];
        dzz = positions[j].z - cfz[j];
        msd += dxx*dxx + dyy*dyy + dzz*dzz;
      }
      msd /= n_part;
      fprintf(wt_f, "%f %e\n", t, msd);

      t += d_tiempo;

      // Actualizar el valor total de promedios
      nprom++;

      // * Calcular la g(r)
      gr(positions, g, n_part, l_caja);

      // Normalizar el virial y calcular el factor de compresibilidad
      total_virial /= (float)(3.0f * n_part);
      big_z = 1.0f + total_virial;
      // Esto es para calcular el promedio después
      z_ave += big_z;

      // * Guardar a archivo
      fprintf(f_ener, "%d,%f,%f\n", i, total_ener / ((float)(n_part)), big_z);
    }
  }

  fclose(f_ener);
  fclose(wt_f);

  printf("%.10f %d %f\n", dr, nprom, z_ave / (float)(nprom));

  printf("Computing g(r)...\n");
  f_gr = fopen(argv[7], "w");
  float *r;
  cudaMallocManaged(&r, nm * sizeof(float));
  float dv = 0.0f;
  float fnorm = 0.0f;

  for (int i = 1; i < nm; i++) {
    r[i] = (i - 1) * dr;
    dv = 4.0f * PI * r[i] * r[i] * dr;
    fnorm = powf(l_caja, 3.0f) / (powf(n_part, 2.0f) * nprom * dv);
    g[i] = g[i] * fnorm;
    fprintf(f_gr, "%.10f %.10f\n", r[i], g[i]);
  }
  fclose(f_gr);
  printf("Done with g(r)...\n");

  cudaDeviceSynchronize();

  // Mean-square displacement
//  printf("Computing MSD...\n");
//  difusion(nprom, n_part, cfx, cfy, cfz, wt);
//  printf("Done with MSD.\n");
//
//  wt_f = fopen(argv[8], "w");
//  for (int i = 0; i < (ncp / ncep); i++) {
//    fprintf(wt_f, "%.10f %.10f\n", t[i], wt[i]);
//  }
//  fclose(wt_f);

  // ! Cleanup
  curandDestroyGenerator(gen);
  cudaFree(positions);
  cudaFree(forces);
  cudaFree(rngvecx_dev);
  cudaFree(rngvecy_dev);
  cudaFree(rngvecz_dev);
  cudaFree(ener);
  cudaFree(virial);
  cudaFree(r);
  cudaFree(g);
  cudaFree(cfx);
  cudaFree(cfy);
  cudaFree(cfz);
  cudaFree(wt);
  //    cudaFree(dif);
  //     cudaDeviceReset();

  return 0;
}
