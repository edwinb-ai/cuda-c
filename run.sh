make clean
make
time ./browniano 0.40 50000 200000 0.00001 123456789ULL 0 gr_1_12p.dat
# ./a.out 0.40 0 100000 0.00001 12398765 1 gr_1.dat
# ./a.out 0.40 0 100000 0.00001 32497068 1 gr_2.dat

    # argv[1] Fracción de empaquetamiento
    # argv[2] Configuraciones para termalizar
    # argv[3] Termalización
    # argv[4] Paso de tiempo
    # argv[5] Semilla
    # argv[6] Configuración existente de termalización
