CC=nvcc
FLAGS=-lcurand -g -G -O1
FILES=langevin_hs.cu functionsl.c kernels.cu

make:
	$(CC) $(FLAGS) $(FILES) -o browniano

clean:
	rm browniano