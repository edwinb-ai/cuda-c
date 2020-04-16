CC=nvcc
FLAGS=-lcurand -g -G -O2
FILES=langevin_hs.cu functionsl.cu

make:
	$(CC) $(FLAGS) $(FILES) -o browniano

clean:
	rm browniano