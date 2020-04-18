CC=nvcc
FLAGS=-lcurand -g -G
FILES=langevin_hs.cu functionsl.cu

make:
	$(CC) $(FLAGS) $(FILES) -o browniano

clean:
	rm browniano