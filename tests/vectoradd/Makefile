NVCCFLAGS := -arch=all -O3 -g -lineinfo

all: vectoradd.cu
	nvcc $(NVCCFLAGS) vectoradd.cu -o vectoradd

clean:
	rm -f vectoradd
