NVCCFLAGS := -arch=all -O3 -g -lineinfo -Wno-deprecated-gpu-targets

all: vectoradd.cu
	nvcc $(NVCCFLAGS) vectoradd.cu -o vectoradd

clean:
	rm -f vectoradd
