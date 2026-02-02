NVCCFLAGS := -arch=all -O3 -g -lineinfo -Wno-deprecated-gpu-targets

all: vectoradd_smem.cu
	nvcc $(NVCCFLAGS) vectoradd_smem.cu -o vectoradd_smem

clean:
	rm -f vectoradd_smem
