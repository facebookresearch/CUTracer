GENCODE   := \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_86,code=compute_86 \
    -gencode arch=compute_90,code=sm_90

NVCCFLAGS := $(GENCODE) -O3 -g -lineinfo

all:
	nvcc $(NVCCFLAGS) vectoradd.cu -o vectoradd

clean:
	rm -f vectoradd
