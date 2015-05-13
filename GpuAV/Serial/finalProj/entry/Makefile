CUDAPATH = /home/apps/fas/GPU/cuda_6.0.37

NVCC = $(CUDAPATH)/bin/nvcc

#nvcc compiler and cuda library path
LFLAGS_NVCC = -L$(CUDAPATH)/lib64 -lcuda -lcudart -lm

#build for SM50 MAXWELL
GENCODE_SM20 = -g -G -gencode=arch=compute_20,code=\"sm_20,compute_20\"
GENCODE = $(GENCODE_SM20)

.SUFFIXES : .cu .ptx .c

all: parallel

parallel: parallel.o
	$(NVCC) $(GENCODE) $(LFLAGS_NVCC) -o $@ $< $(LFLAG)


.cu.o:
	$(NVCC) $(GENCODE) $(NVCCFLAGS) -o $@ -c $<

clean:
	rm -f *.o parallel
