EXECUTABLES = parallel

all: $(EXECUTABLES)

ifdef OPENCL_INC
  CL_CFLAGS = -I$(OPENCL_INC)
endif

ifdef OPENCL_LIB
  CL_LDFLAGS = -L$(OPENCL_LIB)
endif


parallel: parallel.c cl-helper.c
	gcc $(CL_CFLAGS) $(CL_LDFLAGS) -std=gnu99 -o$@ $^ -lrt -lOpenCL

clean:
	rm -f $(EXECUTABLES) *.o
