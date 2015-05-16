#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <stdint.h>
#include "timing.h"
#include "cl-helper.h"

#define TAG_BYTES 10

#define GDIM_X 256
#define GDIM_Y 256
#define LDIM_X 16
#define LDIM_Y 16

const char *MAIN = "mainGPUsig.bin";

//function to load input file and signature files to scan
void loadFile (const char *fileName, uint8_t **buffer, int *size){
    long lSize;
    FILE *fp;
    
    fp = fopen (fileName , "rb" );
    if( !fp ) perror(fileName),exit(1);
    
    //seek the beginning of file
    //fseek(fp, SEEK_SET, 0);
    fseek( fp , 0L , SEEK_END);
    lSize = ftell( fp );
    rewind( fp );
    //printf("%ld\n",lSize);
    (*size) = lSize;

    /* allocate memory for entire content */
    (*buffer) = (uint8_t *) calloc( 1, lSize+1 );
    if( !(*buffer) ) 
        fclose(fp),fputs("memory alloc fails",stderr),exit(1);

    /* copy the file into the buffer */
    if( 1!=fread( (*buffer) , lSize, 1 , fp) )
          fclose(fp),free((*buffer)),fputs("entire read fails",stderr),exit(1);

    fclose(fp);
}


int main(int argc, char **argv)
{
    cl_context ctx;
    cl_command_queue queue;

    //host buffer to store each signature dataset
    uint8_t *mainBuf;
    uint8_t *fileBuf;
    int sizeMb, sizeFb;
    int sigNum;
    int results=0;
    //LDIM is the block dimension, 
    //GDIM is the grid dimension,
    size_t ldim[2];
    size_t gdim[2];    

    create_context_on(CHOOSE_INTERACTIVELY, CHOOSE_INTERACTIVELY, 0, &ctx, &queue, 0);

    //print_device_info_from_queue(queue);

    // --------------------------------------------------------------------------
    // load kernels 
    // --------------------------------------------------------------------------
    char *knl_text = read_file("patMat.cl");
    cl_kernel knl = kernel_from_string(ctx, knl_text, "patternMatching", NULL);
    free(knl_text);

    // --------------------------------------------------------------------------
    // process signautres
    // --------------------------------------------------------------------------

    //load signatures from database
    loadFile(MAIN, &mainBuf, &sizeMb); 

    // allocate device memory for signature
    cl_int status;

    cl_mem devMb = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      sizeof(uint8_t) * sizeMb, 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");

    // transfer signatures to device
    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        queue, devMb, /*blocking*/ CL_TRUE, /*offset*/ 0,
        sizeMb * sizeof(uint8_t), mainBuf,
        0, NULL, NULL));

    sigNum = sizeMb/TAG_BYTES;
    printf("Loaded %d signatures.\n", sigNum);

    ldim[0] = LDIM_X;
    ldim[1] = LDIM_Y;

    gdim[0] = GDIM_X;
    gdim[1] = GDIM_Y;
    
    // --------------------------------------------------------------------------
    // process files
    // --------------------------------------------------------------------------

    //load files from directory
    printf("scanning file ...\n");
    loadFile("badGuy.bin", &fileBuf, &sizeFb);

    // allocate device memory for file
    cl_mem devFb = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      sizeof(uint8_t) * sizeFb, 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");

    // transfer files to device
    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        queue, devFb, /*blocking*/ CL_TRUE, /*offset*/ 0,
        sizeFb * sizeof(uint8_t), fileBuf,
        0, NULL, NULL));

    // --------------------------------------------------------------------------
    // process results flag
    // --------------------------------------------------------------------------

    //allocate device memory for results flag, just 1 int
    cl_mem devResults = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      sizeof(int), 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");

    // transfer results flag to device
    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        queue, devResults, /*blocking*/ CL_TRUE, /*offset*/ 0,
        sizeof(int), &results,
        0, NULL, NULL));

    // --------------------------------------------------------------------------
    // run code on device
    // --------------------------------------------------------------------------
    CALL_CL_GUARDED(clFinish, (queue));

    timestamp_type time1, time2;
    get_timestamp(&time1);

    SET_5_KERNEL_ARGS(knl, devMb, devFb, sigNum, sizeFb, devResults);
    CALL_CL_GUARDED(clEnqueueNDRangeKernel,
    (queue, knl,
     /*dimensions*/ 2, NULL, gdim, ldim,
     0, NULL, NULL));

    CALL_CL_GUARDED(clFinish, (queue));

    get_timestamp(&time2);
    double elapsed = timestamp_diff_in_seconds(time1,time2);
    
    // --------------------------------------------------------------------------
    //get results from device to host
    // --------------------------------------------------------------------------
    CALL_CL_GUARDED(clEnqueueReadBuffer, (
        queue, devResults, /*blocking*/ CL_TRUE, /*offset*/ 0,
        sizeof(int), &results,
        0, NULL, NULL));
    if(results != 0){
        printf("    found virus\n");
    }
}
