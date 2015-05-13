#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TAG_BYTES 10
#define GRID_X 32
#define GRID_Y 32
#define BLOCK_X 32
#define BLOCK_Y 32

#define ALPHABET_LEN 256
#define NOT_FOUND patlen
#define max(a, b) ((a < b) ? b : a)

const char *DAILY = "./dailyPack/dailyGPUsig.bin";
const char *MAIN = "./mainPack/mainGPUsig.bin";

//referenced 
//from http://en.wikipedia.org/wiki/Boyer–Moore_string_search_algorithm
//as the Boyer-Moore pattern matching is a well-established algorithm, 
__device__ void make_delta1(int *delta1, uint8_t *pat, int32_t patlen) {

    int i;
    for (i=0; i < ALPHABET_LEN; i++) {
        delta1[i] = NOT_FOUND;
    }
    for (i=0; i < patlen-1; i++) {
        delta1[pat[i]] = patlen-1 - i;
    }
}

//referenced 
//from http://en.wikipedia.org/wiki/Boyer–Moore_string_search_algorithm
//as the Boyer-Moore pattern matching is a well-established algorithm, 
__device__ int is_prefix(uint8_t *word, int wordlen, int pos) {
    int i;
    int suffixlen = wordlen - pos;
    // could also use the strncmp() library function here
    for (i = 0; i < suffixlen; i++) {
        if (word[i] != word[pos+i]) {
            return 0;
        }
    }
    return 1;
}

//referenced 
//from http://en.wikipedia.org/wiki/Boyer–Moore_string_search_algorithm
//as the Boyer-Moore pattern matching is a well-established algorithm, 
// length of the longest suffix of word ending on word[pos].
// suffix_length("dddbcabc", 8, 4) = 2
__device__ int suffix_length(uint8_t *word, int wordlen, int pos) {
    int i;
    // increment suffix length i to the first mismatch or beginning
    // of the word
    for (i = 0; (word[pos-i] == word[wordlen-1-i]) && (i < pos); i++);
    return i;
}

//referenced 
//from http://en.wikipedia.org/wiki/Boyer–Moore_string_search_algorithm
//as the Boyer-Moore pattern matching is a well-established algorithm, 
__device__ void make_delta2(int *delta2, uint8_t *pat, int32_t patlen) {
    int p;
    int last_prefix_index = patlen-1;

    // first loop
    for (p=patlen-1; p>=0; p--) {
        if (is_prefix(pat, patlen, p+1)) {
            last_prefix_index = p+1;
        }
        delta2[p] = last_prefix_index + (patlen-1 - p);
    }

    // second loop
    for (p=0; p < patlen-1; p++) {
        int slen = suffix_length(pat, patlen, p);
        if (pat[p - slen] != pat[patlen-1 - slen]) {
            delta2[patlen-1 - slen] = patlen-1 - p + slen;
        }
    }
}

//referenced 
//from http://en.wikipedia.org/wiki/Boyer–Moore_string_search_algorithm
//as the Boyer-Moore pattern matching is a well-established algorithm, 
__device__ uint8_t* boyer_moore (uint8_t *string, uint32_t stringlen, uint8_t *pat, uint32_t patlen) {
    int i;
    int delta1[ALPHABET_LEN];
    int *delta2 = (int *)malloc(patlen * sizeof(int));
    make_delta1(delta1, pat, patlen);
    make_delta2(delta2, pat, patlen);

    // The empty pattern must be considered specially
    if (patlen == 0) return string;

    i = patlen-1;
    while (i < stringlen) {
        int j = patlen-1;
        while (j >= 0 && (string[i] == pat[j])) {
            --i;
            --j;
        }
        if (j < 0) {
            free(delta2);
            return (string + i+1);
        }

        i += max(delta1[string[i]], delta2[j]);
    }
    free(delta2);
    return NULL;
}

__global__ void patternMatching(uint8_t *set1, uint8_t *set2, uint8_t *fileBuf, int set1SigNum, int set2SigNum, int fileSize){
    
    //note: blockDim.x = blockDim.y
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = row*GRID_Y*BLOCK_Y + col; //GRID AND BLOCK are hardcoded for convenience
    uint8_t *found; 
    
    //make sure that the idx is within the range of total number of signatures
    if(idx < set1SigNum){
        found = boyer_moore(fileBuf,fileSize,set1+idx*TAG_BYTES,TAG_BYTES);

        if(found != NULL){
            printf("found virus, lookup dailyGPUvirus.ndb line %d for virus type\n",idx);
        }
    }

    //make sure that the idx is within the range of total number of signatures
    if(idx >= set1SigNum && idx < set2SigNum){
        found = boyer_moore(fileBuf, fileSize, set2+(idx-set1SigNum)*TAG_BYTES, TAG_BYTES);

        if(found != NULL){
            printf("found virus, lookup mainGPUvirus.ndb line %d for virus type\n",(idx-set1SigNum)); 
        }
    }
}

//function to load input file and signature files to scan
void loadFile (const char *fileName, uint8_t **buffer, size_t *size){
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
/*
 * Exit codes:
 *  0: clean
 *  1: infected
 *  2: error
 */
//const char *DBDIR = "/home/leon/clamav/share/clamav";

int main(int argc, char **argv)
{
    int gpucount = 0; // Count of available GPUs
    //We only have 3701312 signatures
    //each thread get 1 signature, we need no more than 1024*1024 threads
    //grid size is then fixed to (32,32,1), and block size is (32,32,1)

    int Grid_Dim_x = GRID_X; //Grid dimension, x
    int Grid_Dim_y = GRID_Y; //Grid dimension, y
    int Block_Dim_x = BLOCK_X; //Block dimension, x
    int Block_Dim_y = BLOCK_Y; //Block dimension, y
    cudaEvent_t start, stop; // using cuda events to measure time
    float elapsed_time_ms; // which is applicable for asynchronous code also
    cudaError_t errorcode;

    //host buffer to store each signature dataset
    uint8_t *dailyBuf;
    uint8_t *mainBuf;
    uint8_t *fileBuf;
    uint8_t *devDb, *devMb, *devFb;//device buffer correspoding to the host buffer
    size_t sizeDb, sizeMb, sizeFb;
   
    if(argc != 2) {
        printf("Usage: %s file\n", argv[0]);
        return 2;
    }
    // --------------------SET PARAMETERS AND DATA -----------------------
    //load signatures into host buffer
    loadFile(DAILY, &dailyBuf, &sizeDb);
    loadFile(MAIN, &mainBuf, &sizeMb);
    
    printf("loading signatures in %s\n",DAILY);
    printf("loading signatures in %s\n",MAIN);

    /* 
    for(int i=0; i<11; i++){
        printf("%x ", (unsigned uint8_t) dailyBuf[i]);
    }
    */

    errorcode = cudaGetDeviceCount(&gpucount);
    if (errorcode == cudaErrorNoDevice) {
        printf("No GPUs are visible\n");
        exit(-1);
    }
    
    //alloc mem to GPU
    cudaMalloc((void**)&devDb, sizeDb*sizeof(uint8_t));
    cudaMalloc((void**)&devMb, sizeMb*sizeof(uint8_t)); 

    //copy sigs to GPU mem buffer
    cudaMemcpy(devDb, dailyBuf, sizeDb ,cudaMemcpyHostToDevice);
    cudaMemcpy(devMb, mainBuf, sizeMb ,cudaMemcpyHostToDevice);

    
    printf("Loaded %ld signatures.\n", (sizeDb+sizeMb)/TAG_BYTES);

    if (Block_Dim_x * Block_Dim_y > 1024) {
        printf("Error, too many threads in block\n");
        exit (-1);
    }

    //loading files into file buffer
    loadFile(argv[1], &fileBuf, &sizeFb);
    
    //alloc mem for files on GPU
    cudaMalloc((void**)&devFb, sizeFb*sizeof(uint8_t)); 

    //cp mem from host to GPU
    cudaMemcpy(devFb, fileBuf, sizeFb ,cudaMemcpyHostToDevice);
    
    //declare GPU params
    dim3 Grid(Grid_Dim_x, Grid_Dim_y); //Grid structure
    dim3 Block(Block_Dim_x, Block_Dim_y); //Block structure
    
    cudaEventCreate(&start); // instrument code to measure start time
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    patternMatching<<<Grid, Block>>>(devDb, devMb, devFb, sizeDb/TAG_BYTES, sizeMb/TAG_BYTES, sizeFb);

    // make the host block until the device is finished with foo
    cudaThreadSynchronize();

    // check for error
    errorcode = cudaGetLastError();
    if(errorcode != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(errorcode));
        exit(-1);
    }

    cudaEventRecord(stop, 0); // instrument code to measure end time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop );
    
    printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms); // exec. time
    free(mainBuf);
    free(dailyBuf);
    free(fileBuf);
    cudaFree(devMb);
    cudaFree(devDb);
    cudaFree(devFb);
    return 0;
}
