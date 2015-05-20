//#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_amd_printf :enable

#define ALPHABET_LEN 256
#define PATTERN_LEN 30
#define NOT_FOUND patlen
#define max(a, b) ((a < b) ? b : a)
#define TAG_BYTES PATTERN_LEN
#define POSITIVE 1
#define NEGATIVE 0

//referenced 
//from http://en.wikipedia.org/wiki/Boyer–Moore_string_search_algorithm
//as the Boyer-Moore pattern matching is a well-established algorithm, 
void make_delta1(int *delta1, __global unsigned char *pat, int patlen) {

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
int is_prefix(__global unsigned char *word, int wordlen, int pos) {
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
int suffix_length(__global unsigned char *word, int wordlen, int pos) {
    int i;
    // increment suffix length i to the first mismatch or beginning
    // of the word
    for (i = 0; (word[pos-i] == word[wordlen-1-i]) && (i < pos); i++);
    return i;
}

//referenced 
//from http://en.wikipedia.org/wiki/Boyer–Moore_string_search_algorithm
//as the Boyer-Moore pattern matching is a well-established algorithm, 
void make_delta2(int *delta2, __global unsigned char *pat, int patlen) {
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
void boyer_moore (__global unsigned char *string, unsigned int stringlen, __global unsigned char *pat, unsigned int patlen, unsigned char *found) {
    int i;
    int delta1[ALPHABET_LEN];
    int delta2[PATTERN_LEN];
    int j;


    //FIXME, for variable length, PATTERN_LEN should be 500
    //delta2 should be 500 byte buffer
    //read last byte of delta2 to get the real length
    //patlen will be 
    make_delta1(delta1, pat, patlen);
    make_delta2(delta2, pat, patlen);
    
    // The empty pattern must be considered specially
    if (patlen == 0) return NEGATIVE;

    i = patlen-1;
    while (i < stringlen) {
        j = patlen-1;
        while (j >= 0 && (string[i] == pat[j])) {
            --i;
            --j;
        }
        if (j < 0) {
            *found= string[0];
            return; 
        }

        i += max(delta1[string[i]], delta2[j]);
    }
    *found= NEGATIVE;
}

__kernel void patternMatching(__global unsigned char *set1, __global unsigned char *fileBuf, int set1SigNum, int fileSize, __global unsigned char *devResultsRef){
    
    //note: blockDim.x = blockDim.y
    unsigned char found;
    int col = get_local_id(0) + get_local_size(0) * get_group_id(0);
    int row = get_local_id(1) + get_local_size(1) * get_group_id(1);
    int idx = row*get_num_groups(1)*get_local_size(1) + col; //GRID AND BLOCK are hardcoded for convenience
   
     
    //make sure that the idx is within the range of total number of signatures
    if(idx < set1SigNum){
    //if(idx == 0){
        *devResultsRef = 0;
        
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        boyer_moore(fileBuf,fileSize,set1+idx*TAG_BYTES,TAG_BYTES, &found);
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      
        if(found != NEGATIVE){
            *devResultsRef = found;
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}
