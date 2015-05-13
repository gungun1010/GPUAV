#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <stdint.h>
#include <inttypes.h>

#define ALPHABET_LEN 256
#define NOT_FOUND patlen
#define max(a, b) ((a < b) ? b : a)

const char *DAILY = "./dailyPack/dailyGPUsig.bin";
const char *MAIN = "./mainPack/mainGPUsig.bin";
const char *MOUNT = "/home/leon/GpuAV/serial/files";

//convert hexstring to len bytes of data
//returns 0 on success, -1 on error
//data is a buffer of at least len bytes
//hexstring is upper or lower case hexadecimal, NOT prepended with "0x"
int hex2data(uint8_t **buffer, const unsigned char *hexstring)
{
    unsigned const char *pos = hexstring;
    char *endptr;
    size_t count = 0;
    uint8_t data[26];
    unsigned int len;
    len = strlen(hexstring)/2;
    /* allocate memory for entire content */
    (*buffer) = (uint8_t *) calloc( len, sizeof(uint8_t) );
    if ((hexstring[0] == '\0') || (strlen(hexstring) % 2)) {
        //hexstring contains no data
        //or hexstring has an odd length
        return -1;
    }

    for(count = 0; count < len; count++) {
        char buf[5] = {'0', 'x', pos[0], pos[1], 0};
        (*buffer)[count] = strtol(buf, &endptr, 0);
        pos += 2 * sizeof(char);

        if (endptr[0] != '\0') {
            //non-hexadecimal character encountered
            return -1;
        }
    }

    return 0;

}

void remove_char_from_string(char c, char *str)
{
        int i=0;
        int len = strlen(str)+1;

        for(i=0; i<len; i++)
        {
            if(str[i] == c)
            {
                 // Move all the char following the char "c" by one to the left.
                 strncpy(&str[i],&str[i+1],len-i);
             }
        }
}
//referenced 
//from http://en.wikipedia.org/wiki/Boyer–Moore_string_search_algorithm
//as the Boyer-Moore pattern matching is a well-established algorithm, 
void make_delta1(int *delta1, uint8_t *pat, int32_t patlen) {

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
int is_prefix(uint8_t *word, int wordlen, int pos) {
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
int suffix_length(uint8_t *word, int wordlen, int pos) {
    int i;
    // increment suffix length i to the first mismatch or beginning
    // of the word
    for (i = 0; (word[pos-i] == word[wordlen-1-i]) && (i < pos); i++);
        return i;
}

//referenced 
//from http://en.wikipedia.org/wiki/Boyer–Moore_string_search_algorithm
//as the Boyer-Moore pattern matching is a well-established algorithm, 
void make_delta2(int *delta2, uint8_t *pat, int32_t patlen) {
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
uint8_t* boyer_moore (uint8_t *string, uint32_t stringlen, uint8_t *pat, uint32_t patlen) {
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

int main(int argc, char **argv)
{
    char cmd[1000];
    FILE * fp;
    FILE * sigDb;
    char * fileName = NULL;
    char * sigPattern = NULL;
    size_t len = 0;
    size_t sigLen = 0;
    ssize_t readFile;
    ssize_t readSig;   
    uint8_t *fileBuf;
    uint8_t *sigBuf;
    size_t sizeFb;
    uint8_t *found;
    int count;
    strcpy(cmd, "find ");
    strcat(cmd, MOUNT); 
    strcat(cmd, " -type f > filesToScan.txt");

    system(cmd);

    fp = fopen("filesToScan.txt", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    
    sigDb = fopen("mainCPUsig.ndb","r");

    while ((readFile = getline(&fileName, &len, fp)) != -1) {
        printf("scaning: %s", fileName);

        remove_char_from_string('\n',fileName);
        loadFile(fileName, &fileBuf, &sizeFb);
        
        sigDb = fopen("mainCPUsig.ndb","r");

        count = 0;
        while ((readSig = getline(&sigPattern, &sigLen, sigDb)) != -1){
            remove_char_from_string('\n',sigPattern);
            sigLen = strlen(sigPattern)/2;

            hex2data(&sigBuf, sigPattern);

            found = boyer_moore(fileBuf,sizeFb, sigBuf, sigLen);
            if(found != NULL){
                printf("    found virus in %s\n", fileName);
            }
            free(sigBuf);
            printf("    %d \r ", count); 
            count++;
        }

        fclose(sigDb);
        printf("\n");
        free(fileBuf);
    }
    //loadFile(argv[1], &fileBuf, &sizeFb);
}
