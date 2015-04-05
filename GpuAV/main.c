#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

//function to load all VM files
void attach(char *lvName);
void detach(char *lvName);

int main(int argc, char **argv) {
    if(geteuid() != 0)
    {
        printf("sudo required\n");
        exit(1);
    }


    attach("ent");
    system("ls /mnt/ent/");

    //GPU doing scaning here 
    sleep(5);
    //GPU done scanning this file system

    detach("ent");
    system("ls /mnt/ent/");
    
    /////////////////// 
    //do another VM
    ///////////////////
}

