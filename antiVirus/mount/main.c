#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

//function to load all VM files
void attach(char *lvName, char *lvNum);
void detach(char *lvName, char *lvNum);

int main(int argc, char **argv) {
    if(geteuid() != 0)
    {
        printf("sudo required\n");
        exit(1);
    }


    /////////////////// 
    //do one VM
    ///////////////////
    attach("vm1","a");
    system("ls /mnt/vm1/");

    //GPU doing scaning here 
    printf("GPU is scanning......\n");
    /*
    sleep(5);
    //GPU done scanning this file system

    detach("vm1","a");
    system("ls /mnt/vm1/");
    */
    /////////////////// 
    //do another VM
    ///////////////////
    /*
    attach("vm2","b");
    system("ls /mnt/vm2/");

    //GPU doing scaning here 
    printf("GPU is scanning......\n");
    sleep(5);
    //GPU done scanning this file system

    detach("vm2","b");
    system("ls /mnt/vm2/");
    */
}

