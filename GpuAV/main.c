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
    attach("ent","a");
    system("ls /mnt/ent/");

    //GPU doing scaning here 
    printf("GPU is scanning......\n");
    sleep(5);
    //GPU done scanning this file system

    detach("ent","a");
    system("ls /mnt/ent/");
    
    /////////////////// 
    //do another VM
    ///////////////////
    attach("lv_vm_ubuntu","b");
    system("ls /mnt/lv_vm_ubuntu/");

    //GPU doing scaning here 
    printf("GPU is scanning......\n");
    sleep(5);
    //GPU done scanning this file system

    detach("lv_vm_ubuntu","b");
    system("ls /mnt/lv_vm_ubuntu/");
}

