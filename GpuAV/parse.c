#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void attach(char *lvName){
    
    //attach the VM LV to Dom0
    char cmd[1000];
    system("lvdisplay");
    printf("-----------------------------------------------");
    strcpy(cmd, "xl block-attach 0 phy:/dev/xubuntu-vg/");
    strcat(cmd, lvName);
    strcat(cmd, " xvda w");
    printf("%s\n",cmd);
    system(cmd);
    memset(cmd, 0, strlen(cmd));

    //using fdisk to check if the attachment is successful
    printf("fdisk -l /dev/xvda\n"); 
    system("fdisk -l /dev/xvda");

    //make a mounting point for the VM 
    strcpy(cmd, "mkdir /mnt/");
    strcat(cmd, lvName);

    printf("%s\n",cmd);
    system(cmd);
    memset(cmd, 0, strlen(cmd));
    
    // mount the VM
    strcpy(cmd, "mount /dev/xvda1 /mnt/");
    strcat(cmd, lvName);

    printf("%s\n",cmd);
    system(cmd);
    memset(cmd, 0, strlen(cmd));
    system("xl block-list 0");
}

void detach(char *lvName){
    char cmd[1000];

    //unmount the VM
    strcpy(cmd, "umount /mnt/");
    strcat(cmd, lvName);
    printf("%s\n",cmd);
    system(cmd);
    memset(cmd, 0, strlen(cmd));
    
    //detach from dom0
    system("xl block-detach 0 51712");
}
