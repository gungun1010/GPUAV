#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void attach(char *lvName, char *lvNum){
    
    //attach the VM LV to Dom0
    char cmd[1000];
    system("lvdisplay");
    printf("-----------------------------------------------\n");
    strcpy(cmd, "xl block-attach 0 phy:/dev/xubuntu-vg/");
    strcat(cmd, lvName);
    strcat(cmd, " xvd");
    strcat(cmd, lvNum);
    strcat(cmd, " w");

    printf("%s\n",cmd);
    system(cmd);
    memset(cmd, 0, strlen(cmd));

    //using fdisk to check if the attachment is successful
    strcpy(cmd, "fdisk -l /dev/xvd");
    strcat(cmd, lvNum);
    printf("%s\n",cmd); 
    system(cmd);
    memset(cmd, 0, strlen(cmd));

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

void detach(char *lvName, char *lvNum){
    char cmd[1000];

    //unmount the VM
    strcpy(cmd, "umount /mnt/");
    strcat(cmd, lvName);
    printf("%s\n",cmd);
    system(cmd);
    memset(cmd, 0, strlen(cmd));
    
    //remove mounting point
    strcpy(cmd, "rm -r ");
    strcat(cmd, lvName);
    strcat(cmd, "/");
     
    //detach from dom0
    printf("%s detaching from dom0\n",lvName);
    if(strcmp(lvNum, "a")){
        system("xl block-detach 0 51712");
    }else if(strcmp(lvNum, "b")){
        system("xl block-detach 0 51728");
    }
}
