#!/usr/bin/python
import commands
import time 
import struct
import os
import array
import binascii

#number of meaningful bytes for GPU
PATTERN_BYTES = 30

#hardcoded virus pattern, obtained from clamav database
#I am doing this just to create a infected file
#to prove the detection function
VIRUS_PATTERN = "6265686176696f723a75726c282364656661756c7423616e63686f72636c"
 
def parse(ndb, gpusig, gpusigHu, gpuvirus):
    #sigs = []
    #virus = []
    f = open(ndb)
    fs = open(gpusig,'wb')
    fsHu = open(gpusigHu, 'wb')
    fv = open(gpuvirus,'w')
    count=0
    lines = f.readlines()
    f.close()
    
    offset = PATTERN_BYTES*2

    for idx in range(len(lines)):
        #print idx
        info = lines[idx].split(":")
        #info[0] is virus's name
        #info[3] is virus signature
    
        if(len(info[3]) > offset): 
            sigs = (info[3][:offset])
            virus = (info[0])
            #print sigs
            try:
                sigsBytes = binascii.a2b_hex(sigs)
                fs.write (sigsBytes)
                fsHu.write (sigs+os.linesep)
                fv.write (virus+os.linesep)
            except TypeError:
               count+=1
               next 
            #print sigs[idx]
            #print virus[idx]
            #time.sleep(1)
    fs.close()
    fv.close()
    fsHu.close()
    print "skipped "+str(count)+" signatures"

def injectVirus (victimFile):
    victim = open(victimFile,'ab')
    virusPat = binascii.a2b_hex(VIRUS_PATTERN) 
    victim.write(virusPat)

def main():
    print "start converting for main"
    ndb = "./mainPack/main5k.ndb"
    gpusig = "../GPU/mainGPUsig.bin"
    gpusigHu = "../GPU/mainGPUsig.ndb"
    gpuvirus = "../GPU/mainGPUvirus.ndb"

    parse(ndb, gpusig, gpusigHu, gpuvirus)
    
    
    victimFile = "badGuy.bin" 
    injectVirus(victimFile) 
main()

