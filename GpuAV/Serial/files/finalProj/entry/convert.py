#!/usr/bin/python
import commands
import time 
import struct
import os
import array
import binascii

#number of meaningful bytes for GPU
PATTERN_BYTES = 10

#hardcoded virus pattern, obtained from clamav database
#I am doing this just to create a infected file
#to prove the detection function
VIRUS_PATTERN = "cf63e7726e22a79a6027"
 
def parse(ndb, gpusig, gpusigHu, gpuvirus):
    #sigs = []
    #virus = []
    f = open(ndb)
    fs = open(gpusig,'wb')
    fsHu = open(gpusigHu, 'wb')
    fv = open(gpuvirus,'w')
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
               next 
            #print sigs[idx]
            #print virus[idx]
            #time.sleep(1)
    fs.close()
    fv.close()
    fsHu.close()

def injectVirus (victimFile):
    victim = open(victimFile,'ab')
    virusPat = binascii.a2b_hex(VIRUS_PATTERN) 
    victim.write(virusPat)

def main():
    print "start converting for main"
    ndb = "./mainPack/main.ndb"
    gpusig = "./mainPack/mainGPUsig.bin"
    gpusigHu = "./mainPack/mainGPUsig.ndb"
    gpuvirus = "./mainGPUvirus.ndb"

    parse(ndb, gpusig, gpusigHu, gpuvirus)
    
    print "start converting for daily"
    ndb = "./dailyPack/daily.ndb"
    gpusig = "./dailyPack/dailyGPUsig.bin"
    gpusigHu = "./dailyPack/dailyGPUsig.ndb"
    gpuvirus = "./dailyGPUvirus.ndb"
    parse(ndb, gpusig, gpusigHu, gpuvirus)
    
    #victimFile = "./files/badGuy.bin" 
    #injectVirus(victimFile) 
main()

