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
VIRUS_PATTERN = "7468697320656d61696c20697320666f72206e6f74696669636174696f6e206f6e6c792e20746f20636f6e746163742075732c20706c65617365206c6f6720696e746f20796f7572206163636f756e7420616e642073656e6420612062616e6b206d61696c2e203c2f7072653e"
 
def parse(ndb, gpusig, gpusigHu, gpuvirus):
    #sigs = []
    #virus = []
    f = open(ndb)
    fs = open(gpusig,'wb')
    fsHu = open(gpusigHu, 'wb')
    fv = open(gpuvirus,'w')
    lines = f.readlines()
    f.close()
    

    for idx in range(len(lines)):
        #print idx
        info = lines[idx].split(":")
        #info[0] is virus's name
        #info[3] is virus signature
    
        #if(len(info[3]) > offset): 
        sigs = (info[3][:-1])
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
    gpusig = "./mainPack/mainCPUsig.bin"
    gpusigHu = "./mainPack/mainRCPUsig.ndb"
    gpuvirus = "./mainCPUvirus.ndb"

    parse(ndb, gpusig, gpusigHu, gpuvirus)
    
    print "start converting for daily"
    ndb = "./dailyPack/daily.ndb"
    gpusig = "./dailyPack/dailyCPUsig.bin"
    gpusigHu = "./dailyPack/dailyCPUsig.ndb"
    gpuvirus = "./dailyCPUvirus.ndb"
    parse(ndb, gpusig, gpusigHu, gpuvirus)
    
    victimFile = "../../files/badGuy.bin" 
    injectVirus(victimFile) 
main()

