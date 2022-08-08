import sys
import os
import numpy as np
import csv
import random

def readAnnotations(name):
    # camid: 2
    # filename: 3
    # spring, summer, fall, winter: 4-7
    # night, twilight, sunrise, sunset, day, fullday: 8-13

    season = {}
    season['spring'] = []
    season['summer'] = []
    season['fall'] = []
    season['winter'] = []
    
    tod = {}
    tod['night'] = []
    tod['twilight'] = []
    tod['sunrise'] = []
    tod['sunset'] = []
    tod['day'] = []
    tod['fullday'] = []

    with open(name) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = True
        for row in reader:
            if header:
                header=False
                continue
        
            imgfn = row[2] + '/' + row[3]
        
            if int(row[4])>0: season['spring'].append(imgfn)
            if int(row[5])>0: season['summer'].append(imgfn)
            if int(row[6])>0: season['fall'].append(imgfn)
            if int(row[7])>0: season['winter'].append(imgfn)
        
            if int(row[8])>0: tod['night'].append(imgfn)
            if int(row[9])>0: tod['twilight'].append(imgfn)
            if int(row[10])>0: tod['sunrise'].append(imgfn)
            if int(row[11])>0: tod['sunset'].append(imgfn)
            if int(row[12])>0: tod['day'].append(imgfn)
            if int(row[13])>0: tod['fullday'].append(imgfn)

    return (season,tod)
    

# main

if len(sys.argv)<5:
    print("usage: create_timm_skyfinder.py task annotationfile outputdir sourcedir valshare")
    print("       task: season or tod")
    exit()

task = sys.argv[1]
season,tod = readAnnotations(sys.argv[2])
outdir = sys.argv[3]
srcdir = sys.argv[4]
valshare = float(sys.argv[5])


# keep only full day, not day
del tod['day']


if task=='season': mydict = season
elif task=='tod': mydict = tod
else:
    print('unknown task '+task)
    exit()

    
# create directories
traindir = outdir + '/train'
valdir = outdir + '/val'
os.makedirs(traindir, exist_ok=True)    
os.makedirs(valdir, exist_ok=True)    


for c in mydict.keys():
    os.makedirs(traindir + '/'+ c, exist_ok=True)    
    os.makedirs(valdir + '/'+ c, exist_ok=True)    


for c in mydict.keys():
    items = mydict[c]
    random.shuffle(items)
    
    firstval = int(len(items)*(1-valshare))


    for i in range(firstval):
        trgname = items[i].replace('/','_')
        if os.path.exists(srcdir+'/'+items[i]):
            os.symlink(srcdir+'/'+items[i],traindir+'/'+c+'/'+trgname)
        else:
            print(imgsourcedir+'/'+items[i] + 'does not exist')
    for i in range(firstval,len(items)):
        trgname = items[i].replace('/','_')
        if os.path.exists(srcdir+'/'+items[i]):
            os.symlink(srcdir+'/'+items[i],valdir+'/'+c+'/'+trgname)
        else:
            print(imgsourcedir+'/'+items[i] + 'does not exist')      


