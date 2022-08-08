import numpy as np
import pandas as pd
import ephem
import cv2
import skimage.restoration
import argparse
import matplotlib.pyplot as plt
import os
import random
import xml.etree.ElementTree as ET

dataroot = 'v:/Skyfinder'

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataroot', type=str, default='v:/Skyfinder', help='root for meradata and image data')
    parser.add_argument('--annotate', action='store_true', help='create image level annotations')
    parser.add_argument('--stats', action='store_true', help='determine dataset staistics')
    parser.add_argument('--noiseth', type=float, default=9999, help='keep images with noise estimate less than noiseth')	
    parser.add_argument('--timeth', type=int, default=9999, help='keep images with time difference less timeth')	
    parser.add_argument('--emptyth', type=int, default=1, help='if 0, disard empty images')	
    parser.add_argument('--sample_images', action='store_true', help='sample balanced sets')
    parser.add_argument('--min_height', type=int, default=300, help='Min height of extracted image (300 matches EfficientNet B3)')	
    parser.add_argument('--outdir', type=str, default='sampled', help='image output directoy (relative to dataroot)')
    parser.add_argument('--augment',action='store_true', help='option for sample_images: perform augmentation')
    parser.add_argument('--cvat',action='store_true', help='option for sample_images: write file copy script and XML snippets to patch annotations for CVAT to <outdir>')
    parser.add_argument('--oversample',type=float, help='option for sample_images: sample oversample times the target number of images')
    parser.add_argument('--update',type=str, help='update the specified annotation file from cvatxml')
    parser.add_argument('--cvatxml',type=str, help='CVAT output XML file to use as source for update')
	
	
    args = parser.parse_args()
	
    return args


def season(row):
    s = np.zeros((4,),dtype=int)
	
    if row['Month'] in [3,4,5]: s[0] = 1
    if row['Month'] in [6,7,8]: s[1] = 1
    if row['Month'] in [9,10,11]: s[2] = 1
    if row['Month'] in [12,1,2]: s[3] = 1

    offs = 0
    if row['Latitude']<0: offs = 2
    row['S_Spring'] = s[(0+offs) % 4]
    row['S_Summer'] = s[(1+offs) % 4]
    row['S_Fall'] = s[(2+offs) % 4]
    row['S_Winter'] = s[(3+offs) % 4]
	
    return row
	
def parse_datetime(ephemdt):
    
    dtstr = str(ephemdt)
	
    dtparts = dtstr.split(' ')
    dateparts = dtparts[0].split('/')
    timeparts = dtparts[1].split(':')
	
    return [ int(dateparts[0]), int(dateparts[1]), int(dateparts[2]), int(timeparts[0]), int(timeparts[1]) ]
	
# parse and convert to local time
def parse_datetime_local(ephemdt,utcoffset):

    dt = parse_datetime(ephemdt)

    m,d,hr = apply_time_offset(dt[1],dt[2],dt[3],utcoffset)	
    dt[1] = m
    dt[2] = d
    dt[3] = hr
  
    return dt  
	
	
def apply_time_offset(m,d,hr,utcoffset):

    hr = hr + utcoffset

    if hr<0:
        d = d-1
        hr = hr+24
		
        if d<1:
            m = m -1
            d = d +30
            if m in [1,3,5,7,8,10,12]:
                d = d+1
            elif m == 2:
                d = 28
            if m < 1:
                m = 12
                d = 31
		
    if hr>23:
        d = d+1
        hr = hr-24
		
        if m in [1,3,5,7,8,10,12]:
            if d>31:
                m = m+1
                d = 1				
        elif m == 2:
            if d>28:
                m = m+1
                d = 1
        elif d>30:
            m = m +1
            d = 1
        if m>12:
            m = 1
            d = 1	
			
    return m,d,hr
	
def timeofday(row):
    
	
    myplace = ephem.Observer()
    myplace.horizon = '0' # naval convention uses -0:34, 0 seems to be in line what ZAMG uses

    myplace.lat = str(row['Latitude'])
    myplace.lon = str(row['Longitude'])  
	
    day = int(row['Day'])
    month = int(row['Month'])
    year = int(row['Year'])
	
    # time in filename seems to be more precise than that in metadata for some images, but completely wrong for others
    timestamp = row['Filename'].split('.')[0].split('_')[-1]
    fnhour = int(timestamp[0:2])
    fnmin = int(timestamp[2:4])
	
	# thus we use provided time, but cross-check with filename
    utcOffs = int(row['Timezone'])
    #timestamp = row['Filename'].split('.')[0].split('_')[-1]
    #hour = timestamp[0:2]
    #hour = str(int(hour)).zfill(2)
    #min = timestamp[2:4]
    hour = str(row['Hour']).zfill(2)
	
	
    # adjust date if needed
    utcmonth,utcday,utchr = apply_time_offset(int(month),int(day),int(hour),-utcOffs)
    utchour = str(utchr).zfill(2)
    
    min = str(row['Min']).zfill(2)
	
    # store capture time stamp
    timestr = hour + ':' + min 
    if utcOffs<0:
        timestr = timestr + str(utcOffs)
    else:
        timestr = timestr + '+' + str(utcOffs)
		
    row['TimeStamp'] = timestr
	
	# estimate accuracy
    fntimemin = fnhour*60+fnmin
    timemin = int(utchour)*60+int(min)

    row['DQ_TimeDiffers'] = abs(timemin - fntimemin)
    # if we are over half a day, shift by one day
    if row['DQ_TimeDiffers']>720:
        row['DQ_TimeDiffers'] = 1440 - row['DQ_TimeDiffers']
	
    datestr = str(year).zfill(4)+'/'+str(utcmonth).zfill(2)+'/'+str(utcday).zfill(2)
    datetimestr = datestr + ' '+utchour+':'+min
	
    myplace.date = datetimestr
	
    sunrise = [-1,-1]
    sunset = [-1,-1]

    try: 
        prev_sunrise = parse_datetime_local(myplace.previous_rising(ephem.Sun()),utcOffs)
        next_sunrise = parse_datetime_local(myplace.next_rising(ephem.Sun()),utcOffs)

        if prev_sunrise[1]==month and prev_sunrise[2] == day: sunrise = [ prev_sunrise[3], prev_sunrise[4] ]
        if next_sunrise[1]==month and next_sunrise[2] == day: sunrise = [ next_sunrise[3], next_sunrise[4] ]
	
        prev_sunset = parse_datetime_local(myplace.previous_setting(ephem.Sun()),utcOffs)
        next_sunset = parse_datetime_local(myplace.next_setting(ephem.Sun()),utcOffs)
		
        if prev_sunset[1]==month and prev_sunset[2] == day: sunset = [ prev_sunset[3], prev_sunset[4] ]
        if next_sunset[1]==month and next_sunset[2] == day: sunset = [ next_sunset[3], next_sunset[4] ]
	
        # check if adjusting time provides a value
        datetimestr = datestr + ' '+utchour+':'+min+':01'
        myplace.date = datetimestr
        if sunrise[0] == -1:
            prev_sunrise = parse_datetime_local(myplace.previous_rising(ephem.Sun()),utcOffs)
            if prev_sunrise[1]==month and prev_sunrise[2] == day: sunrise = [ prev_sunrise[3], prev_sunrise[4] ]
        if sunset[0] == -1:
            prev_sunset = parse_datetime_local(myplace.previous_setting(ephem.Sun()),utcOffs)
            if prev_sunset[1]==month and prev_sunset[2] == day: sunset = [ prev_sunset[3], prev_sunset[4] ]
		
    except ephem.CircumpolarError:
        print('warning: horizon not crossed')
	
	
    #print(str(day)+'.'+str(month)+'. '+hour+':'+min+' '+str(myplace.lat)+' '+str(myplace.lon)+' '+str(row['CamId']))
    #print(str(utcday)+'.'+str(utcmonth)+'. '+utchour+':'+min)
    #print(sunrise)
    #print(sunset)
	
    # polar day or night
    if sunrise[0]==-1 or sunset[0]==-1:

        # ask midday altitude
        myplace.date = datestr + ' ' + str(12-utcOffs)+':00'
        sunpos = str(ephem.Sun(myplace))
        
        sunposdeg = int(sunpos.split(':')[0])
        if sunposdeg<0:
            df['D_Night'] = 1
        else:
            df['D_FullDaylight'] = 1		
		
    else:
        # jump to midday
        midday_min = int(sunrise[0]*60+sunrise[1]+0.5*(sunset[0]*60+sunset[1]-(sunrise[0]*60+sunrise[1])))
        midday = [ int(midday_min/60.0), midday_min % 60 ]
        m,d,utcMdHr = apply_time_offset(int(month),int(day),int(hour),-utcOffs)
        datetimestr = datestr + ' '+str(utcMdHr)+':'+str(midday[1])
        # civil dawn and dusk (-6°)
        myplace.horizon = '-6'
        try:
            civil_dawn = parse_datetime_local(myplace.previous_rising(ephem.Sun(), use_center=True),utcOffs)
            civil_dusk = parse_datetime_local(myplace.next_setting(ephem.Sun(), use_center=True),utcOffs)        
        except ephem.CircumpolarError:
            # no civil dawn/dusk means twilight instead of night
            civil_dawn = [int(year),int(month),int(day),int(0),int(0)]
            civil_dusk = [int(year),int(month),int(day),int(23),int(60)]
				
        # assume sunrise/sunset safety margin (+3°)        
        myplace.horizon = '+3'
        sunrise_end = parse_datetime_local(myplace.previous_rising(ephem.Sun(), use_center=True),utcOffs)
        sunset_start = parse_datetime_local(myplace.next_setting(ephem.Sun(), use_center=True),utcOffs)        

        tmin = int(hour)*60+int(min)
	
        if tmin<civil_dawn[3]*60+civil_dawn[4]: 
            row['D_Night'] = 1
        elif tmin<sunrise[0]*60+sunrise[1]:
            row['D_Twilight'] = 1
        elif tmin<sunrise_end[3]*60+sunrise_end[4]: 
            row['D_Sunrise'] = 1
        elif tmin<sunset_start[3]*60+sunset_start[4]: 
            row['D_Day'] = 1		
        elif tmin<sunset[0]*60+sunset[1]:
            row['D_Sunset'] = 1
        elif tmin<civil_dusk[3]*60+civil_dusk[4]: 
            row['D_Twilight'] = 1
        else:  
            row['D_Night'] = 1		
			
    if int(row['D_Day'])>0 or int(row['D_Sunrise'])>0 or int(row['D_Sunset'])>0:
        row['D_FullDaylight'] = 1
		
    return row
	
	
def skyregion(row,dataroot,maskdatacache):

    camid = str(row['CamId'])
    if camid in maskdatacache.keys():
        row['I_Width'] = maskdatacache[camid][0]
        row['I_Height'] = maskdatacache[camid][1]
        row['I_LowestSky'] = maskdatacache[camid][2]
        row['I_Low09Sky'] = maskdatacache[camid][3]
        
        return row
	
    maskname = dataroot + '/skyfinder_masks/'+camid+'.png'

	
    mask = cv2.imread(maskname, cv2.IMREAD_GRAYSCALE)
	
    row['I_Width'] = mask.shape[1]
    row['I_Height'] = mask.shape[0]
	
    maskdatacache[camid] = [ mask.shape[1], mask.shape[0], 0, 0 ]
	
    lowestvalues = []
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]-1,0,-1):
            if mask[j,i]>0:
                lowestvalues.append(j+1)
                break
				
    if (len(lowestvalues)==0): lowestvalues = [ mask.shape[0]-1 ]
				
    lowestvalues = np.array(lowestvalues)
    row['I_LowestSky'] = np.max(lowestvalues)
    row['I_Low09Sky'] = int(np.quantile(lowestvalues,0.9))
	
    maskdatacache[camid][2] = row['I_LowestSky']
    maskdatacache[camid][3] = row['I_Low09Sky']
	
    return row
	
	
def quality(row,dataroot):

    camid = str(row['CamId'])
    maskname = dataroot + '/skyfinder_masks/'+camid+'.png'

    imagename = dataroot + '/'+str(camid)+'/'+str(row['Filename'])

    qscores = assess_quality(imagename,maskname)
	
    row['IQ_Noise'] = qscores[0]
    row['IQ_Empty'] = qscores[1]
	
    return row
	
def augment(row,dfadditional,minH,outdir):

    # add more if sky > 15%
    skyratio = row['I_LowestSky'] / row['I_Height']
	
    os.makedirs(os.path.join(dataroot,outdir,str(row['CamId'])), exist_ok=True)
	
    if skyratio<0.15:
        return
		
    # check if there is enough room for a different sample
    if minH > row['I_Height']*.75:
        return
	
    # determine max height from image width and expected aspect ratio
    ar = (9.0/16.0)
    maxH = int(row['I_Width'] * ar)
	
    # sample with 10% sky region differences
    i = 0.1
    while i<skyratio:
	
        startH = int(i*row['I_Height'])
	
        # ensure that sky ratio does not decrease
        delta_skyratio = skyratio - i 
        thisMinH = int (2*delta_skyratio * row['I_Width'] )
        if (thisMinH<minH): thisMinH = minH

		
        if (int(row['I_Height'] - startH)>thisMinH) and (thisMinH<=maxH):
	
	
            h = random.randint(thisMinH, maxH)
            w = int(h*(1/ar))
            if w>row['I_Height']:
                w = row['I_Height']
				
            dw = row['I_Height']-w
				
            print(dw)
            startW = random.randint(0, dw)
			
				
            img = cv2.imread(dataroot+'/'+str(row['CamId'])+'/'+row['Filename'])
       	    cropped_image = img[startH:startH+h-1, startW:startW+w-1]
	
            newrow = row.copy()
	    
            fnparts = row['Filename'].split('.')
            newrow['Filename'] = fnparts[0]+'_'+"{:1.2f}".format(i)+'.'+fnparts[1]
			
            print('creating image '+str(newrow['CamId'])+'/'+newrow['Filename'])
            cv2.imwrite(dataroot+'/'+outdir+'/'+str(row['CamId'])+'/'+newrow['Filename'], cropped_image)
		
            dfadditional.append(newrow)
		
        i = i + 0.1
	
def unique_fn(row):

    row['UniqueFn'] = str(row['CamId']) + '_' + row['Filename']
	
    return row
	
def cvat_annot(row,trgdir):
    rowxml = '<image id="' + str(row.name) + '" name="' + row['UniqueFn'] + '" ><tag label="'
	
    trgname = trgdir.split(os.path.sep)[-1]
    
	
    if trgname=='season':
        if row['S_Spring']: rowxml = rowxml + "Spring"
        if row['S_Summer']: rowxml = rowxml + "Summer"
        if row['S_Fall']: rowxml = rowxml + "Fall"
        if row['S_Winter']: rowxml = rowxml + "Winter"

    if trgname=='tod':
        if row['D_Night']: rowxml = rowxml + "Night"
        if row['D_Twilight']: rowxml = rowxml + "Twilight"
        if row['D_Sunrise']: rowxml = rowxml + "Sunrise"
        if row['D_Sunset']: rowxml = rowxml + "Sunset"
        if row['D_Day']: rowxml = rowxml + "FullDaylight"
	
    rowxml = rowxml + '" source="manual"></tag></image>'
	
    row['cvat_xml'] = rowxml
	
    rowcopy = 'copy '+ str(row['CamId']) + os.path.sep + str(row['Filename']) + ' ' + trgdir + os.path.sep + row['UniqueFn']
	
    row['cvat_copy'] = rowcopy
	
    return row	
	
def update_from_xml(row,root):

    fullfilename = str(row['CamId']) + '_' + row['Filename']

    found = False
	
    for imgel in root.iter('image'):
        fn = imgel.attrib['name'].split('/')[-1]
				
        if fullfilename==fn:
            found = True
            labels = []
            for labelel in imgel.iter('tag'):
                labels.append(labelel.attrib['label'])
				
            if ('Spring' in labels) or ('Summer' in labels) or ('Fall' in labels) or ('Winter' in labels) or len(labels)==0:
                row['S_Spring'] = 0
                row['S_Summer'] = 0
                row['S_Fall'] = 0
                row['S_Winter'] = 0

            if ('Night' in labels) or ('Twilight' in labels) or ('Sunset' in labels) or ('Sunrise' in labels) or ('FullDaylight' in labels) or len(labels)==0:

                row['D_Night'] = 0
                row['D_Twilight'] = 0 
                row['D_Sunrise'] = 0
                row['D_Sunset'] = 0
                row['D_FullDaylight'] = 0
                row['D_Day'] = 0
    
            if 'Spring' in labels: row['S_Spring'] = 1
            if 'Summer' in labels: row['S_Summer'] = 1
            if 'Fall' in labels: row['S_Fall'] = 1
            if 'Winter' in labels: row['S_Winter'] = 1

            if 'Night' in labels: row['D_Night'] = 1
            if 'Twilight' in labels: row['D_Twilight'] = 1
            if 'Sunset' in labels: row['D_Sunset'] = 1
            if 'Sunrise' in labels: row['D_Sunrise'] = 1
            if 'FullDaylight' in labels: row['D_FullDaylight'] = 1
            if row['D_Sunset'] or row['D_Sunrise'] or row['D_FullDaylight']: row['D_Day'] = 1
			
            break
			
    if not(found):
        row['S_Spring'] = 0
        row['S_Summer'] = 0
        row['S_Fall'] = 0
        row['S_Winter'] = 0

        row['D_Night'] = 0
        row['D_Twilight'] = 0 
        row['D_Sunrise'] = 0
        row['D_Sunset'] = 0
        row['D_FullDaylight'] = 0
        row['D_Day'] = 0       
			
    return row
			
def write_debug_html(df,basepath,filename,nItems):
    
    dfsample = df.sample(nItems)
	
    html = "<html><head>Annotations</head><title>Annotations</title><body><table>"	
	
    for index, row in dfsample.iterrows(): 

        html = html + "<tr><td><img src='"+ str(row['CamId']) + "/" + row['Filename'] +"' height='100'/></td><td>"+str(row['Latitude'])+"</td><td>"+str(row['Month'])+'-'+str(row['Day'])+' '+str(row['Hour'])+':'+str(row['Min'])+"</td><td>" 
	
        if row['S_Spring']: html = html + "Spring "
        if row['S_Summer']: html = html + "Summer "
        if row['S_Fall']: html = html + "Fall "
        if row['S_Winter']: html = html + "Winter "

        if row['D_Night']: html = html + "Night "
        if row['D_Twilight']: html = html + "Twilight "
        if row['D_Sunrise']: html = html + "Sunrise "
        if row['D_Sunset']: html = html + "Sunset "
        if row['D_Day']: html = html + "Day "

        html = html + "</td><td>"		

        html = html + "Noise="+str(row['IQ_Noise'])+" "
        if row['DQ_TimeDiffers']>0: 
             html = html +"TimeDiff="+str(row['DQ_TimeDiffers'])+" "
        if row['IQ_Empty']: html = html + "Empty "
		
        html = html + "</td></tr>\n"
    
    html = html + "</table></body></html>"
	
    with open(filename, 'w') as f:
        f.write(str(html))
    
def assess_quality(imagename,maskname):

    img = cv2.imread(imagename)
    mask = cv2.imread(maskname, cv2.IMREAD_GRAYSCALE)
	
    # noise
	
    # split into 8x8 patches
    patchW = int(img.shape[1]/8)
    patchH = int(img.shape[0]/8)
	
    scores = []
	
    for i in range(8):
        for j in range(8):
            patch = img[i*patchH:(i+1)*patchH,j*patchW:(j+1)*patchW,:]	
            patchmask = mask[i*patchH:(i+1)*patchH,j*patchW:(j+1)*patchW]	
	
            if np.sum(patchmask)<0.8*patchW*patchH: continue

            pscores = skimage.restoration.estimate_sigma(patch, average_sigmas=False, multichannel=True)# note: after v0.19 use channel_axis=2)
            scores.extend(pscores)

    # no sky
    if len(scores)==0:
        scores = skimage.restoration.estimate_sigma(img, average_sigmas=False, multichannel=True)# note: after v0.19 use channel_axis=2)
			
    score = np.median(scores)
    
    if np.sum(np.isnan(score))>0: score = 0
	
    # empty image
    empty = 0	

    npixels = img.shape[0]*img.shape[1]
	
    nMore50 = 0
    for i in range(img.shape[2]):
        hist = cv2.calcHist([img],[i],None,[256],[0,256])
        if max(hist)>npixels*0.5:
            nMore50 += 1
	
    if nMore50==3:
        empty = 1
		
    return [score,empty]

	
def gen_cvat(mydf,outdir,name):

    # add new filename
    mydf = mydf.apply(lambda r: unique_fn(r), axis=1 )
	
    # sort alphabetically
    dfsorted = mydf.sort_values('UniqueFn', axis=0) 
    dfsorted = dfsorted.reset_index(drop=True)

    # create copy and XML
    dfsorted = dfsorted.apply(lambda r: cvat_annot(r,args.outdir+os.path.sep+name), axis=1 ) 
		
    xmlsnippet = dfsorted['cvat_xml'].astype(str)
    xmlsnippet = '\n'.join(xmlsnippet)
	
    with open(outdir+os.path.sep+name+'_cvat.xml', "w") as text_file:
        text_file.write(xmlsnippet)
	
    copyscript = dfsorted['cvat_copy'].astype(str)
    copyscript = '\n'.join(copyscript)
	
    with open(outdir+os.path.sep+name+'_copy.bat', "w") as text_file:
        text_file.write(copyscript)    
	
def sample_balanced(mydf,classlist,number):

    resultdf = pd.DataFrame(columns = mydf.columns.values.tolist() )

    for c in classlist: 

        mydfcl = mydf.loc[mydf[c]==1]
		
        n_to_sample = min(int(number), mydfcl.shape[0])
        
        mydfcls = mydfcl.sample(n_to_sample)

        resultdf = resultdf.append(mydfcls, ignore_index=True)
		 
    resultdf = resultdf.drop_duplicates(subset=['CamId', 'Filename'])
		 
    return resultdf
		
#	
# MAIN
#
	
args = parse_args()
dataroot = args.dataroot

	
if args.annotate:
	
    df = pd.read_csv(dataroot+'/complete_table_with_mcr.csv')

    # DEBUG: limit to s smaller number of samples
    #df = df.sample(100)

    # add new columns
    df['S_Spring'] = np.zeros((len(df.index),),dtype=int)
    df['S_Summer'] = np.zeros((len(df.index),),dtype=int)
    df['S_Fall'] = np.zeros((len(df.index),),dtype=int)
    df['S_Winter'] = np.zeros((len(df.index),),dtype=int)

    df['D_Night'] = np.zeros((len(df.index),),dtype=int)
    df['D_Twilight'] = np.zeros((len(df.index),),dtype=int)
    df['D_Sunrise'] = np.zeros((len(df.index),),dtype=int)
    df['D_Sunset'] = np.zeros((len(df.index),),dtype=int)
    df['D_Day'] = np.zeros((len(df.index),),dtype=int)
    df['D_FullDaylight'] = np.zeros((len(df.index),),dtype=int)

    df['I_Height'] = np.zeros((len(df.index),),dtype=int)
    df['I_Width'] = np.zeros((len(df.index),),dtype=int)

    df['I_LowestSky'] = np.zeros((len(df.index),),dtype=int)
    df['I_Low09Sky'] = np.zeros((len(df.index),),dtype=int)


    df['IQ_Empty'] = np.zeros((len(df.index),),dtype=int)
    df['IQ_Noise'] = np.zeros((len(df.index),),dtype=int)

    df['DQ_TimeDiffers'] = np.zeros((len(df.index),),dtype=int)
    df['TimeStamp'] = np.zeros((len(df.index),),dtype=str)


    # apply seasons
    df = df.apply(season, axis=1)

    # apply time of day
    df = df.apply(timeofday, axis=1)
	 
    # determine image size and sky region
    maskdatacache = {}
    df = df.apply(lambda r: skyregion(r,dataroot,maskdatacache), axis=1 )

    # assess quality
    df = df.apply(lambda r: quality(r,dataroot), axis=1 )
	 
    df.to_csv(path_or_buf=dataroot+'/skyfinder_annotations.csv', columns=['CamId','Filename','S_Spring','S_Summer','S_Fall','S_Winter',
        'D_Night','D_Twilight','D_Sunrise','D_Sunset','D_Day','D_FullDaylight','I_Height','I_Width','I_LowestSky','I_Low09Sky','TimeStamp','DQ_TimeDiffers','IQ_Empty','IQ_Noise',
        'Conds','Fog','Rain','Snow','Hail','Thunder'])
		
    write_debug_html(df,dataroot,dataroot+'/skyfinder_annotations_dbg.html',20)
	
elif args.stats:
    df = pd.read_csv(dataroot+'/skyfinder_annotations.csv')
	
    # apply filters 
    df = df[df['IQ_Empty']<=args.emptyth]
    df = df[df['IQ_Noise']<=args.noiseth]
    df = df[df['DQ_TimeDiffers']<=args.timeth]
    
	
    stats = {}
	
    for timeofday in ['D_Night','D_Twilight','D_Sunrise','D_Sunset','D_FullDaylight']:
        stats[timeofday] = {}
        total = 0
        for season in ['S_Spring','S_Summer','S_Fall','S_Winter']:
            idx = (df[timeofday]==1) & (df[season]==1)
            stats[timeofday][season] = np.sum(idx)
            total += np.sum(idx)
        stats[timeofday]['Total']=total
	
    stats['Total'] = {}
    for season in ['S_Spring','S_Summer','S_Fall','S_Winter']:
        total = 0
        for timeofday in ['D_Night','D_Twilight','D_Sunrise','D_Sunset','D_FullDaylight']:
            total += stats[timeofday][season]
        stats['Total'][season] = total
	
    print(stats)
	
    # determine sky percentages (using 0.9 quantile)
    skyfrac = np.divide(df['I_Low09Sky'], df['I_Height'])

    plt.hist(skyfrac, bins = 20)
    plt.show()
	
elif args.sample_images:

    random.seed(42)
    # set also numpy seed for pandas
    np.random.seed(42)

    outdir = args.outdir
    minH = args.min_height
    n_oversample = 1
    if args.oversample is not None:
        n_oversample = args.oversample
    
    os.makedirs(os.path.join(dataroot,outdir), exist_ok=True)
	
    df = pd.read_csv(dataroot+'/skyfinder_annotations.csv')
	
    # apply filters 
    df = df[df['IQ_Empty']<=args.emptyth]
    df = df[df['IQ_Noise']<=args.noiseth]
    df = df[df['DQ_TimeDiffers']<=args.timeth]

    if args.augment:
        # container for additional images
        dfadditional = pd.DataFrame(columns = df.columns)
	
        df.apply(lambda r: augment(r,dfadditional,minH,outdir), axis=1 )
	
        df = df.append(dfadditional, ignore_index=True)
	
        df = pd.concat([df,dfadditional])
	 
    dfseason = df.copy()
    seasoncounts = [ np.sum(df['S_Spring']), np.sum(df['S_Summer']), np.sum(df['S_Fall']), np.sum(df['S_Winter']) ]
    dfseason = sample_balanced(dfseason,['S_Spring','S_Summer','S_Fall','S_Winter'],n_oversample * min(seasoncounts))
    dfseason.to_csv(path_or_buf=dataroot+'/skyfinder_sampled_season.csv')
	
    if args.cvat:
        gen_cvat(dfseason,dataroot+'/'+outdir,'season')
	
    dftod = df.copy()
    todcounts = [ np.sum(df['D_Night']), np.sum(df['D_Twilight']), np.sum(df['D_Sunrise']), np.sum(df['D_Sunset']), np.sum(df['D_FullDaylight']) ]
    dftod = sample_balanced(dftod,['D_Night','D_Twilight','D_Sunrise','D_Sunset','D_FullDaylight'],n_oversample * min(todcounts))
    dftod.to_csv(path_or_buf=dataroot+'/skyfinder_sampled_tod.csv')
	
    if args.cvat:
        gen_cvat(dftod,dataroot+os.path.sep+outdir,'tod')
		
elif args.update:
		
    outdir = args.outdir
    cvatxml = args.cvatxml
    
    os.makedirs(os.path.join(dataroot,outdir), exist_ok=True)
	
    df = pd.read_csv(os.path.join(dataroot,args.update))

    xmltree = ET.parse(os.path.join(dataroot,cvatxml))
    rootnode = xmltree.getroot()	
	
    df = df.apply(lambda r: update_from_xml(r,rootnode), axis=1 ) 

    outfn = '.'.join(args.update.split('.')[:-1])
    outfn = outfn.split('/')[-1]
    outfn = outfn.split('\\')[-1]
    outfn = outfn + '_updated.csv'
	
    df.to_csv(path_or_buf=dataroot+'/'+outdir+'/'+outfn)
		
	
else:
    print('unsupported action')
    exit(0)
	
	
	