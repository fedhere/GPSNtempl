# myastrotools
# some useful astro functions

# not really "MY" astro toold either. mostly inheritade, as they propagate through generations of astronomers. couldnt tell who was the author of which function....

import datetime
# chomp

####################
def absmag(appmag,d,dunits='pc'):
    from numpy import log10
    if dunits == 'Mpc':
        d=d*1.0e6
    return appmag - 2.5 *log10( (d/10)**2 )


def chomp(ifile):
    lines = []
    for l in [l.strip() for l in open(ifile).readlines()]:
        if len(l)<1:
            continue
        lines.append(l)
    return lines

##################
# compare lists of ra,decl
#####################
def match_radeclists(data1, data2, tol):
    import math
    mdata = []
    for d1 in data1:
        x1 = d1[0]
        y1 = d1[1]
        for d2 in data2:
            x2 = d2[0]
            y2 = d2[1]
            print (x1,x2,y1,y2)
            dist = math.sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) )
            if dist < tol:
 #                print (x1, y1, mag1, x2, y2, mag2, dist/0.0002)
                mdata.append([dist, x1, y1, x2, y2])
        print (mdata)
    return mdata

##################
# print (offset)
#####################
def offsets(data1, data2):
    import math
    mdata = []
    for d1 in data1:
        x1 = d1[0]
        y1 = d1[1]

        for d2 in data2:
            x2 = d2[0]
            y2 = d2[1]
#            print (x1,x2,y1,y2,)
            offx= x1-x2
            offy = y1-y2
            #print (offx,offy)
            return offx,offy
##################
# ra in seg to deg
#####################
def raseg2deg(raseg):
    if type(raseg) is float:
        return raseg
    st = raseg.split(':') # hr, min sec
#    print 'ra2deg: st--> ', st    
    if len(st)<1:
        return -999.0
    
    return 15.0*(float(st[0]) + float(st[1])/60.0 + float(st[2])/3600.0)

#####################
# dec in seg to degree
#####################
def decseg2deg(decseg):
    if type(decseg) is float:
        return decseg
    decseg = decseg.replace(' ','')
    if not (decseg[0] == '+' or decseg[0] == '-'):
        decseg='+'+decseg
    st = decseg[1:].split(':') # deg, min, sec
#    print 'dec2deg: st--> ', st    
    if len(st)<1:
        return -999.0
    parity = decseg[0]

    decdeg = float(st[0])+ float(st[1])/60.0 + float(st[2])/3600.0 

    if parity == '-':
        decdeg *= -1.0

    return decdeg

##################
# ra in deg to seg
def radeg2seg(ra):
# ra
    ra /= 15.0
    try:
        rhr  = int(ra)
    except:
        rhr=list(map(int, ra))
    ra -= rhr
    ra *= 60.0
    try:
        rmn  = int(ra)
    except:
        rmn=list(map(int, ra))
    ra -= rmn
    ra *= 60.0
    rsc  = ra
    try:
        return(':'.join([str('%02d' % rhr), str('%02d' % rmn), '%02d' % (int(rsc)) + ('%.3f' % (rsc-int(rsc)))[-4:]]))
    except:
        newlist=[]
        for i,hr in enumerate(rhr):
            newlist.append(':'.join([str('%02d' % hr), str('%02d' % rmn[i]), '%02d' % (int(rsc[i])) + ('%.3f' % (rsc[i]-int(rsc[i])))[-4:]]))
        return(newlist)
##################
# dec in deg to seg
def decdeg2seg(dec):
# dec
    iamneg = 0
    try:
        if dec<0:
            iamneg = 1
            dec *= -1.0
        ddeg = int(dec)
        parity = '+'
        if iamneg==1:
            parity = '-'
    except:
        ddeg=list(map(int, dec))
        parity=['+']*len(ddeg)
        for i,d in enumerate(ddeg):
            if d<0:
                parity[i]='-'

    dec -= ddeg
    dec *= 60.0
    try:
        dmn = int(dec)
    except:
        dmn=list(map(int, dec))
    dec -= dmn
    dec *= 60.0
    dsc = dec
    try:
        return(parity + ':'.join([str(ddeg), str('%02d' % dmn), '%02d' % (int(dsc)) + ('%.2f' % (dsc-int(dsc)))[-3:]]))
    except:
        newlist=[]
        for i,dg in enumerate(ddeg):
            newlist.append('%s' % str(parity[i])+':'.join([str('%02d' % dg), str('%02d' % dmn[i]), '%02d' % (int(dsc[i])) + ('%.3f' % (dsc[i]-int(dsc[i])))[-4:]]))
        return(newlist)
##################
# list of ra, decl in seg to deg
#####################
def posseg2deg(pos):
    raseg = pos[0]
    decseg = pos[1]
    radeg = raseg2deg(raseg)
    decdeg = decseg2deg(decseg)
    ans = [radeg, raseg] 
    return(ans)

##################
# list of ra, decl in deg to seg
#####################
def posdeg2seg(pos):
    radeg = pos[0]
    decdeg = pos[1]
    raseg = radeg2seg(radeg)
    decseg = decdeg2seg(decdeg)
    ans = [raseg, decseg] 
    return(ans)

#######################
def gregorian_to_ut_mjd(date):
    d0 = datetime.datetime(1858, 11, 17)
    if type(date)==datetime.date:
        d0 = datetime.date(1858, 11, 17)
    date=date-d0

#    print date

 #hours/24.0+date.minuted/1440+(date.seconds)/86400.
    return date.days+ (date.seconds)/86400.



#######################
def jd2gdate(myjd):
    """Julian date to Gregorian calendar date and time of day.

    The input and output are for the proleptic Gregorian calendar.

    Parameters
    ----------
   myjd:
        julian date (float).

    Returns
    -------
    y, m, d, f : int, int, int, float
        Four element tuple containing year, month, day and the
        fractional part of the day in the Gregorian calendar. The first
        three are integers, and the last part is a float.

    """
    from math import modf

    jd_i = int(myjd)

    f = myjd-jd_i

    # Set JD to noon of the current date. Fractional part is the
    # fraction from midnight of the current date.
    if -0.5 < f < 0.5:
        f += 0.5
    elif f >= 0.5:
        jd_i += 1
        f -= 0.5
    elif f <= -0.5:
        jd_i -= 1
        f += 1.5

    l = jd_i + 68569
    n = int((4 * l) / 146097.0)
    l -= int(((146097 * n) + 3) / 4.0)
    i = int((4000 * (l + 1)) / 1461001)
    l -= int((1461 * i) / 4.0) - 31
    j = int((80 * l) / 2447.0)
    day = l - int((2447 * j) / 80.0)
    l = int(j / 11.0)
    month = j + 2 - (12 * l)
    year = 100 * (n - 49) + i + l

    return int(year), int(month), int(day), f



#######################
def get_mjdoff(dt):
    mjdoff = 60*60*dt.hour + 60*dt.minute + dt.second
    mjdoff /= 24.0*3600
    return mjdoff

#######################
def get_cur_epoch(pmjd):
    unow = datetime.datetime.utcnow()
    nmjd = gregorian_to_ut_mjd(unow)
    mjdoff = get_mjdoff(unow)
    nmjd += mjdoff
    if pmjd<0:
        return [nmjd, -1.0]
#    return nmjd-pmjd
    return [nmjd, '%.2f' % (nmjd-pmjd)]

# vacuum to air conversion from SDSS-III website
def vac2air(x):
    ''' vacuum to air conversion 
    as given on the SDSS-III website 
    x in Angstroms
    '''
    tmp = 1.0 +\
        2.735182e-4 +\
        131.4182/x**2 +\
        2.76249e8/x**4
    return x/tmp

# vacuum to air conversion from SDSS-III website
def indexOfRefraction_makee(x):
    ''' index of refraction at 0C as given by makee website
    x in Angstroms
    '''
    n = (2875.66 + 13.412/(x**2*1e-8) + 0.3777/(x**4*1e-16))*1e-7
    return n+1

def indexOfRefraction_morton(x):
    ''' index of refraction at 0C as given by Morton 1991
    x in Angstroms
    '''
    s = 1.0e4/x
    tmp = 6.4328e-5 + 2.94981e-2/(146-s**2) + 2.5540e-4/(41-s**2)
    return 1+tmp

def makeGaussian2d(sizex, fwhm = 20):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    from scipy import arange,newaxis, exp, log
    x=arange(sizex)
    y=x[:,newaxis]

    x0,y0=sizex/2,sizex/2

    g=exp(-4*log(2)*((x-x0)**2+(y-y0)**2)/fwhm**2)
    return g
#def indexOfRefraction_makee(x):
#    ''' index of refraction as given in makee website
#    '''
#   n = ((2875.66 + 13.412/(w**2*1e-8) + 0.3777/(w**4*1e-16))*1e-7)
#    return n+1
def print_timing(func):
    import time
    print ("timingit")
    print (func)
    def wrapper(*arg):
        t1 = time.time()
        res = func(*arg)
        t2 = time.time()
        print ('%s took %0.3f ms' % (func.func_name, (t2-t1)*1000.0))
        return res
    return wrapper
# declare the @ decorator just before the function, invokes print_timing()
