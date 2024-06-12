import pylab as pl
import numpy as np



def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)

def myopenfig(fignumber, xlim, ylim=(23,10)):
      figcounter=1
      fig = pl.figure(fignumber)
      pl.ylim(ylim)
      return fig

def myplot_txtcolumn(x,y,dy,labels,myfig):
    for label in labels:
#          print label
          pl.text(x,y, label, ha='left', fontsize=10,transform=myfig.transFigure) 
          y=y-dy
        

def myplot_setlabel(xlabel=None, ylabel=None, title=None,
                    label=None, xy=(0,0), ax=None,
                    labsize=15, rightticks=False):
    import matplotlib as mpl
    mpl.rcParams['font.size'] = labsize+0.
#    mpl.rcParams['font.family'] = 'serif'# New Roman'
    mpl.rcParams['font.serif'] = 'Bitstream Vera Serif'

    mpl.rcParams['axes.labelsize'] = labsize+1.
    mpl.rcParams['xtick.labelsize'] = labsize+0.
    mpl.rcParams['ytick.labelsize'] = labsize+0.

    if label:
        print ("######################## LABELS HERE##########################")
        
        if xy==(0, 0):
            xy=(0.22, 0.20)
        if not ax:
            print ("WARNING: no axix, cannot place label")
            pl.figtext(xy[0], xy[1], label, fontsize=labsize)
            pl.xlabel(xlabel, fontsize=labsize+1)
            pl.ylabel(ylabel, fontsize=labsize+1)            
        else:
            pl.text(xy[0], xy[1], label, transform=ax.transAxes, fontsize=labsize)
            if rightticks:
                ax.yaxis.tick_right()
            ax.set_xlabel(xlabel, fontsize=labsize+1)
            ax.set_ylabel(ylabel, fontsize=labsize+1)
    if title:
        pl.title(title)

def myplot_err(x, y, yerr=None, xerr=None, xlim=None, ylim=None,
               symbol=None, alpha=0.5, offset=0, fig = None, fcolor=None,
               ms=7, settopx=False, markeredgewidth=1, litsn=None, ax=None):

#    print "litsn", litsn
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 150 
    if symbol:
        color=symbol[:-1]
        marker=symbol[-1]
    else:
        color = 'blue'
        marker='o'
        fcolor=color
    if not fig:
        fig = pl.figure()
        ax = pl.add_subplot(111)
    #print fig
#    print "from myplot_err:",symbol,xlim,ylim
    if xlim :
        ax.set_xlim(xlim)
    if ylim :
        ax.set_ylim(ylim)
    #print xlim,ylim
#,x
    if not isinstance(litsn, np.ndarray):
        litsn=[]
    #print "litsn", litsn

    
    if not symbol :
        symbol='ko'
    if yerr is not None:
          #print color
          ax.errorbar(x, np.asarray(y) + offset, yerr=yerr, xerr=xerr,
                      fmt=None, color=color,
                      ecolor=color, alpha=alpha, markeredgewidth=1.0)

    elif xerr is not None:
          ax.errorbar(x, np.asarray(y) + offset, yerr=yerr, xerr=xerr,
                      fmt=None, color=color,
                      ecolor=color, alpha=alpha, markeredgewidth=1.0)

    thisp = ax.plot(x, np.asarray(y) + offset, ".", marker=marker,
                    alpha=alpha, markersize=ms,
                    markerfacecolor=fcolor, mec=color,
                    markeredgewidth=markeredgewidth)
#    print litsn

    ax.plot(x[litsn], np.asarray(y[litsn]) + offset, ".",
            marker=marker, alpha=alpha, markersize=ms,
            markerfacecolor=fcolor, mec='slategrey',
            markeredgewidth=1)
    return thisp
    #return #pl.plot(x,y+offset,symbol,alpha=alpha, markersize=8)

def myplot_hist(x, y, xlim=None, ylim=None, symbol=None, alpha=1, offset=0,
                fig = None, fcolor=None, ax=None, nbins=None):

    if symbol:
        color=symbol[:-1]
        marker=symbol[-1]
        if not fcolor:
            fcolor=color
    else:
        color = 'blue'
        marker='o'
        fcolor=color
    if fig:
        pl.figure(fig)
#    print "from myplot_err:",symbol,xlim,ylim
    if xlim :
        pl.xlim(xlim)
    if ylim :
        pl.ylim(ylim)
    #print xlim,ylim,x,y
    if not symbol :
        symbol='ko'
    #print nbins
    if nbins is None:
        nbins=int((xlim[1]-xlim[0])/10)
    else:
        nbins=int((xlim[1]-xlim[0])/nbins)
    print (nbins)
    print (xlim[0],xlim[1], np.asarray(x),np.asarray(y),nbins,ax)
    print ("\n\n\n")
    
    X,Y,Ystd=binMeanStdPlot(np.asarray(x),np.asarray(y),numBins=nbins,xmin=xlim[0],xmax=xlim[1], ax=ax)
    
    pl.errorbar(X,Y+offset, yerr=Ystd, fmt=None,color=color, ecolor=color,alpha=alpha)
    pl.errorbar(X,Y+offset, yerr=Ystd, fmt='.',color=color, ecolor=color,alpha=alpha)
    try:
        binsz=X[1]-X[0]
    except:
        binsz=0
    newx=[]
    newy=[]
    newx=[newx+[x,x] for x in X-binsz/2]
    newx =np.asarray(newx).flatten()[1:]

    newy=[newy+[y,y] for y in (Y)]
    newy=np.asarray(newy).flatten()
    newx=np.insert(newx,len(newx),newx[-1]+binsz)
    return pl.step(newx,np.asarray(newy)+offset,"",alpha=alpha,color=color)

    #return #pl.plot(x,y+offset,symbol,alpha=alpha, markersize=8)

def myplotarrow(x,y,label, dx=0,dy=+0.3, color='k'):
#    ax.annotate(label, xy=(x, y), xytext=(x+dx, y+dy),
#                arrowprops=dict(facecolor=color, shrink=0.05),)
    pl.arrow(x,y, dx ,dy, length_includes_head=True, color=color)
    pl.text(x,y+(-2.0*dy), label, ha='center', fontsize=10, color=color)#,transform=myfig.transFigure) 
#    pl.arrow(maxjd+15, maxflux-1, 0, +0.5, length_includes_head=True)


def binMeanPlot(X,Y,numBins=8,xmin=None,xmax=None, binsize=None):
    if xmin is None:
        xmin = X.min()
    if xmax is None:
        xmax = X.max()
    if not binsize is None:
        numBins=int((X.max()-X.min())/binsize)
    bins = np.linspace(xmin,xmax,numBins+1)
#    print bins,Y

    YY = np.array([nanmean(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])
    YYmedian = np.array([nanmedian(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])
    YYstd = np.array([np.std(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])
    return bins[:-1]+(bins[1]-bins[0])*0.5,YY,YYmedian,YYstd
 
def binMeanStdPlot(X,Y,ax=None,numBins=8,xmin=None,xmax=None, binsize=None):
    if xmin is None:
        xmin = X.min()
    if xmax is None:
        xmax = X.max()
    if binsize:
        numBins=int((xmax-xmin)/binsize)

    bins = np.linspace(xmin,xmax,numBins+1)
    XX = np.array([np.mean((bins[binInd], bins[binInd+1])) for binInd in range(numBins)])
    YY = np.array([np.mean(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])
    #XX[np.isnan(YY)]=np.nan
    YYstd = np.array([np.std(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])
    return XX, YY, YYstd

def binWMeanStdPlot(X,Y,std,ax=None,numBins=8,xmin=None,xmax=None, binsize=None):
    if xmin is None:
        xmin = X.min()
    if xmax is None:
        xmax = X.max()
    if binsize:
        numBins=int(((xmax-xmin)/binsize)+0.5)
    else:
        binsize=(xmin-xmax)/numBins
    bins = np.arange(xmin,xmax+binsize,binsize)
    
    XX=[]
    YY=[]
    for binInd in range(numBins):
        XX.append(np.mean((bins[binInd], bins[binInd+1])) )
        thisY=Y[(X > bins[binInd]) & (X <= bins[binInd+1])]
        print ("thislen",XX[-1],len(thisY))
        if len(thisY)>0:            
            YY.append(np.average(thisY, weights=1.0/(np.array(std[(X > bins[binInd]) & (X <= bins[binInd+1])]))**2))
            print (thisY,std[(X > bins[binInd]) & (X <= bins[binInd+1])],
                   1.0/(np.array(std[(X > bins[binInd]) & (X <= bins[binInd+1])]))**2,
                   np.average(thisY, weights=1.0/np.array(std[(X > bins[binInd]) & (X <= bins[binInd+1])])**2),
                   np.average(thisY))

        else:
            YY.append(float('nan'))
        print ("\n\n")

    print ("\n\n\n",YY[-1],"\n\n\n")

    YYstd = np.array([np.std(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])
    return XX, np.array(YY), YYstd

def binMedianStdPlot(X,Y,ax=None,numBins=8,xmin=None,xmax=None,binsize=None):
    if xmin is None:
        xmin = X.min()
    if xmax is None:
        xmax = X.max()
    if binsize:
        numBins=int(((xmax-xmin)/binsize)+0.5)
    else:
        binsize=(xmin-xmax)/numBins
    bins = np.arange(xmin,xmax+binsize,binsize)
    print (xmin,xmax,bins, numBins)
    XX = np.array([np.mean((bins[binInd], bins[binInd+1])) for binInd in range(numBins)])
    YY = np.array([np.median(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])
    #XX[np.isnan(YY)]=np.nan
    YYstd = np.array([np.std(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])

    return XX, YY, YYstd




'''
    if ax is None:
        fig = pylab.figure()
        ax = fig.add_subplot(111)

    lineHandles = ax.plot(XX,YY)
    return lineHandles[0], XX, YY


    if ax is None:
        fig = pylab.figure()
        ax = fig.add_subplot(111)
    lineHandles = ax.plot(XX,YY)
patchHandle = ax.fill_between(XX[~np.isnan(YY)],YY[~npmyplo.isnan(YY)]-YYstd[~np.isnan(YY)],YY[~np.isnan(YY)]+YYstd[~np.isnan(YY)])
    patchHandle.set_facecolor([.8, .8, .8])
    patchHandle.set_edgecolor('none')
    return lineHandles[0], patchHandle, '''
