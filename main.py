"""
Analysis of TESS TTVs

Tansu Daylan
MIT Kavli Institute, Cambridge, MA, 02109, US
tansu.daylan@gmail.com
www.tansudaylan.com
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

import os

import dynesty
from dynesty import plotting as dyplot

import pickle

from astropy.time import Time

def summgene(arry):
    
    print np.amin(arry)
    print np.amax(arry)
    print np.mean(arry)
    print arry.shape


def ttvr_mc18():
    
    perimc18line = 0.94145267 #0.00000011 days
    epocmc18line = 2457319.80197 #0.00021
    
    deltmc18quad = -6e-10 # 2e-10
    
    perimc18quad = 0.94145186 # 0.00000023 days
    epocmc18quad = 2457319.80167 # 0.00026
    
    epocmodlline = epocmc18line
    perimodlline = perimc18line
    
    
    def retr_modlline(itrn, epocline, periline):
        
        modlline = periline * itrn + epocline
        
        return modlline
    
    
    def retr_modlquad(itrn, epocquad, periquad, deltquad):
        
        print 'deltquad'
        print deltquad
        print
        
        modlquad = 0.5 * deltquad * itrn**2 + periquad * itrn + epocquad
        
        return modlquad
    
    
    tranobsv = [ \
                # Hellier et al. 2009
                [2454221.48163, 0.00038, 0.00038], \
                # Triaud et al. 2010
                [2454664.90531, 0.00016, 0.00017], \
               ]
    
    perihellmean = 0.94145299
    perihellstdv = 0.00000087
    peritessmean = 0.9414576
    peritessstdv = 0.0000035
    
    diff = perihellmean - peritessmean
    stdv = np.sqrt(perihellstdv**2 + peritessstdv**2)
    
    #WASP Spitzer Spitzer Warm Spitzer Warm Spitzer Warm Spitzer Warm Spitzer TRAPPIST TRAPPIST TRAPPIST TRAPPIST TRAPPIST HST
    #HST TRAPPIST TRAPPIST
    #Date
    #May - December 2006 December 20, 2008 December 24, 2008 January 23, 2010 January 24, 2010 August 23, 2010 August 24, 2010 September 30, 2010 October 2, 2010 December 23, 2010 January 8, 2011 November 11, 2011 April 22, 2014
    #April 22, 2014 August 20, 2015 October 21, 2015
    #Original Reference(s)
    #Hellier et al. (2009)
    #Nymeyer et al. (2011); Maxted et al. (2013) Nymeyer et al. (2011); Maxted et al. (2013)
    #Orbit BJD (TDB)
    #0 2454221.48163 0.00038
    #636.5 2454820.7168 0.0007
    #640.5 2454824.4815 0.0006
    #1061.5 2455220.8337 0.0006
    #1062 2455221.3042 0.0001
    #1285.5 2455431.7191 0.0003
    #1286 2455432.1897 0.0001
    #1327 2455470.7885 0.00040
    #1330 2455473.6144 0.00090
    #1416 2455554.5786 0.00050
    #1433 2455570.5840 0.00045 -0.00048
    #1758 2455876.5559 0.0013
    #2840.5 2456895.6773 0.0006
    #2841 2456896.1478 0.0008
    #3223 2457255.7832 0.00030 -0.00029
    #3291 2457319.8010 0.00039 -0.00038
    





    tranobsv = np.array(tranobsv)
    itrnobsv = (tranobsv[:, 0] - epocmodlline) // perimodlline
    
    itrn = np.arange(np.amin(itrnobsv), np.amax(itrnobsv))
    tranmodlmc18line = retr_modlline(itrn, epocmc18line, perimc18line)
    tranmodlmc18quad = retr_modlquad(itrn, epocmc18quad, perimc18quad, deltmc18quad)
    
    
    tranmodlmc18lineeval = retr_modlline(itrnobsv, epocmodlline, perimodlline)
    
    
    objttranobsv = Time(tranobsv[:, 0], format='jd', scale='utc')
    
    
    print 'tranobsv'
    summgene(tranobsv)
    print 'itrnobsv'
    summgene(itrnobsv)
    print 'tranmodlmc18lineeval'
    summgene(tranmodlmc18lineeval)
    print 'itrn'
    summgene(itrn)
    print 'tranmodlmc18line'
    summgene(tranmodlmc18line)
    
    resiobsv = tranobsv[:, 0] - tranmodlmc18lineeval
    resimodlline = tranmodlmc18line - tranmodlmc18line
    resimodlquad = tranmodlmc18quad - tranmodlmc18line
    
    print 'resiobsv'
    summgene(resiobsv)
    print 'resimodlline'
    summgene(resimodlline)
    print 'resimodlquad'
    summgene(resimodlquad)
    
    figr, axi1 = plt.subplots()
    figr.subplots_adjust(bottom=0.2)
    
    axi0 = axi1.twiny()
    axi2 = axi1.twiny()
    
    axi0.set_xlabel(r"BJD")
    axi1.set_xlabel(r"Transit Index")
    
    print 'tranmodlmc18quad'
    summgene(tranmodlmc18quad)
    axi1.scatter(itrnobsv, resiobsv, label='Observed', color='black')
    axi1.plot(itrn, resimodlline, label='Linear (McDonald 2018)')
    axi1.plot(itrn, resimodlquad, label='Quadratic (McDonald 2018)')
    axi1.legend()
    
    axi2.xaxis.set_ticks_position("bottom")
    axi2.xaxis.set_label_position("bottom")
    axi2.spines["bottom"].set_position(("axes", -0.15))
    axi2.set_frame_on(True)
    axi2.patch.set_visible(False)
    for sp in axi2.spines.itervalues():
        sp.set_visible(False)
    axi2.spines["bottom"].set_visible(True)
    
    minmyear = np.amin(objttranobsv.decimalyear)
    maxmyear = np.amax(objttranobsv.decimalyear)
    axi2.set_xlim([minmyear, maxmyear])
    arryyear = np.arange(np.round(minmyear), np.round(maxmyear) + 1)
    axi2.set_xticks(arryyear)
    axi2.set_xlabel('Year')
    
    path = '/Users/tdaylan/Desktop/w18b.pdf'
    plt.savefig(path)
    plt.close()


def icdf(para, numbepoc):
    
    icdf = np.empty_like(para)
    icdf[0] = 2. * np.pi * para[0]
    icdf[1] = 10. * para[1]
    icdf[2] = 3. * para[2] + 1.
    icdf[3:3+numbepoc] = 3. * para[2] + 1.

    return icdf

    
def retr_llik(para, time, numbepoc, ttvrobsv, ttvrstdvobsv):
    
    phas = para[0]
    ampl = para[1]
    peri = para[2]
    ttvrmodl = np.sin(phas + 2. * np.pi * time / peri)
    vari = para[3:3+numbepoc]
    weig = 1. / vari
    chi2 = np.sum((ttvrobsv - ttvrmodl)**2 * weig)
    #if boolsymm:
    #else:
    #    chi2 = np.sum(-np.log(vari[None, :]) + (mean[None, :] - samp)**2 / vari[None, :])
    llik = -0.5 * chi2 
    
    #print 'llik'
    #print llik

    return llik


def ttvr_tess():
    
    path = '/Users/tdaylan/Desktop/luke.csv'
    tesspost = np.loadtxt(path, delimiter=',')
    
    pathdata = os.environ['TESS_TTVR_PATH'] + '/'
    os.system('mkdir -p %s' % pathdata)

    pathsave = pathdata + 'save.pickle'
    
    time = tesspost[:, 0]
    ttvrobsv = tesspost[:, 1]
    ttvrstdvobsv = 0.5 * (tesspost[:, 2] + tesspost[:, 3])

    numbepoc = time.size
    numbsamp = 1000
    
    numbdims = numbepoc + 3
    
    if not os.path.exists(pathsave):
        
        dictllik = [time, numbepoc, ttvrobsv, ttvrstdvobsv]
        dicticdf = [numbepoc]
        sampler = dynesty.NestedSampler(retr_llik, icdf, numbdims, logl_args=dictllik, ptform_args=dicticdf, bound='single', dlogz=1000.)
        sampler.run_nested()
        results = sampler.results
        results.summary()
        
        with open(pathsave, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pathsave, 'rb') as handle:
            b = pickle.load(handle)
    
    print 'results'
    print results

    # plot
    rfig, raxes = dyplot.runplot(results)
    tfig, taxes = dyplot.traceplot(results)
    cfig, caxes = dyplot.cornerplot(results)

ttvr_tess()

