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

import time as timemodl

import h5py 

import os

import dynesty

import emcee

import scipy.optimize as optimize

from dynesty import plotting as dyplot

from dynesty import utils as dyutils
import pickle

from astropy.time import Time

def summgene(arry):
    try: 
        print np.amin(arry)
        print np.amax(arry)
        print np.mean(arry)
        print arry.shape
    except:
        print arry


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
    
    path = pathdata + 'w18b.pdf'
    print 'Writing to %s...' % path
    plt.savefig(path)
    plt.close()


def icdf(para, numbepoc):
    
    icdf = limtpara[0, :] + para * (limtpara[1, :] - limtpara[0, :])

    return icdf


def retr_lpos(para, time, numbepoc, ttvrobsv, ttvrstdvobsv, indxswep, listttvrmodl, ttvrtype, binsttvr, histttvr, listskewnorm):
    
    if ((para < limtpara[0, :]) | (para > limtpara[1, :])).any():
        lpos = -np.inf
    else:
        llik = retr_llik(para, time, numbepoc, ttvrobsv, ttvrstdvobsv, indxswep, listttvrmodl, ttvrtype, binsttvr, histttvr, listskewnorm)
        lpos = llik
    
    indxswep[0] += 1
    
    return lpos


def retr_ttvrmodl(time, offs, phas, ampl, peri):
    
    ttvrmodl = offs + ampl * np.sin(phas + 2. * np.pi * time / peri)

    return ttvrmodl


def retr_llik(para, time, numbepoc, ttvrobsv, ttvrstdvobsv, indxswep, listttvrmodl, ttvrtype, binsttvr, histttvr, listskewnorm):
    
    offs = para[0]
    if ttvrtype == 'cons':
        ttvrmodl = offs * np.ones_like(time)
    if ttvrtype == 'sinu':
        phas = para[1]
        ampl = para[2]
        peri = para[3]
        #print 'offs'
        #print offs
        #print 'phas'
        #print phas
        #print 'ampl'
        #print ampl
        #print 'peri'
        #print peri
        #print

        ttvrmodl = retr_ttvrmodl(time, offs, phas, ampl, peri)
    
    #print 'para'
    #print para
    #print 'indxswep'
    #print indxswep
    #print

    if ebartype == 'asym':
        t0 = timemodl.time()
        if intptype == 'bins':
            llik = 0.
            for k in indxepoc:  
                indx = np.digitize(ttvrmodl[k], binsttvr[k])
                #print 'k'
                #print k
                #print 'binsttvr[k]'
                #summgene(binsttvr[k])
                #print 'ttvrmodl[k]'
                #print ttvrmodl[k]
                #print 'indx'
                #print indx
                #print
                llik += histttvr[k][indx]
        elif intptype != 'skew':
            if dimstype == 'odim':
                llik = 1.
                for k in indxepoc:
                    print 'k'
                    print k
                    print 'funcintp[k](ttvrmodl[k])'
                    print funcintp[k](ttvrmodl[k])
                    llik *= funcintp[k](ttvrmodl[k])
            if dimstype == 'tdim':
                llik = funcintp(ttvrmodl)
        if intptype == 'skew':
            llik = 0.
            for k in indxepoc:
                llik += np.log(listskewnorm[k].pdf(ttvrmodl[k]))
        else:
            llik = np.log(llik)
        print 'llik'
        print llik

        llikgaus = -0.5 * np.sum((ttvrmodl - ttvrobsv)**2 / ttvrstdvobsv**2 + np.log(1. / np.sqrt(2 * np.pi) / ttvrstdvobsv))
        print 'llikgaus'
        print llikgaus
        print

    else:
        llik = -0.5 * np.sum((ttvrmodl - ttvrobsv)**2 / ttvrstdvobsv**2)
   
    if indxswep % 1000 == 0:
        listttvrmodl.append(ttvrmodl)
    
    #print 'llik'
    #print llik
    #print type(llik)
    #print

    return llik


pathdata = os.environ['TESS_TTVR_DATA_PATH'] + '/'
os.system('mkdir -p %s' % pathdata)

# get Luke's posterior
path = pathdata + 'luke.csv'
tesspost = np.loadtxt(path, delimiter=',')
time = tesspost[:, 0]
numbepoc = time.size
tesspost[:, 1] -= 0.9414518 * np.arange(numbepoc)
ttvrobsv = tesspost[:, 1]
ttvrstdvobsv = 0.5 * (tesspost[:, 2] + tesspost[:, 3])

# temp
numbsamp = 1000000
numbsampburn = 50000

indxepoc = np.arange(numbepoc)

ttvrtype = 'cons'
ttvrtype = 'sinu'

ebartype = 'symm'
#ebartype = 'asym'

samptype = 'emce'
samptype = 'nest'
samptype = 'mlik'

intptype = 'kdne'
intptype = 'spln'
#intptype = 'bins'
intptype = 'linr'
intptype = 'skew'

dimstype = 'odim'
#dimstype = 'tdim'

if ttvrtype == 'cons':
    numbpara = 1
if ttvrtype == 'sinu':
    numbpara = 4



#summgene(filearry['mcmc/accepted'])
#print 'chan'
#summgene(filearry['mcmc/chain'][:, :, 0])
#summgene(filearry['mcmc/chain'][:, :, 1])
#summgene(filearry['mcmc/chain'][:, :, 2])
#summgene(filearry['mcmc/chain'][:, :, 3])

#path = '/Users/tdaylan/Downloads/wasp-18b_lightcurve_fit_parameters/100100827_mandelagol_fit_empiricalerrs_t000.pickle'
#with open(path, 'rb') as handle:
#    objt = pickle.load(handle)
#print objt
#raise Exception('')

listskewnorm = None
binsttvr = None
histttvr = None

if ebartype == 'asym':
    
    tesspdfn = [[] for t in indxepoc]
    for t in indxepoc:
        path = pathdata + 'wasp-18b_posteriors/100100827_mandelagol_fit_samples_4d_t%03d_empiricalerrs.h5' % t
        filearry = h5py.File(path, 'r')
        tesspdfn[t] = filearry['mcmc/chain'][:, :, 3].flatten()
        tesspdfn[t] -= 0.9414518 * t
    funcintp = [[] for t in indxepoc]
    if intptype == 'kdne':
        print 'Constructing KDE objects for the input posterior...'
        for t in indxepoc:
            print 't'
            print t
            if dimstype == 'odim':
                funcintp[t] = scipy.stats.gaussian_kde(tesspdfn[t][:100])
            if dimstype == 'tdim':
                temp = np.empty((2, tesspdfn[t].size))
                temp[0, :] = tesspdfn[t]
                temp[1, :] = gdat.indxepoc
                funcintp[t] = scipy.stats.gaussian_kde(temp)
    if intptype == 'skew':
        a1 = [[] for t in indxepoc]
        b1 = [[] for t in indxepoc]
        loc1 = [[] for t in indxepoc]
        scale1 = [[] for t in indxepoc]
        listskewnorm = [[] for t in indxepoc]
        for t in indxepoc:
            print 't'
            print t
            a1[t], loc1[t], scale1[t] = scipy.stats.skewnorm.fit(tesspdfn[t][tesspdfn[t] > 0.])
            listskewnorm[t] = scipy.stats.skewnorm(a1[t], loc1[t], scale1[t])

    binsttvr = []
    histttvr = [[] for t in indxepoc]
    for t in indxepoc:
        if (tesspdfn[t] != 0.).any():
            print 'Warning! TTV is zero for time bin %d. Disregarding these samples...' % t
        minmttvr = np.amin(tesspdfn[t][tesspdfn[t] > 0.])
        maxmttvr = np.amax(tesspdfn[t])
        if t == 0:
            print 't'
            print t
            print 'tesspdfn[t]'
            summgene(tesspdfn[t])
            print 'minmttvr'
            print minmttvr
        if minmttvr < 1300.:
            raise Exception('')
        binsttvr.append(np.linspace(minmttvr, maxmttvr, 7))
        meanttvr = (binsttvr[t][1:] + binsttvr[t][:-1]) / 2.
        histttvr[t], xx = np.histogram(tesspdfn[t], bins=binsttvr[t])
        histttvr[t] = histttvr[t].astype(float)
        histttvr[t] /= np.sum(histttvr[t]) 
        if intptype == 'linr':
            funcintp[t] = scipy.interpolate.interp1d(meanttvr, histttvr[t], bounds_error=False, fill_value=0.)
        if intptype == 'spln':
            funcintp[t] = scipy.interpolate.LSQUnivariateSpline(meanttvr, histttvr[t], np.linspace(meanttvr[1], meanttvr[-2], 3))
            #funcintp[t] = scipy.interpolate.UnivariateSpline(meanttvr, histttvr[t], s=meanttvr.size*100)
        
    if True:
    #if False:
        for t in indxepoc:
            figr, axis = plt.subplots()
            wdth = binsttvr[t][1] - binsttvr[t][0]
            meanttvr = (binsttvr[t][1:] + binsttvr[t][:-1]) / 2.
            axis.bar(meanttvr, histttvr[t], width=wdth, label='Posterior', color='b')
            binsttvrfine = np.linspace(minmttvr, maxmttvr, 200)
            # overplot the interpolation
            if intptype == 'skew':
                axis.plot(binsttvrfine, listskewnorm[t].pdf(binsttvrfine), label='Skewed Normal Fit', color='r')
            if intptype == 'kdne' or intptype == 'spln' or intptype == 'linr':
                axis.plot(binsttvrfine, funcintp[t](binsttvrfine), label='Spline', color='r')
            axis.set_yscale('log')
            axis.legend()
            path = pathdata + 'spln_%04d.pdf' % t
            print 'Writing to %s...' % path
            plt.savefig(path)
            plt.close()

limtpara = np.empty((2, numbpara))
# offs
#limtpara[0, 0] = 1354.4560
#limtpara[1, 0] = 1354.4600
limtpara[0, 0] = 1354.4360
limtpara[1, 0] = 1354.4800
if ttvrtype == 'sinu':
    # phas
    limtpara[0, 1] = 0.8 * np.pi
    limtpara[1, 1] = 1.5 * np.pi
    # ampl
    limtpara[0, 2] = 0.
    limtpara[1, 2] = 0.002 * np.pi
    # peri
    limtpara[0, 3] = 14.
    limtpara[1, 3] = 22.

listttvrmodl = []
pathsave = pathdata + 'save.pickle'
if True or not os.path.exists(pathsave):
    
    indxswep = np.array([0])
    dictllik = [time, numbepoc, ttvrobsv, ttvrstdvobsv, indxswep, listttvrmodl, ttvrtype, binsttvr, histttvr, listskewnorm]
    dicticdf = [numbepoc]

    if samptype == 'mlik':
        
        bnds = [limtpara[:, k] for k in indxpara]
        res = optimize.minimize(retr_llik, np.mean(limtpara, axis=0), method='TNC', bounds=bnds, tol=1e-10, args=dictllik)
        paramlik = res.x


    if samptype == 'emce':
        numbwalk = 50
        indxwalk = np.arange(numbwalk)
        parainit = []
        for k in indxwalk:
            parainit.append(np.empty(numbpara))
            meannorm = (limtpara[0, :] + limtpara[1, :]) / 2.
            stdvnorm = (limtpara[0, :] - limtpara[1, :]) / 10.
            parainit[k]  = (scipy.stats.truncnorm.rvs((limtpara[0, :] - meannorm) / stdvnorm, (limtpara[1, :] - meannorm) / stdvnorm)) * stdvnorm + meannorm
            #parainit[k][1]  = 0.0004 * np.random.randn()
            #parainit[k][2]  = 15. * (1. + 1e-1 * np.random.randn())
            #parainit[k][3]  = 1354.45800 * (1. + 1e-6 * np.random.randn())
        numbsampwalk = numbsamp / numbwalk
        numbsampwalkburn = numbsampburn / numbwalk

        objtsamp = emcee.EnsembleSampler(numbwalk, numbpara, retr_lpos, args=dictllik)
        #objtsamp = emcee.EnsembleSampler(numbwalk, numbpara, retr_lpos, args=dictllik, threads=10)
        parainitburn, prob, state = objtsamp.run_mcmc(parainit, numbsampwalkburn)
        objtsamp.reset()
        objtsamp.run_mcmc(parainitburn, numbsampwalk)
        #print objtsamp.get_autocorr_time(low=10, high=None, step=1, c=10, fast=False)
        #print objtsamp.acor
        objtsave = objtsamp
    else:
    
        sampler = dynesty.NestedSampler(retr_llik, icdf, numbpara, logl_args=dictllik, ptform_args=dicticdf, bound='single', dlogz=1000.)
        sampler.run_nested()
        results = sampler.results
        results.summary()
        objtsave = results
    
    if False:
        print 'Writing to %s...' % pathsave
        with open(pathsave, 'wb') as handle:
            pickle.dump(objtsave, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    print 'Reading from %s...' % pathsave
    with open(pathsave, 'rb') as handle:
        objtsave = pickle.load(handle)

if samptype == 'emce':
    numbsamp = objtsave.flatchain.shape[0]
    indxsampwalk = np.arange(numbsampwalk)
else:
    numbsamp = objtsave['samples'].shape[0]

indxsamp = np.arange(numbsamp)
indxpara = np.arange(numbpara)

# plot the posterior

# plot the spline interpolation
#figr, axis = plt.subplots()
#for ttvrmodl in listttvrmodl:
#    axis.plot(np.arange(numbepoc), ttvrmodl, alpha=0.3, color='g')
#axis.plot(np.arange(numbepoc), ttvrobsv, label='Observed', color='black')
#path = pathdata + 'fitt.pdf'
#print 'Writing to %s...' % path
#plt.savefig(path)
#plt.close()

# resample the nested posterior
if samptype == 'nest':
    weights = np.exp(results['logwt'] - results['logz'][-1])
    samppara = dyutils.resample_equal(results.samples, weights)
    assert samppara.size == results.samples.size

## parameter
### trace
for k in indxpara:
    figr, axis = plt.subplots()
    if samptype == 'emce':
        axis.plot(indxsamp, objtsave.flatchain[:, k])
    else:
        axis.plot(indxsamp, samppara[:, k]) 
    path = pathdata + 'trac%04d_%s.pdf' % (k, ttvrtype)
    print 'Writing to %s...' % path
    plt.savefig(path)
    plt.close()
    
    if samptype == 'emce':
        figr, axis = plt.subplots()
        for i in indxwalk:
            axis.plot(indxsampwalk, objtsave.chain[i, :, k])
        path = pathdata + 'tracwalk%04d_%s.pdf' % (k, ttvrtype)
        print 'Writing to %s...' % path
        plt.savefig(path)
        plt.close()
    
        

#niter
#logvol
#information
#samples_id
#logz
#bound
#ncall
#samples_bound
#scale
#nlive
#samples
#bound_iter
#samples_u
#samples_it
#logl
#logzerr
#eff
#logwt


### histogram
for k in indxpara:
    figr, axis = plt.subplots()
    if samptype == 'emce':
        axis.hist(objtsave.flatchain[:, k]) 
    else:
        axis.hist(samppara[:, k]) 
    path = pathdata + 'hist%04d_%s.pdf' % (k, ttvrtype)
    print 'Writing to %s...' % path
    plt.savefig(path)
    plt.close()


if samptype == 'nest':
    for keys in objtsave:
        if isinstance(objtsave[keys], np.ndarray) and objtsave[keys].size == numbsamp:
            figr, axis = plt.subplots()
            axis.plot(indxsamp, objtsave[keys])
            path = pathdata + '%s_%s.pdf' % (keys, ttvrtype)
            print 'Writing to %s...' % path
            plt.savefig(path)
else:
    ## log-likelihood
    figr, axis = plt.subplots()
    if samptype == 'emce':
        axis.plot(indxsamp, objtsave.flatlnprobability)
    else:
        axis.plot(indxsamp, objtsave['logl'])
    path = pathdata + 'llik_%s.pdf' % ttvrtype
    print 'Writing to %s...' % path
    plt.savefig(path)
    plt.close()

    


### sample model ttvr
numbttvrmodl = 100
indxttvrmodl = np.arange(numbttvrmodl)
indxsamprand = np.random.choice(indxsamp, numbttvrmodl, replace=False)
yerr = np.empty((2, numbepoc))
yerr[0, :] = ttvrstdvobsv
yerr[1, :] = ttvrstdvobsv
numbepocfine = 100
indxepocfine = np.linspace(0., numbepoc - 1, numbepocfine)

ttvrmodlfine = np.empty((numbsamp, numbepocfine))
for k in indxttvrmodl:
    if samptype == 'emce':
        objttemp = objtsave.flatchain
    else:
        objttemp = samppara
    
    offs = objttemp[indxsamprand[k], 0]
    #print 'offs'
    #print offs - 1354.45
    #print
    if ttvrtype == 'cons':
        ttvrmodlfine[k, :] = offs * np.ones_like(indxepocfine)
    if ttvrtype == 'sinu':
        phas = objttemp[k, 1]
        ampl = objttemp[k, 2]
        peri = objttemp[k, 3]
        ttvrmodlfine[k, :] = retr_ttvrmodl(indxepocfine, offs, phas, ampl, peri)

figr, axis = plt.subplots()
axis.errorbar(indxepoc, ttvrobsv, yerr=yerr, color='black', marker='o', ls='')
for k in indxttvrmodl:
    axis.plot(indxepocfine, ttvrmodlfine[k, :], alpha=0.05, color='b')
path = pathdata + 'modl_%s.pdf' % ttvrtype
print 'Writing to %s...' % path
plt.savefig(path)
plt.close()

indxepoc = np.delete(indxepoc, [14, 15])
ttvrobsv = np.delete(ttvrobsv, [14, 15])
yerrcopy = np.copy(yerr)
yerr = np.empty((2, numbepoc - 2))
yerr[:, :14] = yerrcopy[:, :14]
yerr[:, 14:] = yerrcopy[:, 16:]
figr, axis = plt.subplots()
axis.errorbar(indxepoc, ttvrobsv, yerr=yerr, color='black', marker='o', ls='')
for k in indxttvrmodl:
    axis.plot(indxepocfine, ttvrmodlfine[k, :], alpha=0.05, color='b')
path = pathdata + 'modlexcl_%s.pdf' % ttvrtype
print 'Writing to %s...' % path
plt.savefig(path)
plt.close()


# plot
if samptype == 'nest':
    rfig, raxes = dyplot.runplot(results)
    path = pathdata + 'dyne_runs_%s.pdf' % ttvrtype
    print 'Writing to %s...' % path
    plt.savefig(path)
    plt.close()
    
    tfig, taxes = dyplot.traceplot(results)
    path = pathdata + 'dyne_trac_%s.pdf' % ttvrtype
    print 'Writing to %s...' % path
    plt.savefig(path)
    plt.close()
    
    cfig, caxes = dyplot.cornerplot(results)
    path = pathdata + 'dyne_corn_%s.pdf' % ttvrtype
    print 'Writing to %s...' % path
    plt.savefig(path)
    plt.close()
    

