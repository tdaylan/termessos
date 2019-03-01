"""
Analysis of short-time scale (i.e., no linear or larger term, only periodic) TTVs

Tansu Daylan
MIT Kavli Institute, Cambridge, MA, 02109, US
tansu.daylan@gmail.com
www.tansudaylan.com
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import h5py 
import os, sys
import emcee

import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyutils

def summgene(arry):
    try: 
        print np.amin(arry)
        print np.amax(arry)
        print np.mean(arry)
        print arry.shape
    except:
        print arry


def icdf(para, numbepoc):
    
    icdf = limtpara[0, :] + para * (limtpara[1, :] - limtpara[0, :])

    return icdf


def retr_lpos(para, time, numbepoc, ttvrobsv, ttvrstdvobsv, ttvrtype):
    
    if ((para < limtpara[0, :]) | (para > limtpara[1, :])).any():
        lpos = -np.inf
    else:
        llik = retr_llik(para, time, numbepoc, ttvrobsv, ttvrstdvobsv, ttvrtype)
        lpos = llik
    
    return lpos


def retr_ttvrmodl(time, offs, phas, ampl, peri):
    
    ttvrmodl = offs + ampl * np.sin(phas + 2. * np.pi * time / peri)

    return ttvrmodl


def retr_llik(para, time, numbepoc, ttvrobsv, ttvrstdvobsv, ttvrtype):
    
    offs = para[0]
    if ttvrtype == 'cons':
        ttvrmodl = offs * np.ones_like(time)
    if ttvrtype == 'sinu':
        phas = para[1]
        ampl = para[2]
        peri = para[3]
        ttvrmodl = retr_ttvrmodl(time, offs, phas, ampl, peri)
    
    llik = -0.5 * np.sum((ttvrmodl - ttvrobsv)**2 / ttvrstdvobsv**2)
    
    return llik


pathdata = os.environ['TESS_TTVR_DATA_PATH'] + '/'
os.system('mkdir -p %s' % pathdata)

# From Table 2
epoc = 2458375.169883
peri = 0.9414526

# get Luke's posterior
path = pathdata + 'WASP-18b_literature_and_TESS_times_O-C_vs_epoch_selected.csv'
objtfile = open(path, 'r')
tesspost = np.empty((44, 5))
for k, line in enumerate(objtfile):
    if k < 12:
        continue
    linesplt = line.split(';')
    tesspost[k-12, :] = linesplt[:-2]
indxepocobsv = tesspost[:, 0]
ttvrobsv = (tesspost[:, 2] - epoc - peri * (indxepocobsv - 396)) * 24. * 60.
ttvrstdvobsv = tesspost[:, 4]

numbepoc = indxepocobsv.size
indxepoc = np.arange(numbepoc)

time = np.copy(indxepocobsv)

# 'emce' or 'nest'
samptype = sys.argv[1]

listttvrtype = ['cons', 'sinu']
for ttvrtype in listttvrtype:
    if ttvrtype == 'cons':
        numbpara = 1
    if ttvrtype == 'sinu':
        numbpara = 4
    
    if samptype == 'emce':
        numbsamp = 100000
        numbsampburn = 50000
    
    numbdoff = numbepoc - numbpara
    
    indxpara = np.arange(numbpara)
    limtpara = np.empty((2, numbpara))
    # offs
    limtpara[0, 0] = -0.2
    limtpara[1, 0] = 0.2
    if ttvrtype == 'sinu':
        # phas
        limtpara[0, 1] = 0.8 * np.pi
        limtpara[1, 1] = 1.5 * np.pi
        # ampl
        limtpara[0, 2] = 0.
        limtpara[1, 2] = 1. * np.pi
        # peri
        limtpara[0, 3] = 14.
        limtpara[1, 3] = 22.
    
    dictllik = [time, numbepoc, ttvrobsv, ttvrstdvobsv, ttvrtype]
    dicticdf = [numbepoc]
    
    if samptype == 'emce':
        numbwalk = 50
        indxwalk = np.arange(numbwalk)
        parainit = []
        for k in indxwalk:
            parainit.append(np.empty(numbpara))
            meannorm = (limtpara[0, :] + limtpara[1, :]) / 2.
            stdvnorm = (limtpara[0, :] - limtpara[1, :]) / 10.
            parainit[k]  = (scipy.stats.truncnorm.rvs((limtpara[0, :] - meannorm) / stdvnorm, (limtpara[1, :] - meannorm) / stdvnorm)) * stdvnorm + meannorm
        numbsampwalk = numbsamp / numbwalk
        numbsampwalkburn = numbsampburn / numbwalk
        objtsamp = emcee.EnsembleSampler(numbwalk, numbpara, retr_lpos, args=dictllik)
        parainitburn, prob, state = objtsamp.run_mcmc(parainit, numbsampwalkburn)
        objtsamp.reset()
        objtsamp.run_mcmc(parainitburn, numbsampwalk)
        #print objtsamp.get_autocorr_time()
        #print objtsamp.acor
        objtsave = objtsamp
    else:
    
        sampler = dynesty.NestedSampler(retr_llik, icdf, numbpara, logl_args=dictllik, ptform_args=dicticdf, bound='single', dlogz=1000.)
        sampler.run_nested()
        results = sampler.results
        results.summary()
        objtsave = results
        
    if samptype == 'emce':
        numbsamp = objtsave.flatchain.shape[0]
        indxsampwalk = np.arange(numbsampwalk)
    else:
        numbsamp = objtsave['samples'].shape[0]
    
    indxsamp = np.arange(numbsamp)
    
    # resample the nested posterior
    if samptype == 'nest':
        weights = np.exp(results['logwt'] - results['logz'][-1])
        samppara = dyutils.resample_equal(results.samples, weights)
        assert samppara.size == results.samples.size
    
    # plot the posterior
    ## parameter
    ### trace
    for k in indxpara:
        if samptype == 'emce':
            figr, axis = plt.subplots()
            for i in indxwalk:
                axis.plot(indxsampwalk[::10], objtsave.chain[i, ::10, k])
            path = pathdata + 'tracwalk%04d_%s.pdf' % (k, ttvrtype)
            print 'Writing to %s...' % path
            plt.savefig(path)
            plt.close()
        
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
            for i in indxwalk:
                axis.plot(indxsampwalk[::10], objtsave.lnprobability[::10, i])
        else:
            axis.plot(indxsamp, objtsave['logl'])
        path = pathdata + 'llik_%s.pdf' % ttvrtype
        print 'Writing to %s...' % path
        plt.savefig(path)
        plt.close()
    
        chi2 = -2. * objtsave.lnprobability
        print 'Posterior-mean chi2: '
        print np.mean(chi2)
        print 'Posterior-mean chi2 per dof: '
        print np.mean(chi2) / numbdoff
    
    
    ### sample model ttvr
    numbttvrmodl = 100
    indxttvrmodl = np.arange(numbttvrmodl)
    indxsamprand = np.random.choice(indxsamp, numbttvrmodl, replace=False)
    yerr = np.empty((2, numbepoc))
    yerr[0, :] = ttvrstdvobsv
    yerr[1, :] = ttvrstdvobsv
    
    numbepocfine = 100
    indxepocfine = np.linspace(np.amin(indxepocobsv), np.amax(indxepocobsv), numbepocfine)
    ttvrmodlfine = np.empty((numbsamp, numbepocfine))
    for k in indxttvrmodl:
        if samptype == 'emce':
            objttemp = objtsave.flatchain
        else:
            objttemp = samppara
        offs = objttemp[indxsamprand[k], 0]
        if ttvrtype == 'cons':
            ttvrmodlfine[k, :] = offs * np.ones_like(indxepocfine)
        if ttvrtype == 'sinu':
            phas = objttemp[k, 1]
            ampl = objttemp[k, 2]
            peri = objttemp[k, 3]
            ttvrmodlfine[k, :] = retr_ttvrmodl(indxepocfine, offs, phas, ampl, peri)
    
    figr, axis = plt.subplots(figsize=(6, 3))
    axis.errorbar(indxepocobsv, ttvrobsv, yerr=yerr, color='black', marker='o', ls='')
    for k in indxttvrmodl:
        axis.plot(indxepocfine, ttvrmodlfine[k, :], alpha=0.05, color='b')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Transit timing residuals [minute]')
    plt.tight_layout()
    path = pathdata + 'modl_%s.pdf' % ttvrtype
    print 'Writing to %s...' % path
    plt.savefig(path)
    plt.close()
    
    ### nested sampling specific
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
    

