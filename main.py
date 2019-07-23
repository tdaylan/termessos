"""
Analysis of short-time scale (i.e., no linear or larger term, only periodic) TTVs

Tansu Daylan
MIT Kavli Institute, Cambridge, MA, 02109, US
tansu.daylan@gmail.com
www.tansudaylan.com
"""

import numpy as np
import scipy
import h5py 
import os, sys

import emcee

import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyutils

import pickle
import astropy
import multiprocessing

import ttvfast

import tdpy.mcmc
from tdpy.util import summgene
import tdpy

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time


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


def retr_llik_sinu(para, time, numbepoc, ttvrobsv, ttvrstdvobsv, ttvrtype):
    
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


def init():
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
    
    pathmetd = pathdata + samptype + '/'
    os.system('mkdir -p %s' % pathmetd)
    
    listttvrtype = ['cons', 'sinu']
    for ttvrtype in listttvrtype:
        if ttvrtype == 'cons':
            numbpara = 1
        if ttvrtype == 'sinu':
            numbpara = 4
        
        if samptype == 'emce':
            numbsamp = 200000
            numbsampburn = 20000
        
        numbdoff = numbepoc - numbpara
        
        indxpara = np.arange(numbpara)
        limtpara = np.empty((2, numbpara))
        # offs
        limtpara[0, 0] = -0.2
        limtpara[1, 0] = 0.2
        listlablpara = ['$C$ [minutes]']
        if ttvrtype == 'sinu':
            # phas
            limtpara[0, 1] = 0.#np.pi
            limtpara[1, 1] = 2. * np.pi
            # ampl
            limtpara[0, 2] = 0.
            limtpara[1, 2] = 1.
            # peri
            limtpara[0, 3] = 2.
            limtpara[1, 3] = 15.
        
            listlablpara += ['$\phi$', '$A$ [minutes]', '$P$ [epochs]']
        
        numbbins = 60
        indxbins = np.arange(numbbins)
        binspara = np.empty((numbbins + 1, numbpara)) 
        for k in indxpara:
            binspara[:, k] = np.linspace(limtpara[0, k], limtpara[1, k], numbbins + 1)
        meanpara = (binspara[1:, :] + binspara[:-1, :]) / 2.
    
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
                parainit[k]  = (scipy.stats.truncnorm.rvs((limtpara[0, :] - meannorm) / stdvnorm, \
                                                (limtpara[1, :] - meannorm) / stdvnorm)) * stdvnorm + meannorm
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
        
        if samptype == 'emce':
            listsamp = objtsave.flatchain
            listllik = objtsave.flatlnprobability
        else:
            listsamp = samppara[:, k]
       
        indxsampmlik = np.argmax(listllik)
    
        # plot the posterior
        ## parameter
        ### trace
        for k in indxpara:
            if samptype == 'emce':
                figr, axis = plt.subplots()
                for i in indxwalk:
                    axis.plot(indxsampwalk[::10], objtsave.chain[i, ::10, k])
                path = pathdata + '%s/tracwalk%04d_%s.pdf' % (samptype, k, ttvrtype)
                print 'Writing to %s...' % path
                plt.savefig(path)
                plt.close()
            
        ### histogram
        for k in indxpara:
            figr, axis = plt.subplots(figsize=(6, 4))
            axis.hist(listsamp[:, k], numbbins) 
            axis.set_ylabel('$N_{samp}$')
            axis.set_xlabel(listlablpara[k])
            path = pathdata + '%s/hist%04d_%s.pdf' % (samptype, k, ttvrtype)
            plt.tight_layout()
            print 'Writing to %s...' % path
            plt.savefig(path)
            plt.close()
        
        path = pathdata + '%s/' % samptype
        strgplot = 'post_%s' % ttvrtype
    
        listparamlik = listsamp[indxsampmlik, :]
        tdpy.mcmc.plot_grid(path, strgplot, listsamp, listlablpara, listvarbdraw=[listparamlik], numbbinsplot=numbbins)
        
        if samptype == 'nest':
            for keys in objtsave:
                if isinstance(objtsave[keys], np.ndarray) and objtsave[keys].size == numbsamp:
                    figr, axis = plt.subplots()
                    axis.plot(indxsamp, objtsave[keys])
                    path = pathdata + '%s/%s_%s.pdf' % (samptype, keys, ttvrtype)
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
            path = pathdata + '%s/llik_%s.pdf' % (samptype, ttvrtype)
            print 'Writing to %s...' % path
            plt.savefig(path)
            plt.close()
        
            chi2 = -2. * objtsave.lnprobability
            
            print 'Posterior-mean chi2: '
            print np.mean(chi2)
            print 'Posterior-mean chi2 per dof: '
            print np.mean(chi2) / numbdoff
            print 'Minimum chi2: '
            print np.amin(chi2)
            print 'Minimum chi2 per dof: '
            print np.amin(chi2) / numbdoff
            print 'Posterior-mean llik: '
            print np.mean(objtsave.lnprobability)
            print 'Maximum llik: '
            print np.amax(objtsave.lnprobability)
        
        
        ### sample model ttvr
        numbttvrmodl = 100
        indxttvrmodl = np.arange(numbttvrmodl)
        indxsamprand = np.random.choice(indxsamp, numbttvrmodl, replace=False)
        yerr = np.empty((2, numbepoc))
        yerr[0, :] = ttvrstdvobsv
        yerr[1, :] = ttvrstdvobsv
        
        numbepocfine = 1000
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
        
        if ttvrtype == 'cons':
            mlikcons = np.amax(listllik)
    
        if ttvrtype == 'sinu':
            llikperi = np.empty(numbbins)
            for m in indxbins:
                indxsampthis = np.where((listsamp[:, -1] < binspara[m+1, -1]) & (listsamp[:, -1] > binspara[m, -1]))
                llikperi[m] = np.amax(listllik[indxsampthis])
            
            figr, axis = plt.subplots(figsize=(6, 3))
            axis.plot(meanpara[:, -1], -2. * llikperi, label='Sinusoidal')
            axis.axhline(-2. * mlikcons, label='Constant', color='orange')
            axis.set_xlabel('Period [epochs]')
            axis.set_ylabel('$\chi^2$')
            plt.tight_layout()
            plt.legend()
            path = pathdata + '%s/chi2peri_%s.pdf' % (samptype, ttvrtype)
            print 'Writing to %s...' % path
            plt.savefig(path)
            plt.close()
        
        figr, axis = plt.subplots(figsize=(6, 3))
        axis.errorbar(indxepocobsv, ttvrobsv, yerr=yerr, color='black', marker='o', ls='')
        for k in indxttvrmodl:
            axis.plot(indxepocfine, ttvrmodlfine[k, :], alpha=0.05, color='b')
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Transit timing residuals [minute]')
        plt.tight_layout()
        path = pathdata + '%s/modl_%s.pdf' % (samptype, ttvrtype)
        print 'Writing to %s...' % path
        plt.savefig(path)
        plt.close()
        
        ### nested sampling specific
        if samptype == 'nest':
            rfig, raxes = dyplot.runplot(results)
            path = pathdata + '%s/dyne_runs_%s.pdf' % (samptype, ttvrtype)
            print 'Writing to %s...' % path
            plt.savefig(path)
            plt.close()
            
            tfig, taxes = dyplot.traceplot(results)
            path = pathdata + '%s/dyne_trac_%s.pdf' % (samptype, ttvrtype)
            print 'Writing to %s...' % path
            plt.savefig(path)
            plt.close()
            
            cfig, caxes = dyplot.cornerplot(results)
            path = pathdata + '%s/dyne_corn_%s.pdf' % (samptype, ttvrtype)
            print 'Writing to %s...' % path
            plt.savefig(path)
            plt.close()
    

def plot_ttvr(gdat, strgplottype, ttvrtype, listvarb, strgpdfn):
    
    print 'plot_ttvr()'
    
    listindxtranmodl, listtimetranmodl, listindxtranmodlproj, listtimetranmodlproj = listvarb

    numbsamp = len(listindxtranmodl)
    indxsamp = np.arange(numbsamp)
    
    print 'indxsamp'
    summgene(indxsamp)
    gdat.timetranline = [[] for j in gdat.indxplan]

    if strgplottype == 'resi':
        strgplot = 'resi'
        figrsize = (12, 8)
    else:
        strgplot = 'raww'
        figrsize = (6, 14)
    
    figr = plt.figure(figsize=figrsize)
    gs = gridspec.GridSpec(gdat.numbplan, 2, width_ratios=[1, 2])
    axis = [[[] for a in range(2)] for j in gdat.indxplan]
    for a in range(2):
        for j in gdat.indxplan:
            axis[j][a] = plt.subplot(gs[j, a])
    if strgplottype == 'resi':
        axis[1][0].set_ylabel('$T_O - T_M$ [min]')
        axis[1][1].set_ylabel('$T_O - T_M$ [min]')
    else:
        axis[1][0].set_ylabel('$T$ [day]')
        axis[1][1].set_ylabel('$T$ [day]')
    
    for a in range(2):
        for j in gdat.indxplan:
            axis[j][a].set_xlabel('$i_T$')
    
            if strgplottype == 'resi':

                ydat = (gdat.timetranobsd[j] - gdat.timetranlineproj[j]) * gdat.facttime
                temp, listcaps, temp = axis[j][a].errorbar(gdat.indxtranobsd[j], ydat, color='k', \
                                                yerr=gdat.stdvtimetranobsd[j]*gdat.facttime, label='Observed - Linear', ls='', markersize=5, marker='o')
                
                if (ydat > 1000).any():
                    raise Exception('')

                for caps in listcaps:
                    caps.set_markeredgewidth(3)
        
                if numbsamp == 1:
                    alph = 1.
                else:
                    alph = 0.1

                for i in indxsamp:
                    if i == 0:
                        lablfrst = 'N-body - Mean N-body'
                        lablseco = 'N-body - Linear'
                        #rasterized = False
                    else:
                        lablfrst = None
                        lablseco = None
                        #rasterized = True
                    if len(listindxtranmodl[i][j]) > 0:
                        gdat.timetranline[j] = gdat.meanepocline[j] + (listindxtranmodl[i][j] - gdat.numbtranoffs[j]) * \
                                                                                            gdat.meanperiline[j] - gdat.timeobsdinit
                        axis[j][a].plot(listindxtranmodl[i][j], (listtimetranmodl[i][j] - listtimetranmodl[i][j][0] - \
                                listindxtranmodl[i][j] * (listtimetranmodl[i][j][-1] - listtimetranmodl[i][j][0]) / \
                                (listindxtranmodl[i][j].size - 1)) * gdat.facttime, \
                                          label=lablfrst, color='b', alpha=alph)
                        axis[j][a].plot(listindxtranmodl[i][j], (listtimetranmodl[i][j] - gdat.timetranline[j]) * gdat.facttime, \
                                                                                                label=lablseco, color='r', alpha=alph)
                axis[j][a].text(0.9, 0.13, gdat.liststrgplan[j], transform=axis[j][a].transAxes)
            else:
                axis[j][a].plot(listindxtranmodl[i][j], timetranmodl[i][j], label='Model')
                axis[j][a].plot(gdat.indxtranobsd[j], gdat.timetranobsd[j], label='Observed')
                axis[j][a].text(0.86, 0.13, gdat.liststrgplan[j], transform=axis[j][a].transAxes)
            
            if a == 0:
                axis[j][a].set_ylim([-30, 30])
                if j == 0:
                    axis[j][a].set_xlim([0., 40])
                if j == 1:
                    axis[j][a].set_xlim([0., 40])
                if j == 2:
                    axis[j][a].set_xlim([0., 15])
            axistwin = axis[j][a].twiny()
            axistwin.set_xlabel('$T$ [day]')
            limtindx = np.array(axis[j][a].get_xlim())
            limttime = limtindx * gdat.meanperiline[j]
            axistwin.set_xlim(limttime)
            limt = np.array([astropy.time.Time('2019-09-01T00:00:00', format='isot', scale='utc').jd, \
                             astropy.time.Time('2020-09-01T00:00:00', format='isot', scale='utc').jd]) - gdat.timeobsdinit
            axistwin.fill_between(limt, -10, 10, color='purple', alpha=0.2)
    axis[0][1].legend()
    
    plt.tight_layout()
    path = gdat.pathimag + 'timetran_%s_%s_%s.pdf' % (strgplot, ttvrtype, strgpdfn)
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()


def icdf(para, numbepoc):
    
    icdf = gdat.limtpara[0, :] + para * (gdat.limtpara[1, :] - gdat.limtpara[0, :])

    return icdf


def retr_lpos(para, gdat, ttvrtype, strgpdfn):
    
    if ((para < gdat.limtpara[0, :]) | (para > gdat.limtpara[1, :])).any():
        lpos = -np.inf
    else:
        llik = retr_llik(para, gdat, ttvrtype, strgpdfn)
        lpos = llik
    
    return lpos


def retr_modl(para, gdat, ttvrtype):
    
    #print 'retr_modl()'
    paraglob = np.copy(gdat.paraglob)
    
    if ttvrtype == 'peri':
        #print 'para'
        #print para
        paraglob[2+1+gdat.indxplan*7] = para
    if ttvrtype == 'perimass':
        paraglob[2+1+gdat.indxplan*7] = para[:gdat.numbplan]
        paraglob[2+gdat.indxplan*7] = para[gdat.numbplan:2*gdat.numbplan]
    
    #print 'paraglob'
    #print paraglob
    planet1 = ttvfast.models.Planet(*paraglob[2:2+7])
    planet2 = ttvfast.models.Planet(*paraglob[2+7:2+14])
    planet3 = ttvfast.models.Planet(*paraglob[2+14:])
    listobjtplan = [planet1, planet2, planet3]
    results = ttvfast.ttvfast(listobjtplan, gdat.massstar, gdat.inittimefastttvr, gdat.delttimefastttvr, gdat.numbstepfastttvr)
    indxtranmodl = [[] for j in gdat.indxplan]
    timetranmodl = [[] for j in gdat.indxplan]
    indxtranmodlproj = [[] for j in gdat.indxplan]
    timetranmodlproj = [[] for j in gdat.indxplan]
    for j in gdat.indxplan:
        indx = np.where(np.array(results['positions'][0]) == j)[0]
        timetranmodl[j] = np.array(results['positions'][2])[indx]
        indxgood = np.where(timetranmodl[j] != -2.)[0]
        numbtran = indxgood.size
        timetranmodl[j] = timetranmodl[j][indxgood]
        indxtranmodl[j] = np.arange(numbtran, dtype=int)
        indxtranmodlproj[j] = np.intersect1d(indxtranmodl[j], gdat.indxtranobsd[j])
        timetranmodlproj[j] = timetranmodl[j][indxtranmodlproj[j]]
        #print 'j'
        #print j
        #print 'indxtranmodl[j]'
        #print indxtranmodl[j]
        #print 'timetranmodl[j]'
        #print timetranmodl[j]
        #print 'gdat.indxtranobsd[j]'
        #print gdat.indxtranobsd[j]
        #print 'indxtranmodlproj[j]'
        #print indxtranmodlproj[j]
        #print 'timetranmodlproj[j]'
        #print timetranmodlproj[j]
        #print 'gdat.timetranobsd[j]'
        #print gdat.timetranobsd[j]
        #print 
        if gdat.diagmode:
            if timetranmodlproj[j].size != gdat.timetranobsd[j].size:
                raise Exception('')
    
    return indxtranmodl, timetranmodl, indxtranmodlproj, timetranmodlproj
            

def retr_llik(para, gdat, ttvrtype, strgpdfn):
    
    #print 'retr_llik()'
    #print 'strgpdfn'
    #print strgpdfn
    #print 'ttvrtype'
    #print ttvrtype
    
    if ttvrtype == 'sigm':
        logtsigm = para[0]
    else:
        logtsigm = 0.

    if strgpdfn == 'init' or ttvrtype != 'sigm':
        indxtranmodl, timetranmodl, indxtranmodlproj, timetranmodlproj = retr_modl(para, gdat, ttvrtype)
    else:
        timetranmodlproj = gdat.inittimetranmodlproj
    
    for j in gdat.indxplan:
        if strgpdfn == 'init' or ttvrtype != 'sigm':
            indx = indxtranmodl[j]
            if indx.size == 0:
                print 'indx.size == 0'
                return -np.inf
    
        if timetranmodlproj[j].size < gdat.timetranobsd[j].size:
            print 'timetranmodlproj[j].size < gdat.timetranobsd[j].size'
            return -np.inf
    
    if strgpdfn == 'post':
        gdat.indxswep += 1
        #if gdat.indxswep % 100 == 0:
        #    print 'gdat.indxswep'
        #    print gdat.indxswep

    llik = 0.
    for j in gdat.indxplan:
        lliktemp = -0.5 * np.sum((gdat.timetranobsd[j] - timetranmodlproj[j])**2 / (np.exp(logtsigm) * gdat.stdvtimetranobsd[j])**2)
        llik += lliktemp
        #print 'j'
        #print j
        #print 'gdat.timetranobsd[j]'
        #print gdat.timetranobsd[j]
        #print 'timetranmodlproj[j]'
        #print timetranmodlproj[j]
        #print 'gdat.timetranobsd[j] - timetranmodlproj[j]'
        #print gdat.timetranobsd[j] - timetranmodlproj[j]
        #print 'logtsigm'
        #print logtsigm
        #print '(np.exp(logtsigm) * gdat.stdvtimetranobsd[j])'
        #summgene(np.exp(logtsigm) * gdat.stdvtimetranobsd[j])
        #print 'lliktemp'
        #print lliktemp
        #print
    return llik


def init():
    # global object
    gdat = tdpy.util.gdatstrt()
    
    # paths
    gdat.pathbase = os.environ['TTVR_DATA_PATH'] + '/'
    gdat.pathdata = gdat.pathbase + 'data/'
    gdat.pathimag = gdat.pathbase + 'imag/'
    
    gdat.numbplan = 3
    gdat.indxplan = np.arange(gdat.numbplan)
    
    gdat.facttime = 24. * 60.
    
    gdat.timeobsdinit = 2458387.0927381925
    
    gdat.liststrgplan = ['b', 'c', 'd']
    # get data

    gdat.diagmode = True
    
    gdat.meanepocline = np.array([1461.01464, 1463.08481, 1469.33834]) + 2457000
    gdat.meanperiline = np.array([3.360080, 5.660172, 11.38014])
    
    objtfile = open(gdat.pathdata + 'measured_allesfit_all_conv.pickle','r')
    datapick = pickle.load(objtfile)
    objtfile = open(gdat.pathdata + 'measured_allesfit_all_ttv_conv.pickle','r')
    datapickttvr = pickle.load(objtfile)
    
    gdat.indxtranobsd = [[] for j in gdat.indxplan]
    gdat.timetranlineproj = [[] for j in gdat.indxplan]
    gdat.timetranobsd = [[] for j in gdat.indxplan]
    gdat.stdvtimetranobsd = [[] for j in gdat.indxplan]
    for j in gdat.indxplan:
        #if j == 0:
        #    gdat.timetranobsd[j] = np.array([2458387.0927381925, 2458390.4528234517, 2458393.812908711, 2458397.17299397, 2458400.5330792293, 2458403.8931644885, 2458413.973420266, 2458417.3335055253, 2458427.413761303, 2458430.773846562, 2458434.133931821, 2458440.8541023396, 2458444.214187599, 2458447.574272858, 2458454.2944433764, 2458457.6545286356, 2458461.0146138947, 2458497.9755517454])
        #if j == 1:
        #    gdat.timetranobsd[j] = np.array([2458389.5025863373, 2458395.162756024, 2458400.822925711, 2458412.1432650844, 2458417.803434771, 2458423.463604458, 2458429.123774145, 2458434.7839438315, 2458440.4441135186, 2458446.1042832052, 2458451.764452892, 2458457.4246225785, 2458463.0847922657, 2458468.7449619523, 2458497.045810386])
        #    gdat.timetranobsd[j] = np.concatenate((gdat.timetranobsd[j], np.array([2458576.29165406, 2458598.93411])))
        #if j == 2:
        #    gdat.timetranobsd[j] = np.array([2458389.677356091, 2458401.0575013356, 2458412.43764658, 2458435.1979370695, 2458446.5780823138, 2458457.9582275585, 2458480.7185180476, 2458503.478808537, 2458537.619244271])
            
        gdat.timetranobsd[j] = datapickttvr[gdat.liststrgplan[j]]['transit_time']
        gdat.stdvtimetranobsd[j] = datapickttvr[gdat.liststrgplan[j]]['transit_time_err']
        if j == 1:
            gdat.timetranobsd[j] = np.concatenate((gdat.timetranobsd[j], np.array([2458576.29165406, 2458598.93411])))
            gdat.stdvtimetranobsd[j] = np.concatenate((gdat.stdvtimetranobsd[j], np.array([0.0027935178950429, 1. / 60. / 24.])))
        gdat.timetranobsd[j] -= gdat.timeobsdinit
        
        #gdat.stdvtimetranobsd[j] = np.ones_like(gdat.timetranobsd[j]) * 10. / 60. / 24.
        
        #if j == 1:
        #    gdat.stdvtimetranobsd[j][-2] = 0.0027935178950429
        #    gdat.stdvtimetranobsd[j][-1] = 1. / 60. / 24.
    
    for j in gdat.indxplan:
        print 
        gdat.indxtranobsd[j] = np.round(gdat.timetranobsd[j] / gdat.meanperiline[j]).astype(int)
    
    # convert Earth mass to Solar mass
    gdat.meanmassradv = np.array([2.47, 5.46, 2.55]) * 0.00000300245
    stdvmassradv = np.array([0.75, 1.30, 0.91])
    
    samptype = 'emce'#sys.argv[1]
    #os.mkdir(gdat.pathimag + '%s' % (samptype))
    
    # setttings
    ## modeling 
    boolecce = False
    maxmecce = 0.1
    gdat.inittimefastttvr = 0
    gdat.delttimefastttvr = 0.03
    gdat.numbstepfastttvr = 1000
    
    gdat.paraglob = [
        0.000295994511,# G
        0.40, # Mstar
        
        #planet b
        gdat.meanmassradv[0],#M planet
        gdat.meanperiline[0],#P
        0,#e
        88.65,#i
        #    np.random.uniform(low=0.0, high=maxmecce),#e
        #    np.random.uniform(low=0.0, high=360.0), #longNode
        #    np.random.uniform(low=0.0, high=360.0),# #argument
        0,#longNode
        0, #argument
        89.99999, #Mean Anomolay
        
        #planet c
        gdat.meanmassradv[1], #M planet
        gdat.meanperiline[1], #P
        0, #e
        89.53,#i
        0,#longNode
        0,#Argument
        296.7324,#Mean Anomaly
    
        #planet d
        gdat.meanmassradv[2],#M planet
        gdat.meanperiline[2],#P
        0,#e
        89.69,#i
        0,#longNode
        0,#Argument
        8.25829165761]#Mean anomaly
    
    
    
    planet1 = ttvfast.models.Planet(*gdat.paraglob[2:9])
    planet2 = ttvfast.models.Planet(*gdat.paraglob[9:16])
    planet3 = ttvfast.models.Planet(*gdat.paraglob[16:])
    
    gravity, gdat.massstar = gdat.paraglob[:2]
    listobjtplan = [planet1, planet2, planet3]
    # run the TTV Simulation
    results = ttvfast.ttvfast(listobjtplan, gdat.massstar, gdat.inittimefastttvr, gdat.delttimefastttvr, gdat.numbstepfastttvr)
    
    # The function ttvfast.ttvfast returns a dictionary containing positions and rv. The positions entry is a tuple of:
    # a list of integer indices for which values correspond to which planet,
    # a list of integers defining the epoch,
    # a list of times,
    # a list of rsky values, and
    # a list of vsky values.
    
    gdat.numbtranoffs = [22, 13, 7]
    for j in gdat.indxplan:
        gdat.timetranlineproj[j] = gdat.meanepocline[j] + (gdat.indxtranobsd[j] - gdat.numbtranoffs[j]) * gdat.meanperiline[j] - gdat.timeobsdinit
    
    # 'emce' or 'nest'
    #numbsamp = np.array([20000, 500])
    #numbsampburn = np.array([1000, 100])
    numbsamp = np.array([20000, 20000])
    numbsampburn = np.array([2000, 2000])
        
    #listttvrtype = ['sigm', 'peri', 'massperi']
    listttvrtype = ['peri', 'massperi']
    for h, ttvrtype in enumerate(listttvrtype):
        
        numbdata = 0
        for j in gdat.indxplan:
            numbdata += gdat.indxtranobsd[j].size
        
        if ttvrtype == 'sigm':
            listlablpara = ['$\ln \sigma$']
        if ttvrtype == 'peri':
            listlablpara = ['$P_b$ []', '$P_c$ []', '$P_d$ []']
        if ttvrtype == 'perimass':
            listlablpara = ['$P_b$ []', '$P_c$ []', '$P_d$ []', '$M_b$ []', '$M_c$ []', '$M_d$ []']
        
        numbpara = len(listlablpara)
        indxpara = np.arange(numbpara)
        gdat.limtpara = np.empty((2, numbpara))
        numbdoff = numbdata - numbpara
        if ttvrtype == 'sigm':
            # ln-sigma
            gdat.limtpara[0, 0] = -10.
            gdat.limtpara[1, 0] = 10.
        if ttvrtype == 'peri':
            # periods
            for j in gdat.indxplan:
                gdat.limtpara[0, j] = gdat.meanperiline[j] - 1e-2 * gdat.meanperiline[j]
                gdat.limtpara[1, j] = gdat.meanperiline[j] + 1e-2 * gdat.meanperiline[j]
        if ttvrtype == 'perimass':
            for j in gdat.indxplan:
                gdat.limtpara[0, j] = gdat.meanmassradv[j] - 1e-1 * gdat.meanmassradv[j]
                gdat.limtpara[1, j] = gdat.meanmassradv[j] + 1e-1 * gdat.meanmassradv[j]
                gdat.limtpara[0, j+3] = gdat.meanperiline[j] - 1e-1 * gdat.meanperiline[j]
                gdat.limtpara[1, j+3] = gdat.meanperiline[j] + 1e-1 * gdat.meanperiline[j]
        
        numbbins = 60
        indxbins = np.arange(numbbins)
        binspara = np.empty((numbbins + 1, numbpara)) 
        for k in indxpara:
            binspara[:, k] = np.linspace(gdat.limtpara[0, k], gdat.limtpara[1, k], numbbins + 1)
        meanpara = (binspara[1:, :] + binspara[:-1, :]) / 2.
    
        dictllik = [gdat, ttvrtype, 'post']
        
        print 'meanpara'
        print meanpara
        if samptype == 'emce':
            numbwalk = 20
            indxwalk = np.arange(numbwalk)
            gdat.parainit = []
            gdat.meanparainit = (gdat.limtpara[0, :] + gdat.limtpara[1, :]) / 2.
            for k in indxwalk:
                gdat.parainit.append(np.empty(numbpara))
                stdvnorm = (gdat.limtpara[0, :] - gdat.limtpara[1, :]) / 10.
                gdat.parainit[k]  = (scipy.stats.truncnorm.rvs((gdat.limtpara[0, :] - gdat.meanparainit) / stdvnorm, \
                                                                    (gdat.limtpara[1, :] - gdat.meanparainit) / stdvnorm)) * stdvnorm + gdat.meanparainit
            numbsampwalk = numbsamp[h] / numbwalk
            numbsampwalkburn = numbsampburn[h] / numbwalk
            if gdat.diagmode:
                if numbsampwalk == 0:
                    raise Exception('')
            gdat.initindxtranmodl, gdat.inittimetranmodl, \
                    gdat.initindxtranmodlproj, gdat.inittimetranmodlproj = retr_modl(gdat.meanparainit, gdat, ttvrtype)
            listvarb = [[gdat.initindxtranmodl], [gdat.inittimetranmodl], [gdat.initindxtranmodlproj], [gdat.inittimetranmodlproj]]
            plot_ttvr(gdat, 'resi', ttvrtype, listvarb, 'init')
            #raise Exception('')
            objtsamp = emcee.EnsembleSampler(numbwalk, numbpara, retr_lpos, args=dictllik, pool=multiprocessing.Pool())
            #objtsamp = emcee.EnsembleSampler(numbwalk, numbpara, retr_lpos, args=dictllik)
            if numbsampwalkburn > 0:
                gdat.indxswep = 0
                gdat.parainitburn, prob, state = objtsamp.run_mcmc(gdat.parainit, numbsampwalkburn, progress=True)
                objtsamp.reset()
            else:
                gdat.parainitburn = gdat.parainit
            gdat.indxswep = 0
            objtsamp.run_mcmc(gdat.parainitburn, numbsampwalk, progress=True)
            objtsave = objtsamp
        else:
        
            sampler = dynesty.NestedSampler(retr_llik, icdf, numbpara, logl_args=dictllik, ptform_args=dictllik, bound='single', dlogz=1000.)
            sampler.run_nested()
            results = sampler.results
            results.summary()
            objtsave = results
       
        indxsamp = np.arange(numbsamp[h])
        
        gdat.parapost = objtsave.flatchain
        if ttvrtype != 'sigm':
            indxsampplot = indxsamp[::100]
            gdat.sampindxtranmodl = [[] for i in indxsampplot]
            gdat.samptimetranmodl = [[] for i in indxsampplot]
            gdat.sampindxtranmodlproj = [[] for i in indxsampplot]
            gdat.samptimetranmodlproj = [[] for i in indxsampplot]
            for i in range(indxsampplot.size):
                gdat.sampindxtranmodl[i], gdat.samptimetranmodl[i], gdat.sampindxtranmodlproj[i], gdat.samptimetranmodlproj[i] = \
                                                                                             retr_modl(gdat.parapost[indxsampplot[i], :], gdat, ttvrtype)
            listvarb = gdat.sampindxtranmodl, gdat.samptimetranmodl, gdat.sampindxtranmodlproj, gdat.samptimetranmodlproj
            plot_ttvr(gdat, 'resi', ttvrtype, listvarb, 'post')
            
        if samptype == 'emce':
            #numbsamp = objtsave.flatchain.shape[0]
            indxsampwalk = np.arange(numbsampwalk)
        else:
            pass
            #numbsamp = objtsave['samples'].shape[0]
        
        # resample the nested posterior
        if samptype == 'nest':
            weights = np.exp(results['logwt'] - results['logz'][-1])
            samppara = dyutils.resample_equal(results.samples, weights)
            assert samppara.size == results.samples.size
        
        if samptype == 'emce':
            listsamp = objtsave.flatchain
            listllik = objtsave.flatlnprobability
        else:
            listsamp = samppara[:, k]
       
        indxsampmlik = np.argmax(listllik)
    
        # plot the posterior
        ## parameter
        ### trace
        for k in indxpara:
            if samptype == 'emce':
                figr, axis = plt.subplots()
                for i in indxwalk:
                    axis.plot(indxsampwalk[::10], objtsave.chain[i, ::10, k])
                path = gdat.pathimag + '%s/tracwalk%04d_%s.pdf' % (samptype, k, ttvrtype)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
            
        ### histogram
        for k in indxpara:
            figr, axis = plt.subplots(figsize=(6, 4))
            axis.hist(listsamp[:, k], numbbins) 
            axis.set_ylabel('$N_{samp}$')
            axis.set_xlabel(listlablpara[k])
            path = gdat.pathimag + '%s/hist%04d_%s.pdf' % (samptype, k, ttvrtype)
            plt.tight_layout()
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
        


        listsampproc = np.empty((numbsamp, gdat.numbplan))
        for j in gdat.indxplan:
            listsampproc[:, j] = tranmodl[j][100]
            listsampproc[:, j] -= np.mean(listsampproc[:, j])
            listlablpara.append('T_{p,%s}' % gdat.liststrgplan[j])
        listsamp = np.concatenate((listsamp, listsampproc))
        
        path = gdat.pathimag + '%s/' % samptype
        strgplot = 'post_%s' % ttvrtype
        
        listparamlik = listsamp[indxsampmlik, :]
        
        tdpy.mcmc.plot_grid(path, strgplot, listsamp, listlablpara, listvarbdraw=[listparamlik], numbbinsplot=numbbins)
        
        if samptype == 'nest':
            for keys in objtsave:
                if isinstance(objtsave[keys], np.ndarray) and objtsave[keys].size == numbsamp:
                    figr, axis = plt.subplots()
                    axis.plot(indxsamp, objtsave[keys])
                    path = gdat.pathimag + '%s/%s_%s.pdf' % (samptype, keys, ttvrtype)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
        else:
            ## log-likelihood
            figr, axis = plt.subplots()
            if samptype == 'emce':
                for i in indxwalk:
                    axis.plot(indxsampwalk, objtsave.lnprobability[:, i])
            else:
                axis.plot(indxsamp, objtsave['logl'])
            path = gdat.pathimag + '%s/llik_%s.pdf' % (samptype, ttvrtype)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
        
            chi2 = -2. * objtsave.lnprobability
            
            print('Posterior-mean chi2: ')
            print(np.mean(chi2))
            print('Posterior-mean chi2 per dof: ')
            print(np.mean(chi2) / numbdoff)
            print('Minimum chi2: ')
            print(np.amin(chi2))
            print('Minimum chi2 per dof: ')
            print(np.amin(chi2) / numbdoff)
            print('Posterior-mean llik: ')
            print(np.mean(objtsave.lnprobability))
            print('Maximum llik: ')
            print(np.amax(objtsave.lnprobability))
        
        ### nested sampling specific
        if samptype == 'nest':
            rfig, raxes = dyplot.runplot(results)
            path = gdat.pathimag + '%s/dyne_runs_%s.pdf' % (samptype, ttvrtype)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
            
            tfig, taxes = dyplot.traceplot(results)
            path = gdat.pathimag + '%s/dyne_trac_%s.pdf' % (samptype, ttvrtype)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
            
            cfig, caxes = dyplot.cornerplot(results)
            path = gdat.pathimag + '%s/dyne_corn_%s.pdf' % (samptype, ttvrtype)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
    
def cnfg_t270():
    pathdata = '/Users/tdaylan/DropboxMITTTT/knownplanets/toi-270/'
    pathtmpt = '%s/allesfit_tmpt/' % pathdata
    liststrgplan = ['b', 'c', 'd']
    offs = [17, 10, 5]
    epoc = [2458444.2140459623, 2458446.104473652, 2458446.5783906463]
    peri = [3.360062366764236, 5.660172076246358, 11.38028139190828]
    liststrginst = ['TESS', 'Trappist-South_z_2', 'LCO-SAAO-1m0_ip', 'Trappist-South_z_1', 'LCO_ip', 'LCO-SSO-1m0_ip_1', 'mko-cdk700_g', 'Myers_B']
    pathtarg = '%s/allesfit_tmpt/' % pathdata
    ttvr.util.ttvr(pathdata, pathtmpt, epoc, peri, offs, liststrginst, booltmptmcmc=False)


init()


