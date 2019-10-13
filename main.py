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
import matplotlib.gridspec as gridspec
import time

import allesfitter


def exec_alle_ttvr(pathdata, pathtmpt, epoc, peri, offs, liststrginst, booltmptmcmc=True, liststrgplan=['b']):
    
    # read ns_table of the global fit, constructing a template params.csv 
    if booltmptmcmc:
        strgtemp = 'mcmc'
    else:
        strgtemp = 'ns'
    objtfile = open(pathtmpt + 'results/%s_table.csv' % strgtemp)
    objtfilesave = open(pathtmpt + 'params.csv', 'w')
    objtfilesave.write('#name,value,fit,bounds,label,unit\n')
    for k, line in enumerate(objtfile):
        if line.startswith('#'):
            objtfilesave.write(line)
        elif not ('CTIO' in line or \
                'PEST' in line or \
                'LCO-SSO-1m0_ip_2' in line or \
                'TESS5' in line or \
                'LCO-SAAO-1m_gp' in line or \
                'LCO-SSO-1m0_gp' in line):
            linesplt = line.split(',')
            if 'TESS34spoc' in linesplt[0]:
                linespltprim = linesplt[0].split('TESS34spoc')
                linesplt[0] = linespltprim[0] + 'TESS' + linespltprim[1]
            lineneww = linesplt[0] + ',' + linesplt[1] + ',0,uniform -1e12 1e12,temp,\n'
            objtfilesave.write(lineneww)
    objtfilesave.close()
    
    numbplan = len(liststrgplan)
    indxplan = np.arange(numbplan)
    
    numbtran = np.empty(numbplan, dtype=int)
    indxtran = []
    numbtran = [10000 for k in range(numbplan)]
    timetran = []
    for k, strgplan in enumerate(liststrgplan):
        indxtran.append(np.arange(numbtran[k]))
        timetran.append(epoc[k] + (indxtran[k] - offs[k]) * peri[k])
    
    timetole = 0.25 # [days]
    
    pathpost = pathdata + 'allesfit_sttv/post/'
    os.system('mkdir -p %s' % pathpost)
    
    numbinst = len(liststrginst)
    arry = {}
    time = {}
    minmtime = 1e100
    maxmtime = -1e100
    for strginst in liststrginst:
        temp = np.loadtxt(pathtmpt + strginst + '.csv', delimiter=',')
        if strginst == 'TESS':
            arry[strginst] = temp
        else:
            arry[strginst] = temp[:, :3]
        time[strginst] = temp[:, 0]
        minmtime = min(np.amin(time[strginst]), minmtime)
        maxmtime = max(np.amax(time[strginst]), maxmtime)
    limttime = [minmtime, maxmtime]
    numbplan = len(liststrgplan)
    indxplan = np.arange(numbplan)
    indxtranskip = [[] for k in indxplan]
    for k, strgplan in enumerate(liststrgplan):
        for l in indxtran[k]:
            pathtran = pathdata + 'allesfit_sttv/allesfit_%s%03d/' % (strgplan, l)
            indx = {}
            booltraninst = np.zeros(numbinst, dtype=bool)
            for m, strginst in enumerate(liststrginst):
                indx[strginst] = np.where(abs(time[strginst] - timetran[k][l]) < timetole)[0]
                if indx[strginst].size > 0:
                    booltraninst[m] = True
                
            if np.where(booltraninst)[0].size == 0:
                indxtranskip[k].append(l)
                continue
    
            if os.path.exists(pathtran):
                continue
    
            cmnd = 'mkdir -p %s' % pathtran
            os.system(cmnd)
    
            cmnd = 'cp %ssettings.csv %s' % (pathtmpt, pathtran)
            os.system(cmnd)
            
            cmnd = 'cp %sparams.csv %s' % (pathtmpt, pathtran)
            os.system(cmnd)
            
            for m, strginst in enumerate(liststrginst):
                if booltraninst[m]:
                    path = '%s%s.csv' % (pathtran, strginst)
                    print('Writing to %s...' % path)
                    np.savetxt(path, arry[strginst][indx[strginst], :], delimiter=',')
    
            for a in range(2):
                if a == 0:
                    pathfile = '%ssettings.csv' % pathtran
                else:
                    pathfile = '%sparams.csv' % pathtran
                objtfile = open(pathfile)
                listline = []
                
                for line in objtfile:
    
                    if a == 0:
                        if line.startswith('companions_phot'):
                            line = 'companions_phot,%s\n' % strgplan
                            listline.append(line)
                        elif line.startswith('inst_phot'):
                            line = 'inst_phot,'
                            cntr = 0
                            for m, strginst in enumerate(liststrginst):
                                if booltraninst[m]:
                                    if cntr == 0:
                                        line += strginst
                                    else:
                                        line += ' ' + strginst
                                    cntr += 1
                            line += '\n'
                            listline.append(line)
                        elif 'TESS' in line:
                            linesplttess = line.split('TESS')
                            for m, strginst in enumerate(liststrginst):
                                if booltraninst[m]:
                                    listline.append(linesplttess[0] + strginst + linesplttess[1])
                        else:
                            listline.append(line)
                            
                    else:
                        if line[2:].startswith('epoch'):
                            strgplantemp = line[0]
                            linesplt = line.split(',')
                            if strgplantemp == strgplan:
                                linesplt[2] = '1'
                                line = ','.join(linesplt)
                            listline.append(line)
                        else:
                            boolfine = True
                            for m, strginst in enumerate(liststrginst):
                                if not booltraninst[m] and strginst in line:
                                    boolfine = False
                            if boolfine:
                                listline.append(line)
                objtfile.close()
                print('Writing to %s...' % pathfile)
                objtfile = open(pathfile, 'w')
                for line in listline:
                    objtfile.write(line)
                objtfile.close()
            print
    
    listtimetran = []
    listtimeresi = []
    for k, strgplan in enumerate(liststrgplan):
        for l in indxtran[k]:
            
            if l in indxtranskip[k]:
                continue
            
            pathtran = pathdata + 'allesfit_sttv/allesfit_%s%03d/' % (strgplan, l)
            
            allesfitter.show_initial_guess(pathtran)
            allesfitter.mcmc_fit(pathtran)
            allesfitter.mcmc_output(pathtran)
            
            for strgtemp in ['mcmc_fit', 'initial_guess']:
                cmnd = 'cp %sallesfit_sttv/allesfit_%s%03d/results/%s_%s.pdf %s%s_%s_%d.pdf' \
                                                    % (pathdata, strgplan, l, strgtemp, strgplan, pathpost, strgtemp, strgplan, l)
                os.system(cmnd)
            
            pathsave = pathdata + 'allesfit_sttv/allesfit_%s%03d/results/mcmc_save.h5' % (strgplan, l)
            print('Reading %s...' % pathsave)
            emceobjt = emcee.backends.HDFBackend(pathsave, read_only=True)
            if cntr == 0:
                numbsamp = emceobjt.get_chain().size
                timeresi = np.zeros((numbtran[k], numbsamp)) - 1e12
            timeresi[l, :] = (epoc[k] - emceobjt.get_chain().flatten()) * 24. * 60. # [min]
        listtimetran.append(timetran[k])
        listtimeresi.append(timeresi)
    
    pathsave = pathdata + 'allesfit_sttv/post/per_transit_ttv.csv'
    objtfile = open(pathsave, 'w')
    for a in range(2):
        figr, axis = plt.subplots(figsize=(12, 6))
        ylim = 0.
        for k, strgplan in enumerate(liststrgplan):
            ydat = np.mean(listtimeresi[k], 1)
            yerr = np.std(listtimeresi[k], 1)
            ylim = max(ylim, np.amax(abs(ydat)))
            axis.errorbar(listtimetran[k], ydat, yerr=yerr, label=strgplan, ls='', marker='o')
            # write to csv file
            if a == 0:
                objtfile.write(strgplan + '\n')
                for n in range(ydat.size):
                    objtfile.write('%g, %g\n' % (listtimetran[k][n], ydat[n]))
        if a == 0:
            strg = 'zoom'
            ylim = 10
        else:
            strg = 'full'
            ylim = 50
        axis.set_xlabel('T [BJD]')
        axis.set_xlim(limttime)
        axis.set_ylabel('O-C [min]')
        axis.set_ylim([-ylim, ylim])
        axis.axhline(0., ls='--', color='gray', alpha=0.5)
        axis.legend()
        path = pathdata + 'allesfit_sttv/post/resi_%s.pdf' % strg
        print('Writing to %s...' % path)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    objtfile.close()


def icdf(para, numbepoc):
    
    icdf = limtpara[0, :] + para * (limtpara[1, :] - limtpara[0, :])

    return icdf


def retr_lpos(para, gdat, ttvrtype, strgpdfn):
    
    if ((para < gdat.limtpara[0, :]) | (para > gdat.limtpara[1, :])).any():
        lpos = -np.inf
    else:
        llik = retr_llik(para, gdat, ttvrtype, strgpdfn)
        lpos = llik
    
    return lpos


def retr_ttvrmodl_sinu(time, offs, phas, ampl, peri):
    
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


def init_sinu():
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
            objtsamp = emcee.EnsembleSampler(numbwalk, numbpara, retr_lpos_sinu, args=dictllik)
            parainitburn, prob, state = objtsamp.run_mcmc(parainit, numbsampwalkburn)
            objtsamp.reset()
            objtsamp.run_mcmc(parainitburn, numbsampwalk)
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
                print('Writing to %s...' % path)
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
            print('Writing to %s...' % path)
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
                    print('Writing to %s...' % path)
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
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
        
            chi2 = -2. * objtsave.lnprobability
            
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
            print('Writing to %s...' % path)
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
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        ### nested sampling specific
        if samptype == 'nest':
            rfig, raxes = dyplot.runplot(results)
            path = pathdata + '%s/dyne_runs_%s.pdf' % (samptype, ttvrtype)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
            
            tfig, taxes = dyplot.traceplot(results)
            path = pathdata + '%s/dyne_trac_%s.pdf' % (samptype, ttvrtype)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
            
            cfig, caxes = dyplot.cornerplot(results)
            path = pathdata + '%s/dyne_corn_%s.pdf' % (samptype, ttvrtype)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
    

def plot_stdv(gdat, ttvrtype, listvarb):
    
    figr, axis = plt.subplots(3, 1, figsize=(12, 10))
    for j in gdat.indxplan:
        axis[j].set_ylabel('$\sigma_T$')
        axis[j].set_xlabel('$i_T$')
        axistwin = axis[j].twiny()
        axistwin.set_xlabel('$T$ [day]')
        listtemptemp = []
        numbsamp = len(listvarb[0])
        sizearry = np.empty(numbsamp, dtype=int)
        for i in range(len(listvarb[0])):
            listtemptemp.append(listvarb[1][i][j])
            sizearry[i] = listvarb[1][i][j].size
        minmsizearry = np.amin(sizearry)
        listtemp = np.empty((numbsamp, minmsizearry))
        for i in range(numbsamp):
            listtemp[i, :] = listtemptemp[i][:minmsizearry]
        axis[j].plot(listvarb[0][0][j][:minmsizearry], np.std(listtemp, 0) * gdat.facttime)
        limtindx = np.array(axis[j].get_xlim())
        limttime = limtindx * gdat.meanperiline[j]
        axistwin.set_xlim(limttime)
        limt = np.array([astropy.time.Time('2019-09-01T00:00:00', format='isot', scale='utc').jd, \
                     astropy.time.Time('2020-09-01T00:00:00', format='isot', scale='utc').jd]) - gdat.timeobsdinit
        axistwin.fill_between(limt, -10, 10, color='purple', alpha=0.2)
    plt.tight_layout()
    path = gdat.pathimag + 'stdvtimetranmodl_%s.pdf' % (ttvrtype)
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()



def plot_ttvr(gdat, strgplottype, ttvrtype, listvarb, strgpdfn):
    
    listindxtranmodl, listtimetranmodl, listindxtranmodlproj, listtimetranmodlproj = listvarb

    numbsamp = len(listindxtranmodl)
    indxsamp = np.arange(numbsamp)
    
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
                                                yerr=gdat.stdvtimetranobsd[j]*gdat.facttime, label='Observed - Linear', ls='', markersize=2, marker='o')
                
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


def retr_modl(para, gdat, ttvrtype):
    
    mass = para[:gdat.numbplan]
    peri = para[1*gdat.numbplan:2*gdat.numbplan]
    ecce = para[2*gdat.numbplan:3*gdat.numbplan]
    incl = para[3*gdat.numbplan:4*gdat.numbplan]
    land = para[4*gdat.numbplan:5*gdat.numbplan]
    argu = para[5*gdat.numbplan:6*gdat.numbplan]
    anom = para[6*gdat.numbplan:7*gdat.numbplan]
    
    paraglob = [
        0.000295994511,# G
        0.40, # Mstar
        
        #planet b
        mass[0],# mass of the planet
        peri[0],# period of the planet
        ecce[0],# mass of the planet
        incl[0],# period of the planet
        land[0],# mass of the planet
        argu[0],# period of the planet
        anom[0],# mass of the planet
        
        
        #planet c
        mass[1],# mass of the planet
        peri[1],# period of the planet
        ecce[1],# mass of the planet
        incl[1],# period of the planet
        land[1],# mass of the planet
        argu[1],# period of the planet
        anom[1],# mass of the planet
    
        #planet d
        mass[2],# mass of the planet
        peri[2],# period of the planet
        ecce[2],# mass of the planet
        incl[2],# period of the planet
        land[2],# mass of the planet
        argu[2],# period of the planet
        anom[2],# mass of the planet
        ]
    
    planet1 = ttvfast.models.Planet(*paraglob[2:9])
    planet2 = ttvfast.models.Planet(*paraglob[9:16])
    planet3 = ttvfast.models.Planet(*paraglob[16:])
    
    gravity, gdat.massstar = paraglob[:2]
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
        if gdat.diagmode:
            if timetranmodlproj[j].size != gdat.timetranobsd[j].size:
                print('j')
                print(j)
                print('timetranmodlproj[j]')
                print(timetranmodlproj[j])
                print('gdat.timetranobsd[j]')
                print(gdat.timetranobsd[j])
                raise Exception('')
    
    return indxtranmodl, timetranmodl, indxtranmodlproj, timetranmodlproj
            

def retr_llik(para, gdat, ttvrtype, strgpdfn):
    
    if strgpdfn == 'init' or ttvrtype != 'sigm':
        indxtranmodl, timetranmodl, indxtranmodlproj, timetranmodlproj = retr_modl(para, gdat, ttvrtype)
    else:
        timetranmodlproj = gdat.inittimetranmodlproj
    
    for j in gdat.indxplan:
        if strgpdfn == 'init' or ttvrtype != 'sigm':
            indx = indxtranmodl[j]
            if indx.size == 0:
                return -np.inf
    
        if timetranmodlproj[j].size < gdat.timetranobsd[j].size:
            return -np.inf
    
    if strgpdfn == 'post':
        gdat.indxswep += 1

    llik = 0.
    for j in gdat.indxplan:
        lliktemp = -0.5 * np.sum((gdat.timetranobsd[j] - timetranmodlproj[j])**2 / (gdat.stdvtimetranobsd[j])**2)
        llik += lliktemp
    
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
    
    path = gdat.pathdata + 'measured_allesfit_all_ttv.pickle'
    objtfile = open(path,'rb')
    datapickttvr = pickle.load(objtfile)
    
    gdat.indxtranobsd = [[] for j in gdat.indxplan]
    gdat.timetranlineproj = [[] for j in gdat.indxplan]
    gdat.timetranobsd = [[] for j in gdat.indxplan]
    gdat.stdvtimetranobsd = [[] for j in gdat.indxplan]
    for j in gdat.indxplan:
        gdat.timetranobsd[j] = datapickttvr[gdat.liststrgplan[j]]['transit_time']
        gdat.stdvtimetranobsd[j] = datapickttvr[gdat.liststrgplan[j]]['transit_time_err']
        gdat.timetranobsd[j] -= gdat.timeobsdinit
    
    #datapickttvr['lin_period']
    
    ttvrtype = 'totl'

    for j in gdat.indxplan:
        gdat.indxtranobsd[j] = np.round(gdat.timetranobsd[j] / gdat.meanperiline[j]).astype(int)
    
    # convert Earth mass to Solar mass
    
    samptype = 'emce'
    
    # setttings
    ## modeling 
    boolecce = False
    maxmecce = 0.1
    gdat.inittimefastttvr = 0
    gdat.delttimefastttvr = 0.03
    gdat.numbstepfastttvr = 1000
    
    # run the TTV Simulation
    #results = ttvfast.ttvfast(listobjtplan, gdat.massstar, gdat.inittimefastttvr, gdat.delttimefastttvr, gdat.numbstepfastttvr)
    
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
    numbsamp = 10000
    numbsampburn = 2000
        
    numbdata = 0
    for j in gdat.indxplan:
        numbdata += gdat.indxtranobsd[j].size
    
    listlablpara = []
    for strgfrst, strgseco in [['P', ' [day]'], ['M', ' [M_\odot]'], ['e', ''], ['i', ' [deg]'], ['l', ' [deg]'], \
                                                                                        ['w', ' [deg]'], ['TA', ' [deg]']]:
        for j in gdat.indxplan:
            listlablpara.append('$%s_%s$%s' % (strgfrst, gdat.liststrgplan[j], strgseco))
    print('listlablpara')
    print(listlablpara)
    numbpara = len(listlablpara)
    indxpara = np.arange(numbpara)
    gdat.limtpara = np.empty((2, numbpara))
    numbdoff = numbdata - numbpara
    
        #0,#e
        #88.65,#i
        #0,#longNode
        #0, #argument
        #89.99999, #Mean Anomolay
        #0, #e
        #89.53,#i
        #0,#longNode
        #0,#Argument
        #296.7324,#Mean Anomaly
        #0,#e
        #89.69,#i
        #0,#longNode
        #0,#Argument
        #8.25829165761]#Mean anomaly
    
    gdat.numbparaplan = 7
    gdat.indxparaplan = np.arange(gdat.numbparaplan)
    gdat.meanparaplan = np.empty((gdat.numbplan, gdat.numbparaplan))
    gdat.stdvparaplan = np.empty((gdat.numbplan, gdat.numbparaplan))
    gdat.meanparaplan[:, 0] = np.array([2.47, 5.46, 2.55]) * 0.00000300245
    gdat.stdvparaplan[:, 0] = np.array([0.75, 1.30, 0.91]) * 0.00000300245
    
    gdat.meanparaplan[:, 1] = gdat.meanperiline
    gdat.stdvparaplan[:, 1] = gdat.meanparaplan[:, 1] * 1e-3
        
    gdat.meanparaplan[:, 2] = np.array([0., 0., 0.])
    gdat.stdvparaplan[:, 2] = gdat.meanparaplan[:, 2] + 1e-3
    
    gdat.meanparaplan[:, 3] = np.array([88.65, 89.53, 89.69])
    gdat.stdvparaplan[:, 3] = gdat.meanparaplan[:, 3] * 1e-3
    
    gdat.meanparaplan[:, 4] = np.array([0., 0., 0.])
    gdat.stdvparaplan[:, 4] = gdat.meanparaplan[:, 4] + 1e-3
    
    gdat.meanparaplan[:, 5] = np.array([0., 0., 0.])
    gdat.stdvparaplan[:, 5] = gdat.meanparaplan[:, 5] + 1e-3
    
    gdat.meanparaplan[:, 6] = np.array([89.99999, 296.7324, 8.25829165761])
    gdat.stdvparaplan[:, 6] = gdat.meanparaplan[:, 6] * 1e-3
    
    for n in gdat.indxparaplan:
        for j in gdat.indxplan:
            gdat.limtpara[0, j+n*gdat.numbplan] = gdat.meanparaplan[j, n] - gdat.stdvparaplan[j, n]
            gdat.limtpara[1, j+n*gdat.numbplan] = gdat.meanparaplan[j, n] + gdat.stdvparaplan[j, n]
   
    print('gdat.meanparaplan')
    print(gdat.meanparaplan)
    print('gdat.stdvparaplan')
    print(gdat.stdvparaplan)
    print('gdat.limtpara')
    print(gdat.limtpara)
    numbbins = 60
    indxbins = np.arange(numbbins)
    binspara = np.empty((numbbins + 1, numbpara)) 
    for k in indxpara:
        binspara[:, k] = np.linspace(gdat.limtpara[0, k], gdat.limtpara[1, k], numbbins + 1)
    meanpara = (binspara[1:, :] + binspara[:-1, :]) / 2.
    
    dictllik = [gdat, ttvrtype, 'post']
    
    if samptype == 'emce':
        numbwalk = 2 * numbpara
        indxwalk = np.arange(numbwalk)
        gdat.parainit = []
        gdat.meanparainit = (gdat.limtpara[0, :] + gdat.limtpara[1, :]) / 2.
        for k in indxwalk:
            gdat.parainit.append(np.empty(numbpara))
            stdvnorm = (gdat.limtpara[0, :] - gdat.limtpara[1, :]) / 10.
            print('k')
            print(k)
            print('stdvnorm')
            print(stdvnorm)
            print('gdat.limtpara')
            print(gdat.limtpara)
            print('gdat.meanparainit')
            print(gdat.meanparainit)
            print

            gdat.parainit[k]  = (scipy.stats.truncnorm.rvs((gdat.limtpara[0, :] - gdat.meanparainit) / stdvnorm, \
                                                                (gdat.limtpara[1, :] - gdat.meanparainit) / stdvnorm)) * stdvnorm + gdat.meanparainit
        numbsampwalk = numbsamp / numbwalk
        numbsampwalkburn = numbsampburn / numbwalk
        if gdat.diagmode:
            if numbsampwalk == 0:
                raise Exception('')
        print('gdat.meanparainit')
        print(gdat.meanparainit)
        gdat.initindxtranmodl, gdat.inittimetranmodl, \
                gdat.initindxtranmodlproj, gdat.inittimetranmodlproj = retr_modl(gdat.meanparainit, gdat, ttvrtype)
        listvarb = [[gdat.initindxtranmodl], [gdat.inittimetranmodl], [gdat.initindxtranmodlproj], [gdat.inittimetranmodlproj]]
        plot_ttvr(gdat, 'resi', ttvrtype, listvarb, 'init')
        #raise Exception('')
        objtsamp = emcee.EnsembleSampler(numbwalk, numbpara, retr_lpos, args=dictllik, pool=multiprocessing.Pool())
        #objtsamp = emcee.EnsembleSampler(numbwalk, numbpara, retr_lpos, args=dictllik)
        if numbsampwalkburn > 0:
            gdat.indxswep = 0
            print('gdat.parainit')
            print(gdat.parainit)
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
    
    timetranwind = [[] for j in gdat.indxplan]
    timetranwind[0] = []
    timetranwind[1] = np.array([
                                  2458830.995990, \
                                  2458836.656162, \
                                  2458842.316334, \
                                  2458847.976506, \
                                  2458887.597710, \
                                  2458893.257882, \
                                  2458898.918054, \
                                  2458904.578226, \
                                  2458921.558742, \
                                  2458927.218914, \
                                  2458938.539258, \
                                  2458944.199430, \
                                  2458955.519774, \
                                  2458961.179946, \
                                  2458966.840118, \
                                  2458972.500290, \
                                  2458978.160462, \
                                  2458983.820634, \
                                  2459006.461322, \
                                  2459012.121494, \
                                  2459017.781666, \
                                  2459023.441838, \
                                  2459057.402870, \
                                  2459063.063042, \
                                  2459068.723214, \
                                  2459074.383386, \
                                  2459080.043558, \
                                 ])
    timetranwind[2] = np.array([
                                  2458890.403520, \
                                  2458901.783660, \
                                  2458924.543940, \
                                  2458935.924080, \
                                  2458958.684360, \
                                  2458970.064500, \
                                  2458981.444640, \
                                  2459004.204920, \
                                  2459015.585060, \
                                  2459026.965200, \
                                  2459061.105620, \
                                  2459072.485760, \
                                 ])
    for j in gdat.indxplan:
        if j == 0:
            continue
        timetranwind[j] -= gdat.timeobsdinit
    indxsamp = np.arange(numbsamp)
    
    gdat.parapost = objtsave.flatchain
    if ttvrtype != 'sigm':
        gdat.indxsampfeww = indxsamp[::100]
        gdat.numbsampfeww = gdat.indxsampfeww.size
        gdat.sampindxtranmodl = [[] for i in gdat.indxsampfeww]
        gdat.samptimetranmodl = [[] for i in gdat.indxsampfeww]
        gdat.sampindxtranmodlproj = [[] for i in gdat.indxsampfeww]
        gdat.samptimetranmodlproj = [[] for i in gdat.indxsampfeww]
        for i in range(gdat.indxsampfeww.size):
            gdat.sampindxtranmodl[i], gdat.samptimetranmodl[i], gdat.sampindxtranmodlproj[i], gdat.samptimetranmodlproj[i] = \
                                                                                     retr_modl(gdat.parapost[gdat.indxsampfeww[i], :], gdat, ttvrtype)
        listvarb = gdat.sampindxtranmodl, gdat.samptimetranmodl, gdat.sampindxtranmodlproj, gdat.samptimetranmodlproj
        plot_ttvr(gdat, 'resi', ttvrtype, listvarb, 'post')
        
        plot_stdv(gdat, ttvrtype, listvarb)
    
    numbsampfeww = gdat.indxsampfeww.size

    for j in gdat.indxplan:
        if j == 0:
            continue
        numbtranwind = timetranwind[j].size
        timetranwindpred = np.empty((numbtranwind, numbsampfeww))
        for k, timetran in enumerate(timetranwind[j]):
            for i in range(numbsampfeww):
                indx = np.argmin(np.abs(gdat.samptimetranmodl[i][j] - timetran))
                timetranwindpred[k, i] = gdat.samptimetranmodl[i][j][indx]

            print('%f %f %g %g' % (gdat.timeobsdinit + timetranwind[j][k], gdat.timeobsdinit + np.mean(timetranwindpred[k, :]), \
                                                                                    np.std(timetranwindpred[k, :]) * gdat.facttime, \
                                                                                    (np.mean(timetranwindpred[k, :]) - timetran) * gdat.facttime))

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
    
    listsampproc = np.empty((gdat.numbsampfeww, gdat.numbplan))
    for j in gdat.indxplan:
        for ii, i in enumerate(gdat.indxsampfeww):
            listsampproc[ii, j] = gdat.samptimetranmodl[ii][j][40]
        listsampproc[:, j] -= np.mean(listsampproc[:, j])
        listsampproc[:, j] *= gdat.facttime
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
    exec_alle_ttvr(pathdata, pathtmpt, epoc, peri, offs, liststrginst, booltmptmcmc=False)


def cnfg_t270_ttvr():
    
    init()

globals().get(sys.argv[1])()

