"""
Dynamical modeling of exoplanet systems

Tansu Daylan
MIT Kavli Institute, Cambridge, MA, 02109, US
tansu.daylan@gmail.com
www.tansudaylan.com
"""

import numpy as np
import scipy
import h5py 
import os, sys

import astropy

import ttvfast

import tdpy.mcmc
from tdpy.util import summgene
import tdpy

import matplotlib as mpl
from matplotlib import pyplot as plt


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
    indxplan = np.arange(numbplan, dtype=int)
    
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


def plot_ttvr(gdat, modltype, listvarb):
    
    listindxtranmodl, listtimetranmodl, listindxtranmodlproj, listtimetranmodlproj = listvarb

    numbsamp = len(listindxtranmodl)
    indxsamp = np.arange(numbsamp)
    
    gdat.timetranline = [[] for j in gdat.indxplan]

    strgplot = 'resi'
    figrsize = (12, 8)
    
    figr = plt.figure(figsize=figrsize)
    gs = mpl.gridspec.GridSpec(gdat.numbplan, 2, width_ratios=[1, 2])
    axis = [[[] for a in range(2)] for j in gdat.indxplan]
    for a in range(2):
        for j in gdat.indxplan:
            axis[j][a] = plt.subplot(gs[j, a])
    axis[1][0].set_ylabel('$\Delta T$ [minute]')
    axis[1][1].set_ylabel('$\Delta T$ [minute]')
    
    for a in range(2):
        for j in gdat.indxplan:
            axis[j][a].set_xlabel('$i_T$')
    
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
                    lablseco = 'N-body - Linear'
                    #rasterized = False
                else:
                    lablseco = None
                    #rasterized = True
                if len(listindxtranmodl[i][j]) > 0:
                    gdat.timetranline[j] = gdat.meanepocline[j] + (listindxtranmodl[i][j] - gdat.numbtranoffs[j]) * \
                                                                                        gdat.meanperiline[j] - gdat.timeobsdinit
                    #axis[j][a].plot(listindxtranmodl[i][j], (listtimetranmodl[i][j] - listtimetranmodl[i][j][0] - \
                    #        listindxtranmodl[i][j] * (listtimetranmodl[i][j][-1] - listtimetranmodl[i][j][0]) / \
                    #        (listindxtranmodl[i][j].size - 1)) * gdat.facttime, \
                    #                  label=lablfrst, color='b', alpha=alph)
                    axis[j][a].plot(listindxtranmodl[i][j], (listtimetranmodl[i][j] - gdat.timetranline[j]) * gdat.facttime, \
                                                                                            label=lablseco, color='r', alpha=alph)
            axis[j][a].text(0.9, 0.13, gdat.liststrgplan[j], transform=axis[j][a].transAxes)
            
            if a == 0:
                axis[j][a].set_ylim([-30, 30])
                if j == 0:
                    axis[j][a].set_xlim([0., 40])
                if j == 1:
                    axis[j][a].set_xlim([0., 40])
                if j == 2:
                    axis[j][a].set_xlim([0., 15])
            axistwin = axis[j][a].twiny()
            axistwin.set_xlabel('Time')
            limtindx = np.array(axis[j][a].get_xlim())
            limttime = limtindx * gdat.meanperiline[j] + gdat.timeobsdinit
            axistwin.set_xlim(limttime)
            timetick = axistwin.get_xticks()
            lablxaxi = astropy.time.Time(timetick, format='jd', scale='utc').isot
            lablxaxi = [xaxi[:7] for xaxi in lablxaxi]
            axistwin.set_xticklabels(lablxaxi)

            axistwin = axis[j][a].twinx()
            axistwin.set_ylabel('$5\sigma_T$')
            axistwin.plot(gdat.indxtranstdv[j], gdat.stdvtime[j] * 5.)

    axis[0][1].legend()
    
    plt.tight_layout()
    path = gdat.pathimag + 'timetran_%s_%s_%s.pdf' % (strgplot, modltype, gdat.modltype)
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()


def retr_modl_sinu(time, offs, phas, ampl, peri):
    
    ttvrmodl = offs + ampl * np.sin(phas + 2. * np.pi * time / peri)

    return ttvrmodl


def retr_modl_full(para, gdat, modltype):
    
    indxtranmodl = [[] for j in gdat.indxplan]
    timetranmodl = [[] for j in gdat.indxplan]
    indxtranmodlproj = [[] for j in gdat.indxplan]
    timetranmodlproj = [[] for j in gdat.indxplan]
    if gdat.modltype == 'line':
        for j in gdat.indxplan:
            indxtranmodl[j] = np.arange(500)
            #print('j')
            #print(j)
            #print('para[j]')
            #print(para[j])
            #print

            timetranmodl[j] = para[j] + para[j+3] * indxtranmodl[j]
        
    if gdat.modltype == 'nbod':
    
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
            ecce[0],# eccentricity s of the planet
            incl[0],# inclination of the planet
            land[0],# longitude of the ascending node, Omega
            argu[0],# argument of periapsis, omega
            anom[0],# mean anomaly of the planet
            
            
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
        for j in gdat.indxplan:
            indx = np.where(np.array(results['positions'][0]) == j)[0]
            timetranmodl[j] = np.array(results['positions'][2])[indx]
            indxgood = np.where(timetranmodl[j] != -2.)[0]
            numbtran = indxgood.size
            timetranmodl[j] = timetranmodl[j][indxgood]
            indxtranmodl[j] = np.arange(numbtran, dtype=int)
            
    for j in gdat.indxplan:
        indxtranmodlproj[j] = np.intersect1d(indxtranmodl[j], gdat.indxtranobsd[j])
        timetranmodlproj[j] = timetranmodl[j][indxtranmodlproj[j]]
        
        #print('j')
        #print(j)
        #print('indxtranmodl[j]')
        #summgene(indxtranmodl[j])
        #print('timetranmodl[j]')
        #summgene(timetranmodl[j])
        #print('gdat.indxtranobsd[j]')
        #summgene(gdat.indxtranobsd[j])
        #print('gdat.timetranobsd[j]')
        #summgene(gdat.timetranobsd[j])
        #print('indxtranmodlproj[j]')
        #summgene(indxtranmodlproj[j])
        #print('timetranmodlproj[j]')
        #summgene(timetranmodlproj[j])
        #print('')

        if gdat.diagmode:
            if timetranmodlproj[j].size != gdat.timetranobsd[j].size:
                print('j')
                print(j)
                print('timetranmodlproj[j]')
                print(timetranmodlproj[j])
                print('gdat.timetranobsd[j]')
                print(gdat.timetranobsd[j])
                raise Exception('')
    #print('')
    #print('')
    #print('')
    
    return indxtranmodl, timetranmodl, indxtranmodlproj, timetranmodlproj
            
    
def retr_ttvrmodlline(gdat, epocline, periline):

    timetranmodlproj = [[] for j in gdat.indxplan]
    for j in gdat.indxplan:
        timetranmodlproj[j] = epocline + periline * np.ones_like(gdat.timetranobsd[j])
    
    return timetranmodlproj


def retr_llik(gdat, para):
    
    if gdat.modltype == 'line' or gdat.modltype == 'sinu':
        if gdat.modltype == 'line':
            epocline = para[0]
            periline = para[1]
            timetranmodlproj = retr_ttvrmodlline(gdat, epocline, periline)
        
        if gdat.modltype == 'sinu':
            epocline = para[0]
            periline = para[1]
            phassinu = para[2]
            amplsinu = para[3]
            perisinu = para[4]
            timetranmodlproj = retr_ttvrmodlsinu(gdat, epocline, periline, phassinu, amplsinu, perisinu)
    else:
        indxtranmodl, timetranmodl, indxtranmodlproj, timetranmodlproj = retr_modl(para, gdat)

        for j in gdat.indxplan:
            if timetranmodlproj[j].size < gdat.timetranobsd[j].size:
                return -np.inf
    
    llik = 0.
    for j in gdat.indxplan:
        lliktemp = -0.5 * np.sum((gdat.timetranobsd[j] - timetranmodlproj[j])**2 / (gdat.stdvtimetranobsd[j])**2)
        llik += lliktemp
        #print('j')
        #print(j)
        #print('gdat.timetranobsd[j]')
        #summgene(gdat.timetranobsd[j])
        #print(gdat.timetranobsd[j])
        #print('timetranmodlproj[j]')
        #summgene(timetranmodlproj[j])
        #print(timetranmodlproj[j])
        #print('gdat.stdvtimetranobsd[j]')
        #summgene(gdat.stdvtimetranobsd[j])
        #print(gdat.stdvtimetranobsd[j])
        #print('(gdat.timetranobsd[j] - timetranmodlproj[j])**2')
        #print((gdat.timetranobsd[j] - timetranmodlproj[j])**2)
        #print('(gdat.stdvtimetranobsd[j])**2')
        #print((gdat.stdvtimetranobsd[j])**2)
        #print('(gdat.timetranobsd[j] - timetranmodlproj[j])**2 / (gdat.stdvtimetranobsd[j])**2')
        #print((gdat.timetranobsd[j] - timetranmodlproj[j])**2 / (gdat.stdvtimetranobsd[j])**2)
        #print('lliktemp')
        #print(lliktemp)
        #print('')
    #print('llik')
    #print(llik)
    #print('')
    #print('')
    
    return llik


def init(
         # data
         timetranobsd, \
         stdvtimetranobsd, \

         # model
         minmpara, \
         maxmpara, \
         meanpara, \
         stdvpara, \

         timeobsdinit=None, \
        
         ## TTV model tyye: 'nbod', 'line' or 'sinu'
         listmodltype = ['line', 'nbod'], \
         # strings for the list of planets
         liststrgplan=None, \
         
         ## if eccentricity is included
         maxmecce = 0.1, \
        
        ):
    
    # construct global object
    gdat = tdpy.util.gdatstrt()
    
    # copy unnamed inputs to the global object
    #for attr, valu in locals().iter():
    for attr, valu in locals().items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    # paths
    gdat.pathbase = os.environ['TTVR_DATA_PATH'] + '/'
    gdat.pathdata = gdat.pathbase + 'data/'
    gdat.pathimag = gdat.pathbase + 'imag/'
    
    gdat.numbplan = len(gdat.meanpara)
    gdat.indxplan = np.arange(gdat.numbplan)
    
    gdat.facttime = 24. * 60.
    
    # setup 
    gdat.diagmode = True
    
    if gdat.liststrgplan is None:
        gdat.liststrgplan = ['b', 'c', 'd'][:gdat.numbplan]
   
    gdat.inittimefastttvr = 0
    gdat.delttimefastttvr = 0.03
    gdat.numbstepfastttvr = 700
    
    # The function ttvfast.ttvfast returns a dictionary containing positions and rv. The positions entry is a tuple of:
    # a list of integer indices for which values correspond to which planet,
    # a list of integers defining the epoch,
    # a list of times,
    # a list of rsky values, and
    # a list of vsky values.
    
    gdat.numbtranoffs = [22, 13, 7]
    #for j in gdat.indxplan:
    #    gdat.timetranlineproj[j] = gdat.meanepocline[j] + (gdat.indxtranobsd[j] - gdat.numbtranoffs[j]) * gdat.meanperiline[j] - gdat.timeobsdinit
    
    numbsampwalk = 15000
    numbsampburnwalk = 5000
    numbsampburnwalkseco = 5000
      
    # number of total data points
    numbdata = 0
    for j in gdat.indxplan:
        numbdata += gdat.timetranobsd[j].size
        
    for y, gdat.modltype in enumerate(gdat.listmodltype):

        listlablpara = []
        if gdat.modltype == 'line':
            listlistlablpara = [['E', ' [day]'], ['P', ' [day]']]
        if gdat.modltype == 'sinu':
            # temp
            listlistlablpara = [['E', ' [day]'], ['P', ' [day]']]
        if gdat.modltype == 'nbod':
            listlistlablpara = [['M', ' [$M_{\odot}$]'], ['P', ' [day]'], ['\epsilon', ''], ['i', ' [deg]'], ['\Omega', ' [deg]'], \
                                                                                                ['\omega', ' [deg]'], ['A', ' [deg]']]
        gdat.numbparaplan = len(listlistlablpara)
        for strgfrst, strgseco in listlistlablpara:
            for j in gdat.indxplan:
                listlablpara.append('$%s_%s$%s' % (strgfrst, gdat.liststrgplan[j], strgseco))
        numbpara = len(listlablpara)
        indxpara = np.arange(numbpara)
        numbdoff = numbdata - numbpara
        
        # impose priors
        gdat.indxparaplan = np.arange(gdat.numbparaplan)
        numbbins = 20
        
        gdat.numbpara = gdat.numbplan * gdat.numbparaplan
        gdat.indxpara = np.arange(gdat.numbpara)
        numbwalk = 2 * numbpara
        indxwalk = np.arange(numbwalk)
        
        #gdat.initindxtranmodl, gdat.inittimetranmodl, \
        #        gdat.initindxtranmodlproj, gdat.inittimetranmodlproj = retr_modl_full(gdat.meanparainit, gdat, modltype)
        #listvarb = [[gdat.initindxtranmodl], [gdat.inittimetranmodl], [gdat.initindxtranmodlproj], [gdat.inittimetranmodlproj]]
        

        listscalpara = ['self' for k in gdat.indxpara]
        print('gdat.minmpara')
        summgene(gdat.minmpara)
        print('listlablpara')
        print(listlablpara)
        parapost = tdpy.mcmc.samp(gdat, gdat.pathimag, numbsampwalk, numbsampburnwalk, numbsampburnwalkseco, retr_llik, \
                            listlablpara, listscalpara, gdat.minmpara[y], gdat.maxmpara[y], gdat.meanpara[y], gdat.stdvpara[y], numbdata)
            
        objtsave = objtsamp
        
        for j in gdat.indxplan:
            if j == 0:
                continue
            timetranwind[j] -= gdat.timeobsdinit
        
        gdat.parapost = objtsave.flatchain
        gdat.indxsampfeww = indxsamp[::100]
        gdat.numbsampfeww = gdat.indxsampfeww.size
        gdat.sampindxtranmodl = [[] for i in gdat.indxsampfeww]
        gdat.samptimetranmodl = [[] for i in gdat.indxsampfeww]
        gdat.sampindxtranmodlproj = [[] for i in gdat.indxsampfeww]
        gdat.samptimetranmodlproj = [[] for i in gdat.indxsampfeww]
        sizearry = np.empty((gdat.numbsampfeww, gdat.numbplan), dtype=int)
        listtemptemp = [[[] for i in gdat.indxsampfeww] for j in gdat.indxplan]
        for ii, i in enumerate(gdat.indxsampfeww):
            gdat.sampindxtranmodl[ii], gdat.samptimetranmodl[ii], gdat.sampindxtranmodlproj[ii], gdat.samptimetranmodlproj[ii] = \
                                                                                                        retr_modl(gdat.parapost[i, :], gdat)
            for j in gdat.indxplan:
                listtemptemp[j][ii] = gdat.samptimetranmodl[ii][j]
                sizearry[ii, j] = gdat.samptimetranmodl[ii][j].size
        minmsizearry = np.amin(sizearry, 0)
        gdat.stdvtime = [[] for j in gdat.indxplan]
        gdat.indxtranstdv = [[] for j in gdat.indxplan]
        for j in gdat.indxplan:
            listtemp = np.empty((gdat.numbsampfeww, minmsizearry[j]))
            for ii, i in enumerate(gdat.indxsampfeww):
                listtemp[ii, :] = listtemptemp[j][ii][:minmsizearry[j]]
            gdat.stdvtime[j] = np.std(listtemp, 0) * gdat.facttime
            gdat.indxtranstdv[j] = np.arange(minmsizearry[j])#gdat.sampindxtranmodl[i][j][minmsizearry[j]]

        listvarb = [gdat.sampindxtranmodl, gdat.samptimetranmodl, gdat.sampindxtranmodlproj, gdat.samptimetranmodlproj]
        plot_ttvr(gdat, gdat.modltype, listvarb)
        
        numbsampfeww = gdat.indxsampfeww.size

        for j in gdat.indxplan:
            # temp
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

        if numbsamp != numbsampwalk * numbwalk:
            raise Exception('')

        listsamp = objtsave.flatchain
        listllik = objtsave.flatlnprobability
        
        listlpos = objtsave.lnprobability
        chi2 = -2. * listlpos
        
        listsampproc = np.empty((gdat.numbsampfeww, gdat.numbplan))
        for j in gdat.indxplan:
            for ii, i in enumerate(gdat.indxsampfeww):
                listsampproc[ii, j] = gdat.samptimetranmodl[ii][j][40]
            listsampproc[:, j] -= np.mean(listsampproc[:, j])
            listsampproc[:, j] *= gdat.facttime
            listlablpara.append('T_{p,%s}' % gdat.liststrgplan[j])
        
        print('Minimum chi2: ')
        print(np.amin(chi2))
        print('Minimum chi2 per dof: ')
        print(np.amin(chi2) / numbdoff)
        print('Maximum aposterior: ')
        print(np.amax(listlpos))
        

def cnfg_WASP4():
    
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
    
    listmodltype = ['line', 'sinu']
    for modltype in listmodltype:
        
        indxpara = np.arange(numbpara)
        limtpara = np.empty((2, numbpara))
        # offs
        limtpara[0, 0] = -0.2
        limtpara[1, 0] = 0.2
        listlablpara = ['$C$ [minutes]']
        if modltype == 'sinu':
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
    
        dictllik = [time, numbepoc, ttvrobsv, ttvrstdvobsv, modltype]
        dicticdf = [numbepoc]
        
        # plot the posterior
        ## parameter
        ### trace
            
        path = pathdata + '%s/' % samptype
        strgplot = 'post_%s' % modltype
    
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
            objttemp = objtsave.flatchain
            offs = objttemp[indxsamprand[k], 0]
            if modltype == 'line':
                ttvrmodlfine[k, :] = offs * np.ones_like(indxepocfine)
            if modltype == 'sinu':
                phas = objttemp[k, 1]
                ampl = objttemp[k, 2]
                peri = objttemp[k, 3]
                ttvrmodlfine[k, :] = retr_ttvrmodl(indxepocfine, offs, phas, ampl, peri)
        
        if modltype == 'line':
            mlikcons = np.amax(listllik)
    
        if modltype == 'sinu':
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
            path = pathdata + '%s/chi2peri_%s.pdf' % (samptype, modltype)
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
        path = pathdata + '%s/modl_%s.pdf' % (samptype, modltype)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        

def cnfg_TOI270():
    
    timeobsdinit = 2458387.0927381925
    
    # perform allesfitter run
    boolalle = False

    # relevant transit times
    liststrgplan = ['b', 'c', 'd']
    numbplan = len(liststrgplan)
    indxplan = np.arange(numbplan, dtype=int)
    timetranwind = [[] for j in indxplan]
    timetranwind[0] = [np.array([])]
    timetranwind[1] = np.array([
                                  2458797.034958, \
                                  2458802.695130, \
                                  2458808.355302, \
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
                                  2458799.362400, \
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
    
    import pickle
    pathdata = os.environ['TTVR_DATA_PATH'] + '/data/'
    path = pathdata + 'measured_allesfit_all_ttv.pickle'
    objtfile = open(path,'rb')
    datapickttvr = pickle.load(objtfile)
    indxtranobsd = [[] for j in indxplan]
    timetranlineproj = [[] for j in indxplan]
    timetranobsd = [[] for j in indxplan]
    stdvtimetranobsd = [[] for j in indxplan]
    for j in indxplan:
        #meanperiline[j] = datapickttvr[liststrgplan[j]]['lin_period']
        timetranobsd[j] = datapickttvr[liststrgplan[j]]['transit_time']
        stdvtimetranobsd[j] = datapickttvr[liststrgplan[j]]['transit_time_err']
        if j == 1:
            timetranobsd[j] = np.concatenate((timetranobsd[j], np.array([2458780.0690228315, 2458797.054231985])))
            stdvtimetranobsd[j] = np.concatenate((stdvtimetranobsd[j], np.array([0.0011934256181120872, 0.0006028260104358196])))
        if j == 2:
            timetranobsd[j] = np.concatenate((timetranobsd[j], np.array([2458799.350335964])))
            stdvtimetranobsd[j] = np.concatenate((stdvtimetranobsd[j], np.array([0.001056972425431013])))
        timetranobsd[j] -= timeobsdinit
    
    # setttings
    ## modeling 
    # get data
    meanepocline = np.array([1461.01464, 1463.08481, 1469.33834]) + 2457000
    meanperiline = np.array([3.360080, 5.660172, 11.38014])

    numbparaplan = 7
    numbconf = 2
    indxconf = np.arange(numbconf)
    minmpara = [[] for k in indxconf]
    maxmpara = [[] for k in indxconf]
    meanpara = [[] for k in indxconf]
    stdvpara = [[] for k in indxconf]
    
    for k in indxconf:
        if k == 0:
            numbparaplanconf = 2
        else:
            numbparaplanconf = 7
        numbparaconf = numbparaplanconf * numbplan
        minmpara[k] = np.empty(numbparaconf)
        maxmpara[k] = np.empty(numbparaconf)
        meanpara[k] = np.empty(numbparaconf)
        stdvpara[k] = np.empty(numbparaconf)
        if k == 0:
            meanpara[k][indxplan] = np.array([timetranobsd[j][0] for j in indxplan])
            stdvpara[k][indxplan] = 1e-2
            
            print('indxplan+numbplan:')
            print(indxplan+numbplan)
            print('stdvpara[k]')
            summgene(stdvpara[k])
            print('meanpara[k]')
            summgene(meanpara[k])
            print('meanperiline')
            print(meanperiline)
            meanpara[k][indxplan+numbplan] = meanperiline
            stdvpara[k][indxplan+numbplan] = meanperiline * 1e-2
            
        else:
            meanpara[k][indxplan + numbplan * 0] = np.array([2.47, 5.46, 2.55]) * 0.00000300245
            stdvpara[k][indxplan + numbplan * 0] = np.array([0.75, 1.30, 0.91]) * 0.00000300245 * 1e-2
            
            meanpara[k][indxplan + numbplan * 1] = meanperiline
            stdvpara[k][indxplan + numbplan * 1] = meanpara[k][indxplan + numbplan * 1] * 1e-2
            
            meanpara[k][indxplan + numbplan * 2] = np.array([0., 0., 0.]) # ecce
            stdvpara[k][indxplan + numbplan * 2] = meanpara[k][indxplan + numbplan * 2] * 1e-2
            
            meanpara[k][indxplan + numbplan * 3] = np.array([88.65, 89.53, 89.69]) # incl
            stdvpara[k][indxplan + numbplan * 3] = meanpara[k][indxplan + numbplan * 3] * 1e-2
            
            meanpara[k][indxplan + numbplan * 4] = np.array([0., 0., 0.]) # ascending long
            stdvpara[k][indxplan + numbplan * 4] = meanpara[k][indxplan + numbplan * 4] * 1e-2
            
            meanpara[k][indxplan + numbplan * 5] = np.array([0., 0., 0.]) # arg periapsis
            stdvpara[k][indxplan + numbplan * 5] = meanpara[k][indxplan + numbplan * 5] * 1e-2
            
            meanpara[k][indxplan + numbplan * 6] = np.array([89.99999, 296.7324, 8.25829165761]) # mean anomaly
            stdvpara[k][indxplan + numbplan * 6] = meanpara[k][indxplan + numbplan * 6] * 1e-2
        
        minmpara[k] = meanpara[k] - stdvpara[k] * 5.
        maxmpara[k] = meanpara[k] + stdvpara[k] * 5.

    indxtranobsd = [[] for j in indxplan]
    for j in indxplan:
        indxtranobsd[j] = np.round(timetranobsd[j] / meanperiline[j]).astype(int)
    
    offs = [17, 10, 5]
    epoc = [2458444.2140459623, 2458446.104473652, 2458446.5783906463]
    peri = [3.360062366764236, 5.660172076246358, 11.38028139190828]
    liststrginst = ['TESS', 'Trappist-South_z_2', 'LCO-SAAO-1m0_ip', 'Trappist-South_z_1', 'LCO_ip', 'LCO-SSO-1m0_ip_1', 'mko-cdk700_g', 'Myers_B']
    pathtmpt = pathdata
    if boolalle:
        exec_alle_ttvr(pathdata, pathtmpt, epoc, peri, offs, liststrginst)
    
    init( \
         timetranobsd, \
         stdvtimetranobsd, \
         minmpara, \
         maxmpara, \
         meanpara, \
         stdvpara, \
        )


def cnfg_TOI1233():
    
    timeobsdinit = 2458387.0927381925
    exec_alle_ttvr(pathdata, pathtmpt, epoc, peri, offs, liststrginst)
    


globals().get(sys.argv[1])()
