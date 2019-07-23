import pickle, os
import os, numpy as np
import tdpy
from tdpy.util import summgene
import emcee
import matplotlib.pyplot as plt
import allesfitter

def conv_pick():
    pathbase = os.environ['TTVR_DATA_PATH'] + '/'
    pathdata = pathbase + 'data/'
    
    objtfile = open(pathdata + 'measured_allesfit_all.pickle','rb')
    datapick = pickle.load(objtfile)
    objtfile = open(pathdata + 'measured_allesfit_all_ttv.pickle','rb')
    datapickttvr = pickle.load(objtfile)
    
    with open(pathdata + 'measured_allesfit_all_conv.pickle', 'wb') as pfile:
        pickle.dump(datapick, pfile, protocol=2)
    with open(pathdata + 'measured_allesfit_all_ttv_conv.pickle', 'wb') as pfile:
        pickle.dump(datapickttvr, pfile, protocol=2)


def ttvr(pathdata, pathtmpt, epoc, peri, offs, liststrginst, booltmptmcmc=True, liststrgplan=['b']):
    
    print 'Starting the pipeline to infer per-transit TTVs...'
    
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
    
    print 'Maximum time away from the mid-transit that will be used in the fit:'
    print timetole
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
                print 'klm'
                print k,l,m
                print 'strginst'
                print strginst
                print 'timetran[k][l]'
                print timetran[k][l]
                print 'time[strginst]'
                summgene(time[strginst])
                print
                if indx[strginst].size > 0:
                    booltraninst[m] = True
                
            if np.where(booltraninst)[0].size == 0:
                indxtranskip[k].append(l)
                continue
    
            print 'Planet %s, transit %d' % (strgplan, l)
            print 'Transit time:'
            print timetran[k][l]
            print 'Copying settings.csv and params.csv to the transit folder...'
    
            if os.path.exists(pathtran):
                continue
    
            cmnd = 'mkdir -p %s' % pathtran
            print cmnd
            os.system(cmnd)
    
            cmnd = 'cp %ssettings.csv %s' % (pathtmpt, pathtran)
            print cmnd
            os.system(cmnd)
            
            cmnd = 'cp %sparams.csv %s' % (pathtmpt, pathtran)
            print cmnd
            os.system(cmnd)
            
            for m, strginst in enumerate(liststrginst):
                if booltraninst[m]:
                    path = '%s%s.csv' % (pathtran, strginst)
                    print 'Writing to %s...' % path
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
                print 'Writing to %s...' % pathfile
                objtfile = open(pathfile, 'w')
                for line in listline:
                    #print line
                    objtfile.write(line)
                objtfile.close()
            print
    
    print 'indxtranskip'
    print indxtranskip

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
                print cmnd
                os.system(cmnd)
            
            pathsave = pathdata + 'allesfit_sttv/allesfit_%s%03d/results/mcmc_save.h5' % (strgplan, l)
            print 'Reading %s...' % pathsave
            emceobjt = emcee.backends.HDFBackend(pathsave, read_only=True)
            if cntr == 0:
                numbsamp = emceobjt.get_chain().size
                timeresi = np.zeros((numbtran[k], numbsamp)) - 1e12
            timeresi[l, :] = (epoc[k] - emceobjt.get_chain().flatten()) * 24. * 60. # [min]
        listtimetran.append(timetran[k])
        listtimeresi.append(timeresi)
    
    
    print 'Making plots...'
    
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
        print 'Writing to %s...' % path
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    objtfile.close()
