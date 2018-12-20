import numpy as np, pandas as pd
from glob import glob
import os, argparse, pickle, h5py

'''
Read pickle files containing summary statistics about the posterior and write to a CSV file
'''

pathpick = os.environ['TESS_TTVR_DATA_PATH'] + '/wasp-18b_posteriors/sector_3/'
listname = np.sort(glob(pathpick + '100100827_mandelagol_and_line_fit_empiricalerrs_t???.pickle'))
listepoc, listepocstdv = [[], []]
for name in listname:
    d = pickle.load(open(name, 'rb'))
    listepoc.append(d['fitinfo']['finalparams']['t0'])
    fiterrs = d['fitinfo']['finalparamerrs']
    listepocstdv.append(max(fiterrs['std_merrs']['t0'], fiterrs['std_perrs']['t0']))
dfrm = pd.DataFrame({'t0_BTJD':np.array(listepoc), 't0_bigerr':np.array(listepocstdv)})
pathoutp = os.environ['TESS_TTVR_DATA_PATH'] + '/'
strgoutp = 'sector_3.csv'
path = pathoutp + strgoutp
dfrm.to_csv(path, index=False)
print('Writing to %s...' % path)
