import os
import ttvr.util

pathdata = '/Users/tdaylan/DropboxMITTTT/knownplanets/toi-270/'
pathtmpt = '%s/allesfit_tmpt/' % pathdata
liststrgplan = ['b', 'c', 'd']
offs = [17, 10, 5]
epoc = [2458444.2140459623, 2458446.104473652, 2458446.5783906463]
peri = [3.360062366764236, 5.660172076246358, 11.38028139190828]
liststrginst = ['TESS', 'Trappist-South_z_2', 'LCO-SAAO-1m0_ip', 'Trappist-South_z_1', 'LCO_ip', 'LCO-SSO-1m0_ip_1', 'mko-cdk700_g', 'Myers_B']
pathtarg = '%s/allesfit_tmpt/' % pathdata
ttvr.util.ttvr(pathdata, pathtmpt, epoc, peri, offs, liststrginst, booltmptmcmc=False)


