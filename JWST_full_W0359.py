#!/usr/bin/env python

"""This is Brewster: the golden retriever of smelly atmospheres"""
from __future__ import print_function
from __future__ import division

from builtins import str
from builtins import range
import multiprocessing
import time
import numpy as np
import scipy as sp
import pymultinest as mn
import nestkit_modified as nestkit
import ciamod
import TPmod
import settings
import os
import gc
import sys
import pickle
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
import mpi4py


runname = "W0359"

obspec = np.asfortranarray(np.loadtxt("/Users/harshil/PycharmProjects/Brewster/OSC_upload/0359/0359_full_spectrum.txt",dtype='d',unpack=True))

# Now the wavelength range
w1 = 0.9
w2 = 20.0

fwhm = 999.0

dist = 13.5
dist_err = 0.05

proftype = 1
knots=5

do_fudge = 2
wavelength_cutoff = 5.2 # in um
sf = 1 # scale factor
sf_err = 0.3 # scale factor deviation

gaslist = ['h2o','ch4','co','co2','nh3','h2s','Na','K','ph3']

outdir = "/Users/harshil/PycharmProjects/Brewster/Brewster_local/Output/"

npatches = 1
nclouds = 1

# set up array for setting patchy cloud answers
do_clouds = np.zeros([npatches],dtype='i')

do_clouds[:] = 0

cloudnum = np.zeros([npatches,nclouds],dtype='i')
cloudtype =np.zeros([npatches,nclouds],dtype='i')

cloudnum[:,0] = 1
cloudtype[:,0] = 1

chemeq = 0

do_bff = 0

pfile = "t400g562nc_m+0.5.dat"

logcoarsePress = np.linspace(-4.0, 2.4, knots)
logfinePress = np.arange(-4.0, 2.4, 0.1)

coarsePress = pow(10,logcoarsePress)
press = pow(10,logfinePress)

xpath = "/Users/harshil/PycharmProjects/Brewster/Brewster_local/Linelists/"
xlist = 'gaslistR10K.dat'

ngas = len(gaslist)

malk = 1

runtest = 0
make_arg_pickle = 2

finalout = runname+".pk1"

use_disort = 0 

prof = np.full(5,100.)
if (proftype == 9):
    modP,modT = np.loadtxt(pfile,skiprows=1,usecols=(1,2),unpack=True)
    tfit = InterpolatedUnivariateSpline(np.log10(modP),modT,k=1)
    prof = tfit(logcoarsePress)


# Now we'll get the opacity files into an array
inlinetemps,inwavenum,linelist,gasnum,nwave = nestkit.get_opacities(gaslist,w1,w2,press,xpath,xlist,malk)

# Get the cia bits
tmpcia, ciatemps = ciamod.read_cia("CIA_DS_aug_2015.dat",inwavenum)
cia = np.asfortranarray(np.empty((4,ciatemps.size,nwave)),dtype='float32')
cia[:,:,:] = tmpcia[:,:,:nwave] 
ciatemps = np.asfortranarray(ciatemps, dtype='float32')

# grab BFF and Chemical grids
bff_raw,ceTgrid,metscale,coscale,gases_myP = nestkit.sort_bff_and_CE(chemeq,"chem_eq_tables_P3K.pic",press,gaslist)

settings.init()
settings.runargs = gases_myP,chemeq,dist,dist_err,cloudtype,do_clouds,gasnum,gaslist,cloudnum,inlinetemps,coarsePress,press,inwavenum,linelist,cia,ciatemps,use_disort,fwhm,obspec,proftype,do_fudge, prof,do_bff,bff_raw,ceTgrid,metscale,coscale, wavelength_cutoff, sf, sf_err

if make_arg_pickle > 0:
    pickle.dump(settings.runargs,open(outdir+runname+"_runargs.pic","wb"))
    if make_arg_pickle == 1:
        sys.exit()

n_params = nestkit.countdims(settings.runargs)

#print(n_params)

result = mn.solve(LogLikelihood=nestkit.lnlike, Prior=nestkit.priormap, n_dims=n_params, n_live_points=500, evidence_tolerance=0.5, log_zero=-1e90, multimodal=False, importance_nested_sampling=False, sampling_efficiency='parameter', const_efficiency_mode=False, outputfiles_basename=outdir+runname, verbose=True)

print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')
for col in zip(result['samples'].transpose()):
    print('%.3f +- %.3f' % (col.mean(), col.std()))

