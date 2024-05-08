#!/usr/bin/env python

""" Module of bits to plug into Brewster """
from __future__ import print_function
import math
import time
import gc
import numpy as np
import scipy as sp
import pickle
import forwardmodel
import cloudnest
import TPmod
import settings
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian1DKernel
from bensconv import conv_non_uniform_R
from scipy.special import erfinv

__author__ = "Ben Burningham"
__copyright__ = "Copyright 2015 - Ben Burningham"
__credits__ = ["Ben Burningham"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Ben Burningham"
__email__ = "burninghamster@gmail.com"
__status__ = "Development"


def priormap(theta):
    """This function translates sample points in the n-dimensional hypercube from PyMultiNest into a set of parameters
    for lnlike (and with that the forward model). By default the priors are uniform, so the prior-map carries out a
    relatively simple task in translating the values from the sampler’s live points that lie between 0 and 1.
    For example, the sample points for uniform-with-altitude gas volume mixing ratios must simply be multiplied by -12
    to translate them into log10(gas-fraction) values within our uniform prior range -12 to 0.  Similar translations
    are carried out for all other parameters. If you wish to add a non-uniform prior, this is where you should do it.
    Be careful not to mess up the counting that keeps track of unpacking the 1D state vector."""

    gases_myP, chemeq, dist, dist_err, cloudtype, do_clouds, gasnum, gaslist, cloudnum, inlinetemps, coarsePress, press, inwavenum, linelist, cia, ciatemps, use_disort, fwhm, obspec, proftype, do_fudge, prof, do_bff, bff_raw, ceTgrid, metscale, coscale, wavelength_cutoff, sf, sf_err = settings.runargs

    phi = np.zeros_like(theta)
    ndim = phi.size

    # here are some priors you may want to edit:
    max_mass = 80.  # jupiters
    min_mass = 1.0  # jupiters
    min_rad = 0.5  # jupiters
    max_rad = 2.5  # jupiters

    # first the gases
    if (chemeq != 0):
        phi[0] = (theta[0] * (metscale[-1] - metscale[0])) + metscale[0]
        phi[1] = (theta[1] * (coscale[-1] - coscale[0])) + coscale[0]
        ng = 2
    else:
        if (gasnum[gasnum.size - 1] == 21):
            ng = gasnum.size - 1
            rem = 1.0
            for i in range(0, ng):
                phi[i] = np.log(rem) - (theta[i] * 12.)
                rem = rem - (10 ** phi[i])
        elif (gasnum[gasnum.size - 1] == 23):
            ng = gasnum.size - 2
            rem = 1.0
            for i in range(0, ng):
                phi[i] = np.log(rem) - (theta[i] * 12.)
                rem = rem - (10 ** phi[i])
        else:
            ng = gasnum.size
            rem = 1.0
            for i in range(0, ng):
                phi[i] = np.log(rem) - (theta[i] * 12.)
                rem = rem - (10 ** phi[i])

                # this is a simple uniform prior on mass
    # we want to use the radius, to set a mass prior.
    # this will correlate these parameters??? Yes. which is fine.
    phi[ng] = (theta[ng] * (max_mass - min_mass)) + min_mass

    # this is if we want log g prior: phi[ng] = theta[ng] * 5.5

    # now we retrieve radius in R_jup
    R_j = ((max_rad - min_rad) * theta[ng + 1]) + min_rad
    phi[ng + 1] = R_j


    # this just copes with normal, single instrument data
    # so do dlam next
    phi[ng + 2] = (theta[ng + 2] * 0.02) - 0.01
    # now fudge
    if (do_fudge == 1):
        # logf here
        minerr = np.log10((0.01 * np.min(obspec[2, :])) ** 2.)
        maxerr = np.log10((100. * np.max(obspec[2, :])) ** 2.)
        phi[ng + 3] = (theta[ng + 3] * (maxerr - minerr)) + minerr
        pc = ng + 4

    elif (do_fudge == 2):

        minerr = np.log10((0.01 * np.min(obspec[2, :])) ** 2.)
        maxerr = np.log10((100. * np.max(obspec[2, :])) ** 2.)
        phi[ng + 3] = (theta[ng + 3] * (maxerr - minerr)) + minerr

        # scale factor for NIRSpec
        if  phi[ng + 4] == 0:
            phi[ng + 4] = (sf + (
                    sf_err * np.sqrt(2.0) * np.sqrt(2) * erfinv(2.0 * theta[ng + 4] + 0.001 - 1.0)))
        elif phi[ng + 4] == 1:
            phi[ng + 4] = (sf + (
                    sf_err * np.sqrt(2.0) * np.sqrt(2) * erfinv(2.0 * theta[ng + 4] - 0.001 - 1.0)))
        else:
            phi[ng + 4] = (sf + (
                        sf_err * np.sqrt(2.0) * np.sqrt(2) * erfinv(2.0 * theta[ng + 4] - 1.0)))

        # scale factor for MIRI
        if  phi[ng + 5] == 0:
            phi[ng + 5] = (sf + (
                    sf_err * np.sqrt(2.0) * np.sqrt(2) * erfinv(2.0 * theta[ng + 5] + 0.001 - 1.0)))
        elif phi[ng + 5] == 1:
            phi[ng + 5] = (sf + (
                    sf_err * np.sqrt(2.0) * np.sqrt(2) * erfinv(2.0 * theta[ng + 5] - 0.001 - 1.0)))
        else:
            phi[ng + 5] = (sf + (
                        sf_err * np.sqrt(2.0) * np.sqrt(2) * erfinv(2.0 * theta[ng + 5] - 1.0)))

        pc = ng + 6

    else:
        pc = ng + 3

    npatches = do_clouds.size
    # only really ready for 2 patches here
    if (npatches > 1):
        phi[pc] = theta[pc]
        pc = pc + 1
    if (cloudtype.size > cloudtype.shape[0]):
        nclouds = cloudtype.shape[1]
    else:
        nclouds = cloudtype.size

    nc = 0
    # use correct unpack method depending on clouds situation
    stop_cloud = 0
    for patch in range(0, npatches):
        # only set up for two patches: cloud & clear, or slab + deck, deck
        # so we stop reading cloud parameters after first pass through here.
        if stop_cloud == 1:
            break
        elif (do_clouds[patch] != 0):
            for cloud in range(0, nclouds):
                if (cloudtype[patch, cloud] == 1):
                    if (cloudnum[patch, cloud] == 89):
                        # cloud tau
                        phi[pc + nc] = theta[pc + nc] * 100.
                        # cloud base
                        phi[pc + nc + 1] = \
                            (theta[pc + nc + 1] * \
                             (np.log10(press[-1]) - np.log10(press[0]))) \
                            + np.log10(press[0])
                        # cloud height
                        phi[pc + nc + 2] = \
                            theta[pc + nc + 2] * (phi[pc + nc + 1] \
                                                  - np.log10(press[0]))
                        # power law
                        phi[pc + nc + 3] = (theta[pc + nc + 3] * 20.) - 10.
                        nc = nc + 4
                        stop_cloud == 1
                    elif (cloudnum[patch, cloud] == 99):
                        # cloud tau
                        phi[pc + nc] = theta[pc + nc] * 100.
                        # cloud base
                        phi[pc + nc + 1] = \
                            (theta[pc + nc + 1] * (np.log10(press[-1]) \
                                                   - np.log10(press[0]))) \
                            + np.log10(press[0])
                        # cloud height
                        phi[pc + nc + 2] = theta[pc + nc + 2] * \
                                           (phi[pc + nc + 1] - np.log10(press[0]))
                        nc = nc + 3
                        stop_cloud == 1
                    elif (cloudnum[patch, cloud] < 80):
                        # cloud tau
                        phi[pc + nc] = theta[pc + nc] * 100.
                        # cloud base
                        phi[pc + nc + 1] = \
                            (theta[pc + nc + 1] * (np.log10(press[-1]) \
                                                   - np.log10(press[0]))) \
                            + np.log10(press[0])
                        # cloud height
                        phi[pc + nc + 2] = theta[pc + nc + 2] * \
                                           (phi[pc + nc + 1] - np.log10(press[0]))
                        # particle effective radius
                        phi[pc + nc + 3] = (theta[pc + nc + 3] * 6.) - 3.
                        # particle spread
                        phi[pc + nc + 4] = theta[pc + nc + 4]
                        nc = nc + 5
                        stop_cloud == 1
                elif (cloudtype[patch, cloud] == 2):
                    if (cloudnum[patch, cloud] == 89):
                        # cloud top
                        phi[pc + nc] = \
                            (theta[pc + nc] * (np.log10(press[-1]) \
                                               - np.log10(press[0]))) \
                            + np.log10(press[0])
                        # cloud height
                        phi[pc + nc + 1] = theta[pc + nc + 1] * 7.
                        # power law
                        phi[pc + nc + 2] = (theta[pc + nc + 2] * 20.) - 10.
                        nc = nc + 3
                        stop_cloud == 1
                    elif (cloudnum[patch, cloud] == 99):
                        # cloud top
                        phi[pc + nc] = \
                            (theta[pc + nc] * (np.log10(press[-1]) \
                                               - np.log10(press[0]))) \
                            + np.log10(press[0])
                        # cloud height
                        phi[pc + nc + 1] = theta[pc + nc + 1] * 7.
                        nc = nc + 2
                        stop_cloud == 1
                    elif (cloudnum[patch, cloud] < 80):
                        # cloud base
                        phi[pc + nc] = \
                            (theta[pc + nc] * (np.log10(press[-1]) \
                                               - np.log10(press[0]))) \
                            + np.log10(press[0])
                        # cloud height
                        phi[pc + nc + 1] = theta[pc + nc + 1] * 7.
                        # particle effective radius
                        phi[pc + nc + 2] = (theta[pc + nc + 2] * 6.) - 3.
                        # particle spread
                        phi[pc + nc + 3] = theta[pc + nc + 3]
                        nc = nc + 4
                        stop_cloud == 1
                    # types 3 and 4 to be added
                elif (cloudtype[patch, cloud] > 2):
                    print("cloudtypes 3 and 4 not yet implemented here")
                    sys.exit()

    # Distance parameter
    if phi[ndim - 6] == 0:
        phi[ndim - 6] = (dist + (
                    dist_err * np.sqrt(2.0) * np.sqrt(2) * erfinv(2.0 * theta[ndim - 6] + 0.001 - 1.0))) * 3.086e16
    elif phi[ndim - 6] == 1:
        phi[ndim - 6] = (dist + (
                    dist_err * np.sqrt(2.0) * np.sqrt(2) * erfinv(2.0 * theta[ndim - 6] - 0.001 - 1.0))) * 3.086e16
    else:
        phi[ndim - 6] = (dist + (dist_err * np.sqrt(2.0) * np.sqrt(2) * erfinv(2.0 * theta[ndim - 6] - 1.0))) * 3.086e16

    if (proftype == 1):
        # This is not Mike Line's profile. This is a simple 5 point spline.
        # We get the bottom and top of the atmosphere,
        # then we get a mid point
        # then two quartile points
        # These define T at define pressure points
        # This does not allow reversals
        phi[ndim - 1] = theta[ndim - 1] * 3000.
        # now top of atmosphere temp, defined as being less than bottom
        phi[ndim - 5] = theta[ndim - 5] * phi[ndim - 1]
        # now middle of atmosphere
        phi[ndim - 3] = phi[ndim - 5] + (theta[ndim - 3] * (phi[ndim - 1] - phi[ndim - 5]))
        # now deep quartile
        phi[ndim - 2] = phi[ndim - 3] + (theta[ndim - 2] * (phi[ndim - 1] - phi[ndim - 3]))
        # and bottom shallow quartile
        phi[ndim - 4] = phi[ndim - 5] + (theta[ndim - 4] * (phi[ndim - 3] - phi[ndim - 5]))
        return phi

    if (proftype == 2):
        # a1
        phi[pc + nc] = 0.25 + (theta[pc + nc] * 0.25)
        # a2
        phi[pc + nc + 1] = (theta[pc + 1 + nc] * 0.05)
        # P1
        phi[pc + nc + 2] = (theta[pc + 2 + nc] * \
                            (np.log10(press[-1]) - np.log10(press[0]))) + np.log10(press[0])
        # P3
        # P3 must be greater than P1
        phi[pc + nc + 3] = (theta[pc + nc + 3] * \
                            (np.log10(press[-1]) - phi[pc + nc + 2])) + phi[pc + nc + 2]
        # T3
        phi[pc + nc + 4] = (theta[pc + 4 + nc] * 3000.) + 500.0
        return phi

    elif (proftype == 9):
        return phi


def lnlike(theta):
    gases_myP, chemeq, dist, dist_err, cloudtype, do_clouds, gasnum, gaslist, cloudnum, inlinetemps, coarsePress, press, inwavenum, linelist, cia, ciatemps, use_disort, fwhm, obspec, proftype, do_fudge, prof, do_bff, bff_raw, ceTgrid, metscale, coscale, wavelength_cutoff, sf, sf_err = settings.runargs

    # get the spectrum
    # for MCMC runs we don't want diagnostics
    gnostics = 0
    shiftspec, photspec, tauspec, cfunc = modelspec(theta, settings.runargs, gnostics)
    # Check if CE or VMR methods
    if chemeq == 0:
        if (gasnum[gasnum.size - 1] == 21):
            ng = gasnum.size - 1
        elif (gasnum[gasnum.size - 1] == 23):
            ng = gasnum.size - 2
        else:
            ng = gasnum.size
        invmr = theta[0:ng]

    else:
        ng = 2

    # Get the scaling factors for the spectra. What is the FWHM? Negative number: preset combination of instruments
    if (do_fudge == 1):
        logf = theta[ng + 3]
        nb = 4

    elif (do_fudge == 2):
        logf = theta[ng + 3]
        sf_NIRSpec = theta[ng + 4]
        sf_MIRI = theta[ng + 5]
        nb = 6

    else:
        nb = 3

    modspec = np.array([shiftspec[0, ::-1], shiftspec[1, ::-1]])

    if (fwhm == 999.0):

        lnLik = 0.0

        if (do_fudge == 0):
            R = obspec[-1, :]
            spec = conv_non_uniform_R(obspec, modspec, R)
            s2 = obspec[2, :] ** 2
            lnLik = -0.5 * np.sum((((obspec[1, :] - spec[:]) ** 2) / s2) + np.log(2. * np.pi * s2))

        elif (do_fudge == 1):
            R = obspec[-1, :]
            spec = conv_non_uniform_R(obspec, modspec, R)
            s2 = obspec[2, :] ** 2 + 10. ** logf
            lnLik = -0.5 * np.sum((((obspec[1, :] - spec[:]) ** 2) / s2) + np.log(2. * np.pi * s2))

        elif (do_fudge == 2):
            mr_NIRSpec = np.where(modspec[0, :] < wavelength_cutoff)[0]
            or_NIRSpec = np.where(obspec[0, :] < wavelength_cutoff)[0]

            mr_MIRI = np.where(np.logical_and(modspec[0, :] > wavelength_cutoff, modspec[0, :] < 13))[0]
            or_MIRI = np.where(np.logical_and(obspec[0, :] > wavelength_cutoff, obspec[0, :] < 13))[0]

            #mr_photometry = np.where(modspec[0, :] > obspec[0,-3])[0]
            #or_photometry = np.where(obspec[0, :] > obspec[0,-3])[0]

            R = obspec[-1, :]

            NIRSpec = sf_NIRSpec * conv_non_uniform_R(obspec[:, or_NIRSpec], modspec[:, mr_NIRSpec], R[or_NIRSpec])
            MIRI = sf_MIRI * conv_non_uniform_R(obspec[:, or_MIRI], modspec[:, mr_MIRI], R[or_MIRI])
            #photometry = conv_non_uniform_R(obspec[:, or_photometry], modspec[:, mr_photometry], R[or_photometry])

            s_NIRSpec = obspec[2, or_NIRSpec] ** 2 + 10. ** logf
            s_MIRI = obspec[2, or_MIRI] ** 2 + 10. ** logf
            #s_photometry = obspec[2, or_photometry] ** 2 + 10. ** logf

            lnLik_NIRSpec = -0.5 * np.sum((((obspec[1, or_NIRSpec] - NIRSpec[:]) ** 2) / s_NIRSpec) + np.log(2. * np.pi * s_NIRSpec))
            lnLik_MIRI = -0.5 * np.sum((((obspec[1, or_MIRI] - MIRI[:]) ** 2) / s_MIRI) + np.log(2. * np.pi * s_MIRI))
            #lnLik_photometry = -0.5 * np.sum((((obspec[1, or_photometry] - photometry[:]) ** 2) / s_photometry) + np.log(2. * np.pi * s_photometry))

            lnLik = lnLik_NIRSpec + lnLik_MIRI #+ lnLik_photometry


    if np.isnan(lnLik):
        lnLik = -np.inf

    return lnLik


def modelspec(theta, args, gnostics=0):
    gases_myP, chemeq, dist, dist_err, cloudtype, do_clouds, gasnum, gaslist, cloudnum, inlinetemps, coarsePress, press, inwavenum, linelist, cia, ciatemps, use_disort, fwhm, obspec, proftype, do_fudge, prof, do_bff, bff_raw, ceTgrid, metscale, coscale, wavelength_cutoff, sf, sf_err = args
    nlayers = press.size

    if chemeq == 0:
        if (gasnum[gasnum.size - 1] == 21):
            ng = gasnum.size - 1
        elif (gasnum[gasnum.size - 1] == 23):
            ng = gasnum.size - 2
        else:
            ng = gasnum.size
        invmr = theta[0:ng]

    else:
        mh = theta[0]
        co = theta[1]
        ng = 2
        mfit = interp1d(metscale, gases_myP, axis=0)
        gases_myM = mfit(mh)
        cfit = interp1d(coscale, gases_myM, axis=0)
        invmr = cfit(co)

    GM = (6.67E-11 * theta[ng] * 1.898e27)
    R = theta[ng + 1] * 69911e3
    logg = np.log10(100. * GM / R ** 2.)

    # D = (dist + (np.random.randn()*dist_err)) * 3.086e16
    R2D2 = R ** 2. / theta[-6] ** 2.

    dlam = theta[ng + 2]

    if (do_fudge == 1):
        logf = theta[ng + 3]
        nb = 4

    elif (do_fudge == 2):
        logf1 = theta[ng + 3]
        sf_NIRSpec = theta[ng + 4]
        sf_MIRI = theta[ng + 5]
        nb = 6

    else:
        nb = 3

    npatches = do_clouds.size
    if (npatches > 1):
        prat = theta[ng + nb]
        pcover = np.array([prat, (1. - prat)])
        pc = ng + nb + 1
    else:
        pc = ng + nb
        pcover = 1.0

    # use correct unpack method depending on situation

    if ((npatches > 1) and np.all(do_clouds != 0)):
        cloudparams, nc = cloudnest.unpack_patchy(theta, pc, cloudtype, cloudnum, do_clouds)
    else:
        cloudparams, nc = cloudnest.unpack_default(theta, pc, cloudtype, cloudnum, do_clouds)

    if (proftype < 8):
        intemp = theta[pc + nc + 1:]
    elif (proftype == 9):
        intemp = prof
    else:
        raise ValueError("not valid profile type %proftype" % (char, string))

    # set the profile
    temp = TPmod.set_prof(proftype, coarsePress, press, intemp)

    ngas = gasnum.size
    bff = np.zeros([3, nlayers], dtype="float64")

    # check if its a fixed VMR or a profile from chem equilibrium
    # VMR is log10(VMR) !!!
    if chemeq == 1:
        # this case is a profile
        ng = invmr.shape[2]
        ngas = ng - 3
        logVMR = np.zeros([ngas, nlayers], dtype='d')
        for p in range(0, nlayers):
            for g in range(0, ng):
                tfit = InterpolatedUnivariateSpline(ceTgrid, invmr[:, p, g])
                if (g < 3):
                    bff[g, p] = tfit(temp[p])
                else:
                    logVMR[g - 3, p] = tfit(temp[p])
    else:
        # This case is fixed VMR
        # chemeq = 0
        logVMR = np.empty((ngas, nlayers), dtype='d')
        alkratio = 16.2  # from Asplund et al (2009)

        # now sort Na and K
        # get the ngas for forward model (ngas, not ng
        if (gasnum[gasnum.size - 1] == 21):
            ngas = invmr.shape[0] + 1
        elif (gasnum[gasnum.size - 1] == 23):
            ngas = invmr.shape[0] + 2
        else:
            ngas = invmr.shape[0]

        tmpvmr = np.empty(ngas, dtype='d')
        if (gasnum[gasnum.size - 1] == 21):
            tmpvmr[0:(ngas - 2)] = invmr[0:(ngas - 2)]
            tmpvmr[ngas - 2] = np.log10(10. ** invmr[ngas - 2] / (alkratio + 1.))  # K
            tmpvmr[ngas - 1] = np.log10(10. ** invmr[ngas - 2] * (alkratio / (alkratio + 1.)))  # Na
        elif (gasnum[gasnum.size - 1] == 23):
            # f values are ratios between Na and (K+Cs) and K and Cs respectively
            f1 = 1.348
            f2 = 8912.5
            tmpvmr[0:(ngas - 3)] = invmr[0:(ngas - 3)]
            tmpvmr[ngas - 1] = np.log10(10. ** invmr[ngas - 3] / ((f1 + 1) * (f2 + 1)))  # Cs
            tmpvmr[ngas - 2] = np.log10(10. ** invmr[ngas - 3] * (f1 / (f1 + 1)))  # Na
            tmpvmr[ngas - 3] = np.log10(
                10. ** invmr[ngas - 3] - 10. ** tmpvmr[ngas - 2] - 10. ** tmpvmr[ngas - 1])  # K
        else:
            tmpvmr[0:ngas] = invmr[0:ngas]

        for i in range(0, ngas):
            logVMR[i, :] = tmpvmr[i]

    # now need to translate cloudparams in to cloud profile even
    # if do_clouds is zero..

    cloudprof, cloudrad, cloudsig = cloudnest.atlas(do_clouds, cloudnum, cloudtype, cloudparams, press)
    cloudprof = np.asfortranarray(cloudprof, dtype='float64')
    cloudrad = np.asfortranarray(cloudrad, dtype='float64')
    cloudsig = np.asfortranarray(cloudsig, dtype='float64')
    pcover = np.asfortranarray(pcover, dtype='float32')
    cloudnum = np.asfortranarray(cloudnum, dtype='i')
    do_clouds = np.asfortranarray(do_clouds, dtype='i')

    # Now get the BFF stuff sorted
    if (chemeq == 0 and do_bff == 1):
        for gas in range(0, 3):
            for i in range(0, nlayers):
                tfit = InterpolatedUnivariateSpline(ceTgrid, bff_raw[:, i, gas], k=1)
                bff[gas, i] = tfit(temp[i])

    bff = np.asfortranarray(bff, dtype='float64')
    press = np.asfortranarray(press, dtype='float32')
    temp = np.asfortranarray(temp, dtype='float64')
    logVMR = np.asfortranarray(logVMR, dtype='float64')

    # Diagnostics below.
    # make_cf = get a contribution function
    # clphot = get pressure for cloud_tau = 1.0 as function of wavelength
    # ^^ i.e the cloud photosphere
    # ophot = get pressures for tau(not cloud) = 1.0 as function of wavelength]
    # ^^ i.e. the photosphere due to other (gas phase) opacities)

    # Set clphot,ophot and cfunc as we don't need these in the emcee run
    if (gnostics == 0):
        clphot = 0
        ophot = 0
        make_cf = 0
    else:
        clphot = 1
        ophot = 1
        make_cf = 1

    # now we can call the forward model
    outspec, tmpclphotspec, tmpophotspec, cf = forwardmodel.marv(temp, logg, R2D2, gasnum, logVMR, pcover, do_clouds,
                                                                 cloudnum, cloudrad, cloudsig, cloudprof, inlinetemps,
                                                                 press, inwavenum, linelist, cia, ciatemps, use_disort,
                                                                 clphot, ophot, make_cf, do_bff, bff)

    # Trim to length where it is defined.
    nwave = inwavenum.size
    trimspec = np.zeros([2, nwave], dtype='d')
    trimspec = outspec[:, :nwave]
    cloud_phot_press = tmpclphotspec[0:npatches, :nwave].reshape(npatches, nwave)
    other_phot_press = tmpophotspec[0:npatches, :nwave].reshape(npatches, nwave)
    cfunc = np.zeros([npatches, nwave, nlayers], dtype='d')
    cfunc = cf[:npatches, :nwave, :nlayers].reshape(npatches, nwave, nlayers)

    # now shift wavelen by delta_lambda
    shiftspec = np.empty_like(trimspec)
    shiftspec[0, :] = trimspec[0, :] + dlam
    shiftspec[1, :] = trimspec[1, :]

    return shiftspec, cloud_phot_press, other_phot_press, cfunc


def get_opacities(gaslist, w1, w2, press, xpath='../Linelists', xlist='gaslistR10K.dat', malk=0):
    # Now we'll get the opacity files into an array
    ngas = len(gaslist)

    totgas = 24
    gasdata = []
    with open(xlist) as fa:
        for line_aa in fa.readlines()[1:totgas + 1]:
            line_aa = line_aa.strip()
            gasdata.append(line_aa.split())

    list1 = []
    for i in range(0, ngas):
        for j in range(0, totgas):
            if (gasdata[j][1].lower() == gaslist[i].lower()):
                list1.append(gasdata[j])

    if (malk == 1):
        for i in range(0, ngas):
            list1[i] = [w.replace('K_', 'K_Mike_') for w in list1[i]]
            list1[i] = [w.replace('Na_', 'Na_Mike_') for w in list1[i]]

    if (malk == 2):
        for i in range(0, ngas):
            list1[i] = [w.replace('K_', 'K_2021_') for w in list1[i]]
            list1[i] = [w.replace('Na_', 'Na_2021_') for w in list1[i]]

    lists = [xpath + i[3] for i in list1[0:ngas]]
    gasmass = [xpath + i[2] for i in list1[0:ngas]]
    gasnum = np.asfortranarray(np.array([i[0] for i in list1[0:ngas]], dtype='i'))

    # get the basic framework from water list
    rawwavenum, inpress, inlinetemps, inlinelist = pickle.load(open(lists[0], "rb"))

    wn1 = 10000. / w2
    wn2 = 10000. / w1
    inwavenum = np.asfortranarray(
        rawwavenum[np.where(np.logical_not(np.logical_or(rawwavenum[:] > wn2, rawwavenum[:] < wn1)))], dtype='float64')
    ntemps = inlinetemps.size
    npress = press.size
    nwave = inwavenum.size
    r1 = np.amin(np.where(np.logical_not(np.logical_or(rawwavenum[:] > wn2, rawwavenum[:] < wn1))))
    r2 = np.amax(np.where(np.logical_not(np.logical_or(rawwavenum[:] > wn2, rawwavenum[:] < wn1))))

    # Here we are interpolating the linelist onto our fine pressure scale.
    # pickles have linelist as 4th entry....
    linelist = (np.zeros([ngas, npress, ntemps, nwave], order='F')).astype('float64', order='F')
    for gas in range(0, ngas):
        inlinelist = pickle.load(open(lists[gas], "rb"))[3]
        for i in range(0, ntemps):
            for j in range(r1, r2 + 1):
                pfit = interp1d(np.log10(inpress), np.log10(inlinelist[:, i, j]))
                linelist[gas, :, i, (j - r1)] = np.asfortranarray(pfit(np.log10(press)))
    linelist[np.isnan(linelist)] = -50.0

    return inlinetemps, inwavenum, linelist, gasnum, nwave


def sort_bff_and_CE(chemeq, ce_table, press, gaslist):
    # Sort out the BFF opacity stuff and chemical equilibrium tables:
    metscale, coscale, Tgrid, Pgrid, gasnames, abunds = pickle.load(open(ce_table, "rb"))
    nabpress = Pgrid.size
    nabtemp = Tgrid.size
    nabgas = abunds.shape[4]
    nmet = metscale.size
    nco = coscale.size
    nlayers = press.size
    ngas = len(gaslist)

    bff_raw = np.zeros([nabtemp, nlayers, 3])
    gases_myP = np.zeros([nmet, nco, nabtemp, nlayers, ngas + 3])
    gases = np.zeros([nmet, nco, nabtemp, nabpress, ngas + 3])

    if (chemeq == 0):
        # Just want the ion fractions for solar metallicity in this case
        ab_myP = np.empty([nabtemp, nlayers, nabgas])
        i1 = np.where(metscale == 0.0)
        i2 = np.where(coscale == 1.0)
        for gas in range(0, nabgas):
            for i in range(0, nabtemp):
                pfit = InterpolatedUnivariateSpline(Pgrid, np.log10(abunds[i1[0], i2[0], i, :, gas]), k=1)
                ab_myP[i, :, gas] = pfit(np.log10(press))

                bff_raw = np.zeros([nabtemp, nlayers, 3])
                bff_raw[:, :, 0] = ab_myP[:, :, 0]
                bff_raw[:, :, 1] = ab_myP[:, :, 2]
                bff_raw[:, :, 2] = ab_myP[:, :, 4]

    else:
        # In this case we need the rows for the gases we're doing and ion fractions
        gases[:, :, :, :, 0] = abunds[:, :, :, :, 0]
        gases[:, :, :, :, 1] = abunds[:, :, :, :, 2]
        gases[:, :, :, :, 2] = abunds[:, :, :, :, 4]
        nmatch = 0
        for i in range(0, ngas):
            for j in range(0, nabgas):
                if (gasnames[j].lower() == gaslist[i].lower()):
                    gases[:, :, :, :, i + 3] = abunds[:, :, :, :, j]
                    nmatch = nmatch + 1

        if (nmatch != ngas):
            print("you've requested a gas that isn't in the Vischer table. Please chaeck and try again.")
            sys.exit()

        for i in range(0, nmet):
            for j in range(0, nco):
                for k in range(0, ngas + 3):
                    for l in range(0, nabtemp):
                        pfit = InterpolatedUnivariateSpline(Pgrid, np.log10(gases[i, j, l, :, k]), k=1)
                        gases_myP[i, j, l, :, k] = pfit(np.log10(press))

    return bff_raw, Tgrid, metscale, coscale, gases_myP


def countdims(runargs, plist=False):
    gases_myP, chemeq, dist, dist_err, cloudtype, do_clouds, gasnum, gaslist, cloudnum, inlinetemps, coarsePress, press, inwavenum, linelist, cia, ciatemps, use_disort, fwhm, obspec, proftype, do_fudge, prof, do_bff, bff_raw, ceTgrid, metscale, coscale, wavelength_cutoff, sf, sf_err = runargs

    knots = len(coarsePress)
    pnames = list()
    # first the gases
    if (chemeq != 0):
        pnames = ['[M/H]', '(C/O)']
        ng = 2
    else:
        if (gasnum[gasnum.size - 1] == 21):
            for i in range(0, gasnum.size - 2):
                pnames.append(gaslist[i])
            pnames.append('Na+K')
            ng = gasnum.size - 1
        elif (gasnum[gasnum.size - 1] == 23):
            for i in range(0, gasnum.size - 3):
                pnames.append(gaslist[i])
            pnames.append('Cs+Na+K')
            ng = gasnum.size - 2
        else:
            for i in range(0, gasnum.size):
                pnames.append(gaslist[i])
            ng = gasnum.size

    pnames.extend(['Mass', 'Radius'])


    if (do_fudge == 0):
        pnames.append('dlam')
        pc = ng + 3
    elif (do_fudge == 1):
        pnames.extend(['dlam', 'logb1'])
        pc = ng + 4
    elif (do_fudge == 2):
        pnames.extend(['dlam', 'logb1', 'sf_NIRSpec', 'sf_MIRI'])
        pc = ng + 6

    npatches = do_clouds.size
    # only really ready for 2 patches here
    if (npatches > 1):
        pc = pc + 1
        pnames.append('Pcov')
    if (cloudtype.size > cloudtype.shape[0]):
        nclouds = cloudtype.shape[1]
    else:
        nclouds = cloudtype.size

    nc = 0
    stop_cloud = 0
    # use correct unpack method depending on clouds situation
    for patch in range(0, npatches):
        # only set up for two patches: cloud & clear, or slab + deck, deck
        # so we stop reading cloud parameters after first pass through here.
        if stop_cloud == 1:
            break

        elif (do_clouds[patch] != 0):
            for cloud in range(0, nclouds):
                if (cloudtype[patch, cloud] == 1):
                    if (cloudnum[patch, cloud] == 89):
                        pnames.append('tauC' + str(cloud + 1) + 'P' + str(patch + 1))
                        pnames.append('PbaseC' + str(cloud + 1) + 'P' + str(patch + 1))
                        pnames.append('dPC' + str(cloud + 1) + 'P' + str(patch + 1))
                        pnames.append('PowC' + str(cloud + 1) + 'P' + str(patch + 1))
                        nc = nc + 4
                        stop_cloud = 1
                    elif (cloudnum[patch, cloud] == 99):
                        pnames.append('tauC' + str(cloud + 1) + 'P' + str(patch + 1))
                        pnames.append('PbaseC' + str(cloud + 1) + 'P' + str(patch + 1))
                        pnames.append('dPC' + str(cloud + 1) + 'P' + str(patch + 1))
                        nc = nc + 3
                        stop_cloud = 1
                    elif (cloudnum[patch, cloud] < 80):
                        pnames.append('tauC' + str(cloud + 1) + 'P' + str(patch + 1))
                        pnames.append('PbaseC' + str(cloud + 1) + 'P' + str(patch + 1))
                        pnames.append('dPC' + str(cloud + 1) + 'P' + str(patch + 1))
                        pnames.append('a_C' + str(cloud + 1) + 'P' + str(patch + 1))
                        pnames.append('b_C' + str(cloud + 1) + 'P' + str(patch + 1))
                        nc = nc + 5
                        stop_cloud = 1
                elif (cloudtype[patch, cloud] == 2):
                    if (cloudnum[patch, cloud] == 89):
                        pnames.append('PdeckC' + str(cloud + 1) + 'P' + str(patch + 1))
                        pnames.append('heightC' + str(cloud + 1) + 'P' + str(patch + 1))
                        pnames.append('PowC' + str(cloud + 1) + 'P' + str(patch + 1))
                        nc = nc + 3
                        stop_cloud = 1
                    elif (cloudnum[patch, cloud] == 99):
                        pnames.append('PdeckC' + str(cloud + 1) + 'P' + str(patch + 1))
                        pnames.append('heightC' + str(cloud + 1) + 'P' + str(patch + 1))
                        nc = nc + 2
                        stop_cloud = 1
                    elif (cloudnum[patch, cloud] < 80):
                        nc = nc + 4
                        pnames.append('PdeckC' + str(cloud + 1) + 'P' + str(patch + 1))
                        pnames.append('heightC' + str(cloud + 1) + 'P' + str(patch + 1))
                        pnames.append('a_C' + str(cloud + 1) + 'P' + str(patch + 1))
                        pnames.append('b_C' + str(cloud + 1) + 'P' + str(patch + 1))
                        stop_cloud = 1
                    # types 3 and 4 to be added
                elif (cloudtype[patch, cloud] > 2):
                    print("cloudtypes 3 and 4 not yet implemented here")
                    sys.exit()
    # now add the proftype params...
    pnames.extend(['Distance'])
    if (proftype == 1):
        for i in range(0, knots):
            pnames.extend(['T_' + str(round(np.log10(coarsePress[i]), 1))])
            ndim = pc + nc + 1 + knots


    elif (proftype == 2):
        pnames.extend(['a1', 'a2', 'P1', 'P3', 'T3'])
        ndim = pc + nc + 5 + 1

    elif (proftype == 9):
        ndim = pc + nc +1

    if plist:
        return ndim, pnames
    else:
        return ndim
