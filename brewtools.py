from __future__ import print_function
import pickle
import numpy as np
import emcee
import os

def get_endchain(runname,fin,results_path='./'):
    if (fin == 1):
        pic = results_path+runname+".pk1"
        sampler = pickle_load(pic)
        nwalkers = sampler.chain.shape[0]
        niter = sampler.chain.shape[1]
        ndim = sampler.chain.shape[2]
        flatprobs = sampler.lnprobability[:,:].reshape((-1))
        max_like = flatprobs[np.argmax(flatprobs)]
        print("maximum likelihood = ", max_like)
        flatendchain = sampler.chain[:,niter-2000:,:].reshape((-1,ndim))
        if (emcee.__version__ == '3.0rc2'):
            flatendprobs = sampler.lnprobability[niter-2000:,:].reshape((-1))
        else:
            flatendprobs = sampler.lnprobability[:, niter-2000:].reshape((-1))
        theta_max_end = flatendchain[np.argmax(flatendprobs)]
        max_end_like = np.amax(flatendprobs)
        print("maximum likelihood in final 2K iterations= ", max_end_like)
        print("Mean autocorrelation time: {0:.3f} steps"
              .format(np.mean(sampler.get_autocorr_time(discard=0,c=10,quiet=True))))

    elif(fin ==0):
        pic = results_path+runname+"_snapshot.pic"
        chain,probs = pickle_load(pic) 
        nwalkers = chain.shape[0]
        ntot = chain.shape[1]
        ndim = chain.shape[2]
        niter = int(np.count_nonzero(chain) / (nwalkers*ndim))
        flatprobs = probs[:,:].reshape((-1))
        max_like = flatprobs[np.argmax(probs)]
        print("Unfinished symphony. Number of successful iterations = ", niter)
        print("maximum likelihood = ", max_like)
        flatendchain = chain[:,(niter-2000):niter,:].reshape((-1,ndim))
        flatendprobs = probs[(niter-2000):niter,:].reshape((-1))
        theta_max_end = flatendchain[np.argmax(flatendprobs)]
        max_end_like = np.amax(flatendprobs)
        print("maximum likelihood in final 2K iterations= ", max_end_like)
    else:
        print("File extension not recognised")
        stop
        
    return flatendchain, flatendprobs,ndim


def proc_spec(shiftspec,theta,fwhm,chemeq,gasnum,obspec):
    import numpy as np
    import scipy as sp
    from bensconv import conv_non_uniform_R

    if chemeq == 0:
        if (gasnum[gasnum.size-1] == 21):
            ng = gasnum.size - 1
        elif (gasnum[gasnum.size-1] == 23):
            ng = gasnum.size -2
        else:
            ng = gasnum.size
            invmr = theta[0:ng]

    else:
        ng = 2

    modspec = np.array([shiftspec[0,::-1],shiftspec[1,::-1]])

     # Modified by Harshil
    if (fwhm == 999):
        sf_NIRSpec = theta[ng + 4]
        sf_MIRI = theta[ng + 5]

        mr_NIRSpec = np.where(modspec[0, :] < wavelength_cutoff)[0]
        or_NIRSpec = np.where(obspec[0, :] < wavelength_cutoff)[0]

        mr_MIRI = np.where(np.logical_and(modspec[0, :] > wavelength_cutoff, modspec[0, :] < 13.0))[0]
        or_MIRI = np.where(np.logical_and(obspec[0, :] > wavelength_cutoff, obspec[0, :] < 13.0))[0]

        mr_photometry = np.where(modspec[0, :] > 13)[0]
        or_photometry = np.where(obspec[0, :] > 13)[0]

        R = obspec[-1, :]

        NIRSpec = sf_NIRSpec * conv_non_uniform_R(obspec[:, or_NIRSpec], modspec[:, mr_NIRSpec], R[or_NIRSpec])
        MIRI = sf_MIRI * conv_non_uniform_R(obspec[:, or_MIRI], modspec[:, mr_MIRI], R[or_MIRI])
        photometry = conv_non_uniform_R(obspec[:, or_photometry], modspec[:, mr_photometry], R[or_photometry])

        outspec = np.array(np.concatenate((NIRSpec,MIRI,photometry),axis=0))

    return outspec
            

class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))
