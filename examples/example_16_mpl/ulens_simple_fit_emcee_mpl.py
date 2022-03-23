"""
Fits PSPL model with parallax using EMCEE sampler.

"""
import os
import sys
import numpy as np
import time
from multiprocessing import Pool

try:
    import emcee
except ImportError as err:
    print(err)
    print("\nEMCEE could not be imported.")
    print("Get it from: http://dfm.io/emcee/current/user/install/")
    print("and re-run the script")
    sys.exit(1)
import matplotlib.pyplot as plt

import MulensModel as mm

import corner

allow_negative_blend_flux = False

# Define likelihood functions
def ln_like(theta, event, parameters_to_fit):
    """ likelihood function """
    for key, val in enumerate(parameters_to_fit):
        setattr(event.model.parameters, val, theta[key])

    chi2 = event.get_chi2()
    return -0.5 * chi2


def ln_prior(theta, parameters_to_fit):
    """priors - we only reject obviously wrong models"""
    if theta[parameters_to_fit.index("t_E")] < 0.:
        return -np.inf
    return 0.0


def get_fluxes(event):
    """
    Taken from example_10
    For given Event instance extract all the fluxes and return them in
    a list. Odd elements are source fluxes and even ones are blending fluxes.
    These fluxes are in units used by MulensModel, where 1 corresponds to
    22 mag.
    """
    fluxes = []
    for dataset in event.datasets:
        (data_source_flux, data_blend_flux) = event.get_flux_for_dataset(
            dataset)
        fluxes.append(data_source_flux[0])
        fluxes.append(data_blend_flux)

    return fluxes


def run_flux_checks_ln_prior(n_fluxes_per_dataset, fluxes):
    """
    KK: Taken from example 16 and modified
    Run the checks on fluxes - are they in the prior?
    """
    inside = 0.
    outside = -np.inf

    blend_index = n_fluxes_per_dataset - 1
    if fluxes[blend_index] < 0.:
        return outside

    return inside

def ln_prob(theta, event, parameters_to_fit):
    """
    Taken from example_10
    combines likelihood and priors; returns ln(prob) and a list of fluxes
    """
    fluxes = [None] * 2 * len(event.datasets)

    ln_prior_ = ln_prior(theta, parameters_to_fit)
    if not np.isfinite(ln_prior_):
        return (-np.inf, fluxes)
    ln_like_ = ln_like(theta, event, parameters_to_fit)
    
    fluxes = get_fluxes(event)
    
    if not allow_negative_blend_flux:
        n_fluxes_per_dataset = 2 + 1
        ln_prior_flux = run_flux_checks_ln_prior(n_fluxes_per_dataset, fluxes)
        if not np.isfinite(ln_prior_flux):
            return (-np.inf, fluxes)

    # In the cases that source fluxes are negative we want to return
    # these as if they were not in priors.
    if np.isnan(ln_like_):
        return (-np.inf, fluxes)

    ln_prob_ = ln_prior_ + ln_like_ + ln_prior_flux

    return (ln_prob_, fluxes)


# Read the data
datasets = []

file_name = "Gaia17bdk_GSA.dat"
gTime, gMags, gErrs = np.genfromtxt(file_name, dtype=np.float, usecols=(0,1,2), unpack=True)
gData = mm.MulensData(
    data_list = (gTime, gMags, gErrs),
    phot_fmt = 'mag',
    add_2450000 = False)
    
datasets.append(gData)

file_name = "Gaia17bdk_OGLE.dat"
bpTime, bpMags, bpErrs = np.genfromtxt(file_name, dtype=np.float, usecols=(0,1,2), unpack=True)
bpData = mm.MulensData(
    data_list = (bpTime, bpMags, bpErrs),
    phot_fmt = 'mag',
    add_2450000 = False)
    
datasets.append(bpData)

file_name = "Gaia17bdk_KMTNet_KMTA.dat"
rpTime, rpMags, rpErrs = np.genfromtxt(file_name, dtype=np.float, usecols=(0,1,2), unpack=True)
rpData = mm.MulensData(
    data_list = (rpTime, rpMags, rpErrs),
    phot_fmt = 'flux',
    add_2450000 = False)
    
datasets.append(rpData)

file_name = "Gaia17bdk_KMTNet_KMTC.dat"
rpTime, rpMags, rpErrs = np.genfromtxt(file_name, dtype=np.float, usecols=(0,1,2), unpack=True)
kData = mm.MulensData(
    data_list = (rpTime, rpMags, rpErrs),
    phot_fmt = 'flux',
    add_2450000 = False)
    
datasets.append(kData)

file_name = "Gaia17bdk_KMTNet_KMTS.dat"
rpTime, rpMags, rpErrs = np.genfromtxt(file_name, dtype=np.float, usecols=(0,1,2), unpack=True)
k2Data = mm.MulensData(
    data_list = (rpTime, rpMags, rpErrs),
    phot_fmt = 'flux',
    add_2450000 = False)
    
datasets.append(k2Data)


coords = "17:42:32.27 -34:06:13.46"

# Starting parameters:
params = dict()
params['t_0'] = 2457878.
params['t_0_par'] = 2457878.
params['u_0'] = 0.1  # Change sign of u_0 to find the other solution.
params['t_E'] = 120.
params['pi_E_N'] = 0.
params['pi_E_E'] = 0.
my_model = mm.Model(params, coords=coords)
my_event = mm.Event(datasets=datasets, model=my_model)

# Which parameters we want to fit?
parameters_to_fit = ["t_0", "u_0", "t_E", "pi_E_N", "pi_E_E"]
# And remember to provide dispersions to draw starting set of points
sigmas = [5.0, 0.2, 100., 0.1, 0.1]

# Initializations for EMCEE
n_dim = len(parameters_to_fit)
n_walkers = 80
n_steps = 1000
n_burn = 500
# Including the set of n_walkers starting points:
start_1 = [params[p] for p in parameters_to_fit]
start = [start_1 + np.random.randn(n_dim) * sigmas
         for i in range(n_walkers)]

with Pool() as pool:
    # Run emcee (this can take some time):
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, ln_prob, args=(my_event, parameters_to_fit), pool=pool)
    
    tStart = time.time()
    sampler.run_mcmc(start, n_steps, progress=True)
    tEnd = time.time()

    serial_time = tEnd - tStart
    print("Multiprocessing took {0:.1f} seconds".format(serial_time))

# Remove burn-in samples and reshape:
samples = sampler.chain[:, n_burn:, :]
blobs = np.array(sampler.blobs)
blobs = np.transpose(blobs, axes=(1, 0, 2))[:, n_burn:, :]
print("Samples: ", samples.shape, "Blobs: ", blobs.shape)
samples = np.dstack((samples, blobs))
samples = samples.reshape((-1, len(parameters_to_fit)+2*len(my_event.datasets)))

# Results:
results = np.percentile(samples, [16, 50, 84], axis=0)
print("Fitted parameters:")
for i in range(n_dim):
    r = results[1, i]
    print("{:.5f} {:.5f} {:.5f}".format(r, results[2, i]-r, r-results[0, i]))

# KK: printing results for fluxes
outFluxes = []    
for i in range(n_dim, n_dim + 2*len(my_event.datasets)):
    r = results[1, i]
    print("{:.5f} {:.5f} {:.5f}".format(r, results[2, i]-r, r-results[0, i]))
    outFluxes.append(r)

for i in range(len(my_event.datasets)):
  fluxs =  outFluxes[2*i]
  fluxb =  outFluxes[2*i+1]
  fs = fluxs/(fluxs+fluxb)
  I0 = 22-2.5*np.log10(fluxs+fluxb)
  print("Dataset %d: I0 = %.5f, fs = %.5f"%(i, I0, fs))


# We extract best model parameters and chi2 from the chain:
prob = sampler.lnprobability[:, n_burn:].reshape((-1))
best_index = np.argmax(prob)
best_chi2 = prob[best_index] / -0.5
best = samples[best_index, :]
print("\nSmallest chi2 model:")
print(*[repr(b) if isinstance(b, float) else b.value for b in best])
print(best_chi2)

labels = parameters_to_fit
for i in range(len(my_event.datasets)):
  labels += ["flux_s_%d"%i, "flux_b_%d"%i]

figure =  corner.corner(samples, bins=60, color='k', fontsize=20, alpha=0.8, 
                        show_titles=False, verbose=True, labels=parameters_to_fit, 
                        plot_datapoints=True, levels=(1-np.exp(-0.5), 1-np.exp(-2.), 1-np.exp(-9./2.)), 
                        fill_contours=True)
                        
figure.savefig("Gaia17bdk_corner_simple_mpl.png")
