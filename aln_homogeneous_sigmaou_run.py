import numpy as np
from neurolib.models.aln import ALNModel
from neurolib.utils.loadData import Dataset




ds =Dataset('hcp')


# Set the parameters (model is set in the down state close to the meeting point
# of EI and adaptation limit cycle)
duration = 70
fmrisize = int(duration/2)

# ScZ parameter changes (global coupling here)
sigma_ous = [x*(0.19)for x in [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]] # default value 0.19


# Run the model (to assess robustness, use different seeds)
seeds = np.load('Seeds.npy') # Use the same seeds
#scores = np.zeros((len(seeds),7))
j=5 # cutoff for the transient signal part
scz_time_series = np.zeros((len(seeds),len(sigma_ous), 80,fmrisize-j))
for k,sigma_ou in enumerate(sigma_ous):
    print(k)
    for i,seed in enumerate(seeds):
        print(i)
        aln = ALNModel(Cmat = ds.Cmat, Dmat = ds.Dmat)
        aln.params['dt'] = 0.1 # Integration time step, ms
        aln.params['duration'] = duration*1000
        aln.params['mue_ext_mean'] = 1.63 # mV/ms mean external input current to E
        aln.params['mui_ext_mean'] = 0.05  # mV/ms mean external input current to I
        aln.params['sigma_ou'] = sigma_ou #0.19 # mV/ms/sqrt(ms) intensity of OU oise
        aln.params['a'] = 28.26
        aln.params['b'] = 24.04
        aln.params['Ke_gl'] = 250.
        aln.params['signalV'] = 20.
        aln.params['seed']=seed
        aln.run(bold=True)
        scz_time_series[i,k,:,:] = aln.BOLD.BOLD[:, j:]

np.save('Results/Aln_150s_40seeds_ScZ_changes_sigma_ou.npy',scz_time_series)
