import numpy as np
from neurolib.models.aln import ALNModel
from neurolib.utils.loadData import Dataset




ds =Dataset('hcp')


# Set the parameters (model is set in the down state close to the meeting point
# of EI and adaptation limit cycle)
duration = 70
fmrisize = int(duration/2)

# ScZ parameter changes (global coupling here)
Ke_gls = [x*(250.)for x in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]] # default value 250.


# Run the model (to assess robustness, use different seeds)
seeds = np.load('Seeds.npy') # Use the same seeds
#scores = np.zeros((len(seeds),7))
j=5 # cutoff for the transient signal part
scz_time_series = np.zeros((len(seeds),len(Ke_gls), 80,fmrisize-j))
for k,Ke_gl in enumerate(Ke_gls):
    print(k)
    for i,seed in enumerate(seeds):
        print(i)
        aln = ALNModel(Cmat = ds.Cmat, Dmat = ds.Dmat)
        aln.params['dt'] = 0.1 # Integration time step, ms
        aln.params['duration'] = duration*1000
        aln.params['mue_ext_mean'] = 1.63 # mV/ms mean external input current to E
        aln.params['mui_ext_mean'] = 0.05  # mV/ms mean external input current to I
        aln.params['sigma_ou'] = 0.19 #0.19 # mV/ms/sqrt(ms) intensity of OU oise
        aln.params['a'] = 28.26
        aln.params['b'] = 24.04
        aln.params['Ke_gl'] = Ke_gl
        aln.params['signalV'] = 20.
        aln.params['seed']=seed
        aln.run(bold=True)
        #print(j)
        scz_time_series[i,k,:,:] = aln.BOLD.BOLD[:, j:]

np.save('Results/Aln_150s_40seeds_ScZ_changes_Ke_gl.npy',scz_time_series)
