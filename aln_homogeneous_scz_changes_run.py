import numpy as np
from neurolib.models.aln import ALNModel
from neurolib.utils.loadData import Dataset



ds =Dataset('hcp')



# Set the parameters (model is set in the down state close to the meeting point
# of EI and adaptation limit cycle)
duration = 70
fmrisize = int(duration/2)

# ScZ parameter changes
Jii_max = [x*(-1.64)for x in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]] # default value -1.64
Jei_max = [x*(-3.3)for x in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]] # default value -3.3

# Run the model (to assess robustness, use different seeds)
seeds = np.load('Seeds.npy') # Use the same seeds
j=5 # cutoff for the transient signal part
scz_time_series = np.zeros((len(seeds),len(Jii_max), 80,fmrisize-j))
for k,jii in enumerate(Jii_max):
	jei = Jei_max[k]
        for i,seed in enumerate(seeds):
            print(i)
            aln = ALNModel(Cmat = ds.Cmat, Dmat = ds.Dmat)
            aln.params['dt'] = 0.1 # Integration time step, ms
            aln.params['duration'] = duration*1000
            aln.params['mue_ext_mean'] = 1.63 # mV/ms mean external input current to E
            aln.params['mui_ext_mean'] = 0.05  # mV/ms mean external input current to I
            aln.params['sigma_ou'] = 0.19 #0.09 # mV/ms/sqrt(ms) intensity of OU oise
            aln.params['a'] = 28.26
            aln.params['b'] = 24.04
            aln.params['Ke_gl'] = 250.
            aln.params['signalV'] = 20.
            aln.params['seed']=seed
            aln.params['Jii_max']=jii
            aln.params['Jei_max']=jei
            aln.run(bold=True)
            scz_time_series[i,k,:,:] = aln.BOLD.BOLD[:, j:]

np.save('Results/Aln_150s_40seeds_ScZ_changes_GABA_range.npy',scz_time_series)
