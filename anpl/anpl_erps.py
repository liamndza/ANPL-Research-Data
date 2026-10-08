import mat73
import scipy.io as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dictS1 = sp.loadmat('/Users/lianamendoza/Desktop/ANPL/S1_out.mat', squeeze_me=True)
features = mat73.loadmat('/Users/lianamendoza/Desktop/ANPL/mmpfeats.mat')
 
dictS1_clean = {k: v for k, v in dictS1.items() if not k.startswith('__s')}
features_clean = {k: v for k, v in features.items() if not k.startswith('__')}
 
var_name = list(dictS1_clean.keys())[0]
arr = np.array(dictS1_clean[var_name])
df = pd.DataFrame.from_records(np.ravel(arr))
 
var_name_features = list(features_clean.keys())[0]
all_onsets = features_clean[var_name_features]['eventOnsetTimes']
 
eegdat = df.loc[0, "eegdat"]    # 214 records, one per stimulus

elec = 23                    # electrode 24
sfreq = 200                  # samples per second
pre = int(0.200 * sfreq)     # 40 samples before onset (200 ms)
post = int(0.500 * sfreq)    # 100 samples after onset (500 ms)
 
erp_response = []
skipped = 0
 
for stim in range(len(eegdat)):
    record = eegdat[stim]
    tvect = np.ravel(record[4]).astype(float)   # time of each sample (s)
    resp = np.asarray(record[5])                # (64, 1428, 2)
 
    if resp.ndim == 2:                         
        resp = resp[:, :, np.newaxis]
    if resp.ndim != 3:                          # empty / unusual: skip
        print("skipping stimulus", stim, record[1], "- resp shape:", resp.shape)
        continue
 
    if all_onsets[stim] is None:
        continue
    onsets = np.ravel(all_onsets[stim]).astype(float) - 0.5

    for rep in range(resp.shape[2]):            # each block/segment
        signal = resp[elec, :, rep]
 
        for onset in onsets:
            center = int(round((onset - tvect[0]) * sfreq))
            start = center - pre
            stop = center + post
 
            if start < 0 or stop > len(signal):
                skipped += 1
                continue

            trial = signal[start:stop]       
            erp_response.append(trial)
 
erp_response = np.array(erp_response)    # (n_trials, 140)
erp = np.mean(erp_response, axis=0)
 
samples = np.arange(1, len(erp) + 1)     # 1..140, like MATLAB's plot(erp)
 
plt.figure(figsize=(9, 3.5))
plt.plot(samples, erp)
plt.axvline(pre + 1, color="black", linestyle="--")   # sound starts at sample 51
plt.xlim(0, 150)
plt.xlabel("time (samples)")
plt.ylabel("amplitude (z-score)")
plt.tight_layout()
plt.savefig("S1_elec24_ERP_stimOnset_python.png", dpi=150)
plt.show()
 