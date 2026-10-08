import mat73
import scipy.io as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

dictS1 = sp.loadmat('/Users/lianamendoza/Desktop/ANPL/S1_out.mat', squeeze_me=True)
dictS1_clean = {k: v for k, v in dictS1.items() if not k.startswith('__')}

var_name = list(dictS1_clean.keys())[0]
arr = np.array(dictS1_clean[var_name])
df = pd.DataFrame.from_records(np.ravel(arr))

eegdat = df.loc[0, "eegdat"]    # 214 records, one per stimulus

channel_names = [f"EEG{i + 1}" for i in range(64)]   #=
elec = 23              

all_presentations = []

print("Number of stimuli:", len(eegdat)) # to confirm
# print(np.asarray(eegdat[0][5])() # to inspect the first stimulus's response data 
print((np.asarray(eegdat[0][5])).shape)



for stim in range(len(eegdat)):
    resp = np.asarray(eegdat[stim][5])          

    if resp.ndim == 2:                       
        resp = resp[:, :, np.newaxis]
   
    if resp.ndim != 3:                          # empty: skip
        print("skipping stimulus", stim)
        continue


    for rep in range(resp.shape[2]):
        if resp.shape[1] < 201:
            print("skipping stimulus", stim, "because it is too short")           
            continue
        
        all_presentations.append(resp[:, 60:201, rep]) # 60 secs and #201 secs

data = np.array(all_presentations)              # (n, 64, 141)
print("epochs:", data.shape)

info = mne.create_info(ch_names=channel_names, sfreq=200, ch_types="eeg")

epochs = mne.EpochsArray(data, info, tmin=-0.2) # 0.2 because we are using a window from -200 ms to 500 ms

evoked = epochs.average()
print(evoked.times)

evoked.plot(picks=[channel_names[elec]],
            scalings=dict(eeg=1),
            units=dict(eeg="z-score"),
            titles=dict(eeg=f"Subject 1, {channel_names[elec]}: stimulus onset"))

erp = evoked.get_data(picks=channel_names[elec])[0]
times = evoked.times               

plt.figure(figsize=(9, 3.5))
plt.plot(times, erp)
plt.axvline(0, color="black", linestyle="--")  
plt.xlabel("time (s)")
plt.ylabel("amplitude (z-score)")
plt.tight_layout()
# plt.show()

# median split