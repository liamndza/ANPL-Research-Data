import mat73 #for new ver of mat files
import scipy.io as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mne 

# for future reference
# 1) make one big df for outputs (s1 for now!)
# 2) make one big df for features

dictS1 = sp.loadmat('/Users/lianamendoza/Desktop/ANPL/S1_out.mat', squeeze_me=True)
features = mat73.loadmat('/Users/lianamendoza/Desktop/ANPL/mmpfeats.mat')

# remove MATLAB internals

dictS1_clean = {k: v for k, v in dictS1.items() if not k.startswith('__')}
# audio containing sound waves, and onset
features_clean = {k: v for k, v in features.items() if not k.startswith('__')}

# pick the real variable, not a metadata key
var_name = list(dictS1_clean.keys())[0]
arr = np.array(dictS1_clean[var_name])

var_name_features = list(features_clean.keys())[0]
features_data = features_clean[var_name_features]

# print(features_data.keys())
# onsvec is index 0

# if this is a structured array, convert each entry to a DataFrame row

# print(arr.dtype.names) # these are all the field names in the structured array

# subject 1 data 
if arr.dtype.names is not None: #if this is a structured array
    df = pd.DataFrame.from_records(np.ravel(arr))
else: 
    df = pd.DataFrame(arr)

# features data
if isinstance(features_data, dict):
    df_features = pd.DataFrame(features_data)
else:
    arr_features = np.asarray(features_data)
    if arr_features.dtype.names is not None:
        df_features = pd.DataFrame.from_records(np.ravel(arr_features))
    else:
        df_features = pd.DataFrame(arr_features)

# as of line 45 have two DataFrames: df for subject 1 outputs 
# and df_features for the features

# more analysis here 

#print(df_features.keys())
#print(df.keys())

### CHANNEL LOCATIONS ###

# so far, what we know
# we have extracted the channel locations from the subject 1 data
# it is stored as a structured array within the DataFrame


channel_locations = (df.loc[0, 'chanlocs']) # structured array containing channel locations 
# think of this as one row in the DataFrame, containing all channel locations, 
# where each element corresponds to a different channel's information  

#  chan1 = channel_locations[1]
# print(chan1)

# print(channel_locations.dtype)
 # print the data type of the structured array containing channel locations
 # copying run for future ref
 # [('labels', 'O'), ('sph_radius', 'O'), ('sph_theta', 'O'), ('sph_phi', 'O'), ('theta', 'O'), 
 # ('radius', 'O'), ('X', 'O'), ('Y', 'O'), ('Z', 'O'), ('type', 'O'), ('ref', 'O')]

# print(chan1[1]) # this is "sph_radius" of channel 1 etc.

# -------------------------------------------------------------------------

## WORKING WITH MNE ## 

# -------------------------------------------------------------------------

eeg_record = df.loc[0, "eegdat"][0]
# print(eeg_record.dtype)
# copying run for future ref
#[('id', 'O'), ('name', 'O'), ('block', 'O'), ('befaft', 'O'), ('tvect', 'O'), ('resp', 'O')]

# print(np.asarray(eeg_record[5])[:, :, 0])

eeg_data = np.asarray(eeg_record[5])[:, :, 0]
#eeg_data = eeg_data * 1e-6
time_values = np.asarray(eeg_record[4])

# info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)

channel_names = [str(label) for label in channel_locations["labels"]]
channel_positions = np.column_stack([channel_locations[field].astype(float) for field in ("X", "Y", "Z")]) / 1000

info = mne.create_info(ch_names=channel_names, sfreq=200, ch_types="eeg")
raw = mne.io.RawArray(eeg_data, info)

montage = mne.channels.make_dig_montage(
    ch_pos=dict(zip(channel_names, channel_positions)),
    coord_frame="head",
)
raw.set_montage(montage)


fig = plt.figure()
ax2d = fig.add_subplot(121)
ax3d = fig.add_subplot(122, projection="3d")
raw.plot_sensors(ch_type="eeg", axes=ax2d, show_names=True, show=False)
raw.plot_sensors(ch_type="eeg", axes=ax3d, kind="3d", show_names=True, show=False)
ax3d.view_init(azim=70, elev=15)

# plt.show()

### WORKING WITH TIME :: ERPS? ### no generative allowed here
 
#print(df_features.keys())
# print(df.keys())

print(df_features['f0'])

# maybe put it into an array? then compare to onset of note # for i in range(len(df_features['f0'])):

plt.figure()
plt.plot(df_features['f0'][0])
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()