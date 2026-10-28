import numpy as np
import matplotlib.pyplot as plt
import mat73

features = mat73.loadmat('/Users/lianamendoza/Desktop/ANPL/mmpfeats.mat')
features_clean = {k: v for k, v in features.items() if not k.startswith('__')}
features_data = features_clean[list(features_clean.keys())[0]]

if isinstance(features_data, dict):
    n = len(features_data["name"])
    features_data = [{field: values[i] for field, values in features_data.items()}
                     for i in range(n)]

absf0 = []

for feat in features_data: 
    name = feat["name"]
    if isinstance(name, (list, np.ndarray)):
        name = str(np.ravel(name)[0])
    if "percussion" in name:        
        continue

    ons = np.flatnonzero(np.ravel(feat["eventOnsets"]))   # note onset indices
    ons[0] = 50                   # MATLAB onsT(1) = 51

    f0 = np.ravel(feat["f0"]).astype(float)

    for t in range(len(ons)):
        end = ons[t + 1]   # note runs until the next onset
        absf0.append(np.nanmedian(f0[ons[t]:end]))          # one pitch average per note

absf0 = np.array(absf0)
print(len(absf0))


plt.figure(figsize=(13, 8))
plt.hist(absf0, bins=60, color="steelblue", edgecolor="white")
plt.xlabel("pitch (Hz)")
plt.ylabel("number of notes")
plt.title("Distribution of note pitches in the stimulus set")
plt.tight_layout()
plt.show()
