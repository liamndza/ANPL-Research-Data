import mat73
import scipy.io as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

# Collect pitch value of every note in dataset
features = mat73.loadmat('/Users/lianamendoza/Desktop/ANPL/mmpfeats.mat')

features_clean = {k: v for k, v in features.items() if not k.startswith('__')}
var_name_features = list(features_clean.keys())[0]
print(var_name_features)
features_data = features_clean[var_name_features]