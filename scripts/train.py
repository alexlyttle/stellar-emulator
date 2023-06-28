import os, json
import numpy as np
import pandas as pd
from datetime import datetime

from eggman.grid import luminosity, log_surface_gravity, calculate_eep
from eggman.grid.defaults import MASS, YINI, ZINI, AGE, TEFF, RAD, DNUF, AMLT

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks

print(f"Using TensorFlow {tf.__version__}")

DIR = "/mnt/data-storage/stellar-emulator/train"
FILENAME = "/mnt/data-storage/yaguangli2023/stellar-models/grid_models_surface_effect_uncorrected/grid.h5"

RSTATE = 0
EEP = "EEP"
PHASE = "phase"
LUM = "L"
GRAV = "g"
LOG = "log"
SEP = "_"

UNITS = 128
LAYERS = 7
ACTIV = "elu"
LRATE = 1e-3
BETA1 = 0.9
BETA2 = 0.999
LOSS = "mean_squared_error"
BATCH_SIZE = 65536   # 65536 corresponds to about 1000 epochs in 10 mins
MAX_EPOCHS = 100000  # about 17 hours
SHUFFLE = True
# PATIENCE = 100
# MIN_DELTA = 1e-8  # min change after PATIENCE epochs

if not os.path.exists(DIR):
    print(f"Making directory '{DIR}'.")
    os.makedirs(DIR)

print(f"Loading grid data from '{FILENAME}'.")
tracks = pd.read_hdf(FILENAME, "tracks")
stars = pd.read_hdf(FILENAME, "stars")
data = tracks.join(stars).dropna().reset_index()

print("Preprocessing data.")
# Create log10 quantities
for key in [ZINI, AGE, TEFF, RAD, DNUF]:
    data[SEP.join([LOG, key])] = np.log10(data[key])

data[SEP.join([LOG, GRAV])] = log_surface_gravity(data)
data[SEP.join([LOG, LUM])] = np.log10(luminosity(data))

# Drop bad tracks
with open("../notebook/central_hydrogen_problem_tracks.txt") as file:
    bad = list(map(int, file.read().split(",")))

data = data.drop(index=data[data.track.isin(bad)].index)

print("Calculating EEP.")
# Add EEP and drop pre-MS and post log_g cutoff
keys = [SEP.join([LOG, key]) for key in [AGE, TEFF, LUM]]
primary, secondary = calculate_eep(data, keys)
data[PHASE] = primary
data[EEP] = secondary

data = data.drop(index=data[data.phase == -1].index)

print("Preparing train and test dataset.")
train = data.sample(frac=0.8, random_state=RSTATE)
test = data.drop(index=train.index)

# Separate features and labels
features = [EEP, MASS, YINI, SEP.join([LOG, ZINI]), AMLT]
labels = [SEP.join([LOG, key]) for key in [AGE, TEFF, RAD, DNUF]]
num_features, num_labels = len(features), len(labels)

train_features = train[features].astype(np.float32).copy()
test_features = test[features].astype(np.float32).copy()

train_labels = train[labels].astype(np.float32).copy()
test_labels = test[labels].astype(np.float32).copy()

# Normalisation and rescaling
print("Normalizing and rescaling dataset.")
normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features), batch_size=len(train_features))

label_offset = train_labels.mean().to_numpy()
label_scale = train_labels.std().to_numpy()

rescaler = layers.Rescaling(label_scale, offset=label_offset)

# Build model
print("Building model.")
model = tf.keras.Sequential(
    [normalizer] +
    [layers.Dense(UNITS, ACTIV) for _ in range(LAYERS)] +
    [layers.Dense(num_labels), rescaler]
)

opt = optimizers.Adam(learning_rate=LRATE, beta_1=BETA1, beta_2=BETA2)
model.compile(opt, loss=LOSS)

project_name = datetime.now().strftime("%Y%m%d-%H%M%S")
directory = os.path.join(DIR, project_name)

tboard = callbacks.TensorBoard(os.path.join(directory, "logs"))
# earlystop = callbacks.EarlyStopping(monitor="loss", patience=PATIENCE, min_delta=MIN_DELTA)
checkpoint = callbacks.ModelCheckpoint(
    os.path.join(directory, "best_model.tf"), 
    monitor="loss", 
    save_best_only=True
)

print(f"Fitting model '{project_name}'.")
history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=MAX_EPOCHS,
    shuffle=SHUFFLE,
    callbacks=[
        tboard, 
        # earlystop, 
        checkpoint
    ],
)

history_filename = os.path.join(directory, "history.json")
print(f"Saving history to '{history_filename}'.")
history_dict = dict(
    history=history.history,
    params=history.params,
)

with open(history_filename, "w") as file:
    s = json.dumps(history_dict)
    file.write(s)
