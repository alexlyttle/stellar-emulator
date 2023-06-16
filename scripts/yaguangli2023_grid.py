import os
import numpy as np
import pandas as pd

PATH = "/mnt/data-storage/yaguangli2023/stellar-models/grid_models_surface_effect_uncorrected"
FILENAME = "grid_clean.h5"
CLEAN = True
TRACK_COLS = [
    "star_mass",
    "Yinit",
    "Zinit",
    "amlt",
    "fov_core",
    "fov_shell"
]
STAR_COLS = [
    'star_age',
    'radius',
    'Teff',
    'center_h1',
    'log_LHe',
    'log_Lnuc',
    'Dnu_freq_o',
    'eps_o',
    'delta_Pg'
]
MODE_COLS = [
    'mode_freq_o'
]
CONST = {
    "msun": 1.9884098706980504e33,
    "rsun": 6.957e10,
    "lsun": 3.8280000000000003e33,
    "Tsun": 5.7720034291e3,
}

output_filename = os.path.join(PATH, FILENAME)
input_filenames = [os.path.join(PATH, name) for name in os.listdir(PATH) if name.endswith(".npy")]

def load_tracks(filename):
    return np.load(filename, allow_pickle=True)

def number_of_modes(tracks):
    return np.vectorize(len)(tracks["mode_n"])

def add_index_cols(df):
    df["track"] = df["index"].astype(int)
    df["star"] = df["profile_number"].astype(int)
    df["n"] = df["mode_n"].astype(int)
    df["l"] = df["mode_l"].astype(int)
    return df

def tracks_to_dataframe(tracks):
    num_modes = number_of_modes(tracks)
    keys = [key for key, dtype in tracks.dtype.descr if dtype != "|O"]
    
    print("Expanding modes")
    expanded = np.repeat(tracks[keys], num_modes, axis=0)
    
    print("Creating dataframe")
    df = pd.DataFrame.from_records(expanded)
    for key in ["mode_n", "mode_l", "mode_freq_o"]:
        df[key] = np.concatenate(tracks[key])
    return add_index_cols(df)

def clean_dataframe(df):
    """Remove pre main sequence points."""
    print("Removing pre-main sequence points")
    df["f_nuc"] = 10**df.log_Lnuc / df.luminosity
    df["delta_X"] = df.Xinit - df.center_h1
    idxs = []
    for _, group in df.groupby("track"):
        hburn = (group["f_nuc"] > 0.999) & (group["delta_X"] > 0.0015)
        mask = group.index < hburn.idxmax()
        idxs.append(group.index[mask].to_numpy())
    return df.drop(index=np.concatenate(idxs))

def append_hdf(filename, df, group, index_cols, data_cols):
    df.drop_duplicates(index_cols).set_index(index_cols)[data_cols].to_hdf(filename, group, append=True, format="table")

def update(input_filename):
    print("Loading", input_filename)
    df = tracks_to_dataframe(load_tracks(input_filename))
    if CLEAN:
        df = clean_dataframe(df)
    
    print("Appending tracks to", output_filename)
    append_hdf(output_filename, df, "tracks", "track", TRACK_COLS)
    
    print("Appending stars to", output_filename)
    append_hdf(output_filename, df, "stars", ["track", "star"], STAR_COLS)
    
    print("Appending modes to", output_filename)
    append_hdf(output_filename, df, "modes", ["track", "star", "n", "l"], MODE_COLS)

def add_metadata(filename):
    print("Adding metadata")
    pd.Series(CONST).to_hdf(filename, "constants", format="table")

def main():
    if os.path.exists(output_filename):
        raise ValueError("File", output_filename, "exists. Delete or try a different file name.")
    add_metadata(output_filename)
    for input_filename in input_filenames:
        update(input_filename)

if __name__ == "__main__":
    main()
