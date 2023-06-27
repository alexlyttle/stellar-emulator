import numpy as np
import pandas as pd

from .defaults import *
from .star import nuclear_luminosity_fraction, delta_hydrogen, log_surface_gravity

def calculate_eep(grid, keys, alpha=0.2, weights=None, scale=None):
    """Calculate the primary and secondary equivalent evolutionary phase for
    tracks in the grid. This groups the grid by track and assumes that each
    track is in chronological order.
    
    d[i] = d[i-1] + sum_over_j(weights[j] * (x[j][i] - x[j][i-1])**2)**alpha
    
    Args:
        grid (pandas.DataFrame): 
            Must contain MASS, RAD, YINI, ZINI, XCEN and all of `keys`.
        keys (list of str):
            Keys from which to calculate secondary EEP distance metric.
        alpha (float): 
            Exponent of distance metric, defaults to 0.2 (Li et al. 2022).            
        weights (list of float):
            Weights to apply to each key for distance metric, defaults to equal
            weighting.
        scale (list of float):
            Scale factor for distance metric across each primary EEP. Defaults
            to 1.0 for each phase.
    Returns:
        primary (pandas.Series): Primary EEP (0 = ZAMS, 1 = IAMS, 2 = TAMS, 
            3 = MAXNB, 4 = END) and -1 means unnassigned phase.
        secondary (pandas.Series): Not yet implemented.
    """
    if weights is None:
        weights = np.ones(len(keys))
    
    if scale is None:
        scale = 4 * [1]
        
    # Location of each primary EEP phase
    loc = [sum(scale[:i]) for i in range(len(scale))]

    primary = pd.Series(-1, grid.index)
    secondary = pd.Series(np.nan, grid.index)
    grid["delta_X"] = delta_hydrogen(grid)
    grid["f_nuc"] = nuclear_luminosity_fraction(grid)
    grid["log_g"] = log_surface_gravity(grid)

    for _, group in grid.groupby("track"):

        zams = ((group.f_nuc > 0.999) & (group.delta_X > 0.0015)).idxmax()

        iams = (group[XCEN] < 0.3).idxmax()
        iams = group.index[-1] if iams == group.index[0] else iams

        tams = (group[XCEN] < 1e-12).idxmax()
        tams = group.index[-1] if tams == group.index[0] else tams

        end = (group.log_g < 2.2).idxmax()
        end = group.index[-1] if end == group.index[0] else end

        # Max nuclear burning
        maxn = group.loc[(group.index > tams) & (group.index <= end), "f_nuc"].idxmax()

        bounds = [zams, iams, tams, maxn, end]

        phase = [
            group.index[
                (group.index >= low) & (group.index < up)
            ] for low, up in zip(bounds[:-1], bounds[1:])
        ]
        distance = ((weights*group[keys].diff()**2).sum(axis=1)**alpha).cumsum()
        
        for i, p in enumerate(phase):
            primary.loc[p] = i

            lower = bounds[i]
            upper = bounds[i+1]
            secondary.loc[p] = (
                loc[i] + scale[i] * (distance.loc[p] - distance.loc[lower])
                / (distance.loc[upper] - distance.loc[lower])
            )

    return primary, secondary
