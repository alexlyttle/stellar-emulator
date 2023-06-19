import pandas as pd

from .defaults import *
from .star import nuclear_luminosity_fraction, delta_hydrogen, log_surface_gravity

def calculate_eep(grid):
    """Calculate the primary and secondary equivalent evolutionary phase for
    tracks in the grid. This groups the grid by track and assumes that each
    track is in chronological order.
    
    Returns:
        primary (pandas.Series): Primary EEP (0 = ZAMS, 1 = IAMS, 2 = TAMS, 
            3 = MAXNB, 4 = END) and -1 means unnassigned phase.
        secondary (pandas.Series): Not yet implemented.
    """
    primary = pd.Series(-1, grid.index)
    secondary = None
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

        phase = [group.index[(group.index >= low) & (group.index < up)] for low, up in zip(bounds[:-1], bounds[1:])]

        for i, phi in enumerate(phase):
            primary.loc[phi] = i

    return primary, secondary
