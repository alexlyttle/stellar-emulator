import numpy as np
import astropy.constants as const

from .defaults import *

Teff_sun = (const.L_sun / 4 / np.pi / const.sigma_sb / const.R_sun**2)**0.25   

def luminosity(grid):
    return grid[RAD]**2 * (grid[TEFF] / Teff_sun)**4

def initial_hydrogen(grid):
    return 1 - grid[YINI] - grid[ZINI]

def nuclear_luminosity_fraction(grid):
    return 10**grid[LLNB] / luminosity(grid)

def delta_hydrogen(grid):
    return initial_hydrogen(grid) - grid[XCEN]

def log_surface_gravity(grid):
    return np.log10(
        const.G.cgs * const.M_sun.cgs * grid[MASS] / const.R_sun.cgs**2 / grid[RAD]**2
    )
