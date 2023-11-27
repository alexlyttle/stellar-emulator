import numpyro
import numpy as np
import jax.numpy as jnp
from copy import deepcopy
from ..distributions import LogSalpeter
from isochrones.mist.bc import MISTBolometricCorrectionGrid
from jax.scipy.interpolate import RegularGridInterpolator
from ..emulator import Emulator

dist = numpyro.distributions
ln10 = jnp.log(10)

def lognorm_from_norm(mu, sigma):
    """Returns shape params for lognorm from mu and sigma of norm."""
    var = sigma**2
    mu2 = mu**2
    return (
        jnp.log(mu2) - 0.5*jnp.log(mu2 + var),
        jnp.sqrt(jnp.log(1 + var/mu2))
    )


class Model:
    def __init__(self):
        pass

    def __call__(self, obs=None):
        pass


class Star(Model):
    _eep_bounds = jnp.arange(0.0, 5.0)
    _x_bounds = jnp.array([0.0, 0.3, 0.6, 0.7, 1.0])
    log_zx_sun = np.log10(0.0181)
    log_teff_sun = np.log10(5772)
    log_g_sun = 4.44
    bol_mag_sun = 4.75
    dof = 5

    def __init__(self, bands=None):
        if bands is None:
            bands = ["G", "BP", "RP"]
        self.bands = bands
        self.bc_interp = self._interp_bc_grid()
        # self._eep_funclist = self._get_eep_funclist()
        self.emulator = Emulator()

    def _get_eep_funclist(self):
        widths = jnp.diff(self._eep_bounds)
        new_widths = jnp.diff(self._x_bounds)
        def linear(a, b, c):
            return lambda x: a + b * (x - c)
        return [linear(l, w/nw, nl) for w, nw, l, nl in zip(widths, new_widths, self._eep_bounds[:-1], self._x_bounds[:-1])]

    def evol_phase(self, x):
        """Get evol_phase from transformed parameter x."""
        return jnp.piecewise(
            x,
            [(lower <= x) & (x < upper) for lower, upper in zip(self._x_bounds[:-1], self._x_bounds[1:])],
            self._eep_funclist,
        )

    def log_luminosity(self, log_teff, log_radius):
        return 2 * log_radius + 4 * (log_teff - self.log_teff_sun)

    def log_gravity(self, log_mass, log_radius):
        return log_mass - 2 * log_radius + self.log_g_sun

    def hydrogen(self, y, z):
        return 1 - y - z

    def metallicity(self, y, z):    
        return jnp.log10(z / self.hydrogen(y, z)) - self.log_zx_sun

    def heavy_elements(self, y, mh):
        return (1 - y) / (10**-(mh + self.log_zx_sun) + 1)

    def bolometric_magnitude(self, log_lum):
        return self.bol_mag_sun - 2.5 * log_lum

    def absolute_magnitude(self, bol_mag, bc):
        return bol_mag - bc

    def apparent_magnitude(self, abs_mag, plx):
        return abs_mag - 5 * (jnp.log10(plx) + 1)

    def _interp_bc_grid(self):    
        bc_grid = MISTBolometricCorrectionGrid(bands=self.bands)
        points = [bc_grid.df.index.unique(level=name).to_numpy() for name in bc_grid.df.index.names]
        shape = [x.shape[0] for x in points]
        values = np.reshape(bc_grid.df[self.bands].to_numpy(), shape + [len(self.bands),])
        return RegularGridInterpolator(points, values)

    def prior(self, const):
        # x = numpyro.sample("x", dist.Uniform(0.0, 0.999))
        s = numpyro.sample("s", dist.Beta(2.0, 5.0))
        # ln_mass = numpyro.sample("ln_mass", LogSalpeter(jnp.log(0.7), jnp.log(2.3), rate=2.35))
        # Good approximation of Chabrier IMF
        ln_mass = numpyro.sample("ln_mass", dist.TruncatedNormal(-0.2, 0.7, low=jnp.log(0.7), high=jnp.log(2.3)))
        y = numpyro.sample("Y", dist.Uniform(0.22, 0.32))
        mh = numpyro.sample("M_H", dist.TruncatedNormal(const["M_H"]["mu"], const["M_H"]["sigma"], low=-0.9, high=0.5))
        a_mlt = numpyro.sample("a_MLT", dist.Uniform(1.3, 2.7))
        
        plx_dist = dist.LogNormal(*lognorm_from_norm(const["plx"]["mu"], const["plx"]["sigma"]))
        plx = numpyro.sample("plx", plx_dist)

        av_dist = dist.LogNormal(*lognorm_from_norm(const["Av"]["mu"], const["Av"]["sigma"]))            
        av = numpyro.sample("Av", av_dist)

        return s, ln_mass, mh, y, a_mlt, plx, av

    def _emulator_inputs(self, s, ln_mass, mh, y, a_mlt):
        # eep = numpyro.deterministic("EEP", self.evol_phase(x))

        mass = numpyro.deterministic("mass", jnp.exp(ln_mass))
        z = numpyro.deterministic("Z", self.heavy_elements(y, mh))
        # log_z = jnp.log10(z)
        return jnp.stack([s, mass, mh, y, a_mlt], axis=-1)

    def emulate_star(self, s, ln_mass, mh, y, a_mlt):

        inputs = self._emulator_inputs(s, ln_mass, mh, y, a_mlt)
        outputs = self.emulator(inputs)

        log_lum = numpyro.deterministic("log_lum", self.log_luminosity(outputs[1], outputs[2]))
        log_rad = numpyro.deterministic("log_rad", outputs[2])
        numpyro.deterministic("rad", 10**log_rad)

        logg = numpyro.deterministic("logg", self.log_gravity(ln_mass/ln10, log_rad))
        lum = numpyro.deterministic("lum", 10**log_lum)
        teff = numpyro.deterministic("Teff", 10**outputs[1])
        dnu = numpyro.deterministic("Dnu", 10**outputs[3])
        
        log_age = numpyro.deterministic("log_age", outputs[0])
        numpyro.deterministic("age", 10**(log_age-9))

        return teff, logg, log_lum
    
    def __call__(self, const, obs=None):
        if obs is None:
            obs = {}

        s, ln_mass, mh, y, a_mlt, plx, av = self.prior(const)
        
        teff, logg, log_lum = self.emulate_star(s, ln_mass, mh, y, a_mlt)

        numpyro.deterministic("dist", 1/plx)

    #     bc = numpyro.deterministic("bol_corr", bc_interp(xx).squeeze())
        bc = self.bc_interp(jnp.stack([teff, logg, mh, av], axis=-1)).squeeze()

        bol_mag = self.bolometric_magnitude(log_lum)
        abs_mag = self.absolute_magnitude(bol_mag, bc)
        mag = self.apparent_magnitude(abs_mag, plx)

        for i, band in enumerate(self.bands):
            numpyro.deterministic(f"{band}_abs", abs_mag[i])
            mu = numpyro.deterministic(band, mag[i])
            if band not in obs:
                continue
            numpyro.sample(
                f"{band}_obs", 
                dist.StudentT(self.dof, mu, const[band]["sigma"]),
                obs=obs[band],
            )
