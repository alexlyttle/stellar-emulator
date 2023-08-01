import os
import jax.numpy as jnp
from jax.nn import elu
from tensorflow.keras.models import load_model

from . import PACKAGEDIR


class Emulator:
    PATH = os.path.join(PACKAGEDIR, "emulator")

    def __init__(self):
        tf_model = load_model(self.PATH)
        self.weights = tf_model.get_weights()
        self.offset = jnp.array(tf_model.layers[-1].offset)
        self.scale = jnp.array(tf_model.layers[-1].scale)
        self.tf_model = tf_model

    def model(self, x):
        """Emulator model.
        
        Args:
            x (array-like): Neural network inputs.
        
        Returns:
            jax.numpy.ndarray: Neural network outputs.
        """
        x -= self.weights[0]
        x /= self.weights[1]**0.5
        for w, b in zip(self.weights[3:-2:2], self.weights[4:-1:2]):
            x = elu(jnp.matmul(x, w) + b)
        x = jnp.matmul(x, self.weights[-2]) + self.weights[-1]
        return self.offset + self.scale * x

    def __call__(self, x):
        """Emulator model.
        
        Args:
            x (array-like): Neural network inputs.
        
        Returns:
            jax.numpy.ndarray: Neural network outputs.
        """
        return self.model(x)
