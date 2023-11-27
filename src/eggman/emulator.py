import os
import numpy as np
import jax.numpy as jnp
from pytensor import tensor as pt
from jax.nn import elu as elu_jax
from jax.nn import reul as relu_jax
from tensorflow.keras.models import load_model

from . import PACKAGEDIR

def elu(x, alpha=1.0):
	return np.where(x >= 0, x, alpha*(np.exp(x) - 1))

def elu_pt(x, alpha=1.0):
    return pt.where(x >= 0, x, alpha*(pt.exp(x) - 1))

def relu(x):
    return np.where(x >= 0, x, 0)

def relu_pt(x):
    return pt.where(x >= 0, x, 0)


class Emulator:
    PATH = os.path.join(PACKAGEDIR, "emulator")

    def __init__(self):
        tf_model = load_model(self.PATH)
        self.weights = tf_model.get_weights()
        self.offset = jnp.array(tf_model.layers[-1].offset)
        self.scale = jnp.array(tf_model.layers[-1].scale)
        self.model_tf = tf_model

    def model(self, x):
        """Emulator model.
        
        Args:
            x (array-like): Neural network inputs.
        
        Returns:
            jax.numpy.ndarray: Neural network outputs.
        """
        x = np.array(x)
        x -= self.weights[0]
        x /= self.weights[1]**0.5
        for w, b in zip(self.weights[3:-2:2], self.weights[4:-1:2]):
            x = relu(np.matmul(x, w) + b)
        x = np.matmul(x, self.weights[-2]) + self.weights[-1]
        return self.offset + self.scale * x

    def model_jax(self, x):
        """Emulator model.
        
        Args:
            x (array-like): Neural network inputs.
        
        Returns:
            jax.numpy.ndarray: Neural network outputs.
        """
        x = jnp.array(x)
        x -= self.weights[0]
        x /= self.weights[1]**0.5
        for w, b in zip(self.weights[3:-2:2], self.weights[4:-1:2]):
            x = relu_jax(jnp.matmul(x, w) + b)
        x = jnp.matmul(x, self.weights[-2]) + self.weights[-1]
        return self.offset + self.scale * x

    def model_pt(self, x):
        """Emulator model.
        
        Args:
            x (array-like): Neural network inputs.
        
        Returns:
            jax.numpy.ndarray: Neural network outputs.
        """
        x = x - self.weights[0]
        x /= self.weights[1]**0.5
        for w, b in zip(self.weights[3:-2:2], self.weights[4:-1:2]):
            x = relu_pt(pt.matmul(x, w) + b)
        x = pt.matmul(x, self.weights[-2]) + self.weights[-1]
        return self.offset + self.scale * x

    def __call__(self, x, backend="jax"):
        """Emulator model.
        
        Args:
            x (array-like): Neural network inputs.
        
        Returns:
            jax.numpy.ndarray: Neural network outputs.
        """
        if backend == "jax":
            return self.model_jax(x)
        elif backend == "numpy":
            return self.model(x)
        elif backend == "pytensor":
            return self.model_pt(x)
        elif backend == "tensorflow":
            return self.model_tf(x)
        else:
            raise NotImplementedError(f"Backend {repr(backend)} not implemented.")
