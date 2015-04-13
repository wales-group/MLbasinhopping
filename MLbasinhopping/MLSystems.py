import numpy as np
import theano
import theano.tensor as T
from pele.systems import BaseSystem

class MLSystem(BaseSystem):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets        