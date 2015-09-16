import numpy as np

from pele.systems import BaseSystem
from MLbasinhopping.potentials import MLPotential

class MLSystem(BaseSystem):
    def __init__(self, model):
        super(MLSystem, self).__init__()
        self.model = model
    
    def get_potential(self):
        return MLPotential(self.model)

    def get_mindist(self):
        # minimum distance is linear distance between two sets of parameter values.
        # currently no symmetries are considered, since they are model-dependent.
        return lambda x1, x2: (np.linalg.norm(x1-x2), x1, x2)

    def get_orthogonalize_to_zero_eigenvectors(self):
        return None
        