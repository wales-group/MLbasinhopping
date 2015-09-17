import numpy as np

from pele.systems import BaseSystem
from pele.potentials import BasePotential

class BaseModel(object):
    
    def __init__(self):
        self.nparams = None
        
    def cost(self, coords):
        return NotImplementedError
    def costGradient(self, coords):
        return NotImplementedError
    def costGradientHessian(self, coords):
        return NotImplementedError

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

    def get_random_configuration(self):
        
        return np.random.random(self.model.nparams)
    
class MLPotential(BasePotential):
    """ This class interfaces the model class to pele: 
        The potential energy = cost function """
    def __init__(self, model):
        
        self.model = model
         
    def getEnergy(self, coords):
        return self.model.cost(coords)
 
    def getEnergyGradient(self, coords):
        return self.model.costGradient(coords)
     
    def getEnergyGradientHessian(self, coords):
        return self.model.costGradientHessian(coords)