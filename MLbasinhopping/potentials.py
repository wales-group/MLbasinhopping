from pele.potentials import BasePotential

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