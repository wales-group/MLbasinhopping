import numpy as np
import theano
import theano.tensor as T
from pele.potentials import BasePotential
from pele.systems import BaseSystem
                   
class BaseTheanoModel(object):
    """ This is an example class for playing around with regression modelling using theano.
        We would eventually like to abstracify this, so that one can input
        a regression model expression string.
    """
    
    def __init__(self, input_data, target_data, starting_params):
        
        self.X = theano.shared(input_data)
        self.t = theano.shared(target_data)
    
        self.x_to_predict = theano.shared(np.random.random(input_data.shape[1]))
        self.model_function = theano.function
        
        self.params = theano.shared(starting_params)
        self._theano_cost = theano.function(inputs=[],
                                           outputs=self.costFunction()
                                           )
        
        gradient = [T.grad(self.costFunction(), self.params)]
        costGradient = [self.costFunction()] + gradient
        
        self._theano_costGradient = theano.function(inputs=[],
                                                   outputs=costGradient
                                                   )

        hess = theano.gradient.hessian(self.costFunction(), self.params)
        outputs = [self.costFunction()] + gradient + [hess]
        self._theano_costGradientHessian = theano.function(inputs=[],
                                                          outputs=outputs
                                                          )
        
    def theano_cost(self, params):
        self.params.set_value(params)
        return self._theano_cost()
    
    def theano_costGradient(self, params):
        self.params.set_value(params)
        return self._theano_costGradient()
    
    def theano_costGradientHessian(self, params):
        self.params.set_value(params)
        return self._theano_costGradientHessian()
            
    def predict(self, xval):
        
        self.x_to_predict.set_value(xval)    
        
        return self.Y(self.x_to_predict).eval()
    
    def Y(self, X):
        """ 
        This defines the model
        inputs : X -- theano shared variable
        returns : Theano shared variable
        """
        raise NotImplementedError

        
    def costFunction(self):
#         print self.params.shape
#         self.params.set_value(params)
        Cost = T.sum((self.Y(self.X) - self.t)**2)
        
        return Cost

            
class RegressionPotential(BasePotential):
    def __init__(self, model):
        
        self.model = model
         
    def getEnergy(self, coords):
        return self.model.theano_cost(coords)
 
    def getEnergyGradient(self, coords):
        ret = self.model.theano_costGradient(coords)
        return ret[0].item(), ret[1]
#         return self.model.theano_costGradient(coords)
     
    def getEnergyGradientHessian(self, coords):
        return self.model.theano_costGradientHessian(coords)
     

class RegressionSystem(BaseSystem):
    def __init__(self, model):
        super(RegressionSystem, self).__init__()
        self.model = model
        self.params.database.accuracy =0.01
#         self.params.double_ended_connect.local_connect_params.tsSearchParams.hessian_diagonalization = True

    def get_potential(self):
        return RegressionPotential(self.model)
    
    def get_mindist(self):
        # no permutations of parameters
        #return mindist_with_discrete_phase_symmetry
        #permlist = []
        return lambda x1, x2: (np.linalg.norm(x1-x2), x1, x2)

    def get_orthogonalize_to_zero_eigenvectors(self):
        return None
    
    def get_minimizer(self, tol=1.0e-6, nsteps=100000, M=4, iprint=0, maxstep=1.0, **kwargs):
        from pele.optimize import lbfgs_cpp as quench
        return lambda coords: quench(coords, self.get_potential(), tol=tol, 
                                     nsteps=nsteps, M=M, iprint=iprint, 
                                     maxstep=maxstep, 
                                     **kwargs)
#     def get_minimizer(self, **kwargs):
#         return lambda coords: myMinimizer(coords, self.get_potential(),**kwargs)
    
    def get_compare_exact(self, **kwargs):
        # no permutations of parameters
        mindist = self.get_mindist()
        return lambda x1, x2: mindist(x1, x2)

