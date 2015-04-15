import numpy as np
import theano
import theano.tensor as T
from pele.potentials import BasePotential
from pele.systems import BaseSystem
                   
class BaseTheanoModel(object):
    """ This is the Base Model class for model parameter estimation,
     used in conjunction with basin-hopping for model parameter estimation.
     Most of the objects in this class are theano functions. 
        We would eventually like to abstracify this, so that one can input
        a regression model expression string.
    """
    
    def __init__(self, input_data, target_data, testX, testt, starting_params):
        
        self.x_to_predict = theano.shared(np.random.random(input_data.shape))
        self.params = theano.shared(starting_params)

        self.X = theano.shared(input_data)
        if target_data == None:
            target_data = self.drawSamples(starting_params, input_data, sigma=0.02)
        self.t = theano.shared(target_data)
    
        self.testX = theano.shared(testX)
        if testt == None:
            testt = self.drawSamples(starting_params, testX, sigma=0.05)
        self.testt = theano.shared(testt)
        
        self._theano_cost = theano.function(inputs=[],
                                           outputs=self._costFunction()
                                           )
        
        self._theano_testset_error = theano.function(inputs=[],
                                           outputs=self._testCostFunction())
        
        gradient = [T.grad(self._costFunction(), self.params)]
        costGradient = [self._costFunction()] + gradient
        
        self._theano_costGradient = theano.function(inputs=[],
                                                   outputs=costGradient
                                                   )

        hess = theano.gradient.hessian(self._costFunction(), self.params)
        outputs = [self._costFunction()] + gradient + [hess]
        self._theano_costGradientHessian = theano.function(inputs=[],
                                                          outputs=outputs
                                                          )
        
    def cost(self, params):
        """ inputs: parameter values (np array)
            returns: value of cost function
        """
        self.params.set_value(params)

        return self._theano_cost()
    
    def testset_error(self, params):
        
        self.params.set_value(params)
        return self._theano_testset_error()
    
    def costGradient(self, params):
        """ inputs: parameter values (np array)
            returns: value of cost function and gradient
        """
        self.params.set_value(params)
        return self._theano_costGradient()
    
    def costGradientHessian(self, params):
        """ inputs: parameter values (np array)
            returns: value of cost function, gradient and hessian
        """
        self.params.set_value(params)
        return self._theano_costGradientHessian()
            
    def predict(self, xval):
        """ inputs: xvals - array-like 
            Returns: cost and gradient (np array)
        """
       
        self.x_to_predict.set_value(xval)    
        
        return self.Y(self.x_to_predict).eval() 
    
    def Y(self, X):
        """ 
            This function defines the model
            inputs : X -- theano shared variable
            returns : theano shared variable
        """
        raise NotImplementedError

    def drawSamples(self, params, X, sigma=0.1):
        
        self.params.set_value(params)
        return self.predict(X) + sigma * np.random.normal(size=X.shape) 
#         pass
        
    def _costFunction(self):
        """ 
            This function defines the model cost function, default is a sum of squared errors
            returns : theano shared variable
        """
        Cost = T.sum((self.Y(self.X) - self.t)**2)
        
        return Cost

    
    def _testCostFunction(self):
        
        Cost = T.sum((self.Y(self.testX) - self.testt)**2)
        
        return Cost    
    
class RegressionPotential(BasePotential):
    def __init__(self, model):
        
        self.model = model
         
    def getEnergy(self, coords):
        return self.model.cost(coords)
 
    def getEnergyGradient(self, coords):
        ret = self.model.costGradient(coords)
        return ret[0].item(), ret[1]
#         return self.model.costGradient(coords)
     
    def getEnergyGradientHessian(self, coords):
        return self.model.costGradientHessian(coords)
     

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

