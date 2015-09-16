import numpy as np
import theano
import theano.tensor as T
from pele.potentials import BasePotential
from pele.systems import BaseSystem
         
class BaseTheanoModel(object):
    """ This is the Base Model class for model parameter estimation,
     used in conjunction with basin-hopping for model parameter estimation.
     Most of the objects in this class are theano functions. 

    The values stored in self.params are updated when calling cost(params) and are not meant to store the values of the underlying model etc.
    
    Information on naming conventions for this class:
        self._theano_<foo> is a theano function which returns the output of shared variable <foo>.
    """
    
    def __init__(self, input_data, target_data, testX, testt, starting_params, sigma=0.02):
        
        
        self.x_to_predict = theano.shared(np.random.random(input_data.shape))
        self.params = theano.shared(starting_params)

        self.X = theano.shared(input_data)
        
        """If no target data is provided, samples are drawn from the model with noise""" 
        if target_data == None:
            target_data = self.drawSamples(starting_params, input_data, sigma=sigma)
        self.t = theano.shared(target_data)
    
        self.testX = theano.shared(testX)
        """If no testset target data is provided, samples are drawn from the model with noise""" 
        if testt == None:
            testt = self.drawSamples(starting_params, testX, sigma=sigma)
        self.testt = theano.shared(testt)
        
        self._theano_cost = theano.function(inputs=[],
                                           outputs=self._cost()
                                           )
        
        self._theano_testSetCost = theano.function(inputs=[],
                                           outputs=self._testSetCost())
        
        gradient = [T.grad(self._cost(), self.params)]
        costGradient = [self._cost()] + gradient
        
        self._theano_costGradient = theano.function(inputs=[],
                                                   outputs=costGradient
                                                   )

        hess = theano.gradient.hessian(self._cost(), self.params)
        outputs = [self._cost()] + gradient + [hess]
        self._theano_costGradientHessian = theano.function(inputs=[],
                                                          outputs=outputs
                                                          )
        
    def _cost(self):
        """ 
            This function defines the model cost function, in this case a sum of squared errors
            returns : theano shared variable
        """
        Cost = T.sum((self.Y(self.X) - self.t)**2)
        
        return Cost

    
    def _testSetCost(self):
        """ 
            This function calculates the error function applied to the test set.
            returns : theano shared variable
        """        
        Cost = T.sum((self.Y(self.testX) - self.testt)**2)
        
        return Cost    
            
    def cost(self, params):
        """ inputs: parameter values (np array)
            returns: value of cost function
        """
        self.params.set_value(params)

        return self._theano_cost()
    
    def testset_error(self, params):
        
        self.params.set_value(params)
        return self._theano_testSetCost()
    
    def costGradient(self, params):
        """ inputs: parameter values (np array)
            returns: value of cost function and gradient
        """
        self.params.set_value(params)
        c, g = self._theano_costGradient()
        return c, g
    
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

    
class MLPotential(BasePotential):
    """ This class interfaces the regression model class: 
        The potential energy = cost function """
    def __init__(self, model):
        
        self.model = model
         
    def getEnergy(self, coords):
        return self.model.cost(coords)
 
    def getEnergyGradient(self, coords):

        return self.model.costGradient(coords)
     
    def getEnergyGradientHessian(self, coords):
        return self.model.costGradientHessian(coords)
     

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
        
class RegressionSystem(MLSystem):
    def __init__(self, model, db_accuracy=0.01, minimizer_tolerance=1.0e-06):
        super(RegressionSystem, self).__init__(model)
        self.params.database.accuracy = db_accuracy
        self.minimizer_tolerance = minimizer_tolerance
#         self.params.double_ended_connect.local_connect_params.tsSearchParams.hessian_diagonalization = True)
    
    def get_minimizer(self, nsteps=1e6, M=4, iprint=0, maxstep=1.0, **kwargs):
        from pele.optimize import lbfgs_cpp as quench
        return lambda coords: quench(coords, self.get_potential(), tol=self.minimizer_tolerance, 
                                     nsteps=nsteps, M=M, iprint=iprint, 
                                     maxstep=maxstep, 
                                     **kwargs)
    
    def get_compare_exact(self, **kwargs):
        # no permutations of parameters
        mindist = self.get_mindist()
        return lambda x1, x2: mindist(x1, x2)

class TestModel(BaseTheanoModel):
    """ An example regression model: exponential decay * product of sinusoids
    """

    def Y(self, X):
        
        return T.exp(-self.params[0]*X) * T.sin(self.params[1]*X+self.params[2]) \
            * T.sin(self.params[3]*X + self.params[4])

class SinModel(BaseTheanoModel):
    """ A simple non-linear regression model: sinusoid.
    """
         
    def Y(self, X):
        
        return T.sin(self.params[0]*X + self.params[1])
              