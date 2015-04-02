import numpy as np
from pele.potentials import BasePotential

class BaseModel(object):
    
    def __init__(self,params0=None,points=None,x=None,npoints=100,sigma=0.3):        

        self.params0 = params0
        self.sigma = sigma
        self.points = points
        self.npoints = npoints     
        
        """ optional x-values to generate training data """
        self.xvals = x
        
        if self.points == None:
            assert self.params0 != None
            self.nparams = len(self.params0)
            self.points = self.generate_points(x=self.xvals)
        else:
            self.npoints = len(self.points)
            
    def generate_points(self,x=None):
        
        if x==None:
            x = np.random.random(self.npoints)

        y = self.model_batch(x,self.params0) + np.random.normal(scale=self.sigma,size=self.npoints)
        
        return zip(x,y)
    
class WaveModel(BaseModel):

    def __init__(self,*args,**kwargs):

        super(WaveModel,self).__init__(*args,**kwargs)
        self.xvals = 3*np.pi*np.random.random(self.npoints)
        self.points = self.generate_points(x=self.xvals)
        
    def model(self,x,params):
        return np.exp(-params[0]*x) * np.sin(params[1]*x+params[2]) * np.sin(params[3]*x+params[4])
    
    def model_batch(self, x, params):
        """evaluate multiple data points at once
        
        this can be much faster than doing them individually
        """

        return np.exp(-params[0]*x) * np.sin(params[1]*x + params[2]) * np.sin(params[3]*x + params[4])
        #return np.exp(-params[0]*x) * np.sin(params[1]*x + params[2])

    def model_gradient_batch(self, x, params):
        """return a matrix of gradients at each point
        
        Returns
        -------
        grad : array
            grad[i,j] is the gradient w.r.t. param[i] at point[j]
        """
        t1 = np.exp(-params[0]*x)
        t2 = np.sin(params[1]*x+params[2])
        t2der = np.cos(params[1]*x+params[2])
        t3 = np.sin(params[3]*x+params[4])
        t3der = np.cos(params[3]*x+params[4])
        
        grad = np.zeros([params.size, x.size])
        grad[0,:] = -x * t1 * t2 * t3
        
        grad[1,:] = x * t1 * t2der * t3
        grad[2,:] = t1 * t2der * t3
        grad[3,:] = x * t1 * t2 * t3der
        grad[4,:] = t1 * t2 * t3der
        
        return grad
      
class MLPotential(BasePotential):
    def __init__(self, inputs, outputs, model):
        
        self.inputs = inputs
        self.outputs = outputs
        self.model = model
        
    def getEnergy(self, coords):
        return self.model.cost(self.inputs, self.outputs, coords)

    def getEnergyGradient(self, coords):
        return self.model.costGradient(self.inputs, self.outputs, coords)
    
    def getEnergyGradientHessian(self, coords):
        return self.model.costGradientHessian(self.inputs, self.outputs, coords)
    
    
class RegressionPotential(MLPotential):
    def __init__(self):
        super(RegressionPotential).__init__()
    
    def getEnergy(self):
        