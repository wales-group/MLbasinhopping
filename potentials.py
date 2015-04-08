import numpy as np
import theano
import theano.tensor as T
from pele.potentials import BasePotential
from pele.systems import BaseSystem

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

class ErrorFunction(BasePotential):
    """a quadratic error function for fitting
    
    V(xi,yi|alpha) = 0.5 * (yi-(f(xi|alpha))**2
    where 
    """
    def __init__(self, model):
        """ instance of model to fit"""
        self.model = model

    def getEnergy(self, params):
        
        cost = self.model.cost(params)
        return cost.eval()

    def getEnergyGradient(self, params):

        e, g = self.model.costGradient(params)
        return e.eval(), g.eval()
    
    def getEnergyGradientHessian(self, params):
        
        e, g, h = self.model.costGradientHessian(params)
        return e.eval(), g.eval(), h.eval()
    
class TheanoTestModel:
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

    def predict2(self, xval):
        
        self.x_to_predict.set_value(xval)    
        
        return self.Y(self.x_to_predict).eval()
        
    def predict(self, xval):

        self.x_to_predict.set_value(xval)
#         xval = T.scalar("x")        
        model = self.params * self.x_to_predict
        model.eval()
        ret = theano.function([], model)
        
        return ret()
    
    def Y(self, X):
        """ 
        This defines the model
        inputs : X -- theano shared variable
        returns : Theano shared variable
        """
        return T.exp(-self.params[0]*X) * T.sin(self.params[1]*X+self.params[2]) \
                    * T.sin(self.params[3]*X + self.params[4])
#         return self.params[0] * T.exp(self.params[1] * X) * T.sin(self.params[2] * X)
#         return self.params * X
        
    def cost(self, params):
#         print self.params.shape
        self.params.set_value(params)
        Cost = T.sum((self.Y(self.X) - self.t)**2)
        
        return Cost
    
    def costGradient(self, params):
        
        self.params.set_value(params)
        
        c = self.cost(params)
        g = T.grad(c, self.params)
        return c, g
    
    def costGradientHessian(self, params):
        
        c, g = self.costGradient(params)
        h = theano.gradient.hessian(c, self.params)
        
        return c, g, h
#         costfunction = 
# class MLPotential(BasePotential):
#     def __init__(self, inputs, outputs, model):
#         
#         self.inputs = inputs
#         self.outputs = outputs
#         self.model = model
#         
#     def getEnergy(self, coords):
#         return self.model.cost(self.inputs, self.outputs, coords)
# 
#     def getEnergyGradient(self, coords):
#         return self.model.costGradient(self.inputs, self.outputs, coords)
#     
#     def getEnergyGradientHessian(self, coords):
#         return self.model.costGradientHessian(self.inputs, self.outputs, coords)
#     
#     
# class RegressionPotential(MLPotential):
#     def __init__(self):
#         super(RegressionPotential).__init__()
#     
#     def getEnergy(self):
#         pass
#     

class RegressionSystem(BaseSystem):
    def __init__(self, model):
        super(RegressionSystem, self).__init__()
        self.model = model
        self.params.database.accuracy =0.01
        self.params.double_ended_connect.local_connect_params.tsSearchParams.hessian_diagonalization = True

    def get_potential(self):
        return ErrorFunction(self.model)
    
    def get_mindist(self):
        # no permutations of parameters
        
        #return mindist_with_discrete_phase_symmetry
        #permlist = []
        return lambda x1, x2: np.linalg.norm(x1-x2)
#         return MinPermDistWaveModel( niter=10)

    def get_orthogonalize_to_zero_eigenvectors(self):
        return None
#         return my_orthog_opt
        #return orthogopt
        #return orthogopt_translation_only
    
    def get_minimizer(self, tol=1.0e-2, nsteps=100000, M=4, iprint=1, maxstep=1.0, **kwargs):
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
        return lambda x1, x2: (mindist(x1, x2)[0] < 1e-3, x1, x2)
    
def main():
    
    xvals = np.random.random((100,1))
    xvals = np.atleast_2d(xvals)
#     real_params = np.array([1.0, 0.1, 2.0])
    real_params=np.array([0.1,1.0,0.0,0.0,0.5*np.pi])
    tvals = np.exp(-real_params[0]*xvals) * np.sin(real_params[1]*xvals+real_params[2]) \
                    * np.sin(real_params[3]*xvals + real_params[4])
#     real_params[0] * np.exp(real_params[1] * xvals) * np.sin(real_params[2] * xvals) \
#                 + 0.1 * np.random.random((100,1))

    starting_params = np.atleast_1d(np.random.random(real_params.shape))
    
    model = TheanoTestModel(input_data = xvals, 
                            target_data = tvals, 
                            starting_params = starting_params
                            )

    system = RegressionSystem(model)
    
    for _ in xrange(10):
        quench = system.get_minimizer()
        ret = quench(np.random.random(starting_params.shape))
        print ret.coords, ret.energy
    exit()
    import matplotlib.pyplot as plt
#     plt.plot(xvals, model.Y(xvals).eval(), 'x')
#     plt.show()
#     print model.Y(xvals).eval()[0], xvals[0], starting_params[0]
#     exit()
    sampled_points = model.Y(xvals).eval() + 0.1 * np.random.random(xvals.shape)
#     plt.plot(xvals,sampled_points,'x')
#     plt.show()
#     print model.cost(starting_params).eval()
    c,g,h = model.costGradientHessian(starting_params)
    print c.eval()
    print g.eval()
    print h.eval()
    
if __name__=="__main__":
    main()
        