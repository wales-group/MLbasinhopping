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
        
#         cost = self.model.cost(params)
        return self.model.evaluated_cost(params)
#         return cost.eval()

    def getEnergyGradient(self, params):

        return self.model.evaluated_costGradient(params)
#         print self.model.evaluated_costGradient(params)
#         exit()
#         e = self.model.cost(params)
#         g = self.model.costGradient(params)
# 
#         return e.eval(), g.eval()
    
    def getEnergyGradientHessian(self, params):
        
        e, g, h = self.model.costGradientHessian(params)
        return e.eval(), g.eval(), h.eval()
    
                   
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
        raise NotImplementedError
    
        return T.exp(-self.params[0]*X) * T.sin(self.params[1]*X+self.params[2]) \
                    * T.sin(self.params[3]*X + self.params[4])
        
    def costFunction(self):
#         print self.params.shape
#         self.params.set_value(params)
        Cost = T.sum((self.Y(self.X) - self.t)**2)
        
        return Cost
    
#     def costGradient(self):
#         
# #         self.params.set_value()
#         
#         g = T.grad(self.cost(), self.params)
#         return g
    
#     def costGradientHessian(self, params):
#         
#         c, g = self.costGradient(params)
#         h = theano.gradient.hessian(c, self.params)
#         
#         return c, g, h


#     def evaluated_cost(self, params):
#         
# #         func = theano.function([],self.cost(params))
#         self.params.set_value(params)
#         return self.theano_cost()
# 
#     def evaluated_costGradient(self, params):
# #         c = self.evaluated_cost(params)
# #         g = theano.function([], self.costGradient(params))
#         self.params.set_value(params)
# #         print c, g
#         return self.theano_costGradient()
    
#         costfunction = 

class TestModel(BaseTheanoModel):
    
    def __init__(self, *args, **kwargs):
        super(TestModel, self).__init__(*args, **kwargs)
        
    def Y(self, X):
        
        return T.exp(-self.params[0]*X) * T.sin(self.params[1]*X+self.params[2]) \
            * T.sin(self.params[3]*X + self.params[4])
            
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
        self.params.double_ended_connect.local_connect_params.tsSearchParams.hessian_diagonalization = True

    def get_potential(self):
        return RegressionPotential(self.model)
    
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
    
    theano.config.mode = 'FAST_RUN'
    xvals = np.random.random((100,1))
    xvals = np.atleast_2d(xvals)
#     real_params = np.array([1.0, 0.1, 2.0])
    real_params=np.array([0.1,1.0,0.0,0.0,0.5*np.pi])
    tvals = np.exp(-real_params[0]*xvals) * np.sin(real_params[1]*xvals+real_params[2]) \
                    * np.sin(real_params[3]*xvals + real_params[4])
#     real_params[0] * np.exp(real_params[1] * xvals) * np.sin(real_params[2] * xvals) \
#                 + 0.1 * np.random.random((100,1))

    starting_params = np.atleast_1d(np.random.random(real_params.shape))
    
    model = TestModel(input_data = xvals, 
                            target_data = tvals, 
                            starting_params = starting_params
                            )

    system = RegressionSystem(model)
    pot = system.get_potential()
    
    coords = np.random.random(starting_params.shape)

    quench = system.get_minimizer()
    for _ in xrange(100):
#         print model.theano_costGradient(coords)[0]
#         print model.theano_costGradient(coords)[1]
#         ret = pot.getEnergyGradient(coords)
#         print ret
#         exit()
        ret = quench(coords)
        print ret.coords, ret.energy

if __name__=="__main__":
    main()
        