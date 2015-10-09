import os
import sys
import time
import gzip
import cPickle

import numpy
import numpy as np

import theano
import theano.tensor as T

from pele.potentials import BasePotential
from pele.systems import BaseSystem

from MLbasinhopping.base import BaseModel, MLSystem
from MLbasinhopping.NN.mlp import MLP

def get_data():
    
    __location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
    
    dataset = "mnist.pkl.gz"
    with gzip.open(os.path.join(__location__, dataset)) as f:
        train_set, valid_set, test_set = cPickle.load(f)
    
    # training data
    x = train_set[0]
    ndata, n_features = x.shape
#     
#     print "Data set size:\n", ndata, " data points\n", n_features, " features\n";
#     exit()
    # labels
    t = train_set[1]
    assert ndata == t.size
    
    # test data
    test_x = test_set[0]
    test_t = test_set[1].astype('int32')

    return x, t, test_x, test_t


class NNModel(BaseModel):
    def __init__(self, ndata=1000, n_hidden=10, L1_reg=0.00, L2_reg=0.0001, bias_reg=0.00):
        
        train_x, train_t, test_x, test_t = get_data()
        train_x = train_x[:ndata,:]
        train_t = train_t[:ndata]
        train_t = np.asarray(train_t, dtype="int32")
    
        self.train_t = train_t
        self.test_t = test_t
        
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.bias_reg = bias_reg
        
        print "range of target values: ", set(train_t)
        # allocate symbolic variables for the data.  
        # Make it shared so it cab be passed only once 
        x = theano.shared(value=train_x, name='x')  # the data is presented as rasterized images
        t = theano.shared(value=train_t, name='t')  # the labels are presented as 1D vector of
                            # [int] labels
    
        
        rng = numpy.random.RandomState(1234)
        
        # construct the MLP class
        classifier = MLP(
            rng=rng,
            input=x,
            n_in=28 * 28,
            n_hidden=n_hidden,
            n_out=10
        )
        self.classifier = classifier
    
        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
        cost = (
            classifier.negative_log_likelihood(t)
            + L1_reg * classifier.L1
            + L2_reg * classifier.L2_sqr
            + bias_reg* classifier.bias_sqr
        )

        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(cost, param) for param in classifier.params]
    
        outputs = [cost] + gparams
        self.theano_cost_gradient = theano.function(
               inputs=(),
               outputs=outputs
               )
        
        # compute the errors applied to test set
        self.theano_testset_errors = theano.function(
               inputs=(),
#                outputs=self.classifier.errors(t),
                outputs=self.classifier.errors_vector(t),
               givens={
                       x: test_x,
                       t: test_t
                       }                                          
               )
        
        # compute the softmax output from test set
        self.theano_softmax_errors = theano.function(
               inputs=(),
#                outputs=self.classifier.errors(t),
                outputs=self.classifier.logRegressionLayer.p_y_given_x,
               givens={
                       x: test_x
                       }                                          
               )        
    #    res = get_gradient(train_x, train_t)
    #    print "result"
    #    print res
    #    print ""
    
        self.nparams = sum([p.get_value().size for p in classifier.params])
        self.param_sizes = [p.get_value().size for p in classifier.params]
        self.param_shapes = [p.get_value().shape for p in classifier.params]             
    
    def _cost_gradient(self):
        ret = self.theano_cost_gradient()
        cost = float(ret[0])
        gradients = ret[1:]
                
        grad = np.zeros(self.nparams)
        
        i = 0
        for g in gradients:
            npar = g.size
            grad[i:i+npar] = g.ravel()
            i += npar
        return cost, grad
        
    def get_params(self):
        params = np.zeros(self.nparams)
        i = 0
        for p in self.classifier.params:
            p = p.get_value()
            npar = p.size
            params[i:i+npar] = p.ravel()
            i += npar
        return params
    
    def set_params(self, params_vec):
        assert params_vec.size == self.nparams
        i = 0
        for count, p in enumerate(self.classifier.params):
            npar = self.param_sizes[count]
            p.set_value(params_vec[i:i+npar].reshape(self.param_shapes[count]))
            i += npar
    
    def cost(self, params):
        return self.costGradient(params)[0]
        
    def getValidationError(self, params):
        # returns the fraction of misassignments for test set
        self.set_params(params)
        return self.theano_testset_errors()
        
    def costGradient(self, params):
        # the params are stored as shared variables so we have to update
        # them in memory before computing the cost.
        self.set_params(params)
        return self._cost_gradient()

def myquench(coords, pot, tol, **kwargs):
    """ This quench checks the rmsgrad condition is satisfied, and keeps quenching until this is the case
    """
    from pele.optimize import lbfgs_cpp as quench

    rmsgrad = 10000.0 

    while rmsgrad > tol:
        ret = quench(coords, pot, tol=tol, **kwargs)
        e,g = pot.getEnergyGradient(ret.coords)
        rmsgrad = np.linalg.norm(g)  
        coords = ret.coords
        
    return ret
  
class NNSystem(MLSystem):
    def __init__(self, model, minimizer_tolerance=1.0e-5, *args, **kwargs):
        super(NNSystem, self).__init__(model)
        self.minimizer_tolerance = minimizer_tolerance

    def get_minimizer(self, nsteps=1e5, M=4, iprint=0, maxstep=1.0, **kwargs):
        from pele.optimize import lbfgs_cpp as quench
        return lambda coords: quench(coords, self.get_potential(), tol=self.minimizer_tolerance, 
                                     nsteps=nsteps, M=M, iprint=iprint, 
                                     maxstep=maxstep, 
                                     **kwargs)
    
def test():
    
    model = NNModel()
    system = NNSystem(model)
    t = system.get_potential()
    
    nparams = model.nparams        
    print "get_energy_gradient"
    newparams = np.random.uniform(-.05, .05, nparams)
    e, g = t.getEnergyGradient(newparams)
    print "cost", e
    
    print "\n\nagain\nget_energy_gradient"
    newparams = np.random.uniform(-.05, .05, nparams)
    e, g = t.getEnergyGradient(newparams)
    print "cost", e
    
    params = model.get_params()
    print params
    dx = np.max(np.abs(params - newparams))
    print dx
    assert dx < 1e-8
    
    # do minimization
    from pele.optimize import lbfgs_py
    res = lbfgs_py(newparams, t, iprint=10, tol=1e-4)
    print res

if __name__ == "__main__":
    test()
