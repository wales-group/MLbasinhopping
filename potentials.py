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

def run_basinhopping(system, xvals, tvals):
    import matplotlib.pyplot as plt
    
    database = system.create_database()
    x0 = np.random.random(system.model.params.eval().size)
    
    from pele.takestep import RandomCluster
    step = RandomCluster(volume=5.0)
    bh = system.get_basinhopping(database=database, takestep=step,coords=x0,temperature = 10.0)
    #bh.stepsize = 20.
    bh.run(500)
    print "found", len(database.minima()), "minima"
    min0 = database.minima()[0]
    print "lowest minimum found has energy", min0.energy
    m0 = database.minima()[0]
    mindist = system.get_mindist()
    for m in database.minima():
        dist = mindist(m0.coords, m.coords.copy())[0]
        print "   ", m.energy, dist, m.coords


#     for m in database.minima():
#         system.model.params.set_value(m.coords)
#         curve = system.model.predict(np.arange(0.,3.0*np.pi,0.01))
#         plt.plot(np.arange(0.,3.0*np.pi,0.01), curve, '-')
    
#     plt.plot(xvals,tvals,'x')
#     plt.show()
    
    return system, database

def run_double_ended_connect(system, database):
    # connect the all minima to the lowest minimum
    from pele.landscape import ConnectManager
    manager = ConnectManager(database, strategy="gmin")
    for i in xrange(database.number_of_minima()-1):
        min1, min2 = manager.get_connect_job()
        connect = system.get_double_ended_connect(min1, min2, database)
        connect.connect()
        
def make_disconnectivity_graph(database):
    from pele.utils.disconnectivity_graph import DisconnectivityGraph, database2graph
    import matplotlib.pyplot as plt
    
    graph = database2graph(database)
    dg = DisconnectivityGraph(graph, nlevels=10, center_gmin=True)
    dg.calculate()
    dg.plot()
    plt.show()
    
def main():
#     import matplotlib.pyplot as plt
    
    theano.config.mode = 'FAST_RUN'
    xvals = 3.0*np.pi*np.random.random((100,1))
    xvals = np.atleast_2d(xvals)
#     real_params = np.array([1.0, 0.1, 2.0])
    real_params=np.array([0.1,1.0,0.0,0.0,0.5*np.pi])
    tvals = np.exp(-real_params[0]*xvals) * np.sin(real_params[1]*xvals+real_params[2]) \
                    * np.sin(real_params[3]*xvals + real_params[4]) 
                   
    tvals = tvals + np.random.normal(scale=0.1,size=(100,1))
#     real_params[0] * np.exp(real_params[1] * xvals) * np.sin(real_params[2] * xvals) \
#                 + 0.1 * np.random.random((100,1))

    starting_params = np.atleast_1d(np.random.random(real_params.shape))
    
    model = TestModel(input_data = xvals, 
                            target_data = tvals, 
                            starting_params = starting_params
                            )

    system = RegressionSystem(model)
    pot = system.get_potential()
    
    system, db = run_basinhopping(system, xvals, tvals)
    run_double_ended_connect(system, db)
    make_disconnectivity_graph(db)
    exit()
    
    coords = 5.0*np.random.random(starting_params.shape)

    quench = system.get_minimizer()
    for _ in xrange(2):
        coords = np.random.random(starting_params.shape)
#         print model.theano_costGradient(coords)[0]
#         print model.theano_costGradient(coords)[1]
#         ret = pot.getEnergyGradient(coords)
#         print ret
#         exit()
        ret = quench(coords)
        model.params.set_value(ret.coords)
        curve = model.predict(np.arange(0.,xvals.max(),0.01))
        plt.plot(np.arange(0.,xvals.max(),0.01), curve, '-')
    
    plt.plot(xvals,tvals,'x')
    plt.show()
if __name__=="__main__":
    main()
        