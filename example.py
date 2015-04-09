import numpy as np
import theano
import theano.tensor as T
from potentials import BaseTheanoModel, RegressionSystem

import matplotlib.pyplot as plt


class TestModel(BaseTheanoModel):
    
    def __init__(self, *args, **kwargs):
        super(TestModel, self).__init__(*args, **kwargs)
        
    def Y(self, X):
        
        return T.exp(-self.params[0]*X) * T.sin(self.params[1]*X+self.params[2]) \
            * T.sin(self.params[3]*X + self.params[4])
            
def run_basinhopping(system):
    
    database = system.create_database()
    x0 = np.random.random(system.model.params.eval().size)
    
    from pele.takestep import RandomCluster
    step = RandomCluster(volume=5.0)
    bh = system.get_basinhopping(database=database, takestep=step,coords=x0,temperature = 10.0)
    #bh.stepsize = 20.
    bh.run(100)
    print "found", len(database.minima()), "minima"
    min0 = database.minima()[0]
    print "lowest minimum found has energy", min0.energy
    m0 = database.minima()[0]
    mindist = system.get_mindist()
    for m in database.minima():
        dist = mindist(m0.coords, m.coords.copy())[0]
        print "   ", m.energy, dist, m.coords
    
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
    
    graph = database2graph(database)
    dg = DisconnectivityGraph(graph, nlevels=10, center_gmin=True)
    dg.calculate()
    dg.plot()
    plt.show()

def visualize_solutions(system, db, xvals, tvals):    
    for m in db.minima():
        system.model.params.set_value(m.coords)
        curve = system.model.predict(np.arange(0.,3.0*np.pi,0.01))
        plt.plot(np.arange(0.,3.0*np.pi,0.01), curve, '-')
        
    plt.plot(xvals,tvals,'x')
    plt.show()
    
def test():
    # generate some test data
    xvals = 3.0*np.pi*np.random.random((100,1))
    xvals = np.atleast_2d(xvals)
    real_params=np.array([0.1,1.0,0.0,0.0,0.5*np.pi])
    tvals = np.exp(-real_params[0]*xvals) * np.sin(real_params[1]*xvals+real_params[2]) \
                    * np.sin(real_params[3]*xvals + real_params[4]) 
    # add some noise to the model output data
    tvals = tvals + np.random.normal(scale=0.1,size=(100,1))
        
    # this is the model we use to fit the data
    model = TestModel(input_data = xvals, 
                            target_data = tvals, 
                            starting_params = np.atleast_1d(np.random.random(real_params.shape))
                            )

    system = RegressionSystem(model)
    
    system, db = run_basinhopping(system)
    visualize_solutions(system, db, xvals, tvals)
    run_double_ended_connect(system, db)
    make_disconnectivity_graph(db)
    
    exit()
    

if __name__=="__main__":
    test()
                    