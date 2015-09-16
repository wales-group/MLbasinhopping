import numpy as np
import theano
import theano.tensor as T
from MLbasinhopping.regressionModels import BaseTheanoModel, RegressionSystem, TestModel, SinModel

import matplotlib.pyplot as plt
    
def run_basinhopping(system, nsteps, database):
    
    x0 = np.random.random(system.model.params.eval().size)
    
    from pele.takestep import RandomCluster
    step = RandomCluster(volume=5.0)
    bh = system.get_basinhopping(database=database, takestep=step,coords=x0,temperature = 10000.0)
    #bh.stepsize = 20.
    bh.run(nsteps)
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
        
def make_disconnectivity_graph(system, database):
    from pele.utils.disconnectivity_graph import DisconnectivityGraph, database2graph
    
    graph = database2graph(database)
    dg = DisconnectivityGraph(graph, nlevels=10, center_gmin=True)
    dg.calculate()
    
    # color DG points by test-set error
    minimum_to_testerror = lambda m: system.model.testset_error(m.coords)
    dg.color_by_value(minimum_to_testerror)
    dg.plot(linewidth=1.5)
#     plt.colorbar()
    plt.show()

def visualize_solutions(system, db, xvals, tvals, real_params):    
    for m in db.minima():
        system.model.params.set_value(m.coords)
        xs = np.arange(0.,3.0*np.pi,0.01)
#         xs = np.atleast_2d(xs)
        curve = system.model.predict(xs)
        plt.plot(xs, curve, '-')
        
    
    plt.plot(xvals,tvals,'x')
    system.model.params.set_value(real_params)
    plt.plot(np.arange(0.,3*np.pi,0.01), system.model.predict(np.arange(0.,3*np.pi,0.01)), '-')
    plt.show()
    
def test():
    
    np.random.seed(12345)
    
    # generate some test data
    xvals = 3.0*np.pi*np.random.random(100)
    testx = 3.0*np.pi*np.random.random(100)
    
    # parameters values which specify the model 
    real_params = np.random.random(2)
    real_params = real_params * np.pi
      
    # this is the model we use to fit the data
    model = SinModel(input_data = xvals, 
                            target_data = None, # generate samples automatically
                            starting_params=real_params,
                            testX = testx,
                            testt = None # generate samples automatically
                            )

    tvals = model.t.get_value()
    testt = model.testt.get_value()

    system = RegressionSystem(model)
    
#     db = system.create_database("SinModel"+str(real_params[0])+"_"+str(real_params[1])+".sqlite")
    db = system.create_database()
    
    # run basin-hopping on this landscape
    nsteps = 10
    system, db = run_basinhopping(system, nsteps, db)
    
    # draw various best-fits from BH run
    visualize_solutions(system, db, xvals, tvals, real_params)
       
    # connect minima
#     run_double_ended_connect(system, db)
        
    # connect minima
#     make_disconnectivity_graph(system, db)
        

if __name__=="__main__":
    test()
