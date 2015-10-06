import numpy as np
import matplotlib.pyplot as plt

def connect_to_gmin(system, db):
    
    print "now connecting all the minima to the  minimum"

    minima = db.minima()

    m1 = minima[0]

        
    for m2 in minima[1:]:

        connect = system.get_double_ended_connect(m1, m2, db, fresh_connect=True)
        connect.connect()
        
def database_stats(system, db):
    
    print "Minimum Energy: "
    for m in db.minima():
        print m.energy
        
    print "Nminima = ", len(db.minima())
    print "Nts = ", len(db.transition_states())
    
    make_disconnectivity_graph(system, db)
    
def run_basinhopping(system, nsteps, database):
    
    x0 = np.random.random(system.model.nparams)
    
    from pele.takestep import RandomCluster
    step = RandomCluster(volume=55.0)
    bh = system.get_basinhopping(database=database, 
                                 takestep=step,
                                 coords=x0,
                                 temperature = 10000.0)
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

def run_double_ended_connect(system, database, strategy='gmin'):
    # connect the all minima to the lowest minimum
    from pele.landscape import ConnectManager
    manager = ConnectManager(database, strategy=strategy)
    for i in xrange(database.number_of_minima()-1):
        min1, min2 = manager.get_connect_job()
        connect = system.get_double_ended_connect(min1, min2, database)
        connect.connect()
        
def make_disconnectivity_graph(system, database):
    from pele.utils.disconnectivity_graph import DisconnectivityGraph, database2graph
    
    graph = database2graph(database)
    dg = DisconnectivityGraph(graph, nlevels=20, center_gmin=False, include_gmin=False)
    dg.calculate()
    
    # color DG points by test-set error
#     minimum_to_testerror = lambda m: system.model.testset_error(m.coords)
#     dg.color_by_value(minimum_to_testerror)
    dg.plot(linewidth=1.5)
#     plt.colorbar()
    plt.show()
