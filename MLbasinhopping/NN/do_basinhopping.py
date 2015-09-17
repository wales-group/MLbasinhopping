import numpy as np
from pele_mpl import NNSystem

from pele.gui import run_gui

def connect_minima(system, db):
    print "now connecting all the minima to the  minimum"

    minima = db.minima()

    m1 = minima[0]

    for m2 in minima[1:]:
        
        connect = system.get_double_ended_connect(m1, m2, db, fresh_connect=True)

        connect.connect()    

def connect_particular_minima(system, db):
    
    print "now connecting all the minima to the  minimum"
#     indices = [3,5,6,11,12,13,16,20,31]
#     indices = [1,2]
    indices = [1 ,  2 ,  8 ,  4 ,  0 ,  14 ,  16 ,  18 ,  3 ,  17 ,  15 ,  13 ,  7 ,  12 ,  9 ,  6 ,  23 ,  10]
    minima = [db.minima()[m] for m in indices]

#     mindices = [m._id for m in db.minima()]
    m1 = minima[0]

#     exit()
    #ts = db.getTransitionStatesMinimum(m1)
#     for t in ts:
#         print mindices.index(t.minimum1._id), mindices.index(t.minimum2._id), t.minimum1._id, t.minimum2._id
        
    for m2 in minima[1:]:
        print "connecting minima with id's", 3, m2
#         ts = db.getTransitionState(m1, db.minima()[m2])
        ts = db.getTransitionState(m1, m2)
        print ts
#         if ts != None:
#             print "hi :", ts.energy
#         exit()
#         print db.getTransitionState(m1, db.minima()[m2]).energy
        connect = system.get_double_ended_connect(m1, m2, db, fresh_connect=True)
#         print connect.graph.areConnected(m1, m2), connect.graph.minima
#         print len(connect.graph.storage.transition_states()), len(db.transition_states())
        connect.connect()
    
def orderbyenergy(db, energy):
    
    minima = db.minima()
        
    return np.argmin([np.abs(m.energy-energy) for m in minima])

def make_only_dg(system, database, color=None):
    from pele.utils.disconnectivity_graph import DisconnectivityGraph, database2graph
    import matplotlib.pyplot as plt
#     m1 = database.minima()[6]
#     tss = database.getTransitionStatesMinimum(m1)
#     for ts in tss:
#         print orderbyenergy(database, ts.minimum1.energy), orderbyenergy(database, ts.minimum2.energy), ts.minimum1._id, ts.minimum2._id, ts.energy
#     exit()    
    for mi,m in enumerate(database.minima()):
        print mi, m.energy
    graph = database2graph(database)
#     dg = DisconnectivityGraph(graph, center_gmin=True, Emax=4.0)
#     dg = DisconnectivityGraph(graph, nlevels=2, center_gmin=True, subgraph_size=3)
    dg = DisconnectivityGraph(graph, nlevels=30, subgraph_size=5, minima=[database.minima()[i] for i in [1,2]])
    dg.calculate()
#     dg.color_by_value(lambda x : np.random.random(), colormap=lambda x : np.random.random(3))
    
    if color != None:
        dg.color_by_value(lambda m : color[0,orderbyenergy(database, m.energy)], colormap=cmap)
    dg.plot()
    plt.show()   
    
def run_basinhopping(system, db):
    
#     quench = system.get_minimizer()
#     db = system.create_database("/home/ab2111/machine_learning_landscapes/neural_net/theano_db/"+"reg"+str(system.potential.L2_reg)+".sqlite")
#     db = system.create_database()
    
    pot = system.get_potential()
    coords0 = np.random.random(pot.nparams)
    pot.set_params(coords0)
    
    stepsize=5.0
#     stepsize=2.0
    step = system.get_takestep(verbose=True, interval=10, stepsize=stepsize)
    temperature=10.0
#     system.get_minimizer().iprint = 10 
    bh = system.get_basinhopping(database=db, takestep=step,
                                 temperature = temperature,
                                 coords = coords0
                                 )
    
#     bh.quench = system.get_minimizer()
    bh.run(100)
    
#     from pele.thermodynamics._normalmodes import normalmodes
#     import matplotlib.pyplot as plt
    
    import matplotlib.pyplot as plt
    for m in db.minima():
#         print system.potential.L2_reg, m.energy, pot.getValidationError(m.coords)
        plt.plot(m.coords)
        plt.show()
#         e,g,h = pot.getEnergyGradientHessian(m.coords)
#         evals, evecs = normalmodes(h)
#         print evals
#         plt.plot(evals)
#     plt.show()
#     exit()


class HammingErrorAnalysis(object):
    """This class is mainly concerned with calculation of HEM = Hamming Error Matrix.
    """
    
    def __init__(self, system, db):
        self.db = db
        self.system = system
        
        self.run_analysis()
        
    def calculateErrors(self):

        Errors = []
        for m in self.db.minima():
                print m._id, m.energy
                errors = self.system.potential.getValidationError(m.coords)
    #             Errors.append([e for ei,e in enumerate(errors) if pot.test_t[ei]==digit])
                Errors.append(errors)
    
        #         print h, np.sum(errors), len(errors), h * 1./len(errors)
    #             print digit, len(errors)
        #         Errors.append(np.sum(errors))
        
        self.Errors = np.array(Errors)        
    
    def run_analysis(self):
        
        self.calculateErrors()
        self.calculateHEM()
    
    
    def calculateHEM(self):
        from scipy.spatial.distance import hamming
        
        N = self.Errors.shape[0]
        self.HEM = np.zeros((N,N))
    
        for i,ei in enumerate(self.Errors):
            for j,ej in enumerate(self.Errors):
                self.HEM[i,j] = hamming(ei,ej)
 
    def plot(self):
        import matplotlib.pyplot as plt
        plt.clf()
        plt.imshow(self.HEM)
        plt.colorbar()
        plt.show()   
    
    def stats(self):
        print "Errors averaged over minima: "
        p100=np.average(np.sum(self.Errors, axis=0)==self.Errors.shape[0])
        p0=np.average(np.sum(self.Errors, axis=0)==0.0)
        print "Fraction of images with 100% misreads ", p100
        print "Fraction of images with 0% misreads", p0
        print "Fraction of images with <50% misreads", np.average(np.average(self.Errors, axis=0)<0.5)
        print "Remaining images", 1.-p100-p0
        print "ave, std of misreads ", np.average(np.sum(self.Errors, axis=1)), np.std(np.sum(self.Errors, axis=1))
        
class HammingDG():
    def __init__(self, system, db, hem):
        self.system = system
        self.db = db
        """ hem is an instance of HammingErrorAnalysis"""
        self.hem = hem
        
    def plot(self):
        from pele.utils.disconnectivity_graph import DisconnectivityGraph, database2graph
        import matplotlib.pyplot as plt

        graph = database2graph(self.db)
#         dg = DisconnectivityGraph(graph, nlevels=30, subgraph_size=5, minima=[self.db.minima()[i] for i in [1,2]])
        dg = DisconnectivityGraph(graph, nlevels=30, subgraph_size=5)
        dg.calculate()
        
        dg.color_by_value(lambda m : self.hem.HEM[0, orderbyenergy(self.db, m.energy)], colormap=self.colormap)
        dg.plot()
        plt.show()       
     
    def colormap(self, x):   
        if x<0.5:
            return [0, x/0.5, 1.-(x/0.5)]
        elif x<=1.0:
            x = x-0.5
            return [x/0.5, 1.-(x/0.5), 0]
        else:
            print x, " is out of bounds"
            exit()
            
def get_minima_stats(system, db):
    pot = system.get_potential()
#     quench = system.get_minimizer(tol=1.0e-10, nsteps=100, iprint=100, M=400)
    import matplotlib.pyplot as plt
    from pele.thermodynamics._normalmodes import normalmodes
    
    for m in db.minima():
            print m._id, m.energy, np.average(pot.getValidationError(m.coords))
#             e,g,h = pot.getEnergyGradientHessian(m.coords)
#             print "got hessian, rms grad:", np.linalg.norm(g)
#             evals, evecs = normalmodes(h)
#             plt.plot(evals)
#             plt.show()
#     

def quench_trajectory(system, db):
    
    import matplotlib.pyplot as plt
    
    m3 = db.minima()[0]
    m4 = db.minima()[1]
    plt.plot(m3.coords)
    plt.plot(m4.coords)
    plt.show()
    exit()
#     coords0 = 10.*np.random.random(system.potential.nparams)
    pot = system.get_potential()

    tol=1.0e-10
    reduced_tol = 10.0
    rmsgrad = 10.00

    quench = system.get_minimizer(tol=tol, iprint=50)
    
    for m in [m3,m4]:
        ret = quench(m.coords)
        print "energy of qu: ", ret.energy
        plt.plot(ret.coords)
    plt.show()
    exit()
#     plt.plot(ret.coords)
#     plt.show()
#     print "Energy: ", ret.energy, np.average(pot.getValidationError(ret.coords))
#     e,g,h = pot.getEnergyGradientHessian(ret.coords)
#     print "got hessian, rms grad:", np.linalg.norm(g)
#     from pele.thermodynamics._normalmodes import normalmodes
#     evals, evecs = normalmodes(h)
#     
#     print evals
#     plt.hist(evals)
#     plt.show()    
#     exit()
    while rmsgrad > tol:
        ret = quench(coords0)
        rmsgrad = np.linalg.norm(pot.getEnergyGradient(ret.coords)[1])
        
        print "grad: ", rmsgrad, "energy: ", ret.energy, "validation: ", pot.getValidationError(ret.coords)

        coords0 = ret.coords
        reduced_tol = 0.1 * reduced_tol
    
def do_hessian_decomp(system, db):
    
    m = db.minima()[10]
    pot = system.get_potential()
    
    e,g,h = pot.getEnergyGradientHessian(m.coords)
    print "got hessian"
    from pele.thermodynamics._normalmodes import normalmodes
    evals, evecs = normalmodes(h)
    
    print evals
    import matplotlib.pyplot as plt
    plt.hist(evals)
    plt.show()
   
def weighted_prediction(system, db):
    
    np.set_printoptions(linewidth=400)

    bin = np.zeros((len(system.potential.test_t),10))
#     SMs = np.zeros((len(db.minima()), len(system.potential.test_t)))
    SMs = np.zeros((len(db.minima()[100:]), len(system.potential.test_t), 10))
    for mi, m in enumerate(db.minima()[100:]):
        system.potential.set_params(m.coords)
        softmax = system.potential.theano_softmax_errors()
        
        SMs[mi, :, :] = softmax
#         SMs[mi, :, :] = softmax
#         SMs.append(softmax)
    
    test_t = system.potential.test_t
    ave_pvecs = np.average(SMs, axis=0)
    print "averaged error rate : ", np.average([np.argmax(p)!=test_t[i] for i,p in enumerate(ave_pvecs)])
#     for i,p in enumerate(ave_pvecs[:100]):
#         if np.argmax(p) != test_t[i]:
#             print test_t[i]
#             print p
            
    print SMs.shape
        
         
def main():
    
    ndata = 50000
    n_hidden = 300
    p = 2
    L2_reg=np.power(1.0e1, -p)
#     L1_reg=np.power(1.0e1, -p)
#     L2_reg=0.0
    L1_reg=0.0
    bias_reg = 0.0
    system = NNSystem(ndata=ndata, n_hidden=n_hidden, L1_reg=L1_reg, L2_reg=L2_reg, bias_reg=bias_reg)
    db = system.create_database("/home/ab2111/machine_learning_landscapes/neural_net/theano_db/"
                                +"data"+str(ndata)
                                +"hidden"+str(n_hidden)
#                                 +"L1reg"+str(system.potential.L2_reg)+".sqlite")
                                +"reg"+str(system.potential.L2_reg)+".sqlite")
#     db = system.create_database("/home/ab2111/machine_learning_landscapes/neural_net/theano_db/"
#                                 +"toDelete.sqlite")
#     hem = HammingErrorAnalysis(system, db)
#     hem.plot()
#     exit()
#     dg = HammingDG(system, db, hem)
#     dg.plot()
    
#     m0 = db.minima()[0]
#     quench = system.get_minimizer(iprint=200)
#     ret = quench(m0.coords)
#     do_hessian_decomp(system, db)
#     run_basinhopping(system, db)     
    get_minima_stats(system, db)
#     weighted_prediction(system, db)
#     get_misread_stats(system, db)
#     connect_particular_minima(system, db)
#     for _ in xrange(10):
#     quench_trajectory(system, db)
#     run_gui(system, db)
#     make_only_dg(system, db)

def mainNew():
    
    ndata = 50000
    n_hidden = 300
    p = 2
    L2_reg=np.power(1.0e1, -p)
#     L1_reg=np.power(1.0e1, -p)
#     L2_reg=0.0
    L1_reg=0.0
    bias_reg = 0.0
    system = NNSystem(ndata=ndata, n_hidden=n_hidden, L1_reg=L1_reg, L2_reg=L2_reg, bias_reg=bias_reg)
    db = system.create_database("/home/ab2111/machine_learning_landscapes/neural_net/theano_db/"
                                +"data"+str(ndata)
                                +"hidden"+str(n_hidden)
#                                 +"L1reg"+str(system.potential.L2_reg)+".sqlite")
                                +"reg"+str(system.potential.L2_reg)+".sqlite")

    run_basinhopping(system, db)     
    connect_minima(system, db)
#     get_minima_stats(system, db)
#     weighted_prediction(system, db)
#     get_misread_stats(system, db)
#     connect_particular_minima(system, db)
#     for _ in xrange(10):
#     quench_trajectory(system, db)
#     run_gui(system, db)
#     make_only_dg(system, db)
        
if __name__=="__main__":
    main()
