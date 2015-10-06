import numpy as np

def orderbyenergy(db, energy):
    
    minima = db.minima()
        
    return np.argmin([np.abs(m.energy-energy) for m in minima])

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
            print m._id, m.energy, np.average(pot.model.getValidationError(m.coords))
#             e,g,h = pot.getEnergyGradientHessian(m.coords)
#             print "got hessian, rms grad:", np.linalg.norm(g)
#             evals, evecs = normalmodes(h)
#             plt.plot(evals)
#             plt.show()
#     

