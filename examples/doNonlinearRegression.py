import numpy as np
import matplotlib.pyplot as plt
from MLbasinhopping.utils import run_basinhopping, run_double_ended_connect, make_disconnectivity_graph

from MLbasinhopping.regression.models import RegressionModel, RegressionSystem, TestModel, SinModel

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
    
def run():
    
    np.random.seed(12345)
    
    # generate some run data
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
    run_double_ended_connect(system, db)
        
    # connect minima
    make_disconnectivity_graph(system, db)
        

if __name__=="__main__":
    run()
