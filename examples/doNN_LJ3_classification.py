import numpy as np
import os

from MLbasinhopping.NN.models import NNSystem, NNModel

from MLbasinhopping.utils import run_basinhopping, run_double_ended_connect, make_disconnectivity_graph

def get_data_LJ3():

    __location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
    
    dataset = "/home/ab2111/machine_learning_landscapes/NN_LJ3/analysis.LJ3.starting_points"
    dat = np.loadtxt(dataset, usecols=(1,2,3,4,5,6,7,8))
    train_set = dat[:len(dat)/2]
    test_set = dat[len(dat)/2:]
    
    # training data
    x = train_set[:,:-1]
    ndata, n_features = x.shape

#     
#     print "Data set size:\n", ndata, " data points\n", n_features, " features\n";
#     exit()
    # labels
    t = train_set[:,-1]

    assert ndata == t.size
    
    # test data
    test_x = test_set[:,:-1]
    test_t = test_set[:,-1].astype('int32')

    return x, t, test_x, test_t

def main():
    
    ndata = 5000
    n_hidden = 10
    p = 2
    L2_reg=np.power(1.0e1, -p)
    L1_reg=0.0
    bias_reg = 0.0
    model = NNModel(ndata=ndata, n_hidden=n_hidden, n_out=4, dataLoader=get_data_LJ3, L1_reg=L1_reg, L2_reg=L2_reg, bias_reg=bias_reg)
    system = NNSystem(model)
    
    db = system.create_database()
    pot = system.get_potential()
    print "Number of adjustable parameters: ", model.nparams
  
    print "Training model..."
    quench = system.get_minimizer(nsteps=1e6, iprint=1000)
    ret = quench(np.random.random(model.nparams))
    print "Testset error: ", np.average(model.getValidationError(ret.coords))

#     nsteps = 10
#     run_basinhopping(system, nsteps, db)     

if __name__=="__main__":
    main()
