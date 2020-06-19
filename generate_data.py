import numpy as np
import numpy.linalg as lin
import os
import pandas as pd
from argparse import ArgumentParser
from kuramoto import Kuramoto

def gen_samples_kuramoto(K, T, W=None, Y0=None):
    assert (K is not None), "please provide a valid coupling matrix"

    n = K[0].shape[0]
    #T_ = np.linspace(0,40,T+1)#np.arange(T)
    T_ = np.linspace(0,0.05*T, T+1)
    dT = T_[1]-T_[0]
    if W is None:
        W = np.random.rand(n) * 19. + 1.
        #W = np.random.rand(n) * 9. + 1.
    if Y0 is None:
        Y0 = np.random.rand(n) * 2 * np.pi
    init_params = {'W':W, 'K': K, 'Y0': Y0}
    kuramoto = Kuramoto(init_params)
    kuramoto.noise = 'normal'#'logistic'
    odePhi, dPhi = kuramoto.solve(T_)
    return (np.diff(odePhi,axis=1)/dT).T

def synthetic_data_kuramoto(nvars, density, T=20000,W=None, Y0=None):
    graph = 5*gen_graph(nvars, density)
    graph = graph.reshape((1,nvars,nvars))
    samples = gen_samples_kuramoto(graph, T, W, Y0)
    return graph, samples
def main(args):
    # create directory for data

    # default directory will be data/kuramoto/nvars20_density0.1_nharmonics1/
    if args.datadir is None:
        datadir = os.path.join('data', args.experiment)
        if not os.path.exists(datadir):
            os.mkdir(datadir)
        datadir = os.path.join(datadir, 'nvars{0}_density{1}_nharmonics{2}'.format(args.nvars, args.density, args.nharmonics))
        if not os.path.exists(datadir):
            os.mkdir(datadir)
    for n in range(args.ndatasets):
        if args.experiment == 'kuramoto':
            graph = np.zeros((args.nharmonics, args.nvars, args.nvars))
            
            for t in range(args.nharmonics):
                graph_ = gen_graph(args.nvars, args.density/args.nharmonics)
                graph[t,:,:] = 5*graph_
                #idx = graph!=0
                #K[t,idx] = 1. # kuramoto does x_t+1 = f(Ax_t), whereas we do x_t+1 = x_t A. hence take transpose of A
            
            samples = gen_samples_kuramoto(graph, args.length)
            samples = samples 
        elif args.experiment == 'danks':
            graph = np.zeros((args.nharmonics, args.nvars, args.nvars))
            
            for t in range(args.nharmonics):
                graph_ = gen_graph(args.nvars, args.density/args.nharmonics)
                graph[t,:,:] = graph_
           
            function_ = lambda x: np.cos(x + 0.5*np.random.standard_normal(x.shape))
            samples = gen_samples(graph[0], args.length, noise_var=args.noise_var, function=function_)
        # save samples and graphs
        samples_filename = os.path.join(datadir, 'samples_{0}.npy'.format(n))
        graph_filename = os.path.join(datadir, 'graph_{0}.npy'.format(n))
        np.save(samples_filename, samples)
        np.save(graph_filename, graph)

if __name__=='__main__':

    parser = ArgumentParser()

    parser.add_argument('--datadir', type=str, default=None,
            help='directory to store data')
    parser.add_argument('--experiment', type=str, default='kuramoto',
            help='type of data to create', choices=['kuramoto','danks'])
    parser.add_argument('--nharmonics', type=int, default=2,
            help='number of harmonics in kuramoto oscillator')
    parser.add_argument('--ndatasets', type=int, default=16,
            help='number of datasets to generate')
    parser.add_argument('--length', type=int, default=1000,
            help='length of run')
    parser.add_argument('--nvars', type=int, default=20,
            help='number of variables')
    parser.add_argument('--noise-var', type=float, default=0.01,
            help='variance of noise')
    parser.add_argument('--density', type=float, default=0.1,
            help='density of graph')
    
    args = parser.parse_args()
    main(args)
