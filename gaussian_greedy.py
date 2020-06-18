import itertools as it
import numpy as np
import time
import os
from tigramite import data_processing as pp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
import gpytorch
from LBFGS import FullBatchLBFGS
import gc
import copy
from gpytorch.utils import linear_cg 
from gpytorch.lazy import delazify 


class GaussianGreedy(object):
    def __init__(self, 
            selected_variables = None,
            tau_min=1, 
            tau_max=1,
            verbosity=0,
            scale_parents=True,
            include_time=False):
        self.selected_variables = selected_variables
        self.tau_min = tau_min
        self.tau_max = tau_max 
        self.verbosity = verbosity
        self.include_time = include_time
        self.scale_parents = scale_parents
    
    def _set_dataframe(self,dataset):
        dataframe = pp.DataFrame(dataset)
        # Set the data for this iteration of the algorithm
        self.dataframe = dataframe
        # Store the shape of the data in the T and N variables
        self.T, self.N = self.dataframe.values.shape
        # Some checks
        if (np.any(np.array(self.selected_variables) < 0) or
           np.any(np.array(self.selected_variables) >= self.N)):
            raise ValueError("selected_variables must be within 0..N-1")

    def _get_train_data(self, parents, target, tau_max):
        Y = [(target,0)] 
        X = [(target, i) for i in range(-1*tau_max,0)]
        X = X + [(parent, i) for parent, i in it.product(parents, range(-1*tau_max,0))]

        array, xyz, XYZ = self.dataframe.construct_array(X=X, Y=Y, Z=Y,
                tau_max=tau_max, return_cleaned_xyz=True, do_checks=False)
        dim, T = array.shape
        
        xx = np.where(xyz == 0)[0]
        yy = np.where(xyz == 1)[0]
        zz = np.where(xyz == 2)[0]
        Xset, Yset, Zset = xx, yy, zz

        arrayT = np.fastCopyAndTranspose(array)
        train_x = arrayT[:,Xset]
        train_y = arrayT[:, Yset]
        del arrayT, array
        return train_x, train_y

    def bic(self, parents, target, tau_max):
        train_x, train_y = self._get_train_data(parents, target, tau_max)
        kernel = RBF(length_scale=1, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1e0, noise_level_bounds=(1e-10, 1e+1))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=2)
        gp.fit(train_x[idx], train_y[idx])
        
        eps = 1e-8
        theta_opt = gp.kernel_.theta
        n_params = theta_opt.shape[0]
        H = np.zeros((n_params, n_params))
        log_mll, mll_grad = gp.log_marginal_likelihood(theta_opt, eval_gradient=True)

        for i in range(n_params):
            e_ii = np.zeros(n_params)
            e_ii[i] = 1 
            log_mll_2, mll_grad_2 = gp.log_marginal_likelihood(np.log(np.exp(theta_opt)+eps*e_ii), eval_gradient=True)
            H[i,i] = (mll_grad_2[i] - mll_grad[i])/eps
        for i in range(n_params):
            for j in range(i, n_params):
                e_ij = np.zeros(n_params)
                e_ij[i], e_ij[j] = 1, 1
                log_mll_2, mll_grad_2 = gp.log_marginal_likelihood(np.log(np.exp(theta_opt)+eps*e_ij), eval_gradient=True)
                H[i,j] = 0.5*((np.dot(e_ij, mll_grad_2-mll_grad))/eps - H[i,i] - H[j,j])
                H[j,i] = H[i,j].copy()
        
        detH = np.abs(np.linalg.det(H)) + 1e-2
        print(H, detH, mll_grad, mll_grad_2, theta_opt)

        return log_mll, log_mll - 0.5*np.log(detH)

            
    def _run_phase1_single(self, j,
                              tau_min=1,
                              tau_max=1):
        if not self.include_time:
            nodes = [node for node in range(self.N) if node!=j]
        else:
            nodes = [node for node in range(1,self.N) if node!=j]
        
        # Ensure tau_min is atleast 1
        tau_min = max(1, tau_min)
    
        # Iteration through increasing number of conditions, i.e. from 
        # [0,max_conds_dim] inclusive
        converged = False
        CPC = []
        max_bic = -1*np.inf 
        bic_scores = dict()
        while not converged:
            converged = True
            max_bic_ = -1*np.inf
            max_node = None
            candidate_list = [node for node in nodes if node not in CPC]
            for x in candidate_list:
                mll, bic = self.bic([x]+CPC, j, tau_max)
                if self.verbosity > 1:
                    print("\t\t link X{} --> X{} , S: {}, MLL: {}, BIC: {}".format(x, j, CPC, mll, bic))
                if bic > max_bic_:
                    max_bic_ = bic
                    max_node = x
                #print("max_node {} max_assoc {}".format(max_node, max_assoc))
            if (max_bic_ > max_bic) & (max_node != None):
                max_bic = max_bic_
                CPC.append(max_node)
                bic_scores[max_node] = max_bic
                if self.verbosity > 1:
                    print("Node X%d added to candidate parents of X%d"% (max_node,j) )
                converged = False
        return {'parents':CPC, 'scores': bic_scores}

    def run_phase1(self, dataset,
            tau_min = 1,
            tau_max = 1):
        self._set_dataframe(dataset)

        tau_min = max(1,tau_min)
        if self.verbosity > 0:
            print('##########################')
            print('# Starting phase 1 of GaussianForwardBackward')
            print('##########################')
            print("\n\nParameters:")
            if len(self.selected_variables) < self.N:
                print("selected_variables = %s" % self.selected_variables)
            print("\ntau_min = %d" % tau_min
                  + "\ntau_max = %d" % tau_max)
            print("\n")
        # Initialize all parents
        self.CPC = dict()
        self.bic_scores_phase1 = dict()
        # Loop through the selected variables
        for j in self.selected_variables:
            # Print the status of this variable
            if self.verbosity > 0:
                print("\n## Variable %s" % j)
            results = \
                self._run_phase1_single(j,
                                           tau_min=tau_min,
                                           tau_max=tau_max)
            # Record the results for this variable
            self.CPC[j] = results['parents']
            self.bic_scores_phase1[j] = results['scores']
        if self.verbosity > 0:
            for j in self.selected_variables:
                print("#candidate parents of variable X%d after Phase 1:"%j)
                for parent in self.CPC[j]:
                    print("variable: %d scores=%f"%(parent,self.bic_scores_phase1[j][parent]))
                print()
        # Return the parents and minimum associations
        return {'parents':self.CPC, 'scores': self.bic_scores_phase1}#results 

    def _run_phase2_single(self, j,
                              tau_min=1,
                              tau_max=1):
        '''
        if not self.include_time:
            nodes = [node for node in range(self.N) if node!=j]
        else:
            nodes = [node for node in range(1,self.N) if node!=j]
        '''
        try:
            CPC = [node for node in self.CPC[j]]
        except AttributeError:
            print('Please run phase 1 of GaussianGreedy')

        # Ensure tau_min is atleast 1
        tau_min = max(1, tau_min)
    
        # Iteration through increasing number of conditions, i.e. from 
        # [0,max_conds_dim] inclusive
        converged = False
        max_bic = max([_ for _ in self.bic_scores_phase1[j].values()]) 
        bic_scores = dict([(node, max_bic) for node in CPC])
        while not converged:
            converged = True
            max_bic_ = -1*np.inf
            max_node = None
            if len(CPC) == 1:
                break
            else:
                for x in CPC:
                    CPC_wo_x = CPC.copy()
                    CPC_wo_x.remove(x)
                    mll, bic = self.bic(CPC_wo_x, j, tau_max)

                    if self.verbosity > 1:
                        print("\t\t no link X{} --> X{} , S: {}, MLL: {}, BIC: {}".format(x, j, CPC_wo_x, mll, bic))
                    if bic > max_bic_:
                        max_bic_ = bic
                        max_node = x
                    #print("max_node {} max_assoc {}".format(max_node, max_assoc))
                if (max_bic_ > max_bic) & (max_node != None):
                    max_bic = max_bic_
                    CPC.remove(max_node)
                    for node in CPC:
                        bic_scores[node] = max_bic
                    if self.verbosity > 1:
                        print("Node X%d removed from candidate parents of X%d"% (max_node,j) )
                    converged = False
        return {'parents':CPC, 'scores': bic_scores}

    def run_phase2(self, dataset,
            tau_min = 1,
            tau_max = 1):

        torch.cuda.empty_cache()
        self._set_dataframe(dataset)

        tau_min = max(1,tau_min)
        if self.verbosity > 0:
            print('##########################')
            print('# Starting phase 2 of GaussianForwardBackward')
            print('##########################')
            print("\n\nParameters:")
            if len(self.selected_variables) < self.N:
                print("selected_variables = %s" % self.selected_variables)
            print("\ntau_min = %d" % tau_min
                  + "\ntau_max = %d" % tau_max)
            print("\n")
        self.trimmed_CPC = dict()
        self.bic_scores_phase2 = dict()
        # Loop through the selected variables
        for j in self.selected_variables:
            # Print the status of this variable
            if self.verbosity > 0:
                print("\n## Variable %s" % j)
            results = \
                self._run_phase2_single(j,
                                           tau_min=tau_min,
                                           tau_max=tau_max)
            # Record the results for this variable
            self.trimmed_CPC[j] = results['parents']
            self.bic_scores_phase2[j] = results['scores']
        if self.verbosity > 0:
            for j in self.selected_variables:
                print("# parents of variable X%d after Phase 2:"%j)
                for parent in self.trimmed_CPC[j]:
                    print("variable: %d scores=%f"%(parent,self.bic_scores_phase2[j][parent]))
                print()
        # Return the parents and minimum associations
        return {'parents':self.trimmed_CPC, 'scores': self.bic_scores_phase2}#results 

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_devices, output_device):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        #lengthscale_constraint = gpytorch.constraints.Interval(1e-3, 3.)
        if len(train_x.shape) <=2:
            base_covar_module = gpytorch.kernels.RBFKernel() #lengthscale_constraint=lengthscale_constraint)#gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        else:
            base_covar_module = gpytorch.kernels.RBFKernel(batch_shape=train_x.size()[:1]) 

        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices),
            output_device=output_device
        )

    def forward(self, x):
        covar_x = self.covar_module(x)
        if len(x.shape)==3:
            covar_x = covar_x.prod(-3)
            mean_x = self.mean_module(x[0])
        else:
            mean_x = self.mean_module(x)#torch.zeros(x.size()[0]).cuda()# 
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGPModelCPU(ExactGPModel):
    def __init__(self, *args, **kwargs):
        super(ExactGPModelCPU, self).__init__(*args, **kwargs)

        #lengthscale_constraint = gpytorch.constraints.Interval(1e-3, 3.)
        if len(train_x.shape) <=2:
            base_covar_module = gpytorch.kernels.RBFKernel() #lengthscale_constraint=lengthscale_constraint)#gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        else:
            base_covar_module = gpytorch.kernels.RBFKernel(batch_shape=train_x.size()[:1])

        self.covar_module = base_covar_module


class Dataset(TorchDataset):
    def __init__(self, train_x, train_y):
        self.train_x = train_x.cpu().numpy()
        self.train_y = train_y.cpu().numpy()

    def __len__(self):
        return self.train_x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        train_x = torch.Tensor(np.array(self.train_x[idx]))
        train_y = torch.Tensor(np.array(self.train_y[idx]))
        
        return train_x, train_y

class GaussianGreedyScalable(GaussianGreedy):
    def __init__(self, full_batch=True, **kwargs):
        for arg in kwargs:
            print(arg)
        GaussianGreedy.__init__(self, **kwargs)
        self._checkpoint_size = None
        self.full_batch = full_batch
        self._batch_size = 500
        self._sgd_lr = 1e-1
        self._sgd_momentum = 0.9
        self._n_training_iter = 20
        self._n_restarts = 1
        self._lbfgs_lr = 0.5

    def _get_train_data(self, parents, target, tau_max, scale_parents):
        train_x, train_y = super()._get_train_data(parents, target, tau_max)
        if scale_parents:
            # the following will return an array of shape
            # (num_parents, T, tau)
            # this is useful because product kernels will have a scale parameter
            # for each parent. 
            # the scale will be constant across time, it only depends on parent
            train_x = np.transpose(train_x.reshape((train_y.shape[0],-1,tau_max)),(1,0,2))
            return train_x, train_y
        else:
            return train_x, train_y

    def find_best_gpu_setting(self,train_x,
                              train_y,
                              n_devices,
                              output_device,
                              preconditioner_size
    ):
        N = train_x.size(-2)

        # Find the optimum partition/checkpoint size by decreasing in powers of 2
        # Start with no partitioning (size = 0)
        settings = [0] + [int(n) for n in np.ceil(N / 2**np.arange(1, np.floor(np.log2(N))))]

        for checkpoint_size in settings:
            if self.verbosity > 2:
                print('Number of devices: {} -- Kernel partition size: {}'.format(n_devices, checkpoint_size))
            try:
                # Try a full forward and backward pass with this setting to check memory usage
                _, _, _ = self.train(train_x, train_y,
                             n_devices=n_devices, output_device=output_device,
                             checkpoint_size=checkpoint_size,
                             preconditioner_size=preconditioner_size, n_training_iter=1)

                # when successful, break out of for-loop and jump to finally block
                break
            except RuntimeError as e:
                print('RuntimeError: {}'.format(e))
            except AttributeError as e:
                print('AttributeError: {}'.format(e))
            finally:
                # handle CUDA OOM error
                gc.collect()
                torch.cuda.empty_cache()
        return checkpoint_size

    def train(self,
          train_x,
          train_y,
          n_devices,
          output_device,
          checkpoint_size,
          preconditioner_size,
          n_training_iter,
          n_restarts=1):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-3)).to(output_device)
        model = ExactGPModel(train_x, train_y, likelihood, n_devices, output_device).to(output_device)
        model.train()
        likelihood.train()

        optimizer = FullBatchLBFGS(model.parameters(), lr=.5)
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


        with gpytorch.beta_features.checkpoint_kernel(checkpoint_size), \
             gpytorch.settings.max_preconditioner_size(preconditioner_size):

            def closure():
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                return loss

            loss = closure()
            loss.backward()

            for i in range(n_training_iter):
                options = {'closure': closure, 'current_loss': loss, 'max_ls': 20}

                loss, _, _, _, _, _, _, fail = optimizer.step(options)

                if self.verbosity > 2:
                    print_lengthscale = ["%.3f"%p.item() for p in model.covar_module.module.lengthscale] 
                    print(f'Iter {i+1}/{n_training_iter} - Loss: {"%.3f"%loss.item()} lengthscale: {print_lengthscale}   noise: {"%.3f"%model.likelihood.noise.item()}')
                if fail:
                    for pname, p in model.named_parameters():
                        print(pname, p.grad)
                    if self.verbosity > 2:
                        print('Convergence reached!')
                    break
        if self.verbosity > 2:
            print("Finished training on {0} data points using {1} GPUs.".format(train_x.size(-2), n_devices) )
        return model, likelihood, mll


    def bic(self, parents, target, tau_max):
        train_x, train_y = self._get_train_data(parents, target, tau_max, scale_parents=self.scale_parents)
        train_y = train_y.ravel()
        output_device = torch.device('cuda:0')
        train_x, train_y = torch.Tensor(train_x).to(output_device), torch.Tensor(train_y).to(output_device)
        
        # make continguous
        train_x, train_y = train_x.contiguous(), train_y.contiguous()
        self.train_x, self.train_y = copy.copy(train_x), copy.copy(train_y)
        
        n_devices = torch.cuda.device_count()
        if self.verbosity > 2:
            print('Planning to run on {} GPUs.'.format(n_devices))
        preconditioner_size = 100
        if self._checkpoint_size is None:
            self._checkpoint_size = self.find_best_gpu_setting(train_x, train_y,
                                                n_devices=n_devices,
                                                output_device=output_device,
                                                preconditioner_size=preconditioner_size)

        self.model, self.likelihood, self.mll = self.train(train_x, train_y,
                          n_devices=n_devices, output_device=output_device,
                          checkpoint_size=self._checkpoint_size,
                          preconditioner_size=100,
                          n_training_iter=self._n_training_iter,
                          n_restarts=self._n_restarts)

        self.model.set_train_data(train_x, train_y, strict=False)
        output = self.mll.likelihood(self.model(train_x))
        preconditioner, _, _ = output.lazy_covariance_matrix._preconditioner()

        num_random_probes = 10
        # z, w are as defined in page 5 of https://arxiv.org/pdf/1711.03481.pdf
        z, w = torch.randn(1,train_y.shape[-1],num_random_probes), torch.randn(1,train_y.shape[-1],num_random_probes)
        #z, w = torch.bernoulli(torch.rand(1,train_y.shape[-1],num_random_probes))*2- 1., torch.bernoulli(torch.rand(1,train_y.shape[-1],num_random_probes))*2. - 1.
        z, w = z.cuda(), w.cuda()
        z, w = z/np.sqrt(train_y.shape[-1]), w/np.sqrt(train_y.shape[-1])

        def closure(rhs):
            with torch.no_grad():
                return output.lazy_covariance_matrix.matmul(rhs)
        def preconditioner_closure(residual):
            with torch.no_grad():
                return preconditioner(residual)

        # alpha are the weights for each sample
        # alpha = (K+sigma^2)^-1 (y- mu)
        alpha = linear_cg(closure, self.train_y-self.model(self.train_x).loc.detach(), max_iter=1000, preconditioner=preconditioner_closure)
        alpha = alpha/np.sqrt(train_y.shape[-1])
        
        # g, h are as defined in page 5 of https://arxiv.org/pdf/1711.03481.pdf
        # g = (K+ sigma^2)^-1 z, h = (K+sigma^2)^-1 w
        g = linear_cg(closure,z, max_iter=1000, preconditioner=preconditioner_closure)
        h = linear_cg(closure,w, max_iter=1000, preconditioner=preconditioner_closure)

        # get lazy version of K
        # K_ijk := exp(-||x_ij - x_ik||^2 / 2*l_i^2),
        # where x_ij= j-th vector for parent i, l_i = lengthscale for parent i
        K = self.model.covar_module(train_x).evaluate()
        K = K.cpu()
        if len(K.shape) == 2:
            K = K.view((1,)+ K.shape)
        # if K is three dimensional, then
        # K_prod_ijk := exp(-sum_{i=1}^p ||x_ij-x_ik||^2 / 2* l_i^2)
        K_prod = K.prod(-3)
        self.alpha = alpha

        if len(K.shape) == 3:
            H = np.zeros((K.shape[0]+1,K.shape[0]+1))
        else:
            H = np.zeros((2,2))

        
        def rbf_derivative(K_prod, K_i, l_i):
            # gives K_prod * (2*||x_ij - x_ik||^2 / 2*l_i^2)
            dlengthscale = K_prod.mul(K_i.log().mul(-2))
            dlengthscale[dlengthscale==float("Inf")] = 0
            # gives K * (2*||x_ij - x_ik||^2 / 2*l_i^3)
            l_i = l_i.item()
            dlengthscale = dlengthscale.__div__(l_i)
            return dlengthscale

        def rbf_double_derivative(K_prod, K_i, dK_di, l_i):
            d2lengthscale = dK_di.mul(K_i.log().mul(-2))
            d2lengthscale[d2lengthscale==float("Inf")] = 0
            # gives K * (4*||x_ij - x_ik||^4 / 4*l_i^6)
            l_i = l_i.item()
            d2lengthscale = d2lengthscale.__div__(l_i)#.exp())
            # gives K * (-6 * ||x_ij - x_ik||^2 / 2*l_i^3)
            temp = dK_di.mul(-3)
            # gives K * (-6 * ||x_ij - x_ik||^2 / 2*l_i^4)
            temp = temp.__div__(l_i)
            # finally gives K * (4 ||x_ij - x_ik||^4 / 4*l_i^6) - K*(6*||x_ij - x_ik||^2/2*l_i^4)
            d2lengthscale = d2lengthscale.__add__(temp)
            return d2lengthscale

        def rbf_cross_double_derivative(K_prod, dK_di, dK_dj):
            # d^2 K/ dl_i dl_j = dK/dl_i * dK/dl_j / K
            temp = dK_di.mul(dK_dj).__div__(K_prod)
            temp[temp==float("Inf")] = 0
            return temp


        grad_alpha = dict()
        grad_g, grad_w, grad_z = dict(),dict(),dict()   
        for i in range(H.shape[0]-1):
            dK_di = rbf_derivative(K_prod, K[i], self.model.covar_module.module.lengthscale[i])
            for j in range(i, H.shape[0]-1):


                dK_dj = rbf_derivative(K_prod, K[j], self.model.covar_module.module.lengthscale[j])
                dK_di = dK_di.cpu()
                if i==j:
                    d2K_didj = rbf_double_derivative(K_prod, K[i], dK_di, self.model.covar_module.module.lengthscale[i])
                else:
                    d2K_didj = rbf_cross_double_derivative(K_prod, dK_di, dK_dj)

                dK_di = dK_di.cuda()
                dK_dj, d2K_didj = dK_dj.cuda(), d2K_didj.cuda()
                grad_alpha[j] = torch.matmul(dK_dj, alpha)
                grad_g[j] = dK_dj.matmul(g)
                grad_w[j], grad_z[j] = dK_dj.matmul(w), dK_dj.matmul(z)

                term1 = torch.matmul(g.view(-1), gpytorch.matmul(d2K_didj,z).view(-1)) / num_random_probes

                term2 = 0.
                for k in range(num_random_probes):
                    term2 = term2 + torch.matmul(g[...,k].view(-1),grad_w[i][...,k].view(-1))*torch.matmul(h[...,k].view(-1),grad_z[j][...,k].view(-1))
                term2 = term2/num_random_probes
                term2 = term2* (train_y.shape[-1])

                term3 = 0.
                for k in range(num_random_probes):
                    term3 = term3 + torch.matmul(alpha, grad_z[i][...,k].view(-1))*torch.matmul(alpha, grad_g[j][...,k].view(-1))
                term3 = term3/num_random_probes
                term3 = term3* (train_y.shape[-1])

                term4 = torch.matmul(alpha.view(-1), torch.matmul(d2K_didj, alpha).view(-1))

                H[i,j] = -0.5*(term1 - term2 + 2*term3 -term4).item()

                H[j,i] = H[i,j].copy()

        # calculate terms of Hessian involving sigma
        # note that d2K/ dsigma dl_i = 0 for all i
        # d2K/dsigma^2 = 0
        # dK/dsigma = I
        for j in range(H.shape[0]-1):
            term2 = 0.
            for k in range(num_random_probes):
                term2 = term2 + torch.matmul(grad_g[j][...,k].view(-1),w[...,k].view(-1))*torch.matmul(h[...,k].view(-1),z[...,k].view(-1))
            term2 = term2/num_random_probes
            term2 = term2* (train_y.shape[-1])

            term3 = 0.
            for k in range(num_random_probes):
                term3 = term3 + torch.matmul(alpha, grad_z[j][...,k].view(-1))*torch.matmul(alpha, g[...,k].view(-1))
            term3 = term3/num_random_probes
            term3 = term3* (train_y.shape[-1])

            H[j,-1] = -0.5*(-term2 + 2*term3).item()
            H[-1,j] = H[j,-1].copy()

        term2 = 0.
        for k in range(num_random_probes):
            term2 = term2 + torch.matmul(g[...,k].view(-1),w[...,k].view(-1))*torch.matmul(h[...,k].view(-1),z[...,k].view(-1))
        term2 = term2/num_random_probes
        term2 = term2* (train_y.shape[-1])

        term3 = 0.
        for i in range(num_random_probes):
            term3 = term3 + torch.matmul(alpha, z[...,i].view(-1))*torch.matmul(alpha, g[...,i].view(-1))
        term3 = term3/num_random_probes
        term3 = term3* (train_y.shape[-1])

        H[-1,-1] = -0.5*(-term2 + 2*term3).item()

        mll_score = self.train_y.shape[-1]* self.mll(self.model(train_x), self.train_y).item()
        reg_score = -0.5* (np.log(np.abs(np.linalg.det(H))) + np.log(train_y.shape[-1])*H.shape[0])
        '''
        num_random_probes = 10
        num_tridiag = 20
        # z, w are as defined in page 5 of https://arxiv.org/pdf/1711.03481.pdf
        #z, w = torch.randn(1,train_x.shape[0],num_random_probes), torch.randn(1,train_x.shape[0],num_random_probes)
        z, w = torch.bernoulli(torch.rand(1,train_y.shape[-1],num_random_probes))*2- 1., torch.bernoulli(torch.rand(1,train_y.shape[-1],num_random_probes))*2. - 1.
        z, w = z.cuda(), w.cuda()
        #z, w = z/np.sqrt(train_y.shape[-1]), w/np.sqrt(train_y.shape[-1])

        def closure(rhs):
            with torch.no_grad():
                return output.lazy_covariance_matrix.matmul(rhs)
        def preconditioner_closure(residual):
            with torch.no_grad():
                return preconditioner(residual)

        # alpha are the weights for each sample
        # alpha = (K+sigma^2)^-1 (y- mu)
        alpha = linear_cg(closure, self.train_y-self.model(self.train_x).loc.detach(), max_iter=1000, preconditioner=preconditioner_closure)
        
        # g, h are as defined in page 5 of https://arxiv.org/pdf/1711.03481.pdf
        # g = (K+ sigma^2)^-1 z, h = (K+sigma^2)^-1 w
        g = linear_cg(closure,z, max_iter=1000, preconditioner=preconditioner_closure)
        h = linear_cg(closure,w, max_iter=1000, preconditioner=preconditioner_closure)

        # get lazy version of K
        # K_ij := exp(-||x_i - x_j||^2 / 2*l^2)
        temp = self.model.covar_module(train_x)
        # gives K * (2*||x_i - x_j||^2 / 2*l^2)
        dlengthscale = temp.mul(delazify(temp).log().mul(-2))
        # gives K * (2*||x_i - x_j||^2 / 2*l^3)
        dlengthscale = dlengthscale.__div__(self.model.covar_module.module.lengthscale)#.exp())
        # gives K * (4*||x_i - x_j||^4 / 4*l^5)
        d2lengthscale = dlengthscale.mul(delazify(temp).log().mul(-2))
        # gives K * (4*||x_i - x_j||^4 / 4*l^6)
        d2lengthscale = d2lengthscale.__div__(self.model.covar_module.module.lengthscale)#.exp())
        # gives K * (-6 * ||x_i - x_j||^2 / 2*l^3)
        temp2 = dlengthscale.mul(-3)
        # gives K * (-6 * ||x_i - x_j||^2 / 2*l^4)
        temp2 = temp2.__div__(self.model.covar_module.module.lengthscale)#.exp())
        # finally gives K * (4 ||x_i - x_j||^4 / 4*l^6) - K*(6*||x_i - x_j||^2/2*l^4)
        d2lengthscale = d2lengthscale.__add__(temp2)

        ### delete this
        #dlengthscale = gpytorch.lazy.diag_lazy_tensor.DiagLazyTensor(torch.ones(1,self.train_x.shape[-2])).cuda()
        #d2lengthscale =gpytorch.lazy.diag_lazy_tensor.DiagLazyTensor(torch.ones(1,self.train_x.shape[-2])).cuda()
        ################
        self.dlengthscale = copy.copy(dlengthscale)
        self.d2lengthscale = copy.copy(d2lengthscale)
        self.alpha = copy.copy(alpha)

        H = np.zeros((2,2))
        grad_alpha = dlengthscale.matmul(alpha)
        grad_g = dlengthscale.matmul(g)
        grad_w, grad_z = dlengthscale.matmul(w), dlengthscale.matmul(z)
        
        #### delete this
        #g, h = z.clone() , w.clone()
        #grad_alpha = alpha.clone()
        #grad_g = g.clone()
        #grad_w, grad_z = w.clone(), z.clone()
        ################
        
        term1 = torch.matmul(g.view(-1), gpytorch.matmul(d2lengthscale,z).view(-1)) / num_random_probes
        term2 = 0.
        for i in range(num_random_probes):
            term2 = term2 + torch.matmul(g[...,i].view(-1),grad_w[...,i].view(-1))*torch.matmul(h[...,i].view(-1),grad_z[...,i].view(-1))
        term2 = term2/num_random_probes

        term3 = 0.
        for i in range(num_random_probes):
            term3 = term3 + torch.matmul(alpha, grad_z[...,i].view(-1))*torch.matmul(alpha, grad_g[...,i].view(-1))
        term3 = term3/num_random_probes

        term4 = torch.matmul(alpha.view(-1), torch.matmul(d2lengthscale.evaluate(), alpha).view(-1))
        H[0,0] = -0.5*(term1 - term2 + 2*term3 -term4).item()

        term2 = 0.
        for i in range(num_random_probes):
            term2 = term2 + torch.matmul(g[...,i].view(-1),grad_w[...,i].view(-1))*torch.matmul(h[...,i].view(-1),z[...,i].view(-1))
        term2 = term2/num_random_probes

        term3 = 0.
        for i in range(num_random_probes):
            term3 = term3 + torch.matmul(alpha, grad_z[...,i].view(-1))*torch.matmul(alpha, g[...,i].view(-1))
        term3 = term3/num_random_probes

        H[0,1] = -0.5*( -term2 + 2*term3).item()
        H[1,0] = H[0,1].copy()

        term2 = 0.
        for i in range(num_random_probes):
            term2 = term2 + torch.matmul(g[...,i].view(-1),w[...,i].view(-1))*torch.matmul(h[...,i].view(-1),z[...,i].view(-1))
        term2 = term2/num_random_probes

        term3 = 0.
        for i in range(num_random_probes):
            term3 = term3 + torch.matmul(alpha, z[...,i].view(-1))*torch.matmul(alpha, g[...,i].view(-1))
        term3 = term3/num_random_probes
        #term3 = matmul(alpha.view(z.shape[:-1]), z).mean()

        H[1,1] = -0.5*(-term2 + 2*term3).item()

        print(H)
        mll_score = self.train_y.shape[-1]* self.mll(self.model(train_x), self.train_y).item()
        reg_score = -0.5* np.log(np.abs(np.linalg.det(H)))
        '''

        return mll_score, mll_score + reg_score
