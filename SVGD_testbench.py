import torch
import numpy as np
import scipy.spatial.distance
import math
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
# from matplotlib.ticker import LinearLocator
# from matplotlib.ticker import LogLocator
from torch import optim
from evotorch import Problem
from evotorch.algorithms import CMAES
from evotorch.logging import StdOutLogger, PandasLogger
import sympy
from collections import deque


class SVGD():

    def __init__(self):
        pass


    def kernel(self, theta, theta2, h = -1):
        pairwise_dists = torch.cdist(theta,theta2)**2
        if h < 0: # if h < 0, using median trick
            h = torch.quantile(pairwise_dists,q=0.5).item()
            h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))

        # compute the rbf kernel
        if abs(h) <= 1e-8:
            h = 1
        pairwise_div = torch.div(pairwise_dists,(2*(h**2)))
        pairwise_neg = - pairwise_div
        Kij = torch.exp(pairwise_neg)
        return Kij

    def distrib_kernel(self, theta1, theta2, sigma1, sigma2, rho = 1, dims = 2):
        det1 = torch.det(sigma1)
        det2 = torch.det(sigma2)
        try:
            sigmat = torch.linalg.inv(torch.sum(torch.linalg.inv(sigma1), torch.linalg.inv(sigma2)))
            mut = torch.sum(torch.matmul(torch.linalg.inv(sigma1),theta1[:,1]), torch.matmul(torch.linalg.inv(sigma2),theta2[:,1]))
            det_sigmat = torch.det(sigmat)
        except RuntimeError:
            print("non-inversible values passed as parameters of the gaussians")
            return 0
        return math.pi**((1-2*rho)*dims/2) * rho ** (-dims/2) * torch.matmul(torch.matmul(torch.matmul(torch.pow(det_sigmat, 0.5), torch.pow(det1, -rho/2)), torch.pow(det2, -rho/2))), torch.exp(-rho/2 * (torch.matmul(torch.matmul(torch.transpose(theta1), torch.linalg.inv(sigma1)), theta1) + torch.matmul(torch.matmul(torch.transpose(theta2), torch.linalg.inv(sigma2)), theta2) - torch.matmul(torch.matmul(torch.transpose(mut), torch.linalg.inv(sigmat)), mut)))

    # SVGD boîte blanche

    def update(self, particles, lntarget, n_iter = 1, stepsize = 1e-5):
        # Check input
        if particles is None or lntarget is None:
            raise ValueError('particles or lnprob cannot be None!')
        theta = particles.clone().detach()
        theta.requires_grad_(True)
        optimizer = optim.SGD([theta], lr=stepsize)
        for iter in range(n_iter):
            kij = self.kernel(theta,theta.detach())
            normali = torch.ones_like(kij).fill_diagonal_(0)
            # normali = torch.ones_like(kij)
            kij.backward(normali)
            f = theta.grad.clone()
            theta.grad = None
            lnp = lntarget(theta)
            lnp.backward(torch.ones_like(lnp))
            gr = theta.grad.clone()
            # theta.grad = None
            theta.grad = torch.add(torch.matmul(kij,gr),f)
            optimizer.step()
            optimizer.zero_grad()
            theta.requires_grad_(True)
        return theta

    def update(self, particles, lntarget, low, n_iter = 1, stepsize = 1e-5, width = 1, echantill = 10):
        # Check input
        if particles is None or lntarget is None:
            raise ValueError('particles or lnprob cannot be None!')
        theta = particles.clone().detach()
        theta.requires_grad_(True)
        optimizer = optim.Adam([theta], lr=stepsize)
        mem = low
        for iter in range(1,n_iter+1):
            kij = self.kernel(theta,theta.detach())
            normali = torch.ones_like(kij).fill_diagonal_(0)
            kij.backward(normali)
            f = theta.grad.clone()
            theta.grad = None
            lnp = lntarget(theta[:,0], theta[:,1])
            lnp.backward(torch.ones_like(lnp))
            gr = theta.grad.clone()
            theta.grad = torch.add(torch.matmul(kij,gr),f)
            mem = min(mem,torch.min(lntarget(theta[:,0], theta[:,1])).item())
            optimizer.step()
            optimizer.zero_grad()
            theta.requires_grad_(True)
        return theta, mem

    # SVGD avec utilisation d'un échantillonage gaussien

    def update_gradient_gauss_variable(self, particles, lntarget, low = 10e9, n_iter = 1, stepsize = 1e0, width = lambda a : 1, echantill = 1, temp = lambda a:1):
        if particles is None or lntarget is None:
            raise ValueError('particles or lnprob cannot be None!')
        theta = particles.clone().detach()
        theta.requires_grad_(True)
        optimizer = optim.Adam([theta], lr=stepsize)
        mem = low
        for iter in range(1,n_iter+1):
            kij = self.kernel(theta,theta.clone().detach())
            kij.sum().backward()
            f = - theta.grad.clone()
            theta.grad = None
            try:
                scalex = torch.zeros((theta.size()[1],theta.size()[1])).fill_diagonal_(width(iter))
            except ZeroDivisionError:
                scalex = torch.zeros((theta.size()[1],theta.size()[1])).fill_diagonal_(10)
            except ValueError:
                scalex = torch.zeros((theta.size()[1],theta.size()[1])).fill_diagonal_(10)
            if torch.cuda.is_available():
                scalex = scalex.to('cuda')
            samples = torch.distributions.multivariate_normal.MultivariateNormal(theta, covariance_matrix = scalex)
            a = samples.sample((echantill,))
            samples.log_prob(a).sum().backward()
            grad = theta.grad.clone().detach()
            lnp = lntarget(a)
            mem = min(mem, torch.min(lnp).item())
            moy = torch.mean(lnp).item()
            vals = (lnp - moy)
            vals = vals.unsqueeze(-1)
            interm = torch.mul(grad, vals)
            tmp = torch.squeeze(torch.mean(interm, 0, keepdim=True),0)
            j = torch.mm(kij,tmp)
            theta.grad = (1/theta.size()[0]) * torch.add(j, temp(iter) * f)
            optimizer.step()
            optimizer.zero_grad()
            theta.requires_grad_(True)
        return theta, mem

    # SVGD avec échantillonage gaussien et adaptation des paramètres de la loi normale


    def update_gradient_gauss_self_adjust(self, particles, lntarget, width = 1, low = 10e9, n_iter = 1, stepsize_med = 1e0, stepsize_var = 1e0, echantill = 1, temp = lambda a:1):
        if particles is None or lntarget is None:
            raise ValueError('particles or lnprob cannot be None!')
        theta = particles.clone().detach()
        theta.requires_grad_(True)
        sigma = width.clone().detach()
        optimizer = optim.SGD([theta], lr=stepsize_med)
        optimizer2 = optim.SGD([sigma], lr = stepsize_var)
        mem = low
        for iter in range(1,n_iter+1):
            theta.requires_grad_(True)
            sigma.requires_grad_(True)
            sigma2 = torch.matmul(sigma, sigma.permute(0, 2, 1))
            control = 1e-4 * torch.eye(sigma2.size()[1])
            if torch.cuda.is_available():
                control = control.to('cuda')
            sigma3 = sigma2 + control
            kij = self.kernel(theta,theta.clone().detach())
            kij.sum().backward()
            f = theta.grad.clone()
            theta.grad = None
            samples = torch.distributions.multivariate_normal.MultivariateNormal(theta, covariance_matrix = sigma3)
            a = samples.sample((echantill,))
            a.requires_grad_(True)
            samples.log_prob(a).sum().backward()
            grad = - a.grad.clone().detach()
            grad2 = sigma.grad.clone().detach()
            lnp = lntarget(a)
            mem = torch.min(lnp).item()
            moy = torch.mean(lnp).item()
            vals = (lnp - moy)
            vals2 = vals.unsqueeze(-1).unsqueeze(0)
            interm = torch.mul(grad, vals2)
            tmp = torch.mean(interm, (0,1))
            interm2 = torch.mul(grad2, torch.mean(vals,0).unsqueeze(-1).unsqueeze(-1))
            tmp2 = interm2
            j = torch.mm(kij,tmp)
            theta.grad = (1/theta.size()[0]) * torch.add(j, temp(iter) * f)
            sigma.grad = tmp2
            optimizer.step()
            optimizer2.step()
            optimizer.zero_grad()
            optimizer2.zero_grad()
            theta.requires_grad_(True)
            sigma.requires_grad_(True)
        return theta, mem, sigma

    # SVGD avec échantillonage gaussien, gradient naturel et adaptation des paramètres de la loi normale

    def update_gradient_gauss_self_adjust_Fisher(self, particles, lntarget, width = 1, low = 10e9, n_iter = 1, stepsize_med = 1e0, stepsize_var = 1e0, echantill = 1, temp = lambda a:1/a**4, opti = optim.Adagrad):
        if particles is None or lntarget is None:
            raise ValueError('particles or lnprob cannot be None!')
        theta = particles.clone().detach()
        theta.requires_grad_(True)
        sigma = width.clone().detach()
        optimizer = opti([theta], lr=stepsize_med)
        optimizer2 = opti([sigma], lr = stepsize_var)
        mem = low
        for iter in range(1,n_iter+1):
            sigma.requires_grad_(True)
            sigma2 = torch.matmul(sigma, sigma.permute(0, 2, 1).clone().detach())
            control = 1e-4 * torch.eye(sigma2.size()[1])
            if torch.cuda.is_available():
                control = control.to('cuda')
            sigma3 = sigma2 + control
            kij = self.kernel(theta,theta.clone().detach())
            kij.sum().backward()
            f = - theta.grad.clone()
            theta.grad = None
            samples = torch.distributions.multivariate_normal.MultivariateNormal(theta, covariance_matrix = sigma3)
            a = samples.sample((echantill,))
            a.requires_grad_(True)
            samples.log_prob(a).sum().backward()
            grad = - a.grad.clone().detach()
            a.grad = None
            a.requires_grad_(False)
            grad2 = sigma.grad.clone().detach()
            lnp = lntarget(a)
            mem = min(mem, torch.min(lnp).item())
            moy = torch.mean(lnp).item()
            vals = (lnp - moy)
            vals2 = vals.unsqueeze(-1).unsqueeze(0)
            interm = torch.mul(grad, vals2)
            tmp = torch.mean(interm, (0,1)) #grad
            fish = torch.matmul(grad,grad.permute(0, 2, 1)) # estimated fisher info
            try:
                fish_inv = torch.inverse(fish)
            except RuntimeError:
                fish_inv = torch.eye(tmp.size()[0])
                if torch.cuda.is_available():
                    fish_inv = fish_inv.to('cuda')
            tmp = torch.matmul(fish_inv, tmp)
            interm2 = torch.mul(grad2, torch.mean(vals,0).unsqueeze(-1).unsqueeze(-1))
            fish2 = torch.matmul(grad2,grad2.permute(0, 2, 1)) # estimated fisher info
            try:
                fish_inv2 = torch.inverse(fish2)
                # print("i")
            except RuntimeError:
                # print("o")
                fish_inv2 = torch.eye(interm2.size()[1])
                if torch.cuda.is_available():
                    fish_inv2 = fish_inv2.to('cuda')
            tmp2 = torch.matmul(fish_inv2, interm2)
            j = torch.matmul(kij,tmp)
            theta.grad = (1/theta.size()[0]) * torch.add(torch.mean(j,0), temp(iter) * f)
            sigma.grad = tmp2
            optimizer.step()
            optimizer2.step()
            optimizer.zero_grad()
            optimizer2.zero_grad()
            theta.requires_grad_(True)
            sigma.requires_grad_(True)
        return theta, mem, sigma

    # SVGD avec échantillonage gaussien, TRPO et adaptation des paramètres de la loi normale

    def update_gradient_gauss_self_adjust_Fisher_TRPO(self, particles, lntarget, width = 1, low = 10e9, n_iter = 1, stepsize_med = 1e0, stepsize_var = 1e0, echantill = 1, temp = lambda a:1/a**4, opti = optim.Adagrad, back = 0.9):
        if particles is None or lntarget is None:
            raise ValueError('particles or lnprob cannot be None!')
        theta = particles.clone().detach()
        theta.requires_grad_(True)
        sigma = width.clone().detach()
        optimizer = opti([theta], lr=stepsize_med)
        optimizer2 = opti([sigma], lr = stepsize_var)
        mem = low
        for iter in range(1,n_iter+1):
            sigma.requires_grad_(True)
            N = int((math.sqrt(8*sigma.size()[1]+1)-1)/2)
            sigma2 = torch.zeros(N, N)
            if torch.cuda.is_available():
                sigma2 = sigma2.to('cuda')
            sigma2[torch.tril_indices(N, N, offset=0).tolist()] = sigma
            kij = self.kernel(theta,theta.clone().detach())
            kij.sum().backward()
            f = - theta.grad.clone()
            theta.grad = None
            samples = torch.distributions.multivariate_normal.MultivariateNormal(theta, scale_tril = torch.abs(sigma2))
            a = samples.sample((echantill,))
            a.requires_grad_(True)
            samples.log_prob(a).sum().backward()
            grad = - a.grad.clone().detach()
            a.grad = None
            a.requires_grad_(False)
            grad2 = sigma.grad.clone().detach()
            lnp = lntarget(a)
            mem = min(mem, torch.min(lnp).item())
            moy = torch.mean(lnp).item()
            vals = (lnp - moy)
            vals2 = vals.unsqueeze(-1).unsqueeze(0)
            interm = torch.mul(grad, vals2)
            tmp = torch.transpose(torch.mean(interm, (0,)), 1, 2) #grad
            fish = torch.matmul(torch.transpose(grad, 1, 2), grad) # estimated fisher info
            try:
                fish_inv = torch.inverse(fish)
                # print("i")
            except RuntimeError:
                filler = torch.ones(tmp.size()[0],tmp.size()[1])
                fish_inv = torch.diag_embed(filler)
                if torch.cuda.is_available():
                    fish_inv = fish_inv.to('cuda')
            tmp = torch.matmul(fish_inv, tmp)
            interm2 = torch.mul(grad2, torch.mean(vals,0).unsqueeze(-1).unsqueeze(-1))
            fish2 = torch.matmul(grad2,torch.transpose(grad2, 0, 1)) # estimated fisher info
            try:
                fish_inv2 = torch.inverse(fish2)
                # print("i")
            except RuntimeError:
                # print("o")
                fish_inv2 = torch.eye(interm2.size()[1])
                if torch.cuda.is_available():
                    fish_inv2 = fish_inv2.to('cuda')
            tmp2 = torch.matmul(fish_inv2, interm2)
            j = torch.matmul(tmp, kij)
            theta_temp = torch.matmul(torch.matmul(grad, fish_inv), torch.transpose(grad,1,2))
            sigma_temp = torch.matmul(torch.matmul(fish_inv2, grad2), torch.transpose(grad2,0,1))
            thetalr = torch.sqrt((2*1e-7)/(abs(theta_temp)+torch.full_like(theta_temp,1e-8))) # 1e-7 est le epsilon de la formule de TRPO
            sigmalr = torch.sqrt((2*1e-7)/(abs(sigma_temp)+torch.full_like(sigma_temp,1e-8)))
            #backtracking
            sigma2 = torch.zeros(N, N)
            if torch.cuda.is_available():
                sigma2 = sigma2.to('cuda')
            sigma2[torch.tril_indices(N, N, offset=0).tolist()] = sigma
            output = torch.distributions.multivariate_normal.MultivariateNormal(theta, scale_tril = abs(sigma2))
            while (torch.distributions.kl.kl_divergence(output, samples)) >= 1e-1:
                theta.grad = back*(1/theta.size()[0]) * torch.add(torch.mean(thetalr * j, 0).t(), temp(iter) * f)
                sigma.grad = back*sigmalr * tmp2
            optimizer.step()
            optimizer2.step()
            optimizer.zero_grad()
            optimizer2.zero_grad()
            theta.requires_grad_(True)
            sigma.requires_grad_(True)
        return theta, mem, sigma

        # SVGD avec échantillonage gaussien, PPO et adaptation des paramètres de la loi normale

    def update_gradient_gauss_TRPO_PPO(self, particles, lntarget, width = 1, low = 10e9, n_iter = 1, stepsize_med = 1e0, stepsize_var = 1e0, echantill = 1, temp = lambda a:1/a**4, opti = optim.Adagrad, resample_rate = 1, beta = 1e-2, back = 0.9):
        if particles is None or lntarget is None:
            raise ValueError('particles or lnprob cannot be None!')
        theta = particles.clone().detach()
        theta.requires_grad_(True)
        sigma = width.clone().detach()
        optimizer = opti([theta], lr=stepsize_med)
        optimizer2 = opti([sigma], lr = stepsize_var)
        mem = low
        for iter in range(1,n_iter+1):
            sigma.requires_grad_(True)
            sigma2 = torch.matmul(sigma, torch.transpose(sigma, 1, 2).clone().detach())
            control = torch.eye(sigma2.size()[1])
            if torch.cuda.is_available():
                control = control.to('cuda')
            sigma3 = torch.add(sigma2, control)
            sigma4 = torch.linalg.cholesky(sigma3)
            kij = self.kernel(theta,theta.clone().detach())
            kij.sum().backward()
            f = - theta.grad.clone()
            theta.grad = None
            samples_ori = torch.distributions.multivariate_normal.MultivariateNormal(theta.detach(), scale_tril = sigma4.detach())
            a = samples_ori.sample((echantill,))
            ori = torch.exp(samples_ori.log_prob(a))
            lnp = lntarget(a)
            mem = min(mem, torch.min(lnp).item())
            moy = torch.mean(lnp).item()
            vals = (lnp - moy)
            vals2 = vals.unsqueeze(-1).unsqueeze(0)
            for timer in range(resample_rate):
                sigma.requires_grad_(True)
                theta.requires_grad_(True)
                sigma2 = torch.matmul(sigma, torch.transpose(sigma, 1, 2).clone().detach())
                control = torch.eye(sigma2.size()[1])
                if torch.cuda.is_available():
                    control = control.to('cuda')
                sigma3 = torch.add(sigma2, control)
                sigma4 = torch.linalg.cholesky(sigma3)
                kij = self.kernel(theta,theta.clone().detach())
                kij.sum().backward()
                f = - theta.grad.clone()
                theta.grad = None
                sigma.requires_grad_(True)
                theta.requires_grad_(True)
                samples_curr = torch.distributions.multivariate_normal.MultivariateNormal(theta, scale_tril = sigma4)
                b = samples_curr.sample((echantill,))
                b.requires_grad_(True)
                curr = torch.exp(samples_curr.log_prob(b))
                KL = torch.distributions.kl.kl_divergence(samples_ori, samples_curr)
                divid = torch.div(curr, ori)
                multip = torch.mul(vals, divid)
                loss = torch.sub(multip, KL, alpha = beta)
                loss.sum().backward()
                grad = b.grad.clone().detach()
                grad2 = sigma.grad.clone().detach()
                fish = torch.matmul(grad, torch.transpose(grad, 1, 2)) # estimated fisher info
                tmp = torch.mul(grad, vals.unsqueeze(-1))
                try:
                    fish_inv = torch.inverse(fish)
                except RuntimeError:
                    fish_inv = torch.ones((tmp.size()[0],tmp.size()[1],tmp.size()[1]))
                    if torch.cuda.is_available():
                        fish_inv = fish_inv.to('cuda')
                tmp = torch.matmul(fish_inv, tmp)
                interm2 = torch.mul(grad2, torch.mean(vals,0).unsqueeze(-1).unsqueeze(-1))
                fish2 = torch.matmul(grad2,torch.transpose(grad2, 1, 2)) # estimated fisher info
                try:
                    fish_inv2 = torch.inverse(fish2)
                except RuntimeError:
                    fish_inv2 = torch.eye(interm2.size()[1])
                    if torch.cuda.is_available():
                        fish_inv2 = fish_inv2.to('cuda')
                tmp2 = torch.matmul(fish_inv2, interm2)
                j = torch.matmul(kij,tmp)
                theta_temp = torch.matmul(torch.mul(grad, fish_inv), torch.transpose(grad, 1, 2))
                sigma_temp = torch.matmul(grad2*fish_inv2, torch.transpose(grad2, 1, 2))
                thetalr = torch.sqrt((2*1e-7)/(abs(theta_temp)+torch.full_like(theta_temp,1e-8))) # 1e-7 est le epsilon de la formule de TRPO
                sigmalr = torch.sqrt((2*1e-7)/(abs(sigma_temp)+torch.full_like(sigma_temp,1e-8)))
                #backtracking
                sigma2 = torch.zeros(N, N)
                if torch.cuda.is_available():
                    sigma2 = sigma2.to('cuda')
                sigma2[torch.tril_indices(N, N, offset=0).tolist()] = sigma
                output = torch.distributions.multivariate_normal.MultivariateNormal(theta, scale_tril = sigma2)
                while (torch.distributions.kl.kl_divergence(output, samples)) >= 1e-1:
                    theta.grad = back*(1/theta.size()[0]) * torch.add(torch.mean(thetalr * j, 0).t(), temp(iter) * f)
                    sigma.grad = back*sigmalr * tmp2
                optimizer.step()
                optimizer2.step()
                optimizer.zero_grad()
                optimizer2.zero_grad()
        return theta, mem, sigma

    # SVGD avec échantillonage gaussien, gradient naturel et adaptation de la moyenne de la loi normale

    def update_gradient_gauss_self_adjust_Fisher2(self, particles, lntarget, width = 1, low = 10e9, n_iter = 1, stepsize_med = 1e0, stepsize_var = 1e0, echantill = 1, temp = lambda a:1/a**4, opti = optim.Adagrad):
        if particles is None or lntarget is None:
            raise ValueError('particles or lnprob cannot be None!')
        theta = particles.clone().detach()
        theta.requires_grad_(True)
        optimizer = opti([theta], lr=stepsize_med)
        mem = low
        for iter in range(1,n_iter+1):
            kij = self.kernel(theta,theta.clone().detach())
            kij.sum().backward()
            f = - theta.grad.clone()
            theta.grad = None
            samples = torch.distributions.multivariate_normal.MultivariateNormal(theta, covariance_matrix = width)
            a = samples.sample((echantill,))
            a.requires_grad_(True)
            samples.log_prob(a).sum().backward()
            grad = - a.grad.clone().detach()
            lnp = lntarget(a)
            mem = min(mem, torch.min(lnp).item())
            moy = torch.mean(lnp).item()
            vals = (lnp - moy)
            vals2 = vals.unsqueeze(-1).unsqueeze(0)
            interm = torch.mul(grad, vals2)
            tmp = torch.mean(interm, (0,1)) #grad
            fish = torch.mean(torch.matmul(grad,grad.permute(0, 2, 1)), 0) # estimated fisher info
            try:
                fish_inv = torch.inverse(fish)
            except RuntimeError:
                fish_inv = torch.eye(tmp.size()[0])
                if torch.cuda.is_available():
                    fish_inv = fish_inv.to('cuda')
            tmp = torch.mm(fish_inv, tmp)
            j = torch.mm(kij,tmp)
            theta.grad = (1/theta.size()[0]) * torch.add(j, temp(iter) * f)
            optimizer.step()
            optimizer.zero_grad()
            theta.requires_grad_(True)
        return theta, mem

      # PERMET DE PRODUIRE UNE REPRESENTATION VISUELLE DE L'EVOLUTION DES PARTICULES

    def test_bench(self, nbparticles, lntarget, plottarget, directory, solv = update_gradient_gauss_variable, xbounds = [-2,2], ybounds = [-2,2], n_iter = 100, stepsize = 1e0, width = 1, nb_sample = 1):
        # TODO check the number of arguments taken by lntarget, assuming two for now
        current_working_directory = directory
        shape = (nbparticles,2)
        sam = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor([4.,4.]), 1*torch.eye(shape[1]))
        g = sam.sample((nbparticles,)).clone().detach()
        s = torch.full((shape[1],shape[1]), 0.).fill_diagonal_(1).unsqueeze(0).expand(shape[0],-1,-1)
        xs = np.linspace(xbounds[0], xbounds[1],1000)
        ys = np.linspace(ybounds[0], ybounds[1],1000)
        x,y = np.meshgrid(xs,ys)
        tx = torch.tensor(x)
        ty = torch.tensor(y)
        if torch.cuda.is_available():
            g = g.to('cuda')
            tx = tx.to('cuda')
            ty = ty.to('cuda')
            s = s.to('cuda')
        z = plottarget(tx,ty)
        res = [g.clone().detach(),10e9,s]
        plt.title("Iteration nÂ°0")
        res2 = torch.round(res[0],decimals=4).cpu().detach().numpy()
        plt.xlim(xbounds[0], xbounds[1])
        plt.ylim(ybounds[0], ybounds[1])
        tx = tx.cpu().numpy()
        ty = ty.cpu().numpy()
        z = z.cpu().numpy()
        plt.contourf(tx,ty,z, locator=ticker.LogLocator())
        plt.colorbar()
        plt.scatter(res2[:,0],res2[:,1], color='red')
        plt.savefig(current_working_directory+'\\Output\\Iteration 0.png')
        plt.clf()
        low = 10e9
        for i in range(n_iter):
            step = 1e-3
            res = tmp.update_gradient_gauss_self_adjust_Fisher(res[0], lntarget, low = low, n_iter=1, stepsize_med = step, stepsize_var = step, width = s, echantill = 120, temp = lambda a:1/math.log(100*i+math.exp(1)), opti = optim.SGD)
            plt.title("Iteration n°"+str(1*(i+1)))
            res2 = torch.round(res[0],decimals=4).cpu().detach().numpy()
            plt.xlim(xbounds[0], xbounds[1])
            plt.ylim(ybounds[0], ybounds[1])
            plt.contourf(tx,ty,z, locator=ticker.LogLocator()) # A decommenter si la fonction cible est strictement positive
            # plt.contourf(tx,ty,z) # Dans le cas contraire, décommenter cette ligne
            low = res[1]
            plt.colorbar()
            plt.scatter(res2[:,0],res2[:,1], color='red')
            plt.savefig(current_working_directory+'\\Output\\Iteration '+str(1*(i+1))+'.png')
            plt.clf()
        # print(res[2])

# FONCTIONS DE TESTS
# CELLE ANNOTE SVGD SONT MES CIBLES POUR LE SVGD
# CELLE NOTE FORPLOT SONT POUR L'ARRIERE PLAN DES DESSINS


def DensiteNormale(x,mu,sigma):
    a = 1/(sigma * math.sqrt(2*math.pi))*math.exp(-0.5*((x-mu)/sigma)**2)
    return a

def test_func(x):
    return (1/3)*DensiteNormale(x,-2,1)+(2/3)*DensiteNormale(x,2,1)

def DensiteNormale3(x,mu,sigma):
    tmp1 = torch.sub(x,mu)
    tmp2 = torch.div(tmp1,sigma)
    tmp3 = tmp2.pow(2)
    tmp35 = torch.mul(tmp3,-0.5)
    tmp4 = torch.exp(tmp35)
    cons = 1/(sigma * math.sqrt(2*math.pi))
    result = torch.mul(tmp4,cons)
    return result

def test_func3(x):
    x1i = DensiteNormale3(x,-2,1)
    x2 = DensiteNormale3(x,2,1)
    x1 = torch.mul(x1i,1/3)
    interm_test_func = torch.add(x1,x2,alpha=2/3)
    result = torch.log(interm_test_func)
    return result

def Rosenbrock(x, a = 1, b = 100):
    tmp1 = torch.pow(x[:,0],2)
    tmp2 = torch.sub(x[:,1],tmp1)
    tmp3 = torch.pow(tmp2,2)
    tmp5 = - torch.sub(x[:,0],a)
    tmp6 = torch.pow(tmp5,2)
    res = torch.add(tmp6, tmp3, alpha = b)
    return res

def Rosenbrock_svgd(x, a = 1, b = 100):
    tmp1 = torch.pow(x[:,:,0],2)
    tmp2 = torch.sub(x[:,:,1],tmp1)
    tmp3 = torch.pow(tmp2,2)
    tmp5 = - torch.sub(x[:,:,0],a)
    tmp6 = torch.pow(tmp5,2)
    res = torch.add(tmp6, tmp3, alpha = b)
    return res

def Sphere_forplot(x,y):
    xs = torch.pow(x,2)
    ys = torch.pow(y,2)
    res = torch.add(xs,ys)
    return res

def Sphere(x):
    xs = torch.pow(x[:,0],2)
    ys = torch.pow(x[:,1],2)
    res = torch.add(xs,ys)
    return res

def Sphere_svgd(x):
    sq = torch.pow(x,2)
    res = torch.sum(sq,2)
    return res

def Himmelblau_svgd(x):
    xs = torch.pow(x[:,:,0],2)
    ys = torch.pow(x[:,:,1],2)
    x1 = torch.add(xs,x[:,:,1])
    x2 = torch.sub(x1,11)
    op1 = torch.pow(x2,2)
    y1 = torch.add(x[:,:,0],ys)
    y2 = torch.sub(y1,7)
    op2 = torch.pow(y2,2)
    res = torch.add(op1,op2)
    return res

def Himmelblau(x):
    xs = torch.pow(x[:,0],2)
    ys = torch.pow(x[:,1],2)
    x1 = torch.add(xs,x[:,1])
    x2 = torch.sub(x1,11)
    op1 = torch.pow(x2,2)
    y1 = torch.add(x[:,0],ys)
    y2 = torch.sub(y1,7)
    op2 = torch.pow(y2,2)
    res = torch.add(op1,op2)
    return res

def Himmelblau_forplot(a,b):
    xs = torch.pow(a,2)
    ys = torch.pow(b,2)
    x1 = torch.add(xs,b)
    x2 = torch.sub(x1,11)
    op1 = torch.pow(x2,2)
    y1 = torch.add(a,ys)
    y2 = torch.sub(y1,7)
    op2 = torch.pow(y2,2)
    res = torch.add(op1,op2)
    return res

def Easom(x):
    cx = torch.cos(torch.angle(x[:,0]))
    cy = torch.cos(torch.angle(x[:,1]))
    x2 = torch.sub(x[:,0], torch.pi)
    y2 = torch.sub(x[:,1], torch.pi)
    opx = torch.pow(x2,2)
    opy = torch.pow(y2,2)
    exp1 = - torch.add(opx,opy)
    exp2 = torch.exp(exp1)
    resp = - torch.mul(cx,cy)
    res = torch.mul(resp,exp2)
    return res

def Easom_forplot(a,b):
    cx = torch.cos(torch.angle(a))
    cy = torch.cos(torch.angle(b))
    resp = - torch.mul(cx,cy)
    x2 = torch.sub(a, torch.pi)
    y2 = torch.sub(b, torch.pi)
    opx = torch.pow(x2,2)
    opy = torch.pow(y2,2)
    exp1 = - torch.add(opx,opy)
    exp2 = torch.exp(exp1)
    res = torch.mul(resp,exp2)
    return res

def Rosenbrock_forplot(x, y, a = 1, b = 100):
    return (a-x)**2+(b*(y-(x**2))**2)

def count_val(x, prec, target):
    r = x.clone()
    prec_tens = torch.tensor(np.array([prec,prec]*(r.size()[1]-1)))
    if torch.cuda.is_available():
        prec_tens = prec_tens.to("cuda")
    compteur = 0
    for d in r:
        a = torch.abs(torch.sub(d,target))
        if a.sum() < prec_tens.sum():
            compteur += 1
    return compteur

def rastrigin_svgd(x):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    a, b = x[:,:,0],x[:,:,1]
    A = 10
    f = (
        A * 2
        + (a**2 - A * torch.cos(a * torch.pi * 2))
        + (b**2 - A * torch.cos(b * torch.pi * 2))
    )
    return f

def rastrigin(x):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    a, b = x[:,0],x[:,1]
    A = 10
    f = (
        A * 2
        + (a**2 - A * torch.cos(a * torch.pi * 2))
        + (b**2 - A * torch.cos(b * torch.pi * 2))
    )
    return f

def rastrigin_forplot(x,y):
    A = 10
    x = x
    y = y
    f = (
        A * 2
        + (x**2 - A * torch.cos(x * torch.pi * 2))
        + (y**2 - A * torch.cos(y * torch.pi * 2))
    )
    return f

def sigmoid(x, ceil):
    return ceil/(1+math.exp(-x))




if __name__ == '__main__':
    for resample_rate in [1]:
        nb_iters = 1000//resample_rate
        test_value = 20
        tmp = SVGD()

        #UTILISATION DU CREATEUR D'ANIMATIONS
        # tmp.test_bench(1, lntarget = rastrigin_svgd, plottarget = rastrigin_forplot, directory = 'PATH\\TO\\DIRECTORY', width = torch.eye(2).to('cuda'), solv = tmp.update_gradient_gauss_self_adjust, xbounds = [-6,6], ybounds = [-6,6], n_iter = 100)

        # UTILISATION DE CMA-ES
        # CONTIENT SA PROPRE IMPLEMENTATION DES PROBLEMES, CHANGER ROSENBROCK PAR LA CIBLE CHOISIE (https://evotorch.ai/#how-to)
        # problem = Problem("min", Rosenbrock, initial_bounds=(-100,100), solution_length=2, vectorized=True, device="cuda:0")
        # searcher = CMAES(problem, popsize = test_value, stdev_init=1)
        # # _ = StdOutLogger(searcher, interval=1)
        # pandas_logger = PandasLogger(searcher)
        # # Get the progress of the evolution into a DataFrame with the
        # # help of the PandasLogger, and then plot the progress.
        # searcher.run(nb_iters)
        # pandas_frame = pandas_logger.to_dataframe()
        # # print(pandas_frame.keys())
        # # pandas_frame['iter'].apply(lambda a : 50*a)
        # # print(pandas_frame)
        # # pandas_frame[[x for x in range(0,5000,50)],'pop_best_eval'].plot()
        # plt.plot([x for x in range(0,test_value*nb_iters,test_value)], pandas_frame['pop_best_eval'].to_numpy(), label = "CMA-ES")


        # SVGD evolution

        for k in sympy.divisors(test_value): # K FIXE LE NOMBRE DE PARTICULES DU SVGD
            shape = (k,2)
            shape2 = (2,2)
            sam = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor([4.,4.]), 1*torch.eye(shape[1]))
            g = sam.sample((k,)).clone().detach()
            g2 = torch.randint(10000*73, 10000*77, shape2)
            s2 = torch.full((k,(shape[1]*(shape[1]+1))//2,), 1e-4)
            s = torch.full((shape[1],shape[1]), 1e-4).fill_diagonal_(1).unsqueeze(0).expand(shape[0],-1,-1)
            if torch.cuda.is_available():
                g = g.to('cuda')
                g2 = g2.to('cuda')
                s = s.to('cuda')
                s2 = s2.to('cuda')
            g2 = g2 / 10000
            tmp = SVGD()
            X = [test_value*x for x in range(nb_iters+1)]
            low = 10e9
            low2 = 10e9
            Y1 = []
            Y2 = []
            Z1 = []
            for st in [1e-3]:
                print(resample_rate)
                for j in range(30):
                    print(str(k) + " : " + str(j))
                    sma = 10e9
                    res = [g.clone().detach(),10e9,s]
                    resb = [g.clone().detach(),10e9,s2]
                    res2 = [g.clone().detach()]
                    o = torch.matmul(res[2], res[2].permute(0, 2, 1))
                    o2 = torch.diagonal(o, dim1=1, dim2=2)
                    Y =[torch.min(rastrigin(resb[0])).item()]
                    Yb = [torch.min(rastrigin(resb[0])).item()]
                    Z = [torch.min(rastrigin(res2[0])).item()]
                    f = deque()
                    step = st
                    for i in range(nb_iters):

                        resb = tmp.update_gradient_gauss_self_adjust_Fisher_TRPO(resb[0], rastrigin_svgd, low = low2, n_iter=1, stepsize_med = 1, stepsize_var = 1, width = resb[2], echantill = test_value//k, temp = lambda a:1/math.log(100*i+math.exp(1)), opti = optim.SGD)
                        low2 = resb[1]
                        Z.append(low2)
                    Y1.append(Z)
                med = np.median(Y1,0)
                moy = np.mean(Y1,0)
                mi = np.min(Y1,0)
                ma = np.max(Y1,0)
                # plt.plot(X, np.where(med>1e-12, med, 1e-12), label = "SVGD_" + str(k) + "-particles_" + str(test_value//k) + "ech_medianvalue_TRPO")
                plt.plot(X, np.mean(Y1,0), label = "SVGD_" + str(k) + "-particles_" + str(test_value//k) + "ech_meanvalue_TRPO")
                # plt.plot(X, np.mean(Z1,0), label = "SVGD_" + str(k) + "-particles_" + str(test_value//k) + "ech_meanvalue_TRPO")
                # plt.plot(X, np.where(mi>1e-12, mi, 1e-12), label = "SVGD_" + str(k) + "-particles_" + str(test_value//k) + "ech_minvalue_TRPO")
                # plt.plot(X, np.where(ma>1e-12, ma, 1e-12), label = "SVGD_" + str(k) + "-particles_" + str(test_value//k) + "ech_maxvalue_TRPO")
                # print(np.std(Y1, 0))
            plt.legend()
            plt.xlabel("Nombre d'appels à la fonction objectif")
            plt.ylabel("Valeur Minimale découverte moyenne (30 runs)")
            plt.title("SVGD Rastrigin Function temp Fisher SGD, PPO " + str(resample_rate) +" steps")
            plt.yscale('log')
            plt.savefig("curve" + str(resample_rate) + str(k) + ".png", format="png")
            plt.clf()



    # POUR VOIR L'EVOLUTION DE CMA-ES, DECOMMENTER CE QUI SUIT
    # res = searcher.population.access_values()
    # plt.scatter(res[:,0].cpu().numpy(),res[:,1].cpu().numpy())
    # xs = np.linspace(-5.12,5.12,1000)
    # ys = np.linspace(-5.12,5.12,1000)
    # x,y = np.meshgrid(xs,ys)
    # tx = torch.tensor(x)
    # ty = torch.tensor(y)
    # if torch.cuda.is_available():
    #     tx = tx.to('cuda')
    #     ty = ty.to('cuda')
    # plt.title("Iteration n°0")
    # z = rastrigin_forplot(tx,ty)
    # plt.xlim(-5.12,5.12)
    # plt.ylim(-5.12,5.12)
    # tx = tx.cpu().numpy()
    # ty = ty.cpu().numpy()
    # z = z.cpu().numpy()
    # res = searcher.population.access_values()
    # plt.contourf(tx,ty,z)
    # plt.colorbar()
    # plt.scatter(res[:,0].cpu().numpy(),res[:,1].cpu().numpy(),color='red')
    # plt.savefig('PATH\\TO\\DIRECTORY\\Iteration 0.png')
    # plt.clf()
    # for i in range(100):
    #     print(i)
    #     searcher.run(1)
    #     res = searcher.population.access_values()
    #     plt.xlim(-5.12,5.12)
    #     plt.ylim(-5.12,5.12)
    #     plt.contourf(tx,ty,z)
    #     plt.colorbar()
    #     plt.title("Iteration n°"+str((i+1)))
    #     plt.scatter(res[:,0].cpu().numpy(),res[:,1].cpu().numpy(),color='red')
    #     plt.savefig('PATH\\TO\\DIRECTORY\\Iteration '+str((i+1))+'.png')
    #     plt.clf()

