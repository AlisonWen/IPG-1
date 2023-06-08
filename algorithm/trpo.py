# modified from https://github.com/ikostrikov/pytorch-trpo/blob/master/main.py 
# and           https://github.com/ikostrikov/pytorch-trpo/blob/master/conjugate_gradients.py

from utils import set_flat_params_to, get_flat_params_from, normal_log_density, get_flat_grad_from
# from torch.autograd
import numpy as np
import scipy
import torch


def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.contiguous().view(-1) for t in g])
    return g


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def linesearch(model, f, x, fullstep,
               expected_improve_rate,
               max_backtracks=10,
               accept_ratio=.1):
    fval = f()
    print("fval before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f()
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            print("fval after", newfval.item())
            return True, xnew
    return False, x


# Set up function for computing TRPO loss (only for logging)
# don't know how to implemented yet.....
def compute_TRPO_loss_pi(self, data, inter_nu, use_cv, plot=True):
    # act, obs, adv = data['act'], data['ods'], data['adv']
    # with torch.no_grad():
    #     if self.discrete:
    #         raise NotImplementedError("Categoical agent for TRPO not implemented yet")
    #     else:
    #         pi = self.pi.mu_net(obs)
    #         pi = self.pi.mu_net(obs)
    #         log_stds = self.pi.log_std.expand_as(pi)
    #         stds = torch.exp(log_stds)
    #         log_prob = normal_log_density(act, pi, log_stds, stds)
    #         act_loss = -adv * torch.exp(log_prob - fix)
    return torch.tensor([0]), dict()
            

# The KL-divergence between two gaussian distribution
def kl_div_gaussian(pi_0, std_0, log_std_0, pi_1, std_1, log_std_1):
    return log_std_1 - log_std_0 + (std_0.pow(2) + (pi_0 - pi_1).pow(2)) / (2.0 * std_1.pow(2)) - 0.5


# The KL-divergence between two categorical distribution
def kl_div_categorical(dist_0, dist_1):
    return (dist_0 * (dist_0.log() - dist_1.log()))


def surrogate_loss(new_log_prob, old_log_prob, learning_signal):
    # the surrogate function L_{\pi_{\theta_k}} 
    return -(torch.exp(new_log_prob - old_log_prob) * learning_signal).mean()


# Set up function for TRPO update
def trpo_update(ac, data_on, data_off, inter_nu, use_cv, beta, max_kl, damping):
    
    obs, act, adv = data_on['obs'], data_on['act'], data_on['adv']
    
    # Caluculate values for interpolated gradient
    if use_cv:
        print("Using Control Variate...")
        crit_based_adv = ac.get_control_variate(data_on)  #returns q(s,t) - E[q(s,t)] for ~1
        learning_signals = (adv - crit_based_adv) * (1 - inter_nu)
    else:
        learning_signals = adv * (1 - inter_nu)  # line 10-12 IPG pseudocode

    if use_cv:
        b = 1
    else:
        b = inter_nu

    # Calculate off policy gradient
    for param in ac.pi.parameters():
        # clear the gradient of pi 
        if param.grad is not None:
            param.grad.zero_()
        else:
            param.grad = torch.zeros_like(param.data)


    # backward to get gradient
    if beta == 'off_policy_sampling':
        pi_off_loss = ac.compute_loss_off_pi(data_off)
    elif beta == 'on_policy_sampling':
        pi_off_loss = ac.compute_loss_off_pi(data_on)
    else:
        assert False, f"'{beta}' is no valid value for beta"


    # get the off policy gradient 
    pi_off_loss.backward()
    pi_off_grad = get_flat_grad_from(ac.pi)


    # Calculate TRPO gradient

    # Although original code uses the LBFGS to optimize the value loss,
    # we do it in core.py through gradient descent 
    if ac.discrete:
        # raise NotImplementedError("no categorical implementation for trpo")
        dist = ac.pi(obs)
        dist = torch.distributions.utils.clamp_probs(dist)

        prob = dist[range(dist.shape[0   ]), act]


    else:
        pi = ac.pi.mu_net(obs)
        log_stds = ac.pi.log_std.expand_as(pi)
        stds = torch.exp(log_stds)
        
        # The log probability of the actions
        log_prob = normal_log_density(act, pi, log_stds, stds)
        fixed_log_prob = log_prob.detach()

        # create calculation graph for grad to cal grad_grad
        KL = kl_div_gaussian(pi.detach(), stds.detach(), log_stds.detach(), pi, stds, log_stds)
        grad_KL = torch.autograd.grad(KL.mean(), ac.pi.parameters(), retain_graph=True, create_graph=True)
        flat_grad_KL = torch.cat([grad.view(-1) for grad in grad_KL])


        def Fvp(v):
            KL_v = (flat_grad_KL * v).sum()
            grad_KL_v = torch.autograd.grad(KL_v, ac.pi.parameters(), retain_graph=True)
            flag_grad_KL_v = torch.cat([grad.contiguous().view(-1) for grad in grad_KL_v]).detach()

            return flag_grad_KL_v + v * damping


        def get_loss():
            with torch.no_grad():
                pi = ac.pi.mu_net(obs)
                log_stds = ac.pi.log_std.expand_as(pi)
                stds = torch.exp(log_stds)
            
            log_prob = normal_log_density(act, pi, log_stds, stds)
            return surrogate_loss(log_prob, fixed_log_prob, learning_signals).detach()
        

        loss = surrogate_loss(log_prob, fixed_log_prob, learning_signals)
        grad_loss = torch.autograd.grad(loss, ac.pi.parameters(), retain_graph=True)
        flat_grad_loss = torch.cat([grad.view(-1) for grad in grad_loss]).detach()
        print(flat_grad_loss.shape)

        stepdir = conjugate_gradients(Fvp, -flat_grad_loss, 10)

        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

        lm = torch.sqrt(shs / max_kl)
        fullstep = stepdir / lm[0]

        neggdotstepdir = (-flat_grad_loss * stepdir).sum(0, keepdim=True)
        print(("lagrange multiplier:", lm[0], "grad_norm:", flat_grad_loss.norm()))

        prev_params = get_flat_params_from(ac.pi)

        success, new_params = linesearch(ac.pi, get_loss, prev_params, fullstep, neggdotstepdir / lm[0])

        set_flat_params_to(ac.pi.mu_net, new_params)# + b * pi_off_grad)
