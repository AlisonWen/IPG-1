from torch.utils.tensorboard import SummaryWriter
from core import MLPActorCritic, QFunction
from buffers import ON_BUFFER, OFF_BUFFER
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import Adam
from copy import deepcopy
import gymnasium  as gym
import numpy as np
import functools
import datetime
import torch
import core
import time
import os

def evaluate(env, policy, max_timesteps, n_traj=1):
    print(f"Evaluating for {n_traj} episode(s), with {max_timesteps} time-steps.")
    rewards = []
    for j in range(n_traj):
        o, _ = env.reset()
        d, ep_ret, ep_len = False, 0, 0
        while not (d or (ep_len == max_timesteps)):
            if policy.discrete:
                a, _, _ = policy.step(torch.as_tensor(o, dtype=torch.float32))
            else:
                a, _, _, _, _ = policy.step(torch.as_tensor(o, dtype=torch.float32))
            #use mu_net for evaluation??
            o, r, ter, tru, _ = env.step(a)
            d = ter or tru
            ep_ret += r
            ep_len += 1
        rewards.append(ep_ret)
    print("Done evaluating")
    return sum(rewards) / len(rewards)


def soft_update(targ_model, model, tau=0.001):
    with torch.no_grad():
        for p, p_targ in zip(model.parameters(), targ_model.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(1.0 - tau)
            p_targ.data.add_(tau * p.data)


def fit_qf(ac, opt, trian_qf_iters, buf_off, batch_size, gamma, tau):
    for i in range(trian_qf_iters):
        data_off = buf_off.sample_batch(batch_size)
        opt.zero_grad()
        loss_qf = ac.compute_loss_qf(data_off, gamma)
        loss_qf.backward()
        opt.step()
        soft_update(ac.qf_targ, ac.qf, tau)


def fit_vf(ac, opt, train_vf_iters, data_on):
    for i in range(train_vf_iters):
        opt.zero_grad()
        loss_v = ac.compute_loss_v(data_on)
        loss_v.backward()
        opt.step()


def fit_pi(ac, opt, data_on, data_off, train_pi_iters, inter_nu, use_cv, beta):
    for i in range(train_pi_iters):
        opt.zero_grad()
        # update with previuos loss only
        loss_pi, pi_info = ac.compute_loss_pi(data_on, inter_nu, use_cv)

        if beta == 'off_policy_sampling':
            loss_off_pi = ac.compute_loss_off_pi(data_off)
        elif beta == 'on_policy_sampling':
            loss_off_pi = ac.compute_loss_off_pi(data_on)
        else:
            assert False, f"'{beta}' is no valid value for beta"

        if use_cv:
            b = 1
        else:
            b = inter_nu

        loss_pi_inter = loss_pi + b * loss_off_pi
        loss_pi_inter.backward()

        opt.step()


def main(conf, hparams):

    # logger config
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    nowname = hparams.exp_name + now
    if not os.path.isdir(hparams.exp_dir):
        os.mkdir(hparams.exp_dir)
    logdir = os.path.join(hparams.exp_dir, nowname)
    if not os.path.isdir(logdir):
        os.mkdir(logdir)
    
    weight_dir = os.path.join(logdir, 'weights')
    tensorboard_dir = os.path.join(logdir, 'tensorboard')
    
    if not os.path.isdir(weight_dir):
        os.mkdir(weight_dir)

    if not os.path.isdir(tensorboard_dir):
        os.mkdir(tensorboard_dir)

    writer = SummaryWriter(log_dir=tensorboard_dir)

    # dump configuration
    config_filename = os.path.join(logdir, "config.yaml")

    # unused info
    hparams.pop('exp_dir')
    hparams.pop('exp_name')

    # dumps to file
    conf.hparams = hparams
    with open(config_filename, "w") as f:
        OmegaConf.save(conf, f)

    # seeding
    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)

    # set up env
    env = gym.make(hparams.env)
    max_ep_len = env._max_episode_steps

    # Set up model
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    ac = MLPActorCritic(env.observation_space, env.action_space, **conf.model.params)

    # select algorithm
    if hparams.algo == 'IPG':
        ac.compute_loss_pi = functools.partial(core.compute_PG_loss_pi, ac)
    elif hparams.algo == 'PPO':
        ac.compute_loss_pi = functools.partial(core.compute_PPO_loss_pi, ac, clip_ratio=hparams.clip_ratio)
    elif hparams.algo == 'TRPO':
        ac.compute_loss_pi = functools.partial(core.compute_PG_loss_pi, ac)
    else:
        raise ValueError("Unknown algorithm")
    
    # count total parameters
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    print('Number of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    #initialize buffers
    buf_on = ON_BUFFER(obs_dim, act_dim, max_ep_len * hparams.num_cycles, hparams.gamma, hparams.lam)
    buf_off = OFF_BUFFER(obs_dim=obs_dim, act_dim=act_dim, size=int(hparams.off_buffer_size))
     
    # Set up optimizers for policy and value function
    pi_opt = Adam(ac.pi.parameters(), lr=hparams.pi_lr)
    vf_opt = Adam(ac.v.parameters(), lr=hparams.vf_lr)
    qf_opt = Adam(ac.qf.parameters(), lr=hparams.qf_lr)

    def update(epoch):
        data_on = buf_on.get()
        data_off = buf_off.sample_batch(hparams.batch_sample_size)
        with torch.no_grad():
            pi_l_old, pi_info_old = ac.compute_loss_pi(data_on, hparams.inter_nu, hparams.use_cv, plot=True)
            pi_l_old = pi_l_old.item()
            v_l_old = ac.compute_loss_v(data_on).item()
            qf_l_old = ac.compute_loss_qf(data_off, hparams.gamma).item()
            pi_l_off_old = ac.compute_loss_off_pi(data_off).item()
            inter_l_old = pi_l_old + hparams.inter_nu * pi_l_off_old
        eval_reward = evaluate(env, ac, max_ep_len)

        # write with tensorbard
        writer.add_scalar("Loss/train (policy)", pi_l_old, epoch)
        writer.add_scalar("Loss/train (value)", v_l_old, epoch)
        writer.add_scalar("Loss/train (q function)", qf_l_old, epoch)
        writer.add_scalar("Loss/train (off policy)", pi_l_off_old, epoch)
        writer.add_scalar("Loss/train (inter)", inter_l_old, epoch)
        writer.add_scalar("Return/Epoch", np.array(average_return).mean(), epoch)
        writer.add_scalar("Return/Epoch", eval_reward, epoch)

        # fit parameter phi of QF  
        fit_qf(ac, qf_opt, hparams.train_qf_iters, buf_off, hparams.batch_sample_size, hparams.gamma, hparams.tau)

        # freeze q network parameters.
        ac.qf.freeze()

        # policy learning
        fit_pi(ac, pi_opt, data_on, data_off, hparams.train_pi_iters, hparams.inter_nu, hparams.use_cv, hparams.beta)

        # Value function learning
        fit_vf(ac, vf_opt, hparams.train_vf_iters, data_on)

        # unfreeze q net params for next iteration
        ac.qf.unfreeze()

    # Main loop
    for epoch in range(hparams.epochs):
        # collect data
        average_return = []
        for _ in range(hparams.num_cycles):
            # reset env
            obs, _ = env.reset()
            ep_ret, ep_len = 0, 0
            for t in range(max_ep_len):
                if ac.discrete:
                    a, v, logp = ac.step(torch.as_tensor(obs, dtype=torch.float32)) 
                else:
                    a, v, logp, mean, std = ac.step(torch.as_tensor(obs, dtype=torch.float32)) 
                next_obs, r, terminated, truncated, _ = env.step(a)
                d = terminated or truncated
                
                ep_ret += r
                ep_len += 1

                # save to buffer
                buf_on.store(obs, a, r, v, logp)
                buf_off.store(obs, a, r, next_obs, d)

                obs = next_obs

                if d:
                    if truncated:
                        if ac.discrete:
                            _, v, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                        else:
                            _, v, _, _, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                    else:
                        v = 0
                    buf_on.finish_path(v)  # computes GAE after episode is done, we want this after all the gather

                    average_return.append(ep_ret)

        # save model
        if epoch > 0 and ((epoch % hparams.save_freq == 0) or (epoch == hparams.epochs - 1)):
            filename = f'epochs={epoch}.pth'
            torch.save(ac.state_dict(), os.path.join(weight_dir, filename))

        # updating model
        update(epoch)

    

    writer.close()
            

        
if __name__ == '__main__':
    import argparse 

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, default="./configs/base.yaml",
                        help="the config file for base model class")

    parser.add_argument('--env', type=str, default='HalfCheetah-v4',
                        help="environment to train on")
    # parser.add_argument('--env', type=str, default='Ant-v3')
    
    parser.add_argument("--batch_sample_size", type=int, default=64,
                        help="")
    
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="the discount fafctor")

    parser.add_argument('--lam', type=float, default=0.97,
                        help="lam")

    parser.add_argument("--tau", type=float, default=0.001,
                        help="tau")

    parser.add_argument('--seed', '-s', type=int, default=69,
                        help="seed for random generator")
      
    parser.add_argument('--epochs', type=int, default=250,
                        help="number of epochs to run")
     
    parser.add_argument('--num_cycles', type=int, default=4,
                        help="number of cycles to run in one epoch")
    
    parser.add_argument('--use_cv', action="store_true",
                        help="use control variate or not")

    parser.add_argument('--cv_type', default='reparam_critic_cv', 
                        help='determines which cv to use. Possible vals: '
                             '"reparam_critic_cv" and "taylor_exp_cv"')

    parser.add_argument('--beta', default='off_policy_sampling', 
                        help='determines sampling for off-policy loss. '
                             'Possible values: "off_policy_sampling", "on_policy_sampling"')
    
    parser.add_argument('--train_pi_iters', type=int, default=80,
                        help="number of steps to update the actor in one update")  # default 80 for all of them
    
    parser.add_argument('--train_vf_iters', type=int, default=80,
                        help="number of steps to update the critic in one update")
    
    parser.add_argument('--train_qf_iters', type=int, default=4000,
                        help="number of steps to update the Q function in one update")  # same as batch size of on-policy collection
    
    parser.add_argument("--off_buffer_size", type=int, default=1000000,
                        help="off_buffer_size")

    parser.add_argument('--inter_nu', type=float, default=1.0,
                        help="the interpolate factor")
    
    parser.add_argument('--exp_name', type=str, default='ipg',
                        help="the name for the experiments")
    
    parser.add_argument('--exp_dir', type=str, default='./exp',
                        help="the directory for the experiment")
    
    parser.add_argument("--pi_lr", type=float, default=3e-4,
                        help="learning rate for the policy")
    
    parser.add_argument("--vf_lr", type=float, default=1e-3,
                        help="learning rate for the value function")
    
    parser.add_argument("--qf_lr", type=float, default=1e-3,
                        help="learning rate for the Q-function")
    
    parser.add_argument("--save_freq", type=int, default=10,
                        help="save frequency")

    parser.add_argument("--algo", type=str, default="IPG",
                        help="algorithm used")
    
    # only for PPO
    parser.add_argument("--clip_ratio", type=float, default=0.2,
                        help="clip ratio for PPO")

    opts = parser.parse_args()

    # read config
    config_path = opts.config
    config = OmegaConf.load(config_path)
    hparams = config.pop("hparams", OmegaConf.create())

    # overwrite hparams from command prompt
    hparams_cmd = vars(opts)
    hparams = OmegaConf.merge(hparams_cmd, hparams)

    main(config, hparams)
