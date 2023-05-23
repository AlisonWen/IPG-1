import torch


# Set up function for computing PPO policy loss
def compute_PPO_loss_pi(self, data, inter_nu, use_cv, clip_ratio=0.2, plot=False):
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
    if plot:
        learning_signals = adv * (1 - inter_nu)
    else:
        if use_cv:
            print("Using Control Variate...")
            crit_based_adv = self.get_control_variate(data)  #returns q(s,t) - E[q(s,t)] for ~1
            learning_signals = (adv - crit_based_adv) * (1 - inter_nu)
        else:
            learning_signals = adv * (1 - inter_nu)  # line 10-12 IPG pseudocode
    
    # Policy loss
    pi, logp = self.pi(obs, act)  # pi is a distribution
    ratio = torch.exp(logp - logp_old)  # same thing
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * learning_signals  # ratio, min, max
    loss_pi = -(torch.min(ratio * learning_signals, clip_adv)).mean()

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()  # with math, this is equal to dist division, like IS.
    ent = pi.entropy().mean().item()
    clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
    return loss_pi, pi_info
