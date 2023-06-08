

# Set up function for computing PG policy loss
def compute_PG_loss_pi(self, data, inter_nu, use_cv, plot=False):
    obs, act, adv, = data['obs'], data['act'], data['adv']
    if plot:
        # when plotting, don't use CV
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

    return -(logp * learning_signals).mean(), dict()