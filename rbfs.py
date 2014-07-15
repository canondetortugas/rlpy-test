#!/usr/bin/env python

"""
Cart-pole balancing with independent discretization
"""
from brick_domain import BrickDomain
from rlpy.Agents import SARSA, Q_LEARNING
from rlpy.Representations import *
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

import matplotlib.pyplot as plt

param_space = {
    "num_rbfs": hp.qloguniform("num_rbfs", np.log(1e1), np.log(1e4), 1),
    'resolution': hp.quniform("resolution", 3, 30, 1),
    'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
    'lambda_': hp.uniform("lambda_", 0., 1.),
    'epsilon': hp.uniform("epsilon", 0.05, 0.5),
    'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}


def make_experiment(
    # Path needs to have this format or hypersearch breaks
        exp_id=1, path="./{domain}/{agent}/{representation}/",
        boyan_N0=340.55,
        initial_learn_rate=.07066,
        lambda_=0.087,
        resolution=6.0, num_rbfs=24.0,
        epsilon=0.386477):
    opt = {}
    opt["exp_id"] = exp_id
    opt["max_steps"] = 30000
    opt["num_policy_checks"] = 20
    opt["checks_per_policy"] = 10
    opt["path"] = path

    domain = BrickDomain()
    opt["domain"] = domain
    representation = RBF(domain, num_rbfs=int(num_rbfs),
                         resolution_max=resolution, resolution_min=resolution,
                         const_feature=False, normalize=True, seed=exp_id)
    policy = eGreedy(representation, epsilon=epsilon)
    opt["agent"] = Q_LEARNING(
        policy, representation, discount_factor=domain.discount_factor,
        lambda_=lambda_, initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    plt.ion()
    # import ipdb
    # ipdb.set_trace()
    # from rlpy.Tools.run import run_profiled
    # run_profiled(make_experiment)
    id = int(np.random.rand()*200)
    experiment = make_experiment(1)
    experiment.run(visualize_learning=True, visualize_performance=1)
    experiment.plot()
    # experiment.save()
