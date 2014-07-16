#!/usr/bin/env python

"""
Cart-pole balancing with independent discretization
"""
from brick_domain import BrickDomain
from rlpy.Agents import SARSA, Q_LEARNING, NaturalActorCritic
from rlpy.Representations import *
from rlpy.Policies import eGreedy, GibbsPolicy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

import matplotlib.pyplot as plt

param_space = {
    "num_rbfs": hp.qloguniform("num_rbfs", np.log(1e1), np.log(500), 1),
    'resolution': hp.quniform("resolution", 3, 30, 1),
   'lambda_': hp.uniform("lambda_", 0., 1.),
    'inv_discount_factor': hp.loguniform('inv_discount_factor', np.log(1e-5), np.log(1e-1)),
    'learn_rate': hp.loguniform("learn_rate", np.log(5e-2), np.log(1e3)),
    'forgetting_rate': hp.uniform('forgetting_rate', 0.0, 1.0)}

max_steps = 1000
min_steps = 100

def make_experiment(
    # Path needs to have this format or hypersearch breaks
        exp_id=1, path="./{domain}/{agent}/{representation}/",
        learn_rate=0.1188,
        lambda_=0.1154,
        resolution=6.0, num_rbfs=17.0,
        inv_discount_factor=0.0097,
        forgetting_rate=0.38):
    opt = {}
    opt["exp_id"] = exp_id
    opt["max_steps"] = 3000000
    opt["num_policy_checks"] = 20
    opt["checks_per_policy"] = 10
    opt["path"] = path

    discount_factor = 1.0 - inv_discount_factor

    domain = BrickDomain()
    domain.discount_factor = discount_factor
    opt["domain"] = domain
    representation = RBF(domain, num_rbfs=int(num_rbfs),
                         resolution_max=resolution, resolution_min=resolution,
                         const_feature=False, normalize=True, seed=exp_id)
    policy = GibbsPolicy(representation)
    opt["agent"] = NaturalActorCritic(
        policy, representation,  discount_factor=domain.discount_factor,
        lambda_=lambda_, learn_rate=learn_rate, forgetting_rate=forgetting_rate,
        min_steps_between_updates=min_steps, max_steps_between_updates=max_steps)
    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    plt.ion()
    # import ipdb
    # ipdb.set_trace()
    # from rlpy.Tools.run import run_profiled
    # run_profiled(make_experiment)
    # id = int(np.random.rand()*200)
    experiment = make_experiment(1)
    experiment.run(visualize_learning=True, visualize_performance=1)
    experiment.plot()
    # experiment.save()
