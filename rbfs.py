#!/usr/bin/env python

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
    'inv_discount_factor': hp.loguniform('inv_discount_factor', np.log(1e-5), np.log(1e-1)),
    'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}


def make_experiment(
    # Path needs to have this format or hypersearch breaks
        exp_id=1, path="./{domain}/{agent}/{representation}/",
        boyan_N0=330,
        initial_learn_rate=0.219,
        lambda_=0.5547,
        resolution=7.0, num_rbfs=86.0,
        epsilon=0.4645,
        inv_discount_factor=3.186e-5):
    opt = {}
    opt["exp_id"] = exp_id
    opt["max_steps"] = 300000
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
    experiment = make_experiment(id)
    experiment.run(visualize_learning=True, visualize_performance=1)
    experiment.plot()
    # experiment.save()
