from rlpy.Tools.hypersearch import find_hyperparameters

best, trials = find_hyperparameters("./rbfs.py",
                                    "./RBFs_hypersearch",
                                    max_evals=10, parallelization="joblib",
                                    trials_per_point=5)
print "Best parameters: ", best
