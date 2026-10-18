import torch
import numpy as np
rng = np.random.default_rng(seed=40)
torch.manual_seed(212)
import json

# --------------------- Parameters --------------------- #
# factors
K = 3
# Units
N = 10
T=7
# number of datasets to generate
S=100
# number of total outcomes 
m=10
# -----------------------------------------------------

# --------------------- Factors ---------------------
def generate_factors():
    scale = 1
    # lists of normal rvs
    eps1 = rng.normal(0, scale, size=T)
    eps2 = rng.normal(0, scale, size=T)
    eps3 = rng.normal(0, scale, size=T)

    f1, f2, f3 = np.zeros(T), np.zeros(T), np.zeros(T)

    for t in range(T):
        # from Rong's suggestion
        f1[t] = eps1[t] + (0.8*f1[t-1] if t >= 1 else 0)
        f2[t] = eps2[t] + (-0.6*f2[t-1] + 0.8*eps2[t-1] if t >= 1 else 0)
        f3[t] = eps3[t] + (0.9*eps3[t-1] if t >= 1 else 0) + (0.4*eps3[t-2] if t >= 2 else 0)
    return f1, f2, f3

# -----------------------------------------------------

# --------------------- Loadings -------------------------------

# a function to create loadings
def generate_loadings(m):
    rho = 0.9
    Sigma = (1 - rho) * np.eye(m) + rho * np.ones((m, m))

    # list in which to collect the loadings
    loadings_all = []

    # mean vector for each factor - length of this vector is M, and we have K of them
    # mean vector is quite similar for each K 
    means = [rng.normal(0,0.5, size=m) for _ in range(K)]
    # for each unit, draw loadings for each factor
    for i in range(N):
        # the loadings for unit i by the end of the loop.
        loadings = [] # K lists of length M
        for k in range(K):
            # draw loadings from normal distribution with mean mean_k
            lambda_ik = rng.multivariate_normal(mean=means[k], cov=Sigma) 
            loadings.append(lambda_ik)
        # add the loadings for unit i to the collection of all loadings.
        loadings_all.append(loadings)
    
    return loadings_all
#--------------------------------------------------------------

# --------------------- Outcomes plus Noise ---------------------

# function to multiply factors by loadings
def generate_outcomes(loadings_all, f1,f2,f3):
    # extract the factors over time
    factors = [np.array([f1[t], f2[t], f3[t]]) for t in range(T)]
    # to store all outcomes
    outcomes = []

    # for each unit
    for i in range(N):
        # initialize list to store outcomes
        outcomes_i = []
        # calculate the outcomes at time t
        for t in range(T):
            # the outcome at time t is the factors at t times the loadings
            f_t = factors[t]
            # K x M (3 x 2). So Lambda_i.T = (2,3). f1[t],f2[t],f3[t] is (3,)?
            Lambda_i = np.array(loadings_all[i]) 
            Y_it = np.dot(Lambda_i.T, f_t) 
            # print(f"{i},{t}, {Y_it}")
            outcomes_i.append(Y_it)
        # add the outcomes for unit i to the collection of all outcomes
        outcomes.append(outcomes_i)
    # transform to np array
    outcomes=np.array(outcomes)

    # turn the outcomes in np arrays
    for i in range(len(outcomes)):
        outcomes[i] = torch.tensor(outcomes[i]) # shape (T, M)

    # add iid noise to the outcomes
    outcomes = outcomes + rng.normal(0, 0.1, size=outcomes.shape)
    
    # print("all outcomes:", outcomes)
    return outcomes

#-------------------------------------------------------------------------

#------------------ Shape the outcomes to be fed to the model ------------
def reshape_data(outcomes):
    # split outcomes into treated outcomes and control outcomes
    treated = torch.tensor(outcomes[0]) # (1, T, M)
    control = torch.tensor(outcomes[1:]) # (N-1, T, M)

    # break outcomes down into observations at each time point:
    treated_outcomes_list = []
    for t in range(T):
        treated_t = treated[t, :].unsqueeze(1) # (M, 1), <class 'torch.Tensor'>
        treated_outcomes_list.append(treated_t)

    # print("treated:", treated_outcomes_list)

    # break control outcomes down into observations at each time point:
    control_outcomes_list = []
    for t in range(T):
        # grab the all outcomes from all control units at time t
        control_t = control[:, t, :].T # (N-1, M), <class 'torch.Tensor'>
        control_outcomes_list.append(control_t)

    return treated_outcomes_list, control_outcomes_list

    # print("control:", control_outcomes_list)

# ------------------------- Generate Data ---------------------- #
# chain together the data generating functions
def generate_data(m):
    f1,f2,f3 = generate_factors()
    treated_outcomes_list, control_outcomes_list = reshape_data(generate_outcomes(generate_loadings(m),f1,f2,f3))
    return treated_outcomes_list, control_outcomes_list

# function that returns data dict
def make_data_dict(S=S, m=m):
    return {s: generate_data(m) for s in range(S)}

if __name__ == "__main__":
    data_dict = make_data_dict()



    
