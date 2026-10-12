import torch
import sys
import os
import numpy as np
stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
torch.set_printoptions(precision=2)
sys.stderr = stderr_backup


def learnQorthogonal(targets, covariates, n_iterations, verbose, num_timepoints = None):
    # -------------------- EXTRACT INFO FROM INPUTS -------------------- #
    if num_timepoints is None:
        num_timepoints = len(covariates)

    # unpacking inputs
    covariate_matrices = covariates[0:num_timepoints] 
    target_vectors = targets[0:num_timepoints]
    target_vectors = [d.flatten() for d in target_vectors]

    # rows (num outcomes)
    Y_1 = covariate_matrices[0]
    # m is number of outcomes
    m = Y_1.shape[0] 
    # this gets the number of donors
    num_donors = Y_1.shape[1] 
    # Embedding dimension

    # -------------------- SET UP VARIABLES AND PROBLEM -------------------- #

    # initialize vector q, what we are optimizing - requires_grad=True tracks gradients
    q_raw = torch.zeros(m, dtype=torch.float64, requires_grad=True)
    # create w vector as a cp variable whose dimension is the number of donors
    w_var = cp.Variable(num_donors)

    # Create parameters for the embedded covariates
    qY_params = [cp.Parameter((num_donors)) for _ in range(len(target_vectors))]
    # Create parameters for the embedded targets
    qd_params = [cp.Parameter((1)) for _ in range(len(target_vectors))]

    # Calculate difference between targets and SC of covariates
    discrepancy = [cp.sum_squares(qd - (qY_param @ w_var)) for qY_param, qd in zip(qY_params, qd_params)]
    
    # how to get sum of q to be constrained to be 1 and non-negative?
    constraints = [cp.sum(w_var) == 1, w_var >= 0]

    objective = cp.Minimize(sum(discrepancy))
    prob = cp.Problem(objective, constraints)
    layer = CvxpyLayer(prob, parameters=qY_params + qd_params, variables=[w_var])

    # -------------------- OPTIMIZATION -------------------- #
    optimizer = torch.optim.Adam([q_raw], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    for _ in range(n_iterations):
        optimizer.zero_grad()
        q = torch.softmax(q_raw, dim=0)
        # transform the covariates using q
        qY_list = [q @ Y for Y in covariate_matrices]
        # transform the targets using q
        qd_list = [q @ d for d in target_vectors]

        # solve for w given the vector q
        qd_list_reshaped = [qd.unsqueeze(0) for qd in qd_list]
        w_sol, = layer(*qY_list, *qd_list_reshaped) 
        
        # loss using the optimal w for this q
        # qd is scalar (shape ()), while the CVXPY layer receives qd.unsqueeze(0) (shape (1,)).
        loss = sum(torch.sum((qd - qY @ w_sol)**2) for qd, qY in zip(qd_list, qY_list)) 
        # this is where q is updated
        loss.backward()                 
        optimizer.step()
        scheduler.step()

    # --- Results --- #
    if (verbose == True):
        print(f"\nFinal Loss: {loss.item():.8f}")
        print(f"Final w:    {w_sol.detach().numpy().round(4)}")
        print(f"Final Q:\n {q.detach().numpy().round(4)}")

    q_final = torch.softmax(q_raw, dim=0).detach().numpy()
    w_final = np.round(w_sol.detach().numpy(), decimals=2)
    return q_final, w_final
