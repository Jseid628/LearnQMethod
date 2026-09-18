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


# should put an option to just fix the weights at 1/n
def learnQorthogonal(targets, covariates, embedding_dim, n_iterations, reg_Q, reg_w, verbose, num_timepoints = None, init_Q = "eye", fixed_weights=False):
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
    D = embedding_dim

    minDim = min(num_donors, D)
    eye_for_ortho_error = torch.eye(minDim, dtype=torch.float64)

    # Q is what we're optimizing - requires_grad=True tracks gradients
    torch.manual_seed(215)
    if init_Q == "eye":
        # Initialize Q as an identity matrix, which is orthogonal
        Q = torch.eye(num_donors, D, dtype=torch.float64, requires_grad=True)   
    else:
        Q = torch.randn(num_donors, D, dtype=torch.float64, requires_grad=True)

    lambda_l2_Q = reg_Q
    lambda_l2_w = reg_w # large penalty, otherwise this overfits to the EBM data
    
    # --- Define the inner QP once (structure never changes) ---
    if fixed_weights:
        w_fixed = torch.ones(D, dtype=torch.float64) / D
    else: 
        w_var = cp.Variable(D)
        # Create a parameter for each target vector
        YQ_params = [cp.Parameter((m, D)) for _ in range(len(target_vectors))]
        discrepancy = [cp.sum_squares(d.numpy() - (YQ_param @ w_var)) for YQ_param, d in zip(YQ_params, target_vectors)]
        # I believe this is where I'll add in the many many target and covariate matrices
        constraints = [cp.sum(w_var) == 1, w_var >= 0]

        objective = cp.Minimize(sum(discrepancy) + (lambda_l2_w * cp.sum_squares(w_var)))
        prob = cp.Problem(objective, constraints)
        layer = CvxpyLayer(prob, parameters=YQ_params, variables=[w_var])  

    # --- Outer optimization loop ---
    optimizer = torch.optim.Adam([Q], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    for step in range(n_iterations):
        optimizer.zero_grad()

        # transform the covariates using Q
        YQ_list = [Y @ Q for Y in covariate_matrices]
        
        # solve for w given the matrix Q
        # * unpacks the list
        if not fixed_weights:
            w_sol, = layer(*YQ_list)

        # use l2 norm to regularize Q
        l2_Q = torch.sum(Q**2)
        
        if fixed_weights:
            # no regularization with fixed weights and orthogonal Q matrix
            loss = sum(torch.sum((d - YQ @ w_fixed)**2) for d, YQ in zip(target_vectors, YQ_list))
        else:
            # loss using the optimal w for this Q
            loss = sum(torch.sum((d - YQ @ w_sol)**2) for d, YQ in zip(target_vectors, YQ_list)) + (lambda_l2_Q * l2_Q)

        # this is where Q is updated
        loss.backward()                 
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            U, _, Vt = torch.linalg.svd(Q, full_matrices=False)
            Q.data = U @ Vt 

        if verbose and step % 200 == 0:
            if num_donors <= D:
                ortho_error = torch.norm(Q @ Q.T - eye_for_ortho_error)
            else:
                ortho_error = torch.norm(Q.T @ Q - eye_for_ortho_error)
            print("ortho error: \n", ortho_error)
            print(f"Step {step:4d} | Loss: {loss.item():.8f}")
            if not fixed_weights:
                print(f"Step {step:4d} | Loss: {loss.item():.8f} | w: {w_sol.detach().numpy().round(3)}")
            print(f"Grad norm: {Q.grad.norm().item():.8f}")

    # --- Results --- #
    if (verbose == True):
        print(f"\nFinal Loss: {loss.item():.8f}")
        if not fixed_weights:
            print(f"Final w:    {w_sol.detach().numpy().round(4)}")
        print(f"Final Q:\n {Q.detach().numpy().round(4)}")

    Q_final = Q.detach().numpy()
    if not fixed_weights:
        w_final = np.round(w_sol.detach().numpy(), decimals = 2)
        return Q_final, w_final
    else:
        w_final = w_fixed.numpy()
        return Q_final, w_final
    
