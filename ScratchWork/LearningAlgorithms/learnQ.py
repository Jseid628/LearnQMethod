import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

def learnQ(targets, covariates, embedding_dim, n_iterations, reg_Q, reg_w, verbose, num_timepoints = None, init_Q = "zeros"):
    if num_timepoints is None:
        num_timepoints = len(covariates)

    # unpacking inputs
    covariate_matrices = covariates[0:num_timepoints] 
    target_vectors = targets[0:num_timepoints]

    # rows (num outcomes)
    Y_1 = covariate_matrices[0]
    # m is number of outcomes
    m = Y_1.shape[0] 
    # this gets the number of donors
    num_donors = Y_1.shape[1] 
    # Embedding dimension
    D = embedding_dim

    # Q is what we're optimizing - requires_grad=True tracks gradients
    torch.manual_seed(215)
    if init_Q == "eye":
        Q = torch.eye(num_donors, D, dtype=torch.float64, requires_grad=True)   
    else:
        Q = torch.randn(num_donors, D, dtype=torch.float64, requires_grad=True)

    # --- Define the inner QP once (structure never changes) ---
    w_var = cp.Variable(D)
    # Create a parameter for each target vector
    YQ_params = [cp.Parameter((m, D)) for _ in range(len(target_vectors))]
    discrepancy = [cp.sum_squares(d.numpy() - YQ_param @ w_var) for YQ_param, d in zip(YQ_params, target_vectors)]
    # I believe this is where I'll add in the many many target and covariate matrices
    constraints = [cp.sum(w_var) == 1, w_var >= 0]
    objective = cp.Minimize(sum(discrepancy))
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
        w_sol, = layer(*YQ_list) 

        # use l2 norm to regularize Q
        # regularization is needed for the EBM data because we want the synthetic covariates
        # to be in a reasonable range.
        lambda_l2_Q = reg_Q
        lambda_l2_w = reg_w # large penalty, otherwise this overfits to the EBM data

        l2_Q = torch.sum(Q**2)
        l2_w = torch.sum(w_sol**2)

        # loss using the optimal w for this Q
        loss = sum(torch.sum((d - YQ @ w_sol)**2) for d, YQ in zip(target_vectors, YQ_list)) + (lambda_l2_Q * l2_Q) + (lambda_l2_w * l2_w)

        # this is where Q is updated
        loss.backward()                 
        optimizer.step()
        scheduler.step()

        if verbose and step % 200 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.8f}")
            print(f"Step {step:4d} | Loss: {loss.item():.8f} | w: {w_sol.detach().numpy().round(3)}")
            print(f"Grad norm: {Q.grad.norm().item():.8f}")

    # --- Results --- #
    if (verbose == True):
        print(f"\nFinal Loss: {loss.item():.8f}")
        print(f"Final w:    {w_sol.detach().numpy().round(4)}")
        print(f"Final Q:\n {Q.detach().numpy().round(4)}")
    
    Q_final = Q.detach().numpy()
    w_final = w_sol.detach().numpy()
    return Q_final, w_final
