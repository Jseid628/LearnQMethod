import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

def synthOutcomes(targets, covariates, embedding_dim, n_iterations, reg_Q, reg_w, verbose, num_timepoints = None,ortho=False):
    if num_timepoints is None:
        num_timepoints = len(covariates)

    alpha = 0.1
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
    Q = torch.eye(D, m, dtype=torch.float64, requires_grad=True)   

    # --- Define the inner QP once (structure never changes) ---
    w_var = cp.Variable(num_donors)
    # Create a parameter for each target vector
    QY_params = [cp.Parameter((D, num_donors)) for _ in range(len(target_vectors))]
    Qd_params = [cp.Parameter(D) for _ in range(len(target_vectors))]
    discrepancy = [cp.sum_squares(Qd_param - QY_param @ w_var)
                   for QY_param, Qd_param in zip(QY_params, Qd_params)]
    # I believe this is where I'll add in the many many target and covariate matrices
    constraints = [cp.sum(w_var) == 1, w_var >= 0]
    objective = cp.Minimize(sum(discrepancy))
    prob = cp.Problem(objective, constraints)
    layer = CvxpyLayer(prob, parameters=QY_params + Qd_params, variables=[w_var]) 

    # --- Outer optimization loop ---
    optimizer = torch.optim.Adam([Q], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)


    for step in range(n_iterations):
        optimizer.zero_grad()

        # transform the covariates using Q
        QY_list = [Q @ Y for Y in covariate_matrices]
        Qd_list = [Q @ d for d in target_vectors]  # BUG 5: was missing entirely
        
        # solve for w given the matrix Q
        # * unpacks the list
        w_sol, = layer(*QY_list, *Qd_list)

        # use l2 norm to regularize Q
        # regularization is needed for the EBM data because we want the synthetic covariates
        # to be in a reasonable range.
        lambda_l2_Q = reg_Q
        lambda_l2_w = reg_w # large penalty, otherwise this overfits to the EBM data

        l2_Q = torch.sum(Q**2)
        l2_w = torch.sum(w_sol**2)

        
        # loss using the optimal w for this Q
        q_space_loss = sum(torch.sum((Qd - QY @ w_sol)**2) 
                   for Qd, QY in zip(Qd_list, QY_list))
        orig_space_loss = sum(torch.sum((d - Y @ w_sol)**2) 
                            for d, Y in zip(target_vectors, covariate_matrices))
        loss = q_space_loss + alpha * orig_space_loss
                # loss = sum(
        #         torch.sum((d - Y @ w_sol)**2)
        #         for d, Y in zip(target_vectors, covariate_matrices)
        #     ) + lambda_l2_Q * torch.sum(Q**2)
        
        # this is where Q is updated
        loss.backward()                 
        optimizer.step()
        scheduler.step()
        
        if ortho==True:
            if D != m and step % 200 == 0:
                print("Q is not square, but orthogonality constraint has been selected.")
            with torch.no_grad():
                U, _, Vt = torch.linalg.svd(Q, full_matrices=False)
                Q.data = U @ Vt
        

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
