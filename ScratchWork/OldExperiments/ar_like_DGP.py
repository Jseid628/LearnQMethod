import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)

def sim_Data(T, N, m, p, f, with_covariates):
    Outcomes = []
    Thetas = []
    Lambdas = []
    latent_terms = []

    # Fixed base loadings
    Theta_base = torch.randn(m, p, dtype=torch.float64)
    Lambda_base = torch.randn(m, f, dtype=torch.float64)

    # step directions
    Theta_step = torch.randn(m, p, dtype=torch.float64)
    Lambda_step = torch.randn(m, f, dtype=torch.float64)

    for t in range(T):
        if t == 0:
            # start out at the base, before taking any steps
            Theta_t = Theta_base
            Lambda_t = Lambda_base
        else:
            # take a step, always in the same direction, but by varying amounts
            Theta_t = Theta_base + (Theta_step * torch.abs(torch.randn(1)))
            Lambda_t = Lambda_base + (Lambda_step * torch.abs(torch.randn(1)))
        # each time point append
        Thetas.append(Theta_t)
        Lambdas.append(Lambda_t)

    # if we aren't using covariates, just do the loadings for the latent factors
    Z_F = torch.randn(p, N, dtype=torch.float64) * bool(with_covariates)  # (p x N)
    M_F= torch.randn(f, N, dtype=torch.float64)  # (f x N)

    for t in range(T):
        # returning the latent terms
        L_t = Lambdas[t] @ M_F
        latent_terms.append(L_t)

        Y_t = Thetas[t] @ Z_F + Lambdas[t] @ M_F
        Outcomes.append(Y_t)

    return Outcomes, latent_terms, M_F, Z_F


