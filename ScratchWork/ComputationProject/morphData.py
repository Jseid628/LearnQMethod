import numpy as np
import torch

def morph(treated_data, control_data):
    # Reshape the data appropriately from EBM's code.
    T = treated_data.shape[0]        # 40
    K = treated_data.shape[1]        # 10
    N_minus_1 = control_data.shape[0] // T  # 1960 // 40 = 49

    if type(treated_data) is not np.ndarray:
        out_trt = treated_data.to_numpy()
    else:
        out_trt = treated_data

    if type(control_data) is not np.ndarray:
        out_control = control_data.to_numpy()
    else:
        out_control = control_data
        
    # Reshape control to (T, N-1, K) then transpose to (T, K, N-1)
    out_control_3d = out_control.reshape(T, N_minus_1, K).transpose(0, 2, 1)  # (40, 10, 49)

    # Add unit dimension to treated and concatenate
    out_trt_3d = out_trt[:, np.newaxis, :].transpose(0, 2, 1)  

    Y = np.concatenate([out_trt_3d, out_control_3d], axis=2) 
    
    # Train/test split
    test = Y[len(Y)-1] 
    train = Y[0:len(Y)-1]

    train_target_vectors = [torch.from_numpy(train[i][0:train[i].shape[0],0:1]).squeeze() for i in range(len(train))]
    train_covariate_matrices = [torch.from_numpy(train[i][0:train[i].shape[0],1:train[i].shape[1]]) for i in range(len(train))]

    test_target_vector = torch.from_numpy(test[0:test.shape[0],0:1]).squeeze()
    test_covariate_matrix = torch.from_numpy(test[0:test.shape[0],1:test.shape[1]])

    return train_target_vectors, train_covariate_matrices, test_target_vector, test_covariate_matrix