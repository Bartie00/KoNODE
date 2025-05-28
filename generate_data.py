import numpy as np
import torch
from torchdiffeq import odeint


def add_noise(traj, noise_level):
    """
    Add Gaussian noise to trajectories.
    """
    noise = noise_level * torch.randn_like(traj)
    return traj + noise


def generate_damped_oscillator_data(params, dataset_type):
    """
    Generate data for the damped oscillator.
    """
   
    num_trajectories = params[dataset_type]['N']
    trajectory_length = params[dataset_type]['T']
    dt = params['dt']
    noise_level = params['noise']
    solver = params['solver']

   
    beta = params['beta']
    delta = params['delta']

   
    alpha_value = params['alpha_value']  
    gamma_value = params['gamma_value']  
    alpha_values = torch.full((num_trajectories,), alpha_value, dtype=torch.float32)
    gamma_values = torch.full((num_trajectories,), gamma_value, dtype=torch.float32)

   
    def damped_oscillator(t, state, alpha, gamma):
        q, p = state.split(1, dim=-1)  
        dq_dt = p
        dp_dt = -alpha * q - beta * q ** 3 - gamma * p + delta * torch.cos(t)
        return torch.cat([dq_dt, dp_dt], dim=-1)

   
    r = torch.sqrt(torch.rand(num_trajectories) * (1.0 - 0.2 ** 2) + 0.2 ** 2)
    theta = 2 * np.pi * torch.rand(num_trajectories)
    q0 = r * torch.cos(theta)
    p0 = r * torch.sin(theta)
    initial_states = torch.stack([q0, p0], dim=-1)

    
    time_points = torch.arange(0, trajectory_length * dt, dt)

    
    trajectories = []
    for i in range(num_trajectories):
        alpha = alpha_values[i]
        gamma = gamma_values[i]
        single_trajectory = odeint(lambda t, x: damped_oscillator(t, x, alpha, gamma), 
                                   initial_states[i:i+1], time_points, method=solver)
        trajectories.append(single_trajectory)

   
    trajectories = torch.cat(trajectories, dim=1)  # Shape: (T, N, 2)

   
    if noise_level > 0.0 and dataset_type == 'train':
        trajectories = add_noise(trajectories, noise_level)

    return trajectories.numpy()


if __name__ == "__main__":
    # Parameters for data generation
    params = {
        'train': {'N': 600, 'T': 30},
        'test': {'N': 200, 'T': 300},
        'val': {'N': 200, 'T': 300},
        'dt': 0.1,
        'alpha_value': -1.0,  
        'gamma_value': 0.0,  
        'beta': 1.0,
        'delta': 0.0,
        'noise': 0.01,  # Gaussian noise level
        'solver': 'dopri5'  
    }

    # Generate datasets
    print("Generating training data...")
    train_data = generate_damped_oscillator_data(params, 'train')
    print("Generating test data...")
    test_data = generate_damped_oscillator_data(params, 'test')
    print("Generating val data...")
    val_data = generate_damped_oscillator_data(params, 'val')

    # Save data to .npy files
    np.save("train_data.npy", train_data)
    np.save("test_data.npy", test_data)
    np.save("val_data.npy", val_data)
    print("Data generation completed. Saved to 'train_data.npy' and 'test_data.npy' and 'val_data.npy'.")
