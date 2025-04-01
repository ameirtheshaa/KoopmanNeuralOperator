import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from datetime import datetime
from timeit import default_timer

# Import our DynamicalOperator class
from dynamical_operator import DynamicalOperator

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def generate_lorenz_data(num_samples=50, time_points=200, dt=0.01, noise_level=0.0):
    """
    Generate training data from the Lorenz system.
    
    Args:
        num_samples (int): Number of trajectories to generate
        time_points (int): Number of time points in each trajectory
        dt (float): Time step for integration
        noise_level (float): Standard deviation of Gaussian noise to add
        
    Returns:
        torch.Tensor: Tensor of shape [num_samples, 3, 1, time_points]
                      containing the Lorenz system trajectories
    """
    # Lorenz system parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    
    def lorenz_system(t, state):
        """The Lorenz system ODEs"""
        x, y, z = state
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return [dx_dt, dy_dt, dz_dt]
    
    # Initialize data array
    data = np.zeros((num_samples, 3, time_points))
    
    # Time points for integration
    t_span = (0, dt * time_points)
    t_eval = np.linspace(t_span[0], t_span[1], time_points)
    
    # Generate multiple trajectories
    for i in range(num_samples):
        # Random initial condition centered around the Lorenz attractor
        initial_state = np.random.randn(3) * 5
        
        # Solve the ODE system
        solution = solve_ivp(
            lorenz_system, 
            t_span, 
            initial_state, 
            t_eval=t_eval, 
            method='RK45'
        )
        
        # Store the solution (transposed to get [3, time_points])
        data[i] = solution.y
        
        # Add noise if specified
        if noise_level > 0:
            data[i] += np.random.normal(0, noise_level, data[i].shape)
    
    # Normalize the data to improve training stability
    data_mean = np.mean(data, axis=(0, 2), keepdims=True)
    data_std = np.std(data, axis=(0, 2), keepdims=True)
    normalized_data = (data - data_mean) / (data_std + 1e-8)
    
    # Reshape for the DynamicalOperator (which expects [samples, width, height, time])
    # For Lorenz, we'll use a dummy spatial dimension of 1
    reshaped_data = normalized_data.reshape(num_samples, 3, 1, time_points)
    
    # Convert to PyTorch tensor
    tensor_data = torch.tensor(reshaped_data, dtype=torch.float32)
    
    # Save normalization constants for later denormalization
    normalization = {
        'mean': data_mean.reshape(1, 3, 1, 1),
        'std': data_std.reshape(1, 3, 1, 1)
    }
    
    return tensor_data, normalization

def denormalize_data(normalized_data, normalization):
    """
    Denormalize the data using saved normalization constants.
    
    Args:
        normalized_data (torch.Tensor): Normalized data
        normalization (dict): Dictionary with 'mean' and 'std' for denormalization
        
    Returns:
        torch.Tensor: Denormalized data
    """
    mean = torch.tensor(normalization['mean'], dtype=torch.float32)
    std = torch.tensor(normalization['std'], dtype=torch.float32)
    
    return normalized_data * std + mean

def plot_lorenz_prediction(ground_truth, prediction, normalization, sample_idx=0, save_path=None):
    """
    Plot the ground truth and prediction for a Lorenz system trajectory.
    
    Args:
        ground_truth (torch.Tensor): Ground truth data [num_samples, 3, 1, time_points]
        prediction (torch.Tensor): Predicted data [num_samples, 3, 1, time_points]
        normalization (dict): Normalization constants for denormalization
        sample_idx (int): Index of the sample to plot
        save_path (str): Path to save the figure, if None the figure is displayed
    """
    # Convert to numpy and remove dummy spatial dimension
    gt = ground_truth[sample_idx].detach().cpu().numpy()[:, 0, :]
    pred = prediction[sample_idx].detach().cpu().numpy()[:, 0, :]
    
    # Denormalize
    mean = normalization['mean'][0, :, 0, 0]
    std = normalization['std'][0, :, 0, 0]
    gt_denorm = gt * std + mean
    pred_denorm = pred * std + mean
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Time series plot for x
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(gt_denorm[0], label='Ground Truth', color='blue')
    ax1.plot(pred_denorm[0], label='Prediction', color='red', linestyle='--')
    ax1.set_title('X Coordinate')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Time series plot for y
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(gt_denorm[1], label='Ground Truth', color='blue')
    ax2.plot(pred_denorm[1], label='Prediction', color='red', linestyle='--')
    ax2.set_title('Y Coordinate')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Time series plot for z
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(gt_denorm[2], label='Ground Truth', color='blue')
    ax3.plot(pred_denorm[2], label='Prediction', color='red', linestyle='--')
    ax3.set_title('Z Coordinate')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 3D plot of the attractor
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.plot(gt_denorm[0], gt_denorm[1], gt_denorm[2], label='Ground Truth', color='blue')
    ax4.plot(pred_denorm[0], pred_denorm[1], pred_denorm[2], label='Prediction', color='red', linestyle='--')
    ax4.set_title('Lorenz Attractor')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)

def evaluate_model(model, test_data, time_horizon, prediction_length, normalization):
    """
    Evaluate the model using MSE on test data.
    
    Args:
        model: Trained DynamicalOperator model
        test_data (torch.Tensor): Test data
        time_horizon (int): Input sequence length
        prediction_length (int): Number of time steps to predict
        normalization (dict): Normalization constants for denormalization
        
    Returns:
        float: MSE between ground truth and predictions
    """
    device = next(model.dyn_model.model.parameters()).device
    
    # Split test data into inputs and targets
    inputs = test_data[:, :, :, :time_horizon].to(device)
    targets = test_data[:, :, :, time_horizon:time_horizon+prediction_length].to(device)
    
    # Generate predictions
    model.dyn_model.model.eval()
    with torch.no_grad():
        predictions = []
        current_input = inputs.clone()
        
        for t in range(prediction_length):
            pred, _ = model.dyn_model.model(current_input)
            predictions.append(pred[..., -1:])
            current_input = torch.cat((current_input[..., 1:], pred[..., -1:]), dim=-1)
        
        predictions = torch.cat(predictions, dim=-1)
    
    # Calculate MSE
    mse = torch.mean((predictions - targets) ** 2).item()
    
    return mse, predictions

def main():
    # Parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Data parameters
    num_samples = 50
    total_time_points = 165
    time_horizon = 65  # Length of input sequence
    prediction_length = 100  # Length of prediction sequence
    
    # Model parameters
    latent_dim = 32
    fourier_modes = 16
    iterations = 8
    batch_size = 10
    epochs = 20
    
    # Generate Lorenz system data
    print("Generating Lorenz system data...")
    data, normalization = generate_lorenz_data(
        num_samples=num_samples,
        time_points=total_time_points,
        dt=0.02,
        noise_level=0.05
    )
    
    # Split data into train and test sets
    train_size = int(0.8 * num_samples)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Create results directory
    results_dir = "lorenz_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create and train the model
    print("Initializing and training the model...")
    model = DynamicalOperator(
        training_data=train_data,
        time_horizon=time_horizon,
        latent_dim=latent_dim,
        fourier_modes=fourier_modes,
        iterations=iterations,
        device=device,
        architecture='DNO1d',  # Use 1D architecture for Lorenz
        batch_size=batch_size,
        epochs=epochs,
        embedding_delay=1,
        use_linear=False,
        use_batch_norm=True,
        data_dir='.',
        results_dir=results_dir
    )
    
    # Evaluate the model
    print("Evaluating model...")
    mse, predictions = evaluate_model(
        model, 
        test_data, 
        time_horizon, 
        prediction_length, 
        normalization
    )
    print(f"Test MSE: {mse:.6f}")
    
    # Plot results for a few test samples
    print("Plotting results...")
    for i in range(min(3, len(test_data))):
        ground_truth = torch.cat(
            (test_data[i:i+1, :, :, :time_horizon], 
             test_data[i:i+1, :, :, time_horizon:time_horizon+prediction_length]), 
            dim=-1
        )
        predicted = torch.cat(
            (test_data[i:i+1, :, :, :time_horizon], 
             predictions[i:i+1]), 
            dim=-1
        )
        
        save_path = os.path.join(results_dir, f"lorenz_prediction_sample_{i}.png")
        plot_lorenz_prediction(
            ground_truth, 
            predicted, 
            normalization, 
            sample_idx=0,  # We're already selecting one sample at a time
            save_path=save_path
        )
    
    print("Done!")

if __name__ == "__main__":
    main()