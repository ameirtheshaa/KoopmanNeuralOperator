class DynamicalOperator:
    """
    DynamicalOperator: A class implementing Koopman-based neural operators for dynamical systems.
    
    This class provides a framework for learning and predicting the evolution of dynamical
    systems using neural networks combined with spectral methods in Fourier space.
    """
    
    # ========= Model Submodules =========
    class EncoderNetwork(nn.Module):
        """
        Simple MLP encoder that maps from input time dimension to latent dimension.
        """
        def __init__(self, input_length, latent_dim):
            super().__init__()
            self.projection = nn.Linear(input_length, latent_dim)

        def forward(self, x):
            return self.projection(x)

    class DecoderNetwork(nn.Module):
        """
        Simple MLP decoder that maps from latent dimension back to output time dimension.
        """
        def __init__(self, output_length, latent_dim):
            super().__init__()
            self.projection = nn.Linear(latent_dim, output_length)

        def forward(self, x):
            return self.projection(x)

    class SpectralOperator2D(nn.Module):
        """
        2D Spectral Operator for evolution in Fourier space.
        
        Implements a learnable spectral operator that acts on the Fourier modes
        of a 2D signal to perform time evolution.
        
        Args:
            latent_dim (int): Dimension of the latent space.
            fourier_modes_x (int): Number of Fourier modes to use in x direction.
            fourier_modes_y (int): Number of Fourier modes to use in y direction.
        """
        def __init__(self, latent_dim, fourier_modes_x, fourier_modes_y):
            super().__init__()
            self.latent_dim = latent_dim
            self.scale = 1 / (latent_dim * latent_dim)
            self.fourier_modes_x = fourier_modes_x
            self.fourier_modes_y = fourier_modes_y
            # Initialize a learnable complex-valued parameter matrix
            self.spectral_matrix = nn.Parameter(
                self.scale * torch.rand(latent_dim, latent_dim, fourier_modes_x, fourier_modes_y, dtype=torch.cfloat)
            )

        def spectral_transform(self, input, weights):
            """Apply spectral transformation using Einstein summation."""
            return torch.einsum("btxy,tfxy->bfxy", input, weights)

        def forward(self, x):
            """
            Forward pass through the spectral operator.
            
            Args:
                x (torch.Tensor): Input tensor of shape (batch, latent_dim, spatial_x, spatial_y).
                
            Returns:
                torch.Tensor: Transformed tensor after applying the spectral operator.
            """
            # Transform to Fourier domain
            x_fourier = torch.fft.rfft2(x)
            result_fourier = torch.zeros_like(x_fourier, dtype=torch.cfloat)
            
            # Process positive frequencies
            result_fourier[:, :, :self.fourier_modes_x, :self.fourier_modes_y] = self.spectral_transform(
                x_fourier[:, :, :self.fourier_modes_x, :self.fourier_modes_y], self.spectral_matrix
            )
            
            # Process negative frequencies
            result_fourier[:, :, -self.fourier_modes_x:, :self.fourier_modes_y] = self.spectral_transform(
                x_fourier[:, :, -self.fourier_modes_x:, :self.fourier_modes_y], self.spectral_matrix
            )
            
            # Transform back to spatial domain
            return torch.fft.irfft2(result_fourier, s=(x.size(-2), x.size(-1)))

    class DynamicalNetwork2D(nn.Module):
        """
        2D Dynamical Neural Operator with encoder-operator-decoder architecture.
        
        This network combines an encoder-decoder structure with a spectral operator
        to learn and predict the evolution of dynamical systems in 2D.
        
        Args:
            encoder (nn.Module): Encoder network.
            decoder (nn.Module): Decoder network.
            latent_dim (int): Dimension of the latent space.
            fourier_modes_x (int): Number of Fourier modes in x direction.
            fourier_modes_y (int): Number of Fourier modes in y direction.
            iterations (int): Number of iterative operator applications.
            embedding_delay (int): Delay for embedding (usually 1).
            use_linear (bool): If True, uses linear combination; if False, applies nonlinearity.
            use_batch_norm (bool): Whether to apply batch normalization.
        """
        def __init__(self, encoder, decoder, latent_dim, fourier_modes_x=10, fourier_modes_y=10,
                     iterations=6, embedding_delay=1, use_linear=False, use_batch_norm=False):
            super().__init__()
            self.latent_dim = latent_dim
            self.iterations = iterations
            self.embedding_delay = embedding_delay
            self.fourier_modes_x = fourier_modes_x
            self.fourier_modes_y = fourier_modes_y
            self.encoder = encoder
            self.decoder = decoder
            self.spectral_layer = DynamicalOperator.SpectralOperator2D(latent_dim, fourier_modes_x, fourier_modes_y)
            self.feature_mixer = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)
            self.use_linear = use_linear
            self.use_batch_norm = use_batch_norm
            if use_batch_norm:
                self.norm = nn.BatchNorm2d(latent_dim)

        def forward(self, x):
            """
            Forward pass for the 2D dynamical network.
            
            Args:
                x (torch.Tensor): Input tensor.
                
            Returns:
                tuple: (prediction, reconstruction) where:
                    - prediction is the predicted future state
                    - reconstruction is the autoencoder reconstruction
            """
            # Autoencoder reconstruction path
            encoded = torch.tanh(self.encoder(x))
            reconstruction = self.decoder(encoded)
            
            # Prediction path
            latent = torch.tanh(self.encoder(x)).permute(0, 3, 1, 2)
            initial_latent = latent.clone()
            
            # Apply spectral operator iteratively
            for _ in range(self.iterations):
                evolved = self.spectral_layer(latent)
                if self.use_linear:
                    latent = latent + evolved
                else:
                    latent = torch.tanh(latent + evolved)
            
            # Mix with initial features
            if self.use_batch_norm:
                latent = torch.tanh(self.norm(self.feature_mixer(initial_latent)) + latent)
            else:
                latent = torch.tanh(self.feature_mixer(initial_latent) + latent)
                
            latent = latent.permute(0, 2, 3, 1)
            prediction = self.decoder(latent)
            
            return prediction, reconstruction

    class SpectralOperator1D(nn.Module):
        """
        1D Spectral Operator for evolution in Fourier space.
        
        Implements a learnable spectral operator that acts on the Fourier modes
        of a 1D signal to perform time evolution.
        
        Args:
            latent_dim (int): Dimension of the latent space.
            fourier_modes (int): Number of Fourier modes to use.
        """
        def __init__(self, latent_dim, fourier_modes=16):
            super().__init__()
            self.latent_dim = latent_dim
            self.fourier_modes = fourier_modes
            self.scale = 1 / (latent_dim * latent_dim)
            # Initialize the spectral matrix as a learnable complex parameter
            self.spectral_matrix = nn.Parameter(
                self.scale * torch.rand(latent_dim, latent_dim, fourier_modes, dtype=torch.cfloat)
            )

        def spectral_transform(self, input, weights):
            """
            Applies complex multiplication for spectral evolution using Einstein summation.
            
            Args:
                input (torch.Tensor): Input tensor of shape (batch, time, modes).
                weights (torch.Tensor): Spectral matrix of shape (latent_dim, latent_dim, modes).
                
            Returns:
                torch.Tensor: Output tensor of shape (batch, latent_dim, modes).
            """
            return torch.einsum("btx,tfx->bfx", input, weights)

        def forward(self, x):
            """
            Forward pass through the spectral operator.
            
            Args:
                x (torch.Tensor): Input tensor of shape (batch, latent_dim, spatial_dim).
                
            Returns:
                torch.Tensor: Transformed tensor after applying the spectral operator.
            """
            # Compute the Fourier transform along the last dimension
            x_fourier = torch.fft.rfft(x)
            
            # Prepare output tensor in Fourier space
            result_fourier = torch.zeros_like(x_fourier, dtype=torch.cfloat)
            
            # Apply the operator only on the selected Fourier modes
            result_fourier[:, :, :self.fourier_modes] = self.spectral_transform(
                x_fourier[:, :, :self.fourier_modes], self.spectral_matrix
            )
            
            # Transform back to spatial domain
            return torch.fft.irfft(result_fourier, n=x.size(-1))

    class DynamicalNetwork1D(nn.Module):
        """
        1D Dynamical Neural Operator with encoder-operator-decoder architecture.
        
        This network combines an encoder-decoder structure with a spectral operator
        to learn and predict the evolution of dynamical systems in 1D.
        
        Args:
            encoder (nn.Module): Encoder network.
            decoder (nn.Module): Decoder network.
            latent_dim (int): Dimension of the latent space.
            fourier_modes (int): Number of Fourier modes to use.
            iterations (int): Number of iterative operator applications.
            embedding_delay (int): Delay for embedding (usually 1).
            use_linear (bool): If True, uses linear combination; if False, applies nonlinearity.
            use_batch_norm (bool): Whether to apply batch normalization.
        """
        def __init__(
            self,
            encoder,
            decoder,
            latent_dim,
            fourier_modes=16,
            iterations=4,
            embedding_delay=1,
            use_linear=False, 
            use_batch_norm=False,
        ):
            super().__init__()
            self.latent_dim = latent_dim
            self.iterations = iterations
            self.use_linear = use_linear
            self.use_batch_norm = use_batch_norm

            self.encoder = encoder
            self.decoder = decoder
            self.spectral_layer = DynamicalOperator.SpectralOperator1D(latent_dim, fourier_modes=fourier_modes)
            # Feature mixing layer
            self.feature_mixer = nn.Conv1d(latent_dim, latent_dim, kernel_size=1)

            if self.use_batch_norm:
                self.norm = nn.BatchNorm1d(latent_dim)

        def forward(self, x):
            """
            Forward pass for the 1D dynamical network.
            
            Args:
                x (torch.Tensor): Input tensor.
                
            Returns:
                tuple: (prediction, reconstruction) where:
                    - prediction is the predicted future state
                    - reconstruction is the autoencoder reconstruction
            """
            # Reconstruction pathway
            encoded = self.encoder(x)
            encoded = torch.tanh(encoded)
            reconstruction = self.decoder(encoded)

            # Prediction pathway
            latent = self.encoder(x)
            latent = torch.tanh(latent)
            # Permute to (batch, channels, length) for Conv1d operations
            latent = latent.permute(0, 2, 1)
            initial_latent = latent.clone()

            # Apply the spectral operator iteratively
            for _ in range(self.iterations):
                evolved = self.spectral_layer(latent)
                if self.use_linear:
                    latent = latent + evolved
                else:
                    latent = torch.tanh(latent + evolved)

            # Mix with initial features
            if self.use_batch_norm:
                latent = torch.tanh(self.norm(self.feature_mixer(initial_latent)) + latent)
            else:
                latent = torch.tanh(self.feature_mixer(initial_latent) + latent)

            # Permute back for decoder
            latent = latent.permute(0, 2, 1)
            prediction = self.decoder(latent)

            return prediction, reconstruction

    class DynamicalModel:
        """
        This inner class encapsulates model compilation, optimizer initialization,
        training, and saving/loading of dynamical systems models.
        """
        def __init__(self, architecture="DNO2d", encoder_type="MLP", latent_dim=16, 
                    fourier_modes=16, iterations=8, time_horizon=1, embedding_delay=1, 
                    use_linear=False, use_batch_norm=False, device='cpu'):
            self.architecture = architecture
            self.encoder_type = encoder_type
            self.latent_dim = latent_dim
            self.fourier_modes = fourier_modes
            self.iterations = iterations
            self.use_linear = use_linear
            self.use_batch_norm = use_batch_norm
            self.embedding_delay = embedding_delay
            self.device = device
            self.time_horizon = time_horizon
            self.loss_fn = nn.MSELoss()
            self.model = None
            self.optimizer = None
            self.scheduler = None
            self.compile()

        def compile(self):
            """
            Compile the model by initializing encoder, decoder and the appropriate network.
            """
            encoder = DynamicalOperator.EncoderNetwork(self.time_horizon, self.latent_dim)
            decoder = DynamicalOperator.DecoderNetwork(self.time_horizon, self.latent_dim)
            
            if self.architecture == "DNO2d":
                self.model = DynamicalOperator.DynamicalNetwork2D(
                    encoder, decoder, self.latent_dim,
                    fourier_modes_x=self.fourier_modes, fourier_modes_y=self.fourier_modes, 
                    iterations=self.iterations, use_linear=self.use_linear, 
                    use_batch_norm=self.use_batch_norm,
                ).to(self.device)
            elif self.architecture == "DNO1d":
                self.model = DynamicalOperator.DynamicalNetwork1D(
                    encoder, decoder, self.latent_dim,
                    fourier_modes=self.fourier_modes, iterations=self.iterations,
                    use_linear=self.use_linear, use_batch_norm=self.use_batch_norm,
                ).to(self.device)
            print("Model successfully compiled.")

        def init_optimizer(self, optimizer_type="Adam", learning_rate=1e-3, step_size=500, decay_factor=0.8):
            """
            Initialize optimizer and learning rate scheduler.
            
            Args:
                optimizer_type (str): Type of optimizer to use (currently only "Adam" supported).
                learning_rate (float): Initial learning rate.
                step_size (int): Number of epochs between learning rate decay.
                decay_factor (float): Factor by which learning rate is multiplied at each step.
            """
            if optimizer_type == "Adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
            if step_size:
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=decay_factor)

        def train(self, epochs, trainloader, step=1, prediction_length=40):
            """
            Revised training loop using delay embedding when step != 1.
            
            Args:
                epochs (int): Number of training epochs.
                trainloader (DataLoader): DataLoader for training data.
                step (int): Step size for delay embedding.
                prediction_length (int): Number of time steps to predict.
            """
            for epoch in range(epochs):
                self.model.train()
                t_start = default_timer()
                train_recon_loss = 0
                train_pred_loss = 0

                for inputs, targets in trainloader:
                    # Move to device
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    batch_size = inputs.shape[0]

                    # Create delay-embedded ground truth from targets
                    targets_subsampled = targets[..., step - 1::step]
                    num_delay_coords = targets_subsampled.shape[-1]

                    recon_loss = 0
                    pred_frames = []  # to accumulate one delay coordinate per iteration

                    # Run prediction loop for each delay coordinate
                    for t in range(num_delay_coords):
                        # Get predictions and reconstruction
                        prediction, reconstruction = self.model(inputs)

                        # Accumulate reconstruction loss
                        recon_loss += self.loss_fn(reconstruction.reshape(batch_size, -1), 
                                                  inputs.reshape(batch_size, -1))

                        # Store the last frame of the predicted block
                        pred_frame = prediction[..., -1:]
                        pred_frames.append(pred_frame)

                        # Update inputs by dropping oldest frames and appending prediction
                        inputs = torch.cat((inputs[..., step:], prediction[..., -step:]), dim=-1)

                    # Concatenate predicted frames along time dimension
                    predictions = torch.cat(pred_frames, dim=-1)

                    # Verify shapes match
                    if predictions.shape != targets_subsampled.shape:
                        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets_subsampled {targets_subsampled.shape}")

                    # Compute prediction loss
                    pred_loss = self.loss_fn(predictions.reshape(batch_size, -1), 
                                          targets_subsampled.reshape(batch_size, -1))
                    
                    # Combined loss with weighting
                    total_loss = 5 * pred_loss + 0.5 * recon_loss

                    # Track metrics
                    train_pred_loss += pred_loss.item()
                    train_recon_loss += recon_loss.item() / num_delay_coords

                    # Optimization step
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                # Compute average losses
                train_pred_loss /= len(trainloader)
                train_recon_loss /= len(trainloader)
                t_end = default_timer()
                
                # Update learning rate
                self.scheduler.step()
                
                # Report progress
                print("Epoch", "Time", "Train Recons MSE", "Train Pred MSE")
                print(epoch, t_end - t_start, train_recon_loss, train_pred_loss)

        def save(self, filepath):
            """
            Save model to disk.
            
            Args:
                filepath (str): Path to save the model.
            """
            torch.save({"model": self.model, "state_dict": self.model.state_dict()}, filepath)
            print(f"Model saved to {filepath}")

        def load(self, filepath):
            """
            Load model from disk.
            
            Args:
                filepath (str): Path to the saved model.
            """
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            print(f"Model loaded from {filepath}")

    # ========= Data Preparation, Training, and Prediction =========
    def __init__(self, training_data, time_horizon, latent_dim=16, fourier_modes=16, 
                iterations=8, device='cpu', architecture='DNO2d', batch_size=32, 
                epochs=1, embedding_delay=1, use_linear=False, use_batch_norm=False, 
                data_dir='.', results_dir='results'):
        """
        Initialize the DynamicalOperator framework.
        
        Args:
            training_data (torch.Tensor): Training data tensor.
            time_horizon (int): Number of time steps to use as input.
            latent_dim (int): Dimension of the latent space.
            fourier_modes (int): Number of Fourier modes to use.
            iterations (int): Number of iterative operator applications.
            device (str): Device to use for computation ('cpu' or 'cuda').
            architecture (str): Architecture to use ('DNO2d' or 'DNO1d').
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
            embedding_delay (int): Delay for embedding.
            use_linear (bool): If True, uses linear combination; if False, applies nonlinearity.
            use_batch_norm (bool): Whether to apply batch normalization.
            data_dir (str): Directory for data.
            results_dir (str): Directory for results.
        """
        # Store parameters
        self.training_data = training_data
        self.time_horizon = time_horizon
        self.latent_dim = latent_dim
        self.fourier_modes = fourier_modes
        self.iterations = iterations
        self.device = device
        self.architecture = architecture
        self.batch_size = batch_size
        self.epochs = epochs
        self.embedding_delay = embedding_delay
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.use_linear = use_linear
        self.use_batch_norm = use_batch_norm
        self.spatial_index = 0  # Default index used in get_spatial_predictions

        # Determine prediction window length
        self.prediction_length = training_data.shape[-1] - time_horizon

        # Process and split the data
        self.processed_data = self.prepare_data(self.training_data)
        self.train_test_split()
        input_sequences = self.train_inputs
        target_sequences = self.train_targets
        print("Training inputs shape:", input_sequences.shape)
        print("Training targets shape:", target_sequences.shape)

        # Create data loaders
        self.train_loader = self.create_data_loader(input_sequences, target_sequences, batch_size=self.batch_size)
        # For demonstration, using same loader for testing
        self.test_loader = self.train_loader
        self.test_prediction_length = self.prediction_length

        # Initialize the model
        self.dyn_model = self.DynamicalModel(
            architecture=self.architecture, encoder_type="MLP", latent_dim=self.latent_dim, 
            fourier_modes=self.fourier_modes, iterations=self.iterations,
            time_horizon=self.time_horizon, device=self.device
        )
        self.dyn_model.compile()
        self.dyn_model.init_optimizer("Adam", learning_rate=10**-3, step_size=500, decay_factor=0.8)

        # Train the model
        self.train(epochs=self.epochs)

    def prepare_data(self, data):
        """
        Prepare data for model training.
        
        Args:
            data (torch.Tensor): Raw data tensor.
            
        Returns:
            torch.Tensor: Processed data.
        """
        # Can add any preprocessing steps here
        processed_data = data
        return processed_data

    def train_test_split(self):
        """Split data into training inputs and targets along the time dimension."""
        self.train_inputs = self.processed_data[:, :, :, :self.time_horizon]
        self.train_targets = self.processed_data[:, :, :, self.time_horizon:]

    def create_data_loader(self, inputs, targets, batch_size=32):
        """
        Create a DataLoader from input and target tensors.
        
        Args:
            inputs (torch.Tensor): Input sequences.
            targets (torch.Tensor): Target sequences.
            batch_size (int): Batch size.
            
        Returns:
            DataLoader: DataLoader for the dataset.
        """
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader

    @staticmethod
    def clear_memory():
        """Clear GPU memory cache if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train(self, epochs=None):
        """
        Train the model.
        
        Args:
            epochs (int): Number of training epochs.
        """
        if epochs is None:
            epochs = self.epochs
        self.clear_memory()
        self.dyn_model.train(
            epochs=epochs, 
            trainloader=self.train_loader, 
            step=self.embedding_delay, 
            prediction_length=self.prediction_length
        )
        save_path = os.path.join(os.getcwd(), f'dynamical_model_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth')
        self.dyn_model.save(save_path)

    def generate_predictions(self, test_loader=None, test_prediction_length=None):
        """
        Generate predictions using the trained model.
        
        Args:
            test_loader (DataLoader): DataLoader for test data.
            test_prediction_length (int): Number of time steps to predict.
            
        Returns:
            torch.Tensor: Predicted sequences.
        """
        # Set evaluation mode
        self.dyn_model.model.eval()
        all_predictions = []
        if test_loader is None:
            test_loader = self.test_loader
        if test_prediction_length is None:
            test_prediction_length = self.test_prediction_length

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                sequence_predictions = []
                
                # Autoregressive prediction
                for t in range(test_prediction_length):
                    prediction, _ = self.dyn_model.model(inputs)
                    sequence_predictions.append(prediction[..., -1:])  # Last timestep
                    inputs = torch.cat((inputs[..., 1:], prediction[..., -1:]), dim=-1)
                
                # Concatenate along time dimension
                sequence_predictions = torch.cat(sequence_predictions, dim=-1)
                all_predictions.append(sequence_predictions)

        # Combine all batch predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        self.predictions = all_predictions
        
        # Save predictions
        save_pred_path = os.path.join(
            self.data_dir,
            self.results_dir,
            f'predictions_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.npy'
        )
        np.save(save_pred_path, self.predictions.cpu().numpy())
        return self.predictions

    def get_spatial_predictions(self, slice_index=None):
        """
        Get predictions for a particular spatial index.
        
        Args:
            slice_index (int): Spatial index to extract.
            
        Returns:
            torch.Tensor: Predictions at the specified spatial index.
        """
        if slice_index is None:
            slice_index = self.spatial_index
        prediction_data = self.predictions
        return prediction_data[:, slice_index, :].unsqueeze(2)