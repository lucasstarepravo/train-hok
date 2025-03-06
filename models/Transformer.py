from models.NN_Base import BaseModel, NN_Topology
from torch import Tensor
import torch
import torch.nn as nn


class KolmogorovArnoldNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_terms: int, inner_arch: list):
        """
        Kolmogorov–Arnold network that represents a function as a sum of univariate transformations.

        Args:
            input_dim (int): Dimension of the input (here, d_model).
            output_dim (int): Desired output dimension.
            num_terms (int): Number of inner networks (terms in the Kolmogorov representation).
            inner_arch (list): A list of hidden layer sizes to use in each inner network.
                               If empty, each inner network is just a linear layer mapping input -> 1.
        """
        super(KolmogorovArnoldNetwork, self).__init__()
        self.num_terms = num_terms
        # Create a list of inner networks.
        # Each inner network maps from input_dim -> 1 using a small MLP with the given architecture.
        self.inner_networks = nn.ModuleList([
            self._build_inner_network(input_dim, inner_arch) for _ in range(num_terms)
        ])
        # Outer layer combines the num_terms scalars to produce the final output.
        self.outer = nn.Linear(num_terms, output_dim)

    def _build_inner_network(self, input_dim: int, inner_arch: list):
        layers = []
        in_features = input_dim
        # Build the inner MLP from the inner_arch list.
        for hidden in inner_arch:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.SiLU())
            in_features = hidden
        # Final linear mapping to a single scalar.
        layers.append(nn.Linear(in_features, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to have shape (B * S, d_model)
        # Process x through each inner network to get a list of scalars.
        terms = [net(x) for net in self.inner_networks]  # each output: (B * S, 1)
        # Concatenate along the last dimension to shape (B * S, num_terms)
        terms_cat = torch.cat(terms, dim=-1)
        # Combine the terms to get the final output.
        out = self.outer(terms_cat)
        return out


class Transformer_Topology(nn.Module):
    def __init__(self, input_size: int, d_model: int, nhead: int,
                 num_layers: int, dim_feedforward: int, seq_len: int,
                 output_size: int, hidden_layers=None):
        """
        Encoder-only attention architecture for mesh-free simulations.

        Args:
            input_size (int): Dimension of the raw input features.
            d_model (int): Projection dimension for the transformer.
            nhead (int): Number of attention heads.
            num_layers (int): Number of self-attention layers.
            dim_feedforward (int): Hidden dimension for feed-forward layers.
            seq_len (int): Sequence length.
            output_size (int): Dimension of the output.
            hidden_layers (list, optional): For standard MLP, a list of hidden sizes.
                For a Kolmogorov–Arnold network, we interpret hidden_layers as:
                [num_terms, inner_hidden1, inner_hidden2, ...].
                If None, a single linear mapping is used.
        """
        super(Transformer_Topology, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        # Transformer encoder layer.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            batch_first=True
        )

        self.embedding = nn.Linear(self.input_size, self.d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if hidden_layers:
            # For the Kolmogorov–Arnold network:
            # Use the first element as the number of terms and the remaining as the inner network architecture.
            num_terms = hidden_layers[0]
            inner_arch = hidden_layers[1:]  # can be an empty list if you want a simple linear mapping per term
            self.ka_network = KolmogorovArnoldNetwork(self.d_model, self.output_size,
                                                      num_terms=num_terms, inner_arch=inner_arch)
        else:
            self.output = nn.Linear(self.d_model, self.output_size)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder-only model.

        Args:
            src (Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, output_size) if using KA network,
                    or (batch_size, seq_len, output_size) with a linear projection.
        """
        # Project the input to d_model dimensions.
        x = self.embedding(src)  # Shape: (batch, seq_len, d_model)

        # Process through the transformer encoder.
        x = self.encoder(x)

        if self.hidden_layers:
            # Flatten batch and sequence dimensions to apply the KA network token-wise.
            B, S, D = x.shape
            x_flat = x.reshape(B * S, D)  # shape: (B*S, d_model)
            # Apply the Kolmogorov–Arnold network.
            x_flat = self.ka_network(x_flat)
            # Reshape back to (batch_size, seq_len, output_size)
            x = x_flat.reshape(B, S, -1)
        else:
            x = self.output(x)

        return x


class Transformer(BaseModel):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 num_layers: int,
                 dim_feedforward: int,
                 optimizer: str,
                 loss_function: str,
                 epochs: int,
                 batch_size: int,
                 train_f: Tensor,
                 train_l: Tensor,
                 hidden_layers=None) -> None:
        """
        Mesh-free attention model using an encoder-only transformer architecture.

        Args:
            d_model (int): Internal projection dimension.
            nhead (int): Number of attention heads.
            num_layers (int): Number of self-attention layers.
            dim_feedforward (int): Hidden dimension in feed-forward layers.
            hidden_layers (list): (Retained for BaseModel compatibility.)
            optimizer (str): Optimizer name.
            loss_function (str): Loss function name.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            train_f (Tensor): Training features.
            train_l (Tensor): Training labels.
        """
        # This is done to accommodate the parent method, which requires a list to be passed
        self.hidden_layers = hidden_layers
        pass_layers = hidden_layers if hidden_layers else [1]
        super().__init__(pass_layers, optimizer, loss_function, epochs, batch_size, train_f, train_l)

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.input_size = int(train_f.shape[-1])
        self.seq_len = int(train_f.shape[1])
        self.output_size = int(train_l.shape[-1])


        # Replace the default model with Transformer
        self.model = Transformer_Topology(
            input_size=self.input_size,
            seq_len=self.seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            output_size=self.output_size,
            hidden_layers=self.hidden_layers
        )

        self.extra_attrs = {'d_model': self.d_model,
                            'seq_len': self.seq_len,
                            'nhead': self.nhead,
                            'num_layers': self.num_layers,
                            'dim_feedforward': self.dim_feedforward}

    def calculate_loss(self, outputs: Tensor, labels: Tensor, inputs: Tensor = None) -> Tensor:
        """
        Compute the loss between model outputs and labels.
        """
        return self.loss_function(outputs, labels)

    def save_checkpoint(self, path_to_save, model_type, model_ID, model_ddp, **kwargs):
        super().save_checkpoint(path_to_save, model_type, model_ID, model_ddp, **self.extra_attrs)

    def save_model(self, path_to_save, model_type, model_ID, **kwargs):
        """Save the best model weights with additional attributes specific to Transformer."""
        super().save_model(path_to_save, model_type, model_ID, **self.extra_attrs)
