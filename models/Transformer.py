from models.NN_Base import BaseModel, NN_Topology
from torch import Tensor
import torch
import torch.nn as nn


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
            output_size (int): Dimension of the output.
        """
        super(Transformer_Topology, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        # Create a stack of transformer encoder layers.
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
            self.mlp = NN_Topology(input_size=self.d_model,
                                   hidden_layers=hidden_layers,
                                   output_size=self.output_size)
        else:
            self.output = nn.Linear(self.d_model, self.output_size)


    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder-only model.

        Args:
            src (Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Project the input to d_model dimensions.
        x = self.embedding(src)  # Shape: (batch, seq_len, d_model)

        # Process through the transformer encoder.
        x = self.encoder(x)

        if self.hidden_layers:
            # Flatten batch and sequence dimensions to apply NN_Topology token-wise.
            B, S, D = x.shape
            x_flat = x.reshape(B * S, D)  # shape: (B*S, d_model)
            # Process each token with the MLP (NN_Topology).
            x_flat = self.mlp(x_flat)  # shape: (B*S, output_size)
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
                 hidden_layers: list,  # Retained for compatibility; not used directly.
                 optimizer: str,
                 loss_function: str,
                 epochs: int,
                 batch_size: int,
                 train_f: Tensor,
                 train_l: Tensor) -> None:
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
        super().__init__(hidden_layers, optimizer, loss_function, epochs, batch_size, train_f, train_l)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.input_size = int(train_f.shape[-1])
        self.seq_len = int(train_f.shape[1])
        self.output_size = int(train_l.shape[-1])
        self.hidden_layers = hidden_layers

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
