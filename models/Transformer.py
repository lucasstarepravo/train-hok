from models.NN_Base import BaseModel
from torch import Tensor
import torch
import torch.nn as nn


class Transformer_Topology(nn.Module):
    def __init__(self, input_size: int, d_model: int, nhead: int,
                 num_layers: int, dim_feedforward: int,
                 output_size: int):
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
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.output_size = output_size

        # Create a stack of transformer encoder layers.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='gelu'
        )

        self.embedding = nn.Linear(self.input_size, self.d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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

        # Transformer expects (seq_len, batch, d_model).
        x = x.transpose(0, 1)
        # Process through the transformer encoder.
        x = self.encoder(x)
        # Transpose back: (batch, seq_len, d_model).
        x = x.transpose(0, 1)

        # For simplicity, we use the representation of the last token.
        output = self.output(x)
        return output


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

        # Replace the default model with MeshFreeAttentionArchitecture.
        self.model = Transformer_Topology(
            input_size=self.input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            output_size=self.output_size
        )

    def calculate_loss(self, outputs: Tensor, labels: Tensor, inputs: Tensor = None) -> Tensor:
        """
        Compute the loss between model outputs and labels.
        """
        return self.loss_function(outputs, labels)

    def save_checkpoint(self, path_to_save, model_type, model_ID, model_ddp, **kwargs):
        extra_attrs = {'d_model': self.d_model,
                       'nhead': self.nhead,
                       'num_layers': self.num_layers,
                       'dim_feedforward': self.dim_feedforward}
        super().save_checkpoint(path_to_save, model_type, model_ID, model_ddp, **extra_attrs)
