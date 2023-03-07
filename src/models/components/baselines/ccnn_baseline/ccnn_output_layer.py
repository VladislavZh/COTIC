import torch
import torch.nn as nn


class PPOutputLayer(nn.Module):
    def __init__(self, embed_dim: int, num_markers: int) -> None:
        """Initialize PPOuputLayer class - output layer for Baseline CCNN model.

        args:
            embed_dim - hidden dimension (output of the convolutional part)
            num_markers - number of event types in the dataset
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_markers = num_markers

        self.linear_y = nn.Linear(self.embed_dim, self.num_markers)
        self.linear_t = nn.Linear(self.embed_dim, 1)

        w_t = torch.empty(1).to(torch.float32)
        self.w_t = nn.Parameter(w_t, requires_grad=True)
        nn.init.uniform_(self.w_t)

    def forward(self, hidden_state: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Forward pass through the final layer.

        args:
            hidden_state: from hidden layer /maybe CNN or RNN
            num_markers: number of types of events
            dt: t - t_j

        return a tuple of:
            multinomial - tensor of size (batch_size, seq_len, num_markers) with event types
            log-likelihood - tensor of size (batch_size, seq_len, 1) with log-likelihood values
        """
        batch_size, seq_len, _ = hidden_state.shape

        flat = hidden_state.reshape(-1, self.embed_dim)

        softmax = nn.Softmax()
        multinomial = softmax(self.linear_y(flat))
        multinomial = multinomial.reshape(batch_size, seq_len, -1)

        plain_d0 = self.linear_t(flat)
        plain_d0 = plain_d0.reshape(batch_size, seq_len, 1)

        dt = torch.unsqueeze(dt, dim=-1)

        plain_dt = plain_d0 + self.w_t * dt

        lambda_d0 = torch.exp(plain_d0)
        lambda_dt = torch.exp(plain_dt)
        exp_it = lambda_d0 / self.w_t - lambda_dt / self.w_t

        # clip values to avoid overflow in exponent
        MAX_CONSTANT = torch.Tensor([70.0]).to(torch.float32).to(plain_dt.device)
        log_likelihood = torch.min(plain_dt + exp_it, MAX_CONSTANT)

        return multinomial, log_likelihood

    def get_lambda_0(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Compute lambda_0 (i.e. intensity) term for return time estimation.

        args:
            hidden_state - hidden state tensor (aka 'encoded_output')

        returns:
            lambda_0 - intensity values
        """
        batch_size, seq_len, _ = hidden_state.shape
        flat = hidden_state.reshape(-1, self.embed_dim)
        plain_d0 = self.linear_t(flat)
        plain_d0 = plain_d0.reshape(batch_size, seq_len, 1)
        lambda_d0 = torch.exp(plain_d0)
        return lambda_d0
