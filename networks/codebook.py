import torch
import torch.nn.functional as F
from torch import nn

class MultiCodebook(nn.Module):
    def __init__(self, num_embeddings, embedding_dims, commitment_cost, decay, epsilon=1e-5):
        super(MultiCodebook, self).__init__()

        self._embedding_dims = embedding_dims
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon

        self.num_levels = len(embedding_dims)
        self.codebooks = nn.ModuleList()
        for idx in range(self.num_levels):
            self.codebooks.append(CodebookEMA(num_embeddings, embedding_dims[idx], commitment_cost, decay))

    def inference(self, inputs):
        quantized_features = []
        for idx in range(self.num_levels):
            input = inputs[idx]
            quantized = self.codebooks[idx].forward_indice(input)
            quantized_features.append(quantized)
        
        return quantized_features


    def forward(self, inputs, return_indices=False, return_features=False):
        total_loss = 0
        quantized_features = []
        code_indices = []
        for idx in range(self.num_levels):
            input = inputs[idx]
            commit_loss, quantized, _, _, code_indice = self.codebooks[idx](input)
            total_loss += commit_loss
            quantized_features.append(quantized)
            code_indices.append(code_indice)
        
        # total_loss = total_loss / self.num_levels

        if return_indices:
            return code_indices
        elif return_features:
            return quantized_features
        else:
            return total_loss, quantized_features


class CodebookEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(CodebookEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.register_buffer('_embedding', torch.empty(
            self._num_embeddings, self._embedding_dim))
        self._embedding.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', torch.empty(
            num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward_indice(self, encoding_indices):
        input_shape = encoding_indices.shape
        encoding_indices = encoding_indices.reshape((-1, 1))
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=encoding_indices.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatte n
        quantized = torch.matmul(encodings, self._embedding).view(
            input_shape[0], input_shape[1], input_shape[2], self._embedding_dim)

        return quantized.permute(0, 3, 1, 2).contiguous()

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input.detach())
            self._ema_w = self._ema_w * self._decay + (1 - self._decay) * dw

            self._embedding = self._ema_w / self._ema_cluster_size.unsqueeze(1)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                               torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings, encoding_indices.view(input_shape[0:3])
