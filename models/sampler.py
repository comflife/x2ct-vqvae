import torch
import torch.nn as nn


class Sampler(nn.Module):
    def __init__(self, H, embedding_weight):
        super().__init__()
        self.latent_shape = H.ct_config.model.latent_shape
        self.emb_dim = H.ct_config.model.emb_dim
        self.codebook_size = H.ct_config.model.codebook_size
        self.embedding_weight = embedding_weight
        self.embedding_weight.requires_grad = False
        self.n_samples = H.diffusion.sampling_batch_size

    def train_iter(self, x, x_target, step):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()

    def class_conditional_train_iter(self, x, y):
        raise NotImplementedError()

    def class_conditional_sample(n_samples, y):
        raise NotImplementedError()

    def embed(self, z):
        with torch.no_grad():
            z_flattened = z.view(-1, self.codebook_size)  # B*H*W, codebook_size
            
            # Handle different latent_shape formats
            if len(self.latent_shape) == 2:
                # [H, W] format -> assume single channel
                h, w = self.latent_shape
                embedded = torch.matmul(z_flattened, self.embedding_weight).view(
                    z.size(0),
                    h,
                    w,
                    self.emb_dim
                ).permute(0, 3, 1, 2).contiguous()
            elif len(self.latent_shape) == 4:
                # [B, C, H, W] format
                embedded = torch.matmul(z_flattened, self.embedding_weight).view(
                    z.size(0),
                    self.latent_shape[1],
                    self.latent_shape[2],
                    self.latent_shape[3],
                    self.emb_dim
                ).permute(0, 4, 1, 2, 3).contiguous()
            else:
                raise ValueError(f"Unsupported latent_shape format: {self.latent_shape}")

        return embedded
