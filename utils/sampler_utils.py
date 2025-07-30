import os
import torch
from einops import rearrange
from tqdm import tqdm
from .log_utils import save_latents, log
from models import Transformer, AbsorbingDiffusion, AutoregressiveTransformer
from torch.utils.data import Dataset, DataLoader  # Add this import if not already present

def get_sampler(H, embedding_weight):
    if H.model.name == 'absorbing':
        denoise_fn = Transformer(H).cuda()
        sampler = AbsorbingDiffusion(
            H, denoise_fn, H.ct_config.model.codebook_size, embedding_weight)

    elif H.model.name == 'autoregressive':
        sampler = AutoregressiveTransformer(H, embedding_weight)

    return sampler


@torch.no_grad()
def get_samples(H, generator, sampler):

    if H.model.name == "absorbing":
        if H.sample_type == "diffusion":
            latents = sampler.sample(sample_steps=H.diffusion.sample_steps, temp=H.diffusion.temp)
        else:
            latents = sampler.sample_mlm(temp=H.diffusion.temp, sample_steps=H.diffusion.sample_steps)

    elif H.model.name == "autoregressive":
        latents = sampler.sample(H.model.temp)

    latents_one_hot = latent_ids_to_onehot(latents, H.ct_config.model.latent_shape, H.ct_config.model.codebook_size)
    q = sampler.embed(latents_one_hot)
    images = generator(q.float())

    return images


def latent_ids_to_onehot(latent_ids, latent_shape, codebook_size):
    min_encoding_indices = latent_ids.view(-1).unsqueeze(1)
    encodings = torch.zeros(
        min_encoding_indices.shape[0],
        codebook_size
    ).to(latent_ids.device)
    encodings.scatter_(1, min_encoding_indices, 1)
    
    # Handle different latent_shape formats
    if len(latent_shape) == 2:
        # [H, W] format -> assume single batch and channel
        h, w = latent_shape
        one_hot = encodings.view(
            latent_ids.shape[0],
            h,
            w,
            codebook_size
        )
    elif len(latent_shape) == 4:
        # [B, C, H, W] format
        one_hot = encodings.view(
            latent_ids.shape[0],
            latent_shape[1],
            latent_shape[2],
            latent_shape[3],
            codebook_size
        )
    else:
        raise ValueError(f"Unsupported latent_shape format: {latent_shape}")
    
    return one_hot.reshape(one_hot.shape[0], -1, codebook_size)


@torch.no_grad()
def generate_latent_ids(H, ae_ct, ae_xray, train_loader, val_loader=None):
    train_latent_ids = generate_latents_from_loader(H, ae_ct, ae_xray, train_loader)
    val_latent_ids = generate_latents_from_loader(H, ae_ct, ae_xray, val_loader)

    save_latents(H, train_latent_ids, val_latent_ids)


def generate_latents_from_loader(H, ae_ct, ae_xray, dataloader):
    latent_ids = []
    for data in tqdm(dataloader):
        xrays, ct = data["xrays"].cuda(), data["ct"].cuda()

        # NOTE: Not using AMP here to get more accurate results

        xray_latents = rearrange(xrays, "b r c h w -> (b r) c h w")
        xray_latents = ae_xray.encoder(xray_latents)
        xray_quant, _, _ = ae_xray.quantize(xray_latents)
        xray_quant = rearrange(xray_quant, "(b r) c h w -> b r (h w) c", b=xrays.size(0))
        # TODO: This can also be saved as min_encoding_indices to save memory,
        # and decompressed when training. Reltively minor since the datasets are small

        ct_latents = ae_ct.encoder(ct)
        _, _, ct_quant_stats = ae_ct.quantize(ct_latents)
        ct_min_encoding_indices = ct_quant_stats["min_encoding_indices"]
        ct_min_encoding_indices = ct_min_encoding_indices.view(ct.size(0), -1)

        latent_ids.append({
            "xray_embed": xray_quant.cpu().contiguous(),
            "ct_codes": ct_min_encoding_indices.cpu().contiguous()
        })

    return latent_ids


class ChunkedLatentDataset(Dataset):
    """Custom Dataset to load latents on-demand from chunk files"""
    def __init__(self, chunk_dir):
        self.chunk_dir = chunk_dir
        self.chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.pt')])
        self.cumulative_sizes = [0]
        self.total_samples = 0
        
        # Precompute sample counts per chunk (without loading full chunks)
        for chunk_file in self.chunk_files:
            chunk_path = os.path.join(self.chunk_dir, chunk_file)
            chunk = torch.load(chunk_path)  # Load briefly to get len
            self.total_samples += len(chunk)
            self.cumulative_sizes.append(self.total_samples)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Find which chunk the idx belongs to
        for chunk_idx, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                break
        chunk_idx -= 1  # Adjust to 0-based index
        
        # Load only the relevant chunk
        chunk_path = os.path.join(self.chunk_dir, self.chunk_files[chunk_idx])
        chunk = torch.load(chunk_path)
        
        # Get local index within the chunk
        local_idx = idx - self.cumulative_sizes[chunk_idx]
        return chunk[local_idx]  # Returns dict: {"radar_embed": tensor, "lidar_codes": tensor}


# @torch.no_grad()
# def get_latent_loaders(H, shuffle=True):
#     train_latents_fp = f'logs/{H.run.name}_{H.run.experiment}/train_latents'
#     val_latents_fp = f'logs/{H.run.name}_{H.run.experiment}/val_latents'

#     train_latent_ids = torch.load(train_latents_fp)
#     train_latent_loader = torch.utils.data.DataLoader(train_latent_ids, batch_size=H.train.batch_size, shuffle=shuffle)

#     val_latent_ids = torch.load(val_latents_fp)
#     val_latent_loader = torch.utils.data.DataLoader(val_latent_ids, batch_size=H.train.batch_size, shuffle=shuffle)

#     return train_latent_loader, val_latent_loader

@torch.no_grad()
def get_latent_loaders(H, shuffle=True):
    # Train chunks
    train_chunk_dir = f'logs/{H.run.name}_{H.run.experiment}/train_latents_chunks'
    train_dataset = ChunkedLatentDataset(train_chunk_dir)
    train_latent_loader = DataLoader(
        train_dataset,
        batch_size=H.train.batch_size,
        shuffle=shuffle,
        num_workers=4,  # Adjust based on CPU cores; use 0 if I/O issues
        pin_memory=True
    )

    # Val chunks (assuming "test" or "val" – match your generate_latents_from_dataset split_name)
    val_chunk_dir = f'logs/{H.run.name}_{H.run.experiment}/test_latents_chunks'  # Or 'val_latents_chunks' if you used "val"
    val_dataset = ChunkedLatentDataset(val_chunk_dir)
    val_latent_loader = DataLoader(
        val_dataset,
        batch_size=H.train.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

    return train_latent_loader, val_latent_loader


# TODO: rethink this whole thing - completely unnecessarily complicated
def retrieve_autoencoder_components_state_dicts(H, components_list, remove_component_from_key=False):
    state_dict = {}
    # default to loading ema models first
    ae_load_path = f"logs/{H.run.name}_{H.run.experiment}/saved_models/vqgan_ema_{H.model.sampler_load_step}.th"
    if not os.path.exists(ae_load_path):
        f"logs/{H.run.name}_{H.run.experiment}/saved_models/vqgan_{H.model.sampler_load_step}.th"
    log(f"Loading VQGAN from {ae_load_path}")
    full_vqgan_state_dict = torch.load(ae_load_path, map_location="cpu")

    for key in full_vqgan_state_dict:
        for component in components_list:
            if component in key:
                new_key = key[3:]  # remove "ae."
                if remove_component_from_key:
                    new_key = new_key[len(component)+1:]  # e.g. remove "quantize."

                state_dict[new_key] = full_vqgan_state_dict[key]

    return state_dict
