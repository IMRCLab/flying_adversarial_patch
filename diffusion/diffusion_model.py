import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional
import numpy as np

from tqdm import trange

# source for UNet: https://github.com/jbergq/simple-diffusion-model/

class ConvBlock(nn.Module):
    """Simple convolutional block: Conv2D -> BatchNorm -> Activation."""

    def __init__(self, in_size: int, out_size: int, activation: Callable = nn.ReLU) -> None:
        """Constructs the ConvBlock.

        Args:
            in_size (int): Size of input feature map.
            out_size (int): Size of output feature map.
            activation (Callable, optional): Activation function. Defaults to nn.ReLU.
        """
        super().__init__()

        self.conv = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.act = activation(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        # print("conv block x shape: ", x.shape)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

class PositionalEncoding(nn.Module):
    """Transformer sinusoidal positional encoding."""

    def __init__(self, max_time_steps: int, embedding_size: int, device: torch.device, n: int = 10000) -> None:
        """Constructs the PositionalEncoding.

        Args:
            max_time_steps (int): Number of timesteps that can be uniquely represented by encoding.
            embedding_size (int): Size of returned time embedding.
            n (int, optional): User-defined scalar. Defaults to 10000.
        """
        super().__init__()

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        i = torch.arange(embedding_size // 2)
        k = torch.arange(max_time_steps).unsqueeze(dim=1)

        # Pre-compute the embedding vector for each possible time step.
        # Store in 2D tensor indexed by time step `t` along 0th axis, with embedding vectors along 1st axis.
        self.pos_embeddings = torch.zeros(max_time_steps, embedding_size, requires_grad=False).to(device)
        self.pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))

        # self.register_buffer('pos_embeddings', self.pos_embeddings)

    def forward(self, t: Tensor) -> Tensor:
        """Returns embedding encoding time step `t`.

        Args:
            t (Tensor): Time step.

        Returns:
            Tensor: Returned position embedding.
        """
        # print("t in embedding shape: ", t.shape, t.dtype)
        return self.pos_embeddings[t.int(), :]


class TargetEncoding(nn.Module):
    def __init__(self, patch_size: tuple[int, int], embed_channels: int = 1):
        super().__init__()

        # the whole purpose of this is to learn to encode the target 
        # and bring it into a shape that is easy to concatenate with the
        # image before processing it

        self.embed_channels = embed_channels
        self.h, self.w = patch_size
        self.linear = nn.Linear(3, 512)
        self.conv = nn.Conv2d(512, 64, kernel_size=2, stride=1, padding=5)

    def forward(self, target: Tensor) -> Tensor:
        out = self.linear(target).unsqueeze(2).unsqueeze(2)
        out = self.conv(out).view(target.shape[0], self.embed_channels, self.h, self.w)
        return out

def conv3x3(
    in_size: int,
    out_size: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_size,
        out_size,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_size: int, out_size: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_size, out_size, kernel_size=1, stride=stride, bias=False)


class ResNetBlockUp(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        skip_size: int,
        t_size: Optional[int] = None,
        activation: Callable = nn.SiLU,
    ) -> None:
        super().__init__()

        self.up = nn.Upsample(scale_factor=2)
        self.block = ResNetBlock(in_size + skip_size, out_size, activation, t_size=t_size)

    def forward(self, x: Tensor, x_skip: Tensor = None, t_emb: Tensor = None) -> Tensor:
        x = self.up(x)

        # Concatenate with encoder skip connection.
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        out = self.block(x, t_emb)
        return out


class ResNetBlockDown(nn.Module):
    def __init__(
        self, in_size: int, out_size: int, t_size: Optional[int] = None, activation: Callable = nn.SiLU
    ) -> None:
        super().__init__()

        self.block = ResNetBlock(in_size, out_size, activation, stride=2, t_size=t_size)

    def forward(self, x: Tensor, t_emb: Tensor = None) -> Tensor:
        out = self.block(x, t_emb)

        return out


class ResNetBlock(nn.Module):
    """ResNet block with injection of positional encoding."""

    def __init__(
        self, in_size: int, out_size: int, activation: Callable = nn.SiLU, stride: int = 1, t_size: Optional[int] = None
    ) -> None:
        """Constructs the ResNetBlock.

        Args:
            in_size (int): Size of input feature map.
            out_size (int): Size of output feature map.
            activation (Callable, optional): Activation function. Defaults to nn.SiLU.
            stride (int): Stride of first convolutional layer (and skip convolution if in_size != out_size).
            t_size (int): Size of time positional embedding.
        """
        super().__init__()

        self.act = activation(inplace=False)

        self.t_proj = nn.Sequential(self.act, nn.Linear(t_size, out_size)) if t_size is not None else None

        self.conv1 = conv3x3(in_size, out_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.conv2 = conv3x3(out_size, out_size)
        self.bn2 = nn.BatchNorm2d(out_size)

        self.skip_conv: Optional[nn.Sequential] = None
        if in_size != out_size:
            self.skip_conv = nn.Sequential(conv1x1(in_size, out_size, stride), nn.BatchNorm2d(out_size))

    def forward(self, x: Tensor, t_emb: Tensor = None) -> Tensor:
        x_skip = x

        if self.skip_conv is not None:
            x_skip = self.skip_conv(x_skip)

        # First hidden layer.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        # Inject positional encoding in hidden state.
        if t_emb is not None and self.t_proj is not None:
            t_emb = self.t_proj(t_emb).unsqueeze(2).unsqueeze(2) # shape was (b, 128), now is (b, 128, 1, 1)
            # print(t_emb.shape)
            # print(x.shape)
            x = t_emb + x #rearrange(t_emb, "b c -> b c 1 1") + x

        # Second hidden layer.
        x = self.conv2(x)
        x = self.bn2(x)

        # Residual connection + activation.
        x += x_skip
        out = self.act(x)

        return out

class UNet(nn.Module):
    """UNet with ResNet blocks and injection of positional encoding."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: torch.device,
        num_layers: int = 5,
        features_start: int = 64,
        t_emb_size: int = 512,
        max_time_steps: int = 1000,
        target_emb : bool = True
    ) -> None:
        super().__init__()

        self.t_embedding = nn.Sequential(
            PositionalEncoding(max_time_steps, t_emb_size, device), nn.Linear(t_emb_size, t_emb_size)
        )

        if target_emb:
            self.target_embedding = TargetEncoding([80, 80])

        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")
        self.num_layers = num_layers

        if target_emb:
            self.conv_in = nn.Sequential(ConvBlock(in_size + 1, features_start), ConvBlock(features_start, features_start))
        else:
            self.conv_in = nn.Sequential(ConvBlock(in_size, features_start), ConvBlock(features_start, features_start))

        # Create encoder and decoder stages.
        layers = []
        feats = features_start
        for _ in range(num_layers - 1):  # Encoder
            layers.append(ResNetBlockDown(feats, feats * 2, t_size=t_emb_size))
            feats *= 2
        for _ in range(num_layers - 1):  # Decoder
            layers.append(ResNetBlockUp(feats, feats // 2, skip_size=feats // 2, t_size=t_emb_size))
            feats //= 2
        self.layers = nn.ModuleList(layers)

        self.conv_out = nn.Conv2d(feats, out_size, kernel_size=1)
        #self.position_layer = nn.Linear(, 3)

        # small network to denoise the position
        # in size: 7 -> noisy position + target as condition + timestep
        #self.position_net = nn.Sequential(nn.Linear(7, 512), nn.LeakyReLU(), nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 3))
        #self.position_net = nn.Sequential(nn.Linear(7, 32), nn.LeakyReLU(), nn.Linear(32, 64), nn.LeakyReLU(), nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 3))

    def forward(self, x: Tensor, target: Tensor = None, t: Tensor = None) -> Tensor:
        # print(x.shape, target.shape, t.shape)
        if t is not None:
            # Create time embedding using positional encoding.
            # t = torch.concat([t - 0.5, torch.cos(2*torch.pi*t), torch.sin(2*torch.pi*t), -torch.cos(4*torch.pi*t)], axis=1)
            # print(t.shape)
            t_emb = self.t_embedding(t.flatten()) # shape is (b, 512)
        # print("t_emb shape: ", t_emb.shape)
        if target is not None:
            target_emb = self.target_embedding(target)
            x = torch.concat((x, target_emb), dim=1)

        x = self.conv_in(x)

        # Store hidden states for U-net skip connections.
        x_i = [x]

        # Encoder stage.
        for layer in self.layers[: self.num_layers - 1]:
            x_i.append(layer(x=x_i[-1], t_emb=t_emb))

        # Decoder stage.
        for i, layer in enumerate(self.layers[self.num_layers - 1 :]):
            x_i[-1] = layer(x=x_i[-1], x_skip=x_i[-2 - i], t_emb=t_emb)

        #print(x_i[-1].shape)
        noise_patch = self.conv_out(x_i[-1])
        #position = self.position_net(torch.concat([position, target, t], axis=-1))

       
        return noise_patch#, position

class DiffusionModel():
    def __init__(self, device, in_size=1, out_size=1, lr=1e-3):
        self.in_size = in_size    # number of channels (1 -> grayscale)
        self.out_size = out_size  # number of channels
        self.lr = lr    

        self.device = device

        self.model = UNet(in_size=self.in_size, out_size=self.out_size, device=self.device).to(device)
        self.position_net = nn.Sequential(nn.Linear(7, 32), nn.LeakyReLU(), nn.Linear(32, 64), nn.LeakyReLU(), nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 3)).to(device)
    
    # def forward(self, x: Tensor, position: Tensor, target: Tensor, t: Tensor) -> Tensor:
    #     noise_patch = self.model(x, target, t)
    #     noise_position = self.position_net(torch.cat([position, target, t], axis=-1))
    #     return noise_patch, noise_position


    def get_alpha_betas(self, N: int):
        """Schedule from the original paper. Commented out is sigmoid schedule from:

        'Score-Based Generative Modeling through Stochastic Differential Equations.'
        Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar,
        Stefano Ermon, Ben Poole (https://arxiv.org/abs/2011.13456)
        """
        beta_min = 0.1
        beta_max = 20.
        betas = np.array([beta_min/N + i/(N*(N-1))*(beta_max-beta_min) for i in range(N)])
        #betas = np.random.uniform(10e-4, .02, N)  # schedule from the 2020 paper
        alpha_bars = np.cumprod(1 - betas)
        return alpha_bars, betas
    

    # def get_alpha_betas(N: int):
    #     """Schedule from the original paper. Commented out is sigmoid schedule from:

    #     'Score-Based Generative Modeling through Stochastic Differential Equations.'
    #     Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar,
    #     Stefano Ermon, Ben Poole (https://arxiv.org/abs/2011.13456)
    #     """
    #     beta_min = 0.1
    #     beta_max = 20.0
    #     betas = np.array(
    #         [beta_min / N + i / (N * (N - 1)) * (beta_max - beta_min) for i in range(N)]
    #     )
    #     # betas = np.random.uniform(10e-4, 0.02, N)  # schedule from the 2020 paper
    #     alpha_bars = np.cumprod(1 - betas)
    #     return alpha_bars, betas
        
    
    def train(self, data_loader: torch.utils.data.DataLoader, device: torch.device, nepochs: int = 10, denoising_steps: int = 1_000):
        """Alg 1 from the DDPM paper"""
        self.model.train()
        optimizer_patch = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        optimizer_position = torch.optim.Adam(self.position_net.parameters(), lr=self.lr)
        alpha_bars, _ = self.get_alpha_betas(denoising_steps)      # Precompute alphas

        all_losses = []

        losses = []
        individual_losses = []
        try:
            for epoch in trange(nepochs):
                for [patches, targets, positions] in data_loader:
                    patches = patches.to(device)
                    targets = targets.to(device)
                    positions = positions.to(device)
                    # print(patches.shape, targets.shape, positions.shape)
                    optimizer_patch.zero_grad()
                    optimizer_position.zero_grad()
                    # Fwd pass
                    t = torch.randint(denoising_steps, size=(patches.shape[0],))  # sample timesteps - 1 per datapoint
                    alpha_t_patch = torch.index_select(torch.Tensor(alpha_bars), 0, t).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)    # Get the alphas for each timestep
                    alpha_t_position = torch.index_select(torch.Tensor(alpha_bars), 0, t).unsqueeze(1).to(device)

                    noise_patch = torch.randn(*patches.shape, device=device)   # Sample DIFFERENT random noise for each datapoint
                    noise_position = torch.rand(*positions.shape, device=device)

                    model_in = alpha_t_patch**.5 * patches + noise_patch*(1-alpha_t_patch)**.5   # Noise corrupt the data (eq14)
                    noisy_position = alpha_t_position**.5 * positions + noise_position*(1-alpha_t_position)**.5

                    # print(model_in.shape, noisy_position.shape, targets.shape, t.shape)

                    #pred_noise_patch, pred_noise_position = self.model(model_in, noisy_position, targets, t.unsqueeze(1).to(device))
                    pred_noise_patch = self.model(model_in, targets, t.unsqueeze(1).to(device))
                    pred_noise_position = self.position_net(torch.cat([noisy_position, targets, t.unsqueeze(1).to(device)], axis=-1))

                    loss_patch = torch.mean((noise_patch - pred_noise_patch)**2) # Compute loss on prediction (eq14)
                    loss_position = torch.mean((noise_position - pred_noise_position)**2)
                    loss = loss_patch + loss_position  
                    
                    individual_losses.append([loss_patch.detach().cpu().numpy(), loss_position.detach().cpu().numpy()])
                    losses.append(loss.detach().cpu().numpy())
                    all_losses.append(loss.detach().cpu().numpy())

                    # Bwd pass
                    loss_patch.backward()
                    optimizer_patch.step()
                    loss_position.backward()
                    optimizer_position.step()

                if (epoch+1) % 10 == 0:
                    mean_loss = np.mean(np.array(losses))
                    print("Epoch %d,\t Loss %f, Patch Loss %f, Position Loss %f " % (epoch+1, mean_loss, np.mean(np.array(individual_losses)[:, 0]), np.mean(np.array(individual_losses)[:, 1])))
                    losses = []
                    individual_losses = []

                    # Evaluation: generate 4 patches for 4 random targets
                    n_samples = 4
                    x = np.random.uniform(0, 2, n_samples)
                    y = np.random.uniform(-1, 1, n_samples)
                    z = np.random.uniform(-0.5, 0.5, n_samples)
                    r_targets = torch.tensor(np.stack((x, y, z)).T, dtype=torch.float32).to(device)

                    sample_patches, sample_positions = self.sample(n_samples, r_targets, device, patch_size=(80, 80), n_steps=denoising_steps)
                    sample_patches = sample_patches.detach().cpu().numpy()
                    # print(sample_patches.shape)

                    # Plot the patches
                    fig, axs = plt.subplots(1, n_samples, figsize=(15, 5))
                    for i, sample in enumerate(sample_patches):
                        axs[i].imshow(sample[0], cmap='gray')
                        axs[i].set_title(f'Sample {i+1}')
                        axs[i].axis('off')
                    plt.tight_layout()
                    plt.savefig(f'results/diffusion_training/epochs{nepochs}_ds{denoising_steps}/samples_epoch_{epoch+1}.pdf')
                    plt.close()
        except KeyboardInterrupt:
            "Interrupting training..."
            mean_loss = np.mean(np.array(losses))
            print("Epoch %d,\t Loss %f, Patch Loss %f, Position Loss %f " % (epoch+1, mean_loss, np.mean(np.array(individual_losses)[:, 0]), np.mean(np.array(individual_losses)[:, 1])))
        
        return all_losses

    def sample(self, n_samples: int, targets: torch.tensor, device: torch.device, patch_size: tuple[int, int], n_steps: int=1_000):
        """Alg 2 from the DDPM paper."""
        self.model.eval()
        # make sure that targets is a tensor
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)
        # and of shape (n_samples, 3)
        if len(targets.shape) == 1 or targets.shape[0] == 1:
            targets = targets.repeat(n_samples, 1)

        with torch.no_grad():
            patch = torch.randn((n_samples, 1, *patch_size)).to(device)
            position = torch.randn((n_samples, 3)).to(device)
            #print("Initial position: ", position)
            targets = targets.to(device)
            alpha_bars, betas = self.get_alpha_betas(n_steps)
            #alphas = 1 - betas
            alphas = np.clip(1 - betas, 1e-8, np.inf)
            for t in range(len(alphas))[::-1]:
                ts = t * torch.ones((n_samples, 1), dtype=torch.int32).to(device)
                ab_t = alpha_bars[t] * torch.ones((n_samples, 1), dtype=torch.int32).to(device)  # Tile the alpha to the number of samples
                z_patch = (torch.randn((n_samples, 1, *patch_size)) if t > 1 else torch.zeros((n_samples, 1, *patch_size))).to(device)
                z_position = (torch.randn((n_samples, 3)) if t > 1 else torch.zeros((n_samples, 3))).to(device)
                
                #pred_noise, pred_pos = self.model(patch, position, targets, ts)
                pred_noise = self.model(patch, targets, ts)
                pred_pos = self.position_net(torch.cat([position, targets, ts], axis=-1))
                
                patch = 1 / alphas[t]**.5 * (patch - (betas[t]/(1-ab_t)**.5).unsqueeze(2).unsqueeze(2) * pred_noise)
                patch += betas[t]**0.5 * z_patch

                position = 1 / alphas[t]**.5 * (position - (betas[t]/(1-ab_t)**.5) * pred_pos)
                position += betas[t]**0.5 * z_position

                # print("Intermediate position:", position)
                #position.clip_(-0.999999, 0.999999)

            # postprocessing
            #print(patch.min(), patch.max())
            #print(position)
            patch = torch.stack([(x - torch.min(x)) / (torch.max(x) - torch.min(x)) for x in patch]) # normalize
            #position = self._reverse_normalization(position)
            print("Position range: ", position.min(), position.max())
            return patch, position.clip_(-0.999999, 0.999999)

    def _reverse_normalization(self, sampled_position):

        # print(sampled_position, sampled_position.shape)
        position = torch.arctanh(sampled_position)
        # print(position)
        position = torch.stack(norm_transformation(*position.T, scale_min=0.2, scale_max=0.7)).T
        # print(position, position.shape)

        return position

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    def save(self, path):
        torch.save(self.model.state_dict(), path)
    

def norm_transformation(sf, tx, ty, scale_min=0.3, scale_max=0.5, tx_min=-10., tx_max=100., ty_min=-10., ty_max=80.):
    scaling_norm = (scale_max - scale_min) * (torch.tanh(sf) + 1) * 0.5 + scale_min # normalizes scaling factor to range [0.3, 0.5]
    tx_norm = (tx_max - tx_min) * (torch.tanh(tx) + 1) * 0.5 + tx_min
    ty_norm = (ty_max - ty_min) * (torch.tanh(ty) + 1) * 0.5 + ty_min

    return scaling_norm, tx_norm, ty_norm

def inverse_norm(val, minimum, maximum):
    return np.arctanh(2* ((val - minimum) / (maximum - minimum)) - 1)
    
if __name__ == '__main__':

    import pickle
    import argparse
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='Paths to the dataset pickle files.')
    parser.add_argument('--epochs', type=int, default=1_000, help='Number of epochs to train.')
    parser.add_argument('--denoise_steps', type=int, default=1_000, help='Number of denoising steps.')
    parser.add_argument('--output', type=str, default='trained_model.pth', help='Path to save the model.')
    args = parser.parse_args()

    data = []
    for dataset_path in args.datasets:
        with open(dataset_path, 'rb') as f:
            data += pickle.load(f)


    patches = []
    targets = []
    positions = []
    for i in range(len(data)):
        patches.append(data[i][0])
        targets.append(data[i][1])
        positions.append(data[i][2])

    # print(patches.min(), patches.max())
    
    patches = np.array(patches) #/ 255.
    patches = (patches - 0.5) * 2  # Normalize to range [-1, 1]
    targets = np.array(targets)
    positions = np.array(positions)

    unnorm_positions = []
    for [sf, tx, ty] in positions:
        sf_unnorm = inverse_norm(sf, 0.2, 0.7)
        tx_unnorm = inverse_norm(tx, -10., 100.)
        ty_unnorm = inverse_norm(ty, -10., 100.)
        unnorm_positions.append([sf_unnorm, tx_unnorm, ty_unnorm])

    unnorm_positions = np.tanh((np.array(unnorm_positions)))

    patch_size = patches.shape[-2:]

    # patches = np.array([patch - np.min(patch)) / (np.max(patch) - np.min(patch) for patch in patches]) # normalize
    patches = torch.tensor(patches).unsqueeze(1)
    targets = torch.tensor(targets)
    positions = torch.tensor(unnorm_positions, dtype=torch.float32)


    #patches = torch.stack([(x - torch.min(x)) / (torch.max(x) - torch.min(x)) for x in patches]) # normalize

    print(patches.shape, targets.shape, positions.shape)

    print(patches.shape, patches.min(), patches.max())
    print(targets.shape, targets.min(), targets.max())
    print(positions.shape, positions.min(), positions.max())

    # print(patches.shape)

    # Define dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = torch.utils.data.TensorDataset(patches, targets, positions)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    # model = UNet(in_size=1, out_size=1, device=device)
    # model.to(device)

    model = DiffusionModel(device, lr=1e-5)

    # Ensure the directory exists
    output_dir = Path(f'results/diffusion_training/epochs{args.epochs}_ds{args.denoise_steps}/')
    output_dir.mkdir(parents=True, exist_ok=True)

    # training
    print("Start training..")
    all_losses = model.train(loader, device, nepochs=args.epochs, denoising_steps=args.denoise_steps)
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)

    # model.load('diffusion/frontnet_100e.pth')
    
    n_samples = 20
    x = np.random.uniform(0,2,n_samples)
    y = np.random.uniform(-1,1,n_samples)
    z = np.random.uniform(-0.5,0.5,n_samples)

    r_targets = torch.tensor(np.stack((x, y, z)).T, dtype=torch.float32)

    print(r_targets)

    sample_patches, sample_positions = model.sample(n_samples, r_targets, device, patch_size=patch_size, n_steps=args.denoise_steps)
    sample_patches = sample_patches.detach().cpu().numpy()
    sample_positions = sample_positions.detach().cpu().numpy()
    
    print(sample_patches.shape, sample_patches.min(), sample_patches.max())
    print(sample_positions)


    # Plot all sample patches in a grid
    fig, axs = plt.subplots(4, 5, figsize=(15, 12))
    for i, sample in enumerate(sample_patches):
        ax = axs[i // 5, i % 5]
        ax.imshow(sample[0], cmap='gray')
        ax.set_title(f'Sample {i+1}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'final_samples.pdf')
    plt.show()
    

    # # # running into memory issues with this sample function! fix: don't compute gradients
    # # with torch.no_grad():
    # #     samples = sample(model, targets, device, n_samples=n_samples, patch_size=patch_size, n_steps=1_000).detach().cpu().numpy()
    # # print(samples.shape)
    # # print(np.min(samples), np.max(samples))

    # # # fig = plt.figure(constrained_layout=True)
    # # # subfigs = fig.subfigures(2, 1)
    # # # axs_gt = subfigs[0].subplots(1, 2)
    # # # for i, gt_patch in enumerate(gt_patches):
    # # #     axs_gt[i].imshow(gt_patch, cmap='gray')
    # # #     axs_gt[i].set_title(f'ground truth {i}')

    

    # fig = plt.figure(constrained_layout=True)
    # axs_samples = fig.subplots(1, n_samples)
    # for i, sample in enumerate(samples):
    #     axs_samples[i].imshow(sample[0], cmap='gray')
    #     axs_samples[i].set_title(f'sample {i}')
    # fig.savefig(f'results/diffusion_training/samples1.png', dpi=200)
    # plt.show()