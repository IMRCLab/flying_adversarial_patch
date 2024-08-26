import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional
import numpy as np
import yaml

from tqdm import trange

# source for UNet: https://github.com/jbergq/simple-diffusion-model/

DATA_PATH = "results/yolo_patches"

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
    def __init__(self, patch_size: [int, int], embed_channels: int = 1):
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

        out = self.conv_out(x_i[-1])

        return out

class DiffusionModel():
    def __init__(self, device, in_size=1, out_size=1, lr=1e-4):
        self.in_size = in_size    # number of channels (1 -> grayscale)
        self.out_size = out_size  # number of channels
        self.lr = lr    

        self.device = device

        self.model = UNet(in_size=self.in_size, out_size=self.out_size, device=self.device).to(device)

    def get_alpha_betas(self, N: int):
        """Schedule from the original paper. Commented out is sigmoid schedule from:

        'Score-Based Generative Modeling through Stochastic Differential Equations.'
        Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar,
        Stefano Ermon, Ben Poole (https://arxiv.org/abs/2011.13456)
        """
        beta_min = 0.1
        beta_max = 20.
        #betas = np.array([beta_min/N + i/(N*(N-1))*(beta_max-beta_min) for i in range(N)])
        betas = np.random.uniform(10e-4, .02, N)  # schedule from the 2020 paper
        alpha_bars = np.cumprod(1 - betas)
        return alpha_bars, betas
        
    
    def train(self, data_loader: torch.utils.data.DataLoader, device: torch.device, nepochs: int = 10, denoising_steps: int = 1_000):
        """Alg 1 from the DDPM paper"""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        alpha_bars, _ = self.get_alpha_betas(denoising_steps)      # Precompute alphas

        all_losses = []

        losses = []
        for epoch in trange(nepochs):
            for [patches, targets] in data_loader:
                # print('a')
                patches = patches.to(device)
                targets = targets.to(device)

                # print('train patches shape', patches.shape)
                # print('train targets shape', targets.shape)
                optimizer.zero_grad()
                # Fwd pass
                t = torch.randint(denoising_steps, size=(patches.shape[0],))  # sample timesteps - 1 per datapoint
                alpha_t = torch.index_select(torch.Tensor(alpha_bars), 0, t).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)    # Get the alphas for each timestep

                noise = torch.randn(*patches.shape, device=device)   # Sample DIFFERENT random noise for each datapoint
                
                model_in = alpha_t**.5 * patches + noise*(1-alpha_t)**.5   # Noise corrupt the data (eq14)
                out = self.model(model_in, targets, t)
                loss = torch.mean((noise - out)**2)     # Compute loss on prediction (eq14)
                losses.append(loss.detach().cpu().numpy())
                all_losses.append(loss.detach().cpu().numpy())

                # Bwd pass
                loss.backward()
                optimizer.step()

                if (epoch+1) % 10 == 0:
                    mean_loss = np.mean(np.array(losses))
                    losses = []
                    print("Epoch %d,\t Loss %f " % (epoch+1, mean_loss))

        return all_losses

    def sample(self, n_samples: int, targets: torch.tensor, device: torch.device, patch_size: (int, int), n_steps: int=1_000):
        """Alg 2 from the DDPM paper."""
        self.model.eval()
        with torch.no_grad():
            x_t = torch.randn((n_samples, 1, *patch_size)).to(device)
            targets = targets.to(device)
            alpha_bars, betas = self.get_alpha_betas(n_steps)
            alphas = 1 - betas
            for t in range(len(alphas))[::-1]:
                ts = t * torch.ones((n_samples, 1), dtype=torch.int32).to(device)
                ab_t = alpha_bars[t] * torch.ones((n_samples, 1), dtype=torch.int32).to(device)  # Tile the alpha to the number of samples
                z = (torch.randn((n_samples, 1, *patch_size)) if t > 1 else torch.zeros((n_samples, 1, *patch_size))).to(device)
                model_prediction = self.model(x_t, targets, ts.squeeze(1))
                x_t = 1 / alphas[t]**.5 * (x_t - (betas[t]/(1-ab_t)**.5).unsqueeze(2).unsqueeze(2) * model_prediction)
                x_t += betas[t]**0.5 * z

            x_t = (x_t - torch.min(x_t)) / (torch.max(x_t) - torch.min(x_t)) # normalize
            return x_t


    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    
    
if __name__ == '__main__':

    import pickle
    import matplotlib.pyplot as plt

    # with open('diffusion_data/FAP_combined.pickle', 'rb') as f:
    with open('all_yolo_patches.pkl', 'rb') as f:
        data = pickle.load(f)    


    patches = []
    targets = []
    for i in range(len(data)):
        patches.append(data[i][0])
        targets.append(data[i][1])
    
    patches = np.array(patches)
    targets = np.array(targets)

    patch_size = patches.shape[-2:]

    patches = (patches - np.min(patches)) / (np.max(patches) - np.min(patches)) # normalize
    patches = torch.tensor(patches).unsqueeze(1)
    targets = torch.tensor(targets)

    print('patches shape', patches.shape)
    print('targets shape', targets.shape)

    # Define dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = torch.utils.data.TensorDataset(patches, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    # model = UNet(in_size=1, out_size=1, device=device)
    # model.to(device)

    model = DiffusionModel(device)

    # training
    print("Start training..")
    # all_losses = model.train(loader, device, 2, denoising_steps=1_000)
    all_losses = model.train(loader, device, nepochs=1000)

    # print(all_losses)
    torch.save(model.model.state_dict(), f'yolo_conditioned_unet_{patch_size[0]}x{patch_size[1]}_{1_000}_3256i_255.pth')
    
    n_samples = 5
    x = np.random.uniform(0,2,n_samples)
    y = np.random.uniform(-1,1,n_samples,)
    z = np.random.uniform(-0.5,0.5,n_samples,)

    r_targets = torch.tensor(np.stack((x, y, z)).T, dtype=torch.float32)

    samples = model.sample(n_samples, r_targets, device, patch_size=patch_size, n_steps=1000)
    # print(samples.min(), samples.max())
    print('samples min', samples.min())
    print('samples max', samples.max())

    

    # # running into memory issues with this sample function! fix: don't compute gradients
    # with torch.no_grad():
    #     samples = sample(model, targets, device, n_samples=n_samples, patch_size=patch_size, n_steps=1_000).detach().cpu().numpy()
    # print(samples.shape)
    # print(np.min(samples), np.max(samples))

    # # fig = plt.figure(constrained_layout=True)
    # # subfigs = fig.subfigures(2, 1)
    # # axs_gt = subfigs[0].subplots(1, 2)
    # # for i, gt_patch in enumerate(gt_patches):
    # #     axs_gt[i].imshow(gt_patch, cmap='gray')
    # #     axs_gt[i].set_title(f'ground truth {i}')

    fig = plt.figure(constrained_layout=True)
    axs_samples = fig.subplots(1, n_samples)
    for i, sample in enumerate(samples):
        axs_samples[i].imshow(sample[0].cpu(), cmap='gray')
        axs_samples[i].set_title(f'sample {i}')
    fig.savefig(f'samples_conditioning_4_{patch_size[0]}x{patch_size[1]}.png', dpi=200)
    plt.show()


