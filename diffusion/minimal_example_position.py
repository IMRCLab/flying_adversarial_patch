# Base code by Cornelius Braun 

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange

def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            # m.weight.data.fill_(0.0)
            # m.bias.data.fill_(0.0)
# Define net
class Net(nn.Module):
  def __init__(self, nhidden: int = 512):
    super().__init__()
    # layers = [nn.Linear(3, nhidden)] # Change this to 6 if you want to use the fourier embeddings of t
    # for _ in range(5):
    #   layers.append(nn.Linear(nhidden, nhidden))
    # layers.append(nn.Linear(nhidden, 2))
    # self.linears = nn.ModuleList(layers)

    # #Iinit using kaiming
    # for layer in self.linears:
    #   nn.init.kaiming_uniform_(layer.weight)

    self.position_net = nn.Sequential(nn.Linear(7, 512), nn.LeakyReLU(), nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 3))
    #self.position_net = nn.Sequential(nn.Linear(7, 32), nn.LeakyReLU(), nn.Linear(32, 64), nn.LeakyReLU(), nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 3))
    #self.position_net = nn.Sequential(nn.Linear(4, 32), nn.LeakyReLU(), nn.Linear(32, 64), nn.LeakyReLU(), nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 3))

    #self.position_net.apply(weights_init)

  def forward(self, pos, target, t):
    # Optional: Use Fourier feature embeddings for t, cf. transformers
    #t = torch.concat([t - 0.5, torch.cos(2*torch.pi*t), torch.sin(2*torch.pi*t), -torch.cos(4*torch.pi*t)], axis=1)
    # x = torch.concat([x, t], axis=-1)
    # for l in self.linears[:-1]:
    #   x = nn.LeakyReLU()(l(x))
    # return self.linears[-1](x)

    #print(pos.shape, target.shape, t.shape)
    x = torch.concat([pos, target, t], axis=-1)
    # print(x.shape)
    return self.position_net(x)
  
def get_alpha_betas(N: int):
  """Schedule from the original paper. Commented out is sigmoid schedule from:

  'Score-Based Generative Modeling through Stochastic Differential Equations.'
   Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar,
   Stefano Ermon, Ben Poole (https://arxiv.org/abs/2011.13456)
  """
  beta_min = 0.1
  beta_max = 20.
  betas = np.array([beta_min/N + i/(N*(N-1))*(beta_max-beta_min) for i in range(N)])
  # betas = np.random.uniform(10e-4, .02, N)  # schedule from the 2020 paper
  alpha_bars = np.cumprod(1 - betas)
  return alpha_bars, betas

def train(nepochs: int, loader, device: torch.device, denoising_steps: int = 1_000):
  """Alg 1 from the DDPM paper"""
  model = Net()
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
  alpha_bars, _ = get_alpha_betas(denoising_steps)      # Precompute alphas
  # print("Alpha bars shape: ", alpha_bars.shape)
  plt.plot(alpha_bars**0.5, label="Amount Signal")
  plt.plot((1 - alpha_bars)**0.5, label="Amount Noise")
  plt.legend()
  plt.savefig("scheduler.png")
  plt.close()
  losses = []
  print("Start training...")
  for epoch in trange(nepochs):
    for [data] in loader:
      data = data.to(device)
    #   print("Data: ", data.shape)
      positions, targets = torch.split(data, [3, 3], dim=1)
    #   print("Position", positions.shape)
    #   print("Target", target.shape)
      optimizer.zero_grad()

      # Fwd pass
      t = torch.randint(denoising_steps, size=(data.shape[0],))  # sample timesteps - 1 per datapoint
      alpha_t = torch.index_select(torch.Tensor(alpha_bars), 0, t).unsqueeze(1).to(device)    # Get the alphas for each timestep
      noise = torch.randn(*positions.shape, device=device)   # Sample DIFFERENT random noise for each datapoint
      # print("Positions:", positions)
      # print("Anteil Position: ", positions*alpha_t)
      # print("Noise:", noise)
      # print("Amount of noise: ", noise*(1-alpha_t))
      # print("Model in: ", alpha_t * positions + noise*(1-alpha_t))
      
      model_in = alpha_t**.5 * positions + noise*(1-alpha_t)**.5   # Noise corrupt the data (eq14)
      out = model(model_in, targets, t.unsqueeze(1).to(device))
      loss = torch.mean((noise - out)**2)     # Compute loss on prediction (eq14)
      losses.append(loss.detach().cpu().numpy())

      # Bwd pass
      loss.backward()
      optimizer.step()

    if (epoch+1) % 100 == 0:
        mean_loss = np.mean(np.array(losses[-100:]))
        print("Epoch %d,\t Loss %f " % (epoch+1, mean_loss))
        samples = sample(model, targets[:10], device, len(targets[:10]), n_steps=denoising_steps).to('cpu')
        print("Sampled positions: ", samples, samples.min(), samples.max())

  return model, np.array(losses)


def sample(model: nn.Module, targets: torch.tensor, device: torch.device, n_samples: int = 50, n_steps: int=100):
    """Alg 2 from the DDPM paper."""
    with torch.no_grad():
        x_t = torch.randn((n_samples, 3)).to(device)
        targets = targets.to(device)
        alpha_bars, betas = get_alpha_betas(n_steps)
        alphas = 1 - betas
        for t in range(len(alphas))[::-1]:
            ts = t * torch.ones((n_samples, 1)).to(device)
            ab_t = alpha_bars[t] * torch.ones((n_samples, 1)).to(device)  # Tile the alpha to the number of samples
            z = (torch.randn((n_samples, 3)) if t > 1 else torch.zeros((n_samples, 3))).to(device)
            model_prediction = model(x_t, targets, ts)
            x_t = 1 / alphas[t]**.5 * (x_t - betas[t]/(1-ab_t)**.5 * model_prediction)
            x_t += betas[t]**0.5 * z

        return x_t#.clip(-0.999999, 0.9999999)
    

def load_pickle(path):
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)

    patches = []
    targets = []
    positions = []
    for i in range(len(data)):
        patches.append(data[i][0])
        targets.append(data[i][1])
        positions.append(data[i][2])

    return np.array(patches)*255., np.array(targets), np.array(positions)

def norm_transformation(sf, tx, ty, scale_min=0.3, scale_max=0.5, tx_min=-10., tx_max=100., ty_min=-10., ty_max=80.):
    # tx_tanh = torch.tanh(tx) #* 0.8
    # ty_tanh = torch.tanh(ty) #* 0.8

    # new patch placement implementation might need different tx, ty limits!:
    tx_norm = (tx_max - tx_min) * (torch.tanh(tx) + 1) * 0.5 + tx_min
    ty_norm = (ty_max - ty_min) * (torch.tanh(ty) + 1) * 0.5 + ty_min

    scaling_norm = (scale_max - scale_min) * (torch.tanh(sf) + 1) * 0.5 + scale_min # normalizes scaling factor to range [0.3, 0.5]

    return scaling_norm, tx_norm, ty_norm

def reverse_normalization(sampled_position):

  # print(sampled_position, sampled_position.shape)
  position = torch.arctanh(sampled_position)
  # print(position)
  position = torch.stack(norm_transformation(*position.T, scale_min=0.2, scale_max=0.7)).T
  # print(position, position.shape)

  return position

if __name__ == "__main__":
    torch.manual_seed(2211)
    np.random.seed(2211)
    import matplotlib.pyplot as plt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    patches_f, targets_f, positions_f = load_pickle('FAP_combined.pickle')
    patches_y, targets_y, positions_y = load_pickle('yolopatches.pickle')

    print(targets_f[:3], positions_f[:3])
    print(targets_y[:3], positions_y[:3])
    positions = np.vstack((positions_f, positions_y))
    targets = np.vstack((targets_f, targets_y))

    from exp_finetuning import inverse_norm
    scale_min = 0.2
    scale_max = 0.7
    tx_min=-10.
    tx_max=100.
    ty_min=-10.
    ty_max=80.

    unnorm_positions = []
    for [sf, tx, ty] in positions:
        sf_unnorm = inverse_norm(sf, scale_min, scale_max)
        tx_unnorm = inverse_norm(tx, tx_min, tx_max)
        ty_unnorm = inverse_norm(ty, ty_min, ty_max)
        unnorm_positions.append([sf_unnorm, tx_unnorm, ty_unnorm])

    unnorm_positions = np.tanh((np.array(unnorm_positions)))

    # # def inverse_tanh(x):
    # #   return 0.5 * np.log((1 + x) / (1 - x))

    # sanity_check_positions = np.arctanh(unnorm_positions)
    # sanity_check_positions = np.concatenate([norm_transformation(*torch.Tensor(sanity_check_positions).T, scale_min=0.2, scale_max=0.7)]).T

    # np.testing.assert_almost_equal(positions, sanity_check_positions, decimal=5)

    data = np.hstack((unnorm_positions, targets))
    print(data.shape)
    print(data[:3])

    # data = np.hstack((positions, targets))
    # print(data[:3])

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data).float())
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    trained_model, losses = train(1_000, loader, device, denoising_steps=50)
    trained_model = trained_model.eval()


    plt.plot(losses)
    plt.savefig("losses.png")


    data = next(iter(loader))[0]
    positions, targets = torch.split(data, [3, 3], dim=1)

    samples = sample(trained_model, targets[:10], device, len(targets[:10]), n_steps=50).to('cpu')
    
    print(positions[:10])
    print(samples)

    error = torch.mean((positions[:10] - samples)**2)
    print(error)

    # print(sanity_check_positions[:10])
    # reversed_samples = reverse_normalization(samples)
    # print(reversed_samples.shape)
    # print(reversed_samples)

    # error = np.mean((reversed_samples.numpy() - sanity_check_positions[:10])**2)
    # print(error)


# # Get data from a circle
# thetas = np.random.uniform(0, 2*np.pi, 50)
# x = np.cos(thetas) + np.random.normal(0, 3e-2, 50)
# y = np.sin(thetas) + np.random.normal(0, 3e-2, 50)
# data = np.vstack((x, y)).T
# print(data.shape)

# # plt.figure(figsize=(5, 5))
# # plt.scatter(x, y)
# # plt.show()

# # Define dataset
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dataset = torch.utils.data.TensorDataset(torch.Tensor(data))
# loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)


# # a, b = get_alpha_betas(100)
# # plt.plot(a, label="Amount Signal")
# # plt.plot(1 - a, label="Amount Noise")
# # plt.legend()
# # plt.show()

# # training
# trained_model = train(15_000).eval()


# # inference
# samples = sample(trained_model).detach().cpu().numpy()
# print(samples.shape)
# plt.figure(figsize=(5,5))
# plt.scatter(x, y)
# plt.scatter(*(samples.T))
# plt.show()