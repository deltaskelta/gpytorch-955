from typing import List, Tuple

import gpytorch  # type: ignore
import numpy as np  # type: ignore
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm, trange  # type: ignore

device = torch.device("cuda:5")


class GPLoader(Dataset):
    def __init__(self, years: List[str], device: torch.device) -> None:
        self.x = torch.Tensor()
        self.y = torch.Tensor()

        for year in years:
            x_arr = np.load(f"./x_{year}.npy")[:1000, :]
            y_arr = np.load(f"./y_{year}.npy")[:1000, :]

            self.x = torch.cat((self.x, torch.from_numpy(x_arr)))
            self.y = torch.cat((self.y, torch.log(torch.from_numpy(y_arr))))

            for i in range(12, 24):
                max_ = self.x[:, i].max()
                self.x[:, i] = self.x[:, i] / max_

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[i].cuda(), self.y[i].cuda()

    def get_inducing_points(self, n: int) -> torch.Tensor:
        return self.x[:n, :]


class LargeFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super(LargeFeatureExtractor, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(707, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.layers(x)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self, train_x, train_y, likelihood, feature_extractor: nn.Module
    ) -> None:
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
            num_dims=2,
            grid_size=20,
        )
        self.feature_extractor = feature_extractor

    def forward(self, x):
        proj_x = self.feature_extractor(x)
        proj_x = proj_x - proj_x.min().item()
        proj_x = 2 * (proj_x / proj_x.max().item()) - 1

        mean_x = self.mean_module(proj_x)
        covar_x = self.covar_module(proj_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train() -> None:
    dataset = GPLoader(["2017"], device)
    train_x = dataset.x
    train_y = dataset.y

    feature_extractor = LargeFeatureExtractor()  # type: ignore
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood, feature_extractor)

    model.train()
    likelihood.train()

    # Use the adam optimizer. Includes GaussianLikelihood parameters
    optimizer = torch.optim.Adam(
        [
            {"params": model.feature_extractor.parameters()},
            {"params": model.covar_module.parameters()},
            {"params": model.mean_module.parameters()},
            {"params": model.likelihood.parameters()},
        ],
        lr=0.1,
    )

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    loss_log = tqdm(total=0, position=3)
    training_iter = 10000
    for i in trange(training_iter, desc="Iter", position=0):
        # Zero gradients, call model, calculate loss
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y).sum()

        d = (
            f"loss: {loss.item():.2f} "
            f"med: {torch.median(output.loc):.2f}, "
            f"minmax: {output.loc.min():.2f} {output.loc.max():.2f} "
            f"noise: {model.likelihood.noise.item():.2f}"
        )
        loss_log.set_description(d)

        loss.backward()
        optimizer.step()


def predict() -> None:
    pass


if __name__ == "__main__":
    train()
