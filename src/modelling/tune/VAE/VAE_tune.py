import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import MinCovDet
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.distributions import Normal
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset

import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser(description="Tune VAE")

wandb.login(key="dfe8bf8c4e2c8958e50eade06c8caa4cbfdd6bef")

# python src/modelling/tune/ucl_cluster_tune/VAE/VAE_tune.py --data_path "data/processed_data" --feature_type "rsfmri" --project_title "VAE_local_cluster_test"


def parse_list_of_lists(arg_value):
    # Converts a string format like "30-30;40-40-40" into a list of lists [[30, 30], [40, 40, 40]]
    return [[int(y) for y in x.split("-")] for x in arg_value.split(";")]


def float_parse_list(arg_value):
    return [float(x) for x in arg_value.split("-")]


def int_parse_list(arg_value):
    return [int(x) for x in arg_value.split("-")]


parser.add_argument(
    "--data_path",
    type=str,
    default="",
    help="Path to processed data",
)
parser.add_argument(
    "--feature_type",
    type=str,
    default="",
    help="Brain features of interest, options: 'cortical_thickness, cortical_volume, rsfmri'",
)
parser.add_argument(
    "--project_title",
    type=str,
    default="Tune VAE",
    help="Title of the project for weights and biases",
)
parser.add_argument(
    "--batch_size",
    type=int_parse_list,
    default=[64, 128, 256],
    help="Batch size for training, e.g., '64-128-256', which will be parsed into [64, 128, 256]",
)
parser.add_argument(
    "--learning_rate",
    type=float_parse_list,
    default=[
        0.0005,
        0.0001,
        0.00005,
        0.00001,
    ],
    help="Learning rate for the optimizer, e.g., '0.005-0.001-0.0005-0.0001', which will be parsed into [0.005, 0.001, 0.0005, 0.0001]",
)
parser.add_argument(
    "--latent_dim",
    type=int_parse_list,
    default=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    help="Dimensions of the latent space, e.g., '10-11-12', which will be parsed into [10, 11, 12]",
)
parser.add_argument(
    "--hidden_dim",
    type=parse_list_of_lists,
    default=[
        [30],
        [30, 30],
        [40],
        [40, 40],
        [50],
        [50, 50],
        [60],
        [60, 60],
    ],
    help="Pass dimensions of multiple hidden layers, use ';' to separate layers and '-' to separate dimensions within layers, e.g., '30-30;40-40-40'",
)


args = parser.parse_args()
processed_data_path = Path(args.data_path)
feature_type = args.feature_type

feature_sets_map = {
    "cortical_thickness": "t1w_cortical_thickness_rois",
    "cortical_volume": "t1w_cortical_volume_rois",
    "rsfmri": "gordon_net_subcor_limbic_no_dup",
}


brain_features_of_interest_path = Path(
    processed_data_path,
    "brain_features_of_interest.json",
)

with open(brain_features_of_interest_path, "r") as f:
    brain_features_of_interest = json.load(f)

FEATURES = brain_features_of_interest[feature_sets_map[feature_type]]

TRAIN_DATA_PATH = Path(
    processed_data_path,
    "all_brain_features_resid.csv",
)

data_splits_path = Path(
    processed_data_path,
    "data_splits_with_clinical_val.json",
)

with open(data_splits_path, "r") as f:
    data_splits = json.load(f)

if "cortical" in feature_type:
    modality_data_split = data_splits["structural"]

else:
    modality_data_split = data_splits["functional"]

    print("functional")


TRAIN_SUBS = modality_data_split["train"]
INTER_VAL_SUBS = modality_data_split["internalising_val"]
EXTER_VAL_SUBS = modality_data_split["externalising_val"]
HIGH_VAL_SUBS = modality_data_split["high_symptom_val"]

### Model and Dataset class


def compute_ll(x, x_recon):
    return x_recon.log_prob(x).sum(1, keepdims=True).mean(0)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, variational=True, non_linear=False):
        super().__init__()

        self.input_size = input_dim
        self.hidden_dims = hidden_dim
        self.z_dim = hidden_dim[-1]
        self.variational = variational
        self.non_linear = non_linear
        self.layer_sizes_encoder = [input_dim] + hidden_dim
        lin_layers = [
            nn.Linear(dim0, dim1, bias=True)
            for dim0, dim1 in zip(
                self.layer_sizes_encoder[:-1], self.layer_sizes_encoder[1:]
            )
        ]

        self.encoder_layers = nn.Sequential(*lin_layers[0:-1])
        self.enc_mean_layer = nn.Linear(
            self.layer_sizes_encoder[-2], self.layer_sizes_encoder[-1], bias=True
        )
        self.enc_logvar_layer = nn.Linear(
            self.layer_sizes_encoder[-2], self.layer_sizes_encoder[-1], bias=True
        )

    def forward(self, x):
        h1 = x
        for it_layer, layer in enumerate(self.encoder_layers):
            h1 = layer(h1)
            if self.non_linear:
                h1 = F.relu(h1)

        mu = self.enc_mean_layer(h1)
        logvar = self.enc_logvar_layer(h1)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, non_linear=False, init_logvar=-3):
        super().__init__()
        self.input_size = input_dim
        self.hidden_dims = hidden_dim
        self.non_linear = non_linear
        self.init_logvar = init_logvar
        self.layer_sizes_decoder = hidden_dim[::-1] + [input_dim]
        lin_layers = [
            nn.Linear(dim0, dim1, bias=True)
            for dim0, dim1 in zip(
                self.layer_sizes_decoder[:-1], self.layer_sizes_decoder[1:]
            )
        ]
        self.decoder_layers = nn.Sequential(*lin_layers[0:-1])
        self.decoder_mean_layer = nn.Linear(
            self.layer_sizes_decoder[-2], self.layer_sizes_decoder[-1], bias=True
        )
        tmp_noise_par = torch.FloatTensor(1, self.input_size).fill_(self.init_logvar)
        self.logvar_out = Parameter(data=tmp_noise_par, requires_grad=True)

    def forward(self, z):
        x_rec = z
        for it_layer, layer in enumerate(self.decoder_layers):
            x_rec = layer(x_rec)
            if self.non_linear:
                x_rec = F.relu(x_rec)

        mu_out = self.decoder_mean_layer(x_rec)
        return Normal(loc=mu_out, scale=self.logvar_out.exp().pow(0.5))


class VAE(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, latent_dim, learning_rate=0.001, non_linear=False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim + [latent_dim]
        self.latent_dim = latent_dim
        self.non_linear = non_linear
        self.encoder = Encoder(
            input_dim=input_dim, hidden_dim=self.hidden_dim, non_linear=non_linear
        )
        self.decoder = Decoder(
            input_dim=input_dim, hidden_dim=self.hidden_dim, non_linear=non_linear
        )
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
        )

    def encode(self, x):
        return self.encoder(x)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def calc_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)

    def calc_ll(self, x, x_recon):
        return compute_ll(x, x_recon)

    def forward(self, x):
        self.zero_grad()
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_recon = self.decode(z)
        fwd_rtn = {"x_recon": x_recon, "mu": mu, "logvar": logvar}
        return fwd_rtn

    def sample_from_normal(self, normal):
        return normal.loc

    def loss_function(self, x, fwd_rtn):
        x_recon = fwd_rtn["x_recon"]
        mu = fwd_rtn["mu"]
        logvar = fwd_rtn["logvar"]

        kl = self.calc_kl(mu, logvar)
        recon = self.calc_ll(x, x_recon)

        total = kl - recon
        losses = {"total": total, "kl": kl, "ll": recon}
        return losses

    def pred_latent(self, x, DEVICE):
        x = torch.FloatTensor(x.to_numpy()).to(DEVICE)
        with torch.no_grad():
            mu, logvar = self.encode(x)
        latent = mu.cpu().detach().numpy()
        latent_var = logvar.exp().cpu().detach().numpy()
        return latent, latent_var

    def pred_recon(self, x, DEVICE):
        x = torch.FloatTensor(x.to_numpy()).to(DEVICE)
        with torch.no_grad():
            mu, _ = self.encode(x)
            x_pred = self.decode(mu).loc.cpu().detach().numpy()
        return x_pred


class MyDataset(Dataset):
    def __init__(self, data, indices=False, transform=None):
        self.data = data
        if isinstance(data, list) or isinstance(data, tuple):
            self.data = [
                torch.from_numpy(d).float() if isinstance(d, np.ndarray) else d
                for d in self.data
            ]
            self.N = len(self.data[0])
            self.shape = np.shape(self.data[0])
        else:
            if isinstance(data, np.ndarray):
                self.data = torch.from_numpy(self.data).float()
            self.N = len(self.data)
            self.shape = np.shape(self.data)

        self.transform = transform
        self.indices = indices

    def __getitem__(self, index):
        if isinstance(self.data, list):
            x = [d[index] for d in self.data]
        else:
            x = self.data[index]

        if self.transform:
            x = self.transform(x)

        if self.indices:
            return x, index
        return x

    def __len__(self):
        return self.N


def build_model(config, input_dim):
    # Set random seed for CPU
    torch.manual_seed(123)

    # Set random seed for CUDA (GPU) if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)

    # Initialize the model
    model = VAE(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        learning_rate=config.learning_rate,
        non_linear=True,
    ).to(DEVICE)

    if isinstance(model, nn.Module):
        first_layer_params = list(model.parameters())[0]
        print("Parameter values of the first layer:", first_layer_params)

    return model


def latent_deviations_mahalanobis_across(cohort, train):
    dists = calc_robust_mahalanobis_distance(cohort[0], train[0])
    return dists


def calc_robust_mahalanobis_distance(values, train_values):
    robust_cov = MinCovDet(random_state=42).fit(train_values)
    mahal_robust_cov = robust_cov.mahalanobis(values)
    return mahal_robust_cov


def glass_delta(
    control: np.ndarray,
    experimental: np.ndarray,
) -> float:
    """Calculate Glass's Delta, an effect size measure comparing the mean difference
    between an experimental group and a control group to the standard deviation
    of the control group.
    """
    # Calculate the mean of the experimental and control groups
    mean_experimental = np.mean(experimental)
    mean_control = np.mean(control)

    # Calculate the standard deviation of the control group
    sd_control = np.std(control, ddof=1)  # ddof=1 for sample standard deviation

    # Compute Glass's Delta
    delta = (mean_experimental - mean_control) / sd_control

    return delta


def get_degree_of_separation(
    model,
    train_data: np.ndarray,
    val_data: np.ndarray,
    inter_val_data: np.ndarray,
    exter_val_data: np.ndarray,
    high_val_data: np.ndarray,
) -> float:
    """This function is used to monitor the degree to which the deviations scores
    computed from the model can seperate the clinical cohorts from the low symptom
    control during cross-validation.

    Glass's delta is used to compute the degree of separation between the control
    and clinical cohorts.

    Returns:
        float: returns the glass's delta score
    """
    train_data_pd = pd.DataFrame(train_data, columns=FEATURES)
    val_data_pd = pd.DataFrame(val_data, columns=FEATURES)
    inter_val_data_pd = pd.DataFrame(inter_val_data, columns=FEATURES)
    exter_val_data_pd = pd.DataFrame(exter_val_data, columns=FEATURES)
    high_val_data_pd = pd.DataFrame(high_val_data, columns=FEATURES)

    train_latent, _ = model.pred_latent(train_data_pd, DEVICE)
    val_latent, _ = model.pred_latent(val_data_pd, DEVICE)
    inter_val_latent, _ = model.pred_latent(inter_val_data_pd, DEVICE)
    exter_val_latent, _ = model.pred_latent(exter_val_data_pd, DEVICE)
    high_val_latent, _ = model.pred_latent(high_val_data_pd, DEVICE)

    val_distance = latent_deviations_mahalanobis_across([val_latent], [train_latent])
    inter_val_distance = latent_deviations_mahalanobis_across(
        [inter_val_latent], [train_latent]
    )
    exter_val_distance = latent_deviations_mahalanobis_across(
        [exter_val_latent], [train_latent]
    )
    high_val_distance = latent_deviations_mahalanobis_across(
        [high_val_latent], [train_latent]
    )

    inter_glass_delta = glass_delta(val_distance, inter_val_distance)

    exter_glass_delta = glass_delta(val_distance, exter_val_distance)

    high_glass_delta = glass_delta(val_distance, high_val_distance)

    return (inter_glass_delta + exter_glass_delta + high_glass_delta) / 3


def validate(model, val_loader, device):
    model.eval()
    total_val_loss = 0.0
    total_recon = 0.0
    total_kl_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            fwd_rtn = model.forward(batch)
            val_loss = model.loss_function(batch, fwd_rtn)
            batch_val_loss = val_loss["total"].item()
            total_val_loss += batch_val_loss
            total_recon += val_loss["ll"].item()
            total_kl_loss += val_loss["kl"].item()

    mean_val_loss = total_val_loss / len(val_loader)
    mean_recon = total_recon / len(val_loader)
    mean_kl_loss = total_kl_loss / len(val_loader)

    wandb.log({"val_total_loss": mean_val_loss})
    wandb.log({"val_recon": mean_recon})
    wandb.log({"val_kl_loss": mean_kl_loss})

    return mean_val_loss


def train(
    config,
    model,
    train_loader,
    val_loader,
    train_data,
    val_data,
    inter_val_data,
    exter_val_data,
    high_val_data,
    tolerance=50,
):

    model.to(DEVICE)

    best_val_loss = float("inf")

    epochs_no_improve = 0

    for epoch in range(1, config.epochs + 1):

        total_loss = 0.0
        total_recon = 0.0
        kl_loss = 0.0

        model.train()

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(DEVICE)
            fwd_rtn = model.forward(batch)
            loss = model.loss_function(batch, fwd_rtn)
            model.optimizer.zero_grad()
            loss["total"].backward()
            model.optimizer.step()

            total_loss += loss["total"].item()
            total_recon += loss["ll"].item()
            kl_loss += loss["kl"].item()

        val_loss = validate(model, val_loader, DEVICE)

        if val_loss < best_val_loss:
            epochs_no_improve = 0
            best_val_loss = val_loss

            print("New best val loss:", val_loss)
            print("at epoch:", epoch)

            best_model = model.state_dict()

            print("Improved at epoch:", epoch)

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= tolerance:
            print("Early stopping at epoch:", epoch)

            break

        wandb.log(
            {
                "Epoch_total_loss": total_loss / len(train_loader),
                "Epoch_recon_loss": total_recon / len(train_loader),
                "Epoch_kl_loss": kl_loss / len(train_loader),
            }
        )

    model.load_state_dict(best_model)

    # Get degree of separation
    separation_metric = get_degree_of_separation(
        model,
        train_data,
        val_data,
        inter_val_data,
        exter_val_data,
        high_val_data,
    )

    return best_val_loss, separation_metric


def train_k_fold(
    config,
    n_splits=10,
):

    data = pd.read_csv(
        Path(TRAIN_DATA_PATH),
        index_col=0,
        low_memory=False,
    )

    train_dataset = data.loc[
        TRAIN_SUBS,
        FEATURES,
    ].to_numpy()

    data["strata"] = (
        data["demo_sex_v2"].astype(str) + "_" + data["demo_comb_income_v2"].astype(str)
    )

    data["strata"] = pd.Categorical(data["strata"])

    train_strata = data.loc[TRAIN_SUBS, "strata"].cat.codes

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold = 0
    total_loss = 0.0
    total_separation = 0.0

    for train_index, val_index in kf.split(train_dataset, train_strata):
        print(f"Training on fold {fold+1}...")
        # Split dataset into training and validation sets for the current fold

        train_data, val_data = (
            train_dataset[train_index],
            train_dataset[val_index],
        )

        scaler = StandardScaler()

        train_data = scaler.fit_transform(train_data)

        val_data = scaler.transform(val_data)

        inter_val_dataset = data.loc[
            INTER_VAL_SUBS,
            FEATURES,
        ].to_numpy()

        exter_val_dataset = data.loc[
            EXTER_VAL_SUBS,
            FEATURES,
        ].to_numpy()

        high_val_dataset = data.loc[
            HIGH_VAL_SUBS,
            FEATURES,
        ].to_numpy()

        inter_val_data = scaler.transform(inter_val_dataset)

        exter_val_data = scaler.transform(exter_val_dataset)

        high_val_data = scaler.transform(high_val_dataset)

        train_loader = DataLoader(
            MyDataset(train_data), batch_size=config.batch_size, shuffle=True
        )

        val_loader = DataLoader(
            MyDataset(val_data), batch_size=config.batch_size, shuffle=False
        )

        # Get input_dim based on the dataset
        input_dim = train_data.shape[1]

        model = build_model(config, input_dim)

        val_loss, separation_metric = train(
            config,
            model,
            train_loader,
            val_loader,
            train_data,
            val_data,
            inter_val_data,
            exter_val_data,
            high_val_data,
        )

        total_loss += val_loss

        total_separation += separation_metric

        fold += 1

    return total_loss / n_splits, total_separation / n_splits


def main():
    wandb.init()
    average_val_loss, average_separation = train_k_fold(wandb.config)
    wandb.log({"average_val_loss": average_val_loss})
    wandb.log({"average_separation": average_separation})


if __name__ == "__main__":

    project_title = args.project_title
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    latent_dim = args.latent_dim
    hidden_dim = args.hidden_dim

    sweep_configuration = {
        "method": "grid",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "average_separation"},
        "parameters": {
            "batch_size": {"values": batch_size},
            "learning_rate": {"values": learning_rate},
            "latent_dim": {"values": latent_dim},
            "epochs": {"value": 5000},
            "hidden_dim": {"values": hidden_dim},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_title)

    wandb.agent(sweep_id, function=main)
