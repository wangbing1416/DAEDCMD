import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer
import copy
from .model_Dynamics import NeuralDynamics
from torch.distributions import MultivariateNormal


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        # discriminator (without classifier)
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # generator (MLP-based VAE structure)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # 假设隐变量维度为hidden_dim
        )
        self.enc_fc_z_mean = nn.Linear(hidden_dim, hidden_dim)
        self.enc_fc_z_log_var = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.discriminator(x)

    def encode(self, x):
        h = self.encoder(x)
        z_mean = self.enc_fc_z_mean(h)
        z_log_var = self.enc_fc_z_log_var(h)
        return z_mean, z_log_var

    def decode(self, z):
        h = self.decoder(z)
        return h

    def reparameterize(self, z_mean, z_log_var, num_samples=1):
        z_std = (z_log_var * 0.5).exp()
        z_std = z_std.unsqueeze(1).expand(-1, num_samples, -1)
        z_mean = z_mean.unsqueeze(1).expand(-1, num_samples, -1)
        unit_normal = torch.randn_like(z_std)
        z = z_mean + unit_normal * z_std
        z = z.view(-1, z_std.size(2))
        return z

    def forward_generate(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        recon_x = self.decode(z)
        return recon_x, z_mean, z_log_var


def gaussian_kl(q_mean, q_log_var, p_mean=None, p_log_var=None):
    # p defaults to N(0, 1)
    zeros = torch.zeros_like(q_mean)
    p_mean = p_mean if p_mean is not None else zeros
    p_log_var = p_log_var if p_log_var is not None else zeros
    # calcaulate KL(q, p)
    kld = 0.5 * (
            p_log_var - q_log_var +
            (q_log_var.exp() + (q_mean - p_mean) ** 2) / p_log_var.exp() - 1
    )
    kld = kld.sum(1)
    return kld


def compute_gaussian_params(data):
    mean = data.mean(dim=0)  # average value
    cov = torch.cov(data.T)  # co-variance
    return mean, cov


def mix_gaussians_high_dim(means, covariances):
    """
    mix multiple gaussian to one gaussian

    params:
    - means: (K, D), K gaussian D dimensions.
    - covariances: co-variance, (K, D, D)

    return:
    - mixed_mean: mixed mean  (D,)
    - mixed_covariance: mixed co-variance (D, D)
    """
    K, D = means.shape
    mixed_mean = means.mean(dim=0)  # (D,)
    mixed_covariance = torch.zeros((D, D)).to(means.device)  # initialized as zero
    for k in range(K):
        mu_k = means[k]  # (D,)
        Sigma_k = covariances[k]  # (D, D)

        # (μ_k - μ_mix)(μ_k - μ_mix)^T
        deviation = (mu_k - mixed_mean).unsqueeze(1)  # (D, 1)
        deviation_outer_product = deviation @ deviation.T  # (D, D)

        # Σ_k + (μ_k - μ_mix)(μ_k - μ_mix)^T
        mixed_covariance += Sigma_k + deviation_outer_product

    mixed_covariance /= K

    return mixed_mean, mixed_covariance


def model_gaussian(img_txt_z_0, img_txt_z_1, img_z_0, img_z_1, txt_z_0, txt_z_1):
    """
        modeling gaussian distributions with a batched data

        params:
        - img_txt_z_0: (B, D), B batch size, D dimensions.

        return:
        - mixed_mean: mixed mean  (D,)
        - mixed_covariance: mixed co-variance (D, D)
    """
    img_txt_mu_0, img_txt_sigma_0 = compute_gaussian_params(img_txt_z_0)
    img_txt_mu_1, img_txt_sigma_1 = compute_gaussian_params(img_txt_z_1)
    img_mu_0, img_sigma_0 = compute_gaussian_params(img_z_0)
    img_mu_1, img_sigma_1 = compute_gaussian_params(img_z_1)
    txt_mu_0, txt_sigma_0 = compute_gaussian_params(txt_z_0)
    txt_mu_1, txt_sigma_1 = compute_gaussian_params(txt_z_1)
    # mix gaussian dist
    mu_0, sigma_0 = mix_gaussians_high_dim(torch.stack([img_txt_mu_0, img_mu_0, txt_mu_0]),
                                           torch.stack([img_txt_sigma_0, img_sigma_0, txt_sigma_0]))
    mu_1, sigma_1 = mix_gaussians_high_dim(torch.stack([img_txt_mu_1, img_mu_1, txt_mu_1]),
                                           torch.stack([img_txt_sigma_1, img_sigma_1, txt_sigma_1]))
    return mu_0, sigma_0, mu_1, sigma_1, img_txt_sigma_1, img_sigma_1, txt_sigma_1


# our proposed model
class OurModel(nn.Module):
    def __init__(self, args, shared_dim=128, sim_dim=64, k=2):
        super(OurModel, self).__init__()
        self.args = args

        self.event_num = args.event_num

        C = args.class_num
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19
        # bert
        if self.args.dataset == 'weibo':
            bert_model = BertModel.from_pretrained('../../huggingface/bert-base-chinese')
        else:
            bert_model = BertModel.from_pretrained('../../huggingface/bert-base-uncased')

        self.bert_hidden_size = args.bert_hidden_dim
        self.shared_text_linear = nn.Sequential(
            nn.Linear(self.bert_hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU()
        )

        # TODO: whether bert need fine-tuning
        self.bertModel = bert_model.requires_grad_(False)
        for name, param in self.bertModel.named_parameters():
            if name.startswith("encoder.layer.11") or \
                    name.startswith("encoder.layer.10") or \
                    name.startswith("encoder.layer.9"):
                param.requires_grad = True

        self.dropout = nn.Dropout(args.dropout)

        # IMAGE
        resnet = torchvision.models.resnet34(pretrained=True)

        num_ftrs = resnet.fc.out_features
        self.visualmodal = resnet
        self.shared_image = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU()
        )

         # fusion
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(sim_dim * 2, sim_dim * 2),
            nn.BatchNorm1d(sim_dim * 2),
            nn.ReLU(),
            nn.Linear(sim_dim * 2, sim_dim),
            nn.ReLU()
        )

        self.sim_classifier = nn.Sequential(
            nn.Linear(sim_dim + int(sim_dim / 4), sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU(),
            nn.Linear(sim_dim, 2)
        )

        # initialize shared and task-specific experts
        self.event_flag = {}
        self.experts = nn.ModuleList([Expert(input_dim=sim_dim, output_dim=sim_dim, hidden_dim=8) for _ in range(k)])
        self.routing = nn.Sequential(
            nn.Linear(sim_dim, sim_dim),
            nn.ReLU(),
            nn.Linear(sim_dim, 1),
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.routing_map = nn.Sequential(
            nn.Linear(sim_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU(),
            nn.Linear(sim_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )

        self.map_dist = nn.Sequential(
            nn.Linear(sim_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU(),
            nn.Linear(sim_dim, int(sim_dim / 2)),
            nn.BatchNorm1d(int(sim_dim / 2)),
            nn.ReLU(),
        )
        self.dynamics_model = nn.ModuleList(
            [NeuralDynamics(input_size=int(sim_dim / 2), hidden_size=int(sim_dim / 2)) for _ in range(4)])
        self.criterion_dyn = torch.nn.L1Loss()
        self.dynamics_feature_map = nn.Sequential(
            nn.Linear(sim_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU(),
            nn.Linear(sim_dim, int(sim_dim / 4)),
            nn.BatchNorm1d(int(sim_dim / 4)),
            nn.ReLU()
        )

    def forward(self, text, image, mask, eventi=None, label=None,
                mu_0=None, sigma_0=None, mu_1=None, sigma_1=None):
        # IMAGE
        image = self.visualmodal(image)  # [N, 512]
        image_z = self.shared_image(image)
        image_z = self.image_aligner(image_z)
        # Text
        last_hidden_state = torch.mean(self.bertModel(text)[0], dim=1, keepdim=False)
        text_z = self.shared_text_linear(last_hidden_state)
        text_z = self.text_aligner(text_z)
        # TODO: design a powerful fusion function
        text_image = torch.cat((text_z, image_z), 1)
        text_image = self.fusion(text_image)

        # dynamics model
        # calculate P0 with the 0-th event
        loss_dyn = None
        img_txt_sigma_1, img_sigma_1, txt_sigma_1, dynamics_label_sigma_1 = None, None, None, None
        map_img_txt_dist, map_img_dist, map_txt_dist = self.map_dist(text_image), self.map_dist(image_z), self.map_dist(text_z)
        if label is not None and len(label) - 1 > torch.sum(label) > 1:
            if mu_0 is None:
                dynamics_pred_mu_0, dynamics_pred_sigma_0, dynamics_pred_mu_1, dynamics_pred_sigma_1, img_txt_sigma_1, img_sigma_1, txt_sigma_1 = model_gaussian(
                    map_img_txt_dist[label == 0], map_img_txt_dist[label == 1], map_img_dist[label == 0],
                    map_img_dist[label == 1], map_txt_dist[label == 0], map_txt_dist[label == 1])
                mu_0, sigma_0, mu_1, sigma_1 = dynamics_pred_mu_0, dynamics_pred_sigma_0, dynamics_pred_mu_1, dynamics_pred_sigma_1
            else:
                dynamics_label_mu_0, dynamics_label_sigma_0, dynamics_label_mu_1, dynamics_label_sigma_1, img_txt_sigma_1, img_sigma_1, txt_sigma_1 = model_gaussian(
                    map_img_txt_dist[label == 0], map_img_txt_dist[label == 1], map_img_dist[label == 0],
                    map_img_dist[label == 1], map_txt_dist[label == 0], map_txt_dist[label == 1])

                dyn_t = torch.tensor([eventi]).to(self.args.device)
                dynamics_pred_mu_0 = self.dynamics_model[0](dyn_t, mu_0)
                dynamics_pred_sigma_0 = self.dynamics_model[1](dyn_t, sigma_0)
                dynamics_pred_mu_1 = self.dynamics_model[2](dyn_t, mu_1)
                dynamics_pred_sigma_1 = self.dynamics_model[3](dyn_t, sigma_1)
                loss_dyn = self.criterion_dyn(dynamics_pred_mu_0, dynamics_label_mu_0) + self.criterion_dyn(
                    dynamics_pred_sigma_0, dynamics_label_sigma_0) + \
                           self.criterion_dyn(dynamics_pred_mu_1, dynamics_label_mu_1) + self.criterion_dyn(
                    dynamics_pred_sigma_1, dynamics_label_sigma_1)
        else:
            dyn_t = torch.tensor([eventi]).to(self.args.device)
            dynamics_pred_mu_0 = self.dynamics_model[0](dyn_t, mu_0)
            dynamics_pred_sigma_0 = self.dynamics_model[1](dyn_t, sigma_0)
            dynamics_pred_mu_1 = self.dynamics_model[2](dyn_t, mu_1)
            dynamics_pred_sigma_1 = self.dynamics_model[3](dyn_t, sigma_1)

        gaussian_feature = torch.cat([dynamics_pred_mu_0 + dynamics_pred_sigma_0 @ torch.randn(
            dynamics_pred_mu_0.shape[0]).to(dynamics_pred_mu_0.device),
                                      dynamics_pred_mu_1 + dynamics_pred_sigma_1 @ torch.randn(
                                          dynamics_pred_mu_1.shape[0]).to(dynamics_pred_mu_1.device)], dim=-1)
        gaussian_feature = self.dynamics_feature_map(gaussian_feature.unsqueeze(0).expand_as(text_image))

        if label is not None:
            # Dirichlet expert expend
            if eventi not in self.event_flag.keys():
                self.event_flag[eventi] = 0

            if self.event_flag[eventi] == 0:
                # we regard G0 as the shared expert
                losses = []

                for expert in self.experts:
                    # calculate p(y|x; φk^D)
                    disc_hidden = expert(text_image)
                    logits = self.sim_classifier(torch.cat([disc_hidden, gaussian_feature], dim=-1))
                    loss_d = self.ce_loss(logits, label)

                    # calculate p(x; φk^G)
                    recon_x, z_mean, z_log_var = expert.forward_generate(text_image)
                    loss_g = torch.mean(torch.exp(-torch.mean((text_image - recon_x) ** 2, dim=-1)))
                    loss_kl = torch.mean(gaussian_kl(z_mean, z_log_var))
                    loss = loss_d + loss_g + loss_kl
                    if loss_dyn:
                        loss_dyn += loss

                    losses.append(loss)
                responsibilities = torch.softmax(torch.stack(losses), dim=0)
                if torch.argmin(responsibilities) == 0:
                    self.experts.append(copy.deepcopy(self.experts[0]))
                    self.event_flag[eventi] = 1

        # Fake or real
        route_weight = self.routing(text_image)
        route_feature = self.routing_map(
            route_weight * self.experts[0](text_image) + (1 - route_weight) * self.experts[-1](text_image))
        feature = torch.cat([route_feature, gaussian_feature], dim=-1)

        # Fake or real
        class_output = self.sim_classifier(feature)

        class_output = self.dropout(class_output)
        sim = (torch.matmul(image_z.unsqueeze(1), text_z.unsqueeze(2)).squeeze() + torch.norm(image_z, dim=1) * torch.norm(text_z, dim=1)) / (2 * torch.norm(image_z, dim=1) * torch.norm(text_z, dim=1))

        return class_output, text_image, image_z, text_z, torch.cat([sim.unsqueeze(1), (1 - sim).unsqueeze(1)], dim=1), \
            mu_0.detach(), sigma_0.detach(), mu_1.detach(), sigma_1.detach(), loss_dyn,\
            [img_txt_sigma_1, img_sigma_1, txt_sigma_1, dynamics_label_sigma_1, dynamics_pred_sigma_1]
