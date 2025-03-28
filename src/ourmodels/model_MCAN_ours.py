"""
    Partial code is copied from MCAN (Multimodal Fusion with Co-Attention Networks for Fake News Detection)
    their released code: https://github.com/wuyang45/MCAN_code, thanks for their efforts.
    (We directly use a pre-trained InceptionNetV3 to encoder the frequency-domain images.)
"""
import torchvision
from torchvision.transforms import Resize
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from scipy.fftpack import fft, dct

from transformers import BertModel, BertTokenizer
import copy
from .model_Dynamics import NeuralDynamics
from torch.distributions import MultivariateNormal

def process_dct_img(img):
    img = img.numpy()  # size = [1, 224, 224]
    height = img.shape[1]
    width = img.shape[2]
    N = 8
    step = int(height / N)  # 28

    dct_img = np.zeros((1, N * N, step * step, 1), dtype=np.float32)  # [1,64,784,1]
    fft_img = np.zeros((1, N * N, step * step, 1))

    i = 0
    for row in np.arange(0, height, step):
        for col in np.arange(0, width, step):
            block = np.array(img[:, row:(row + step), col:(col + step)], dtype=np.float32)
            block1 = block.reshape(-1, step * step, 1)  # [batch_size,784,1]
            dct_img[:, i, :, :] = dct(block1)  # [batch_size, 64, 784, 1]

            i += 1

    # for i in range(64):
    fft_img[:, :, :, :] = fft(dct_img[:, :, :, :]).real  # [batch_size,64, 784,1]

    fft_img = torch.from_numpy(fft_img).float()  # [batch_size, 64, 784, 1]
    new_img = F.interpolate(fft_img, size=[250, 1])  # [batch_size, 64, 250, 1]
    new_img = new_img.squeeze(0).squeeze(-1)  # torch.size = [64, 250]

    return new_img

class multimodal_attention(nn.Module):
    """
    dot-product attention mechanism
    """

    def __init__(self, attention_dropout=0.5):
        super(multimodal_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):

        attention = torch.matmul(q, k.transpose(-2, -1))
        if scale:
            attention = attention * scale

        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        attention = torch.matmul(attention, v)

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=256, num_heads=8, dropout=0.5):
        super(MultiHeadAttention, self).__init__()

        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(1, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention = multimodal_attention(dropout)
        self.linear_final = nn.Linear(model_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        residual = query
        query = query.unsqueeze(-1)
        key = key.unsqueeze(-1)
        value = value.unsqueeze(-1)

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(-1, num_heads, self.model_dim, dim_per_head)
        value = value.view(-1, num_heads, self.model_dim, dim_per_head)
        query = query.view(-1, num_heads, self.model_dim, dim_per_head)

        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        attention = self.dot_product_attention(query, key, value,
                                               scale, attn_mask)

        attention = attention.view(-1, self.model_dim, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(attention).squeeze(-1)

        # dropout
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output


class PositionalWiseFeedForward(nn.Module):
    """
    Fully-connected network
    """

    def __init__(self, model_dim=256, ffn_dim=2048, dropout=0.5):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x

        x = self.w2(F.relu(self.w1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        output = x
        return output


class multimodal_fusion_layer(nn.Module):
    """
    A layer of fusing features
    """

    def __init__(self, model_dim=256, num_heads=8, ffn_dim=2048, dropout=0.5):
        super(multimodal_fusion_layer, self).__init__()
        self.attention_1 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attention_2 = MultiHeadAttention(model_dim, num_heads, dropout)

        self.feed_forward_1 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.feed_forward_2 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

        self.fusion_linear = nn.Linear(model_dim * 2, model_dim)

    def forward(self, image_output, text_output, attn_mask=None):
        output_1 = self.attention_1(image_output, text_output, text_output,
                                    attn_mask)

        output_2 = self.attention_2(text_output, image_output, image_output,
                                    attn_mask)

        output_1 = self.feed_forward_1(output_1)
        output_2 = self.feed_forward_2(output_2)

        output = torch.cat([output_1, output_2], dim=1)
        output = self.fusion_linear(output)

        return output


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

        self.dct_img = torchvision.models.inception_v3(pretrained=True)
        self.shared_dct = nn.Sequential(
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
        self.dct_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.fusion_layers = nn.ModuleList([
            multimodal_fusion_layer(model_dim=sim_dim, num_heads=8, ffn_dim=2048, dropout=args.dropout)
            for _ in range(1)
        ])

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

        self.resize = Resize([299, 299])

    def forward(self, text, image, mask, eventi=None, label=None,
                mu_0=None, sigma_0=None, mu_1=None, sigma_1=None):
        # IMAGE
        image_resnet = self.visualmodal(image)  # [N, 512]
        image_z = self.shared_image(image_resnet)
        image_z = self.image_aligner(image_z)
        # dct
        image_dct = self.resize(image)
        image_inception = self.dct_img(image_dct)[0]
        if len(image_inception.shape) == 1: image_inception = image_inception.unsqueeze(0)
        image_dct_z = self.shared_dct(image_inception)
        image_dct_z = self.dct_aligner(image_dct_z)
        # Text
        last_hidden_state = torch.mean(self.bertModel(text)[0], dim=1, keepdim=False)
        text_z = self.shared_text_linear(last_hidden_state)
        text_z = self.text_aligner(text_z)

        for fusion_layer in self.fusion_layers:
            output = fusion_layer(image_z, image_dct_z, attn_mask=None)
        for fusion_layer in self.fusion_layers:
            output = fusion_layer(output, text_z, attn_mask=None)

        text_image = output
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
        return class_output, text_image, image_z, text_z, mu_0.detach(), sigma_0.detach(), mu_1.detach(), sigma_1.detach(), \
            loss_dyn, [img_txt_sigma_1, img_sigma_1, txt_sigma_1, dynamics_label_sigma_1, dynamics_pred_sigma_1]
