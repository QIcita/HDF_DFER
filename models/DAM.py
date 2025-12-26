import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from dct_filter import DCT7x7

class AdaptiveDynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last=True, alpha_init_value=0.5, eps=1e-6):
        """
        自适应 DynamicTanh, 通过动态调整 alpha 以适应数据分布偏移
        :param normalized_shape: 归一化维度 (如通道数 C)
        :param channels_last: 是否采用通道最后的格式 (True: [B, H, W, C], False: [B, C, H, W])
        :param alpha_init_value: 初始 alpha 值
        :param eps: 防止除零
        """
        super().__init__()
        self.channels_last = channels_last
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        if self.channels_last:
            std = torch.sqrt(x.var(dim=-1, keepdim=True) + self.eps)
            adaptive_alpha = self.alpha / (std + self.eps)
            x_tanh = torch.tanh(adaptive_alpha * x)
            return x_tanh * self.gamma + self.beta
        else:
            std = torch.sqrt(x.var(dim=1, keepdim=True) + self.eps)
            adaptive_alpha = self.alpha / (std + self.eps)

            std = std.expand_as(x)
            adaptive_alpha = adaptive_alpha.expand_as(x)

            x_tanh = torch.tanh(adaptive_alpha * x)

            return x_tanh * self.gamma[:, None, None] + self.beta[:, None, None]


 

class LocalGlobalRobustModule(nn.Module):
    def __init__(self, step, wasserstein_reg=0.05):
        super().__init__()
        self.wasserstein_reg = wasserstein_reg
        self.sigmoid = nn.Sigmoid()

        # Learnable fusion gates
        self.gamma = nn.Parameter(torch.ones(1, step, step, 1, 1))
        self.delta = nn.Parameter(torch.zeros(1, step, step, 1, 1))

        self.alpha = nn.Parameter(torch.tensor([0.5]))
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, avg_map, max_map):
        b, t, t_, _, _ = avg_map.shape
        assert t == t_, "Temporal dimensions must match."
        mean_map = avg_map.mean(dim=2, keepdim=True)
        wasserstein_dist = torch.norm(avg_map - mean_map, p=2, dim=2, keepdim=True)
        wasserstein_dist = wasserstein_dist.repeat(1, 1, t, 1, 1)
        fused_map = self.alpha * avg_map + self.beta * max_map
        temporal_diff = torch.norm(
            avg_map[:, :, 1:] - avg_map[:, :, :-1],
            p=2, dim=[3, 4], keepdim=True
        )
        temporal_diff = F.pad(temporal_diff, (0, 0, 0, 0, 1, 0))
        interaction_map = (
            self.gamma * fused_map
            + self.delta * wasserstein_dist
            - temporal_diff
        )

        return self.sigmoid(interaction_map)
    
class DROAdversarialBranch(nn.Module):
    def __init__(self, dct_channels, strength=0.1, wasserstein_reg=0.05):
        super().__init__()
        self.strength = strength
        self.wasserstein_reg = wasserstein_reg

        self.noise = nn.Parameter(
            torch.randn(1, dct_channels, 1, 1) * strength
        )
        self.alpha = nn.Parameter(torch.ones(1, dct_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dct_channels, 1, 1))

    def forward(self, dct_feat):
        feat_mean = dct_feat.mean(dim=[2, 3], keepdim=True)
        wasserstein_dist = torch.norm(
            dct_feat - feat_mean, p=2, dim=[2, 3], keepdim=True
        )

        noise_term = self.noise * self.alpha * torch.sign(dct_feat)
        wasserstein_term = self.beta * wasserstein_dist * torch.sign(dct_feat)

        perturbation = noise_term + self.wasserstein_reg * wasserstein_term
        return dct_feat + perturbation

class FreConv(nn.Module):
    def __init__(self, channels, reduction, k=1, p=0):
        super().__init__()
        if reduction == 1:
            self.net = nn.Conv2d(channels, 1, kernel_size=k, padding=p, bias=False)
        else:
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels // reduction, kernel_size=k, padding=p, bias=False),
                nn.ReLU(),
                nn.Conv2d(channels // reduction, 1, kernel_size=k, padding=p, bias=False)
            )

    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = self.softmax(
            torch.bmm(q, k.transpose(1, 2)) / (x.shape[-1] ** 0.5)
        )
        return torch.bmm(attn, v)
    

class DAM(nn.Module):
    def __init__(self, freq_num, channel, step,
                 reduction=1, groups=1, select_method='all'):
        super().__init__()

        self.channel = channel
        self.step = step
        self.select_method = select_method

        self.dct_filter = DCT7x7()
        self.padding = (self.dct_filter.freq_range - 1) // 2

        if select_method == 'all':
            self.dct_channels = self.dct_filter.freq_num
        elif 's' in select_method:
            self.dct_channels = 1
        elif 'top' in select_method:
            self.dct_channels = int(select_method.replace('top', ''))

        self.freq_attn = FreConv(self.dct_channels, reduction, k=7, p=3)

        self.avg_pool = nn.AdaptiveAvgPool3d((step, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((step, 1, 1))

        self.fc_t = nn.Linear(step, step, bias=False)
        self.temporal_attn = SelfAttention(embed_dim=step)

        self.t_weight = nn.Parameter(torch.tensor([0.8]))
        self.s_weight = nn.Parameter(torch.tensor([0.3]))

        self.al = nn.Parameter(torch.tensor([0.5]))
        self.be = nn.Parameter(torch.tensor([0.5]))

        self.temporal_robust = LocalGlobalRobustModule(step)
        self.adv_branch = DROAdversarialBranch(self.dct_channels)
        self.dynamic_tanh = AdaptiveDynamicTanh(
            normalized_shape=self.dct_channels,
            channels_last=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        t, b, c, h, w = x.shape
        x = rearrange(x, 't b c h w -> b t c h w')

        avg_map = self.avg_pool(x)
        max_map = self.max_pool(x)

        temporal_map = self.temporal_robust(avg_map, max_map)
        temporal_map = rearrange(temporal_map, 'b t c 1 1 -> b c t')

        temporal_map = self.fc_t(temporal_map).transpose(1, 2)
        temporal_map = self.temporal_attn(temporal_map)

        t_gate = self.sigmoid(temporal_map.mean(dim=2))
        t_gate = rearrange(t_gate, 'b t -> b t 1 1 1')
        t_gate = t_gate.repeat(1, 1, c, h, w)

        x_t = x * t_gate + x

        dct_weight = self.dct_filter.filter.unsqueeze(1)
        dct_weight = dct_weight.repeat(1, self.channel, 1, 1)
        dct_bias = torch.zeros(self.dct_channels, device=x.device)

        dct_feat = F.conv2d(
            torch.mean(x, dim=1),
            dct_weight,
            dct_bias,
            stride=1,
            padding=self.padding
        )

        dct_adv = self.adv_branch(dct_feat)
        dct_feat = (self.al * dct_feat + self.be * dct_adv) / 2
        dct_feat = self.dynamic_tanh(dct_feat)

        freq_map = self.freq_attn(dct_feat)
        freq_map = freq_map.unsqueeze(1).repeat(1, t, c, 1, 1)

        x_s = x * self.sigmoid(freq_map) + x

        x = (self.t_weight * x_t + self.s_weight * x_s) / 2
        x = rearrange(x, 'b t c h w -> t b c h w')

        return x
