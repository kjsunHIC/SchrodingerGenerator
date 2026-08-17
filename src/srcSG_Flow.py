import torch
import torch.nn as nn
import numpy as np
import scipy as sp
import itertools
import math
import os
import copy
import matplotlib.pyplot as plt
import time
import logging
import torch.nn.functional as F
from torch.distributions import MultivariateNormal 
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser

from srcSG_utilities import  _to_tensor,check_device_and_seed,device_info
from srcSG_AIS import _safe_norm_weights,ais_vectorized_batch,systematic_resample,residual_resample,metropolis_resample

from srcSG_AIS import  SG_AIS_sampler,SG_AIS_resampler,SG_AIS_inverse_mapping,SG_AIS_mapping,SG_AIS_integrator

 


# ===========================
# Create Flow with RQS Coupling
# ===========================

 
# ----------------------------
# Improved NSF block (drop-in replacement)
# - Modern Durkan RQS
# - stable numeric defaults
# - ActNorm (data-dependent init)
# - OneByOneConv (LU paramization)
# - NSF_CL coupling using RQS
# - NormalizingFlowModel with option for ClusterDistribution prior
# ----------------------------


# keep your defaults
DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3
EPS = 1e-12 

def _searchsorted(bin_locations, inputs, eps=1e-6):
    # Safe vectorized searchsorted equivalent used by Durkan RQS
    bin_locations = bin_locations.clone()
    bin_locations[..., -1] = bin_locations[..., -1] + eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


# ----------------------------
# Stable Durkan RQS (analytic inverse)
# Vectorized; supports forward and inverse
# inputs: 1D tensor (or any shape); widths/heights/derivatives: last dim = K (or K+1 for derivatives)
# ----------------------------
def RQS(inputs, unnormalized_widths, unnormalized_heights,
        unnormalized_derivatives, inverse=False, left=0., right=1.,
        bottom=0., top=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT, min_derivative=DEFAULT_MIN_DERIVATIVE,
        eps=EPS):
    """
    Durkan et al. rational-quadratic spline for 1-D inputs.
    - inputs: [B] (or any shape flattened)
    - unnormalized_widths/heights: [B, K]
    - unnormalized_derivatives: [B, K+1] (preferred) or [B, K] (will be padded)
    Returns: (outputs, logabsdet) for forward, or (outputs, -logabsdet) for inverse
    """
    if inputs.numel() == 0:
        return inputs, torch.zeros_like(inputs)

    K = unnormalized_widths.shape[-1]

    # validate min sizes
    if min_bin_width * K > 1.0:
        raise ValueError("min_bin_width too large for K")
    if min_bin_height * K > 1.0:
        raise ValueError("min_bin_height too large for K")

    # widths
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * K) * widths  # ensure min width
    cumwidths = torch.cumsum(widths, dim=-1)  # [B, K]
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)  # [B, K+1]
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]  # [B, K]

    # heights
    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * K) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]  # [B, K]

    # derivatives
    derivs = min_derivative + F.softplus(unnormalized_derivatives)
    # ensure length K+1 for derivatives
    if derivs.shape[-1] < K + 1:
        # pad by repeating edges (safe fallback)
        pad_left = derivs[..., :1]
        pad_right = derivs[..., -1:]
        derivs = torch.cat([pad_left, derivs, pad_right], dim=-1)
    elif derivs.shape[-1] > K + 1:
        derivs = derivs[..., :K+1]

    # choose bin indices
    if inverse:
        bin_idx = _searchsorted(cumheights, inputs)
    else:
        bin_idx = _searchsorted(cumwidths, inputs)
    bin_idx = bin_idx.clamp(0, K - 1)
    idx = bin_idx.unsqueeze(-1)  # [B, 1]

    # gather per-bin params
    input_cumwidths = cumwidths.gather(-1, idx)[..., 0]
    input_bin_widths = widths.gather(-1, idx)[..., 0]
    input_cumheights = cumheights.gather(-1, idx)[..., 0]
    input_heights = heights.gather(-1, idx)[..., 0]

    delta = heights / widths
    input_delta = delta.gather(-1, idx)[..., 0]

    input_derivatives = derivs.gather(-1, idx)[..., 0]
    input_derivatives_plus_one = derivs[..., 1:].gather(-1, idx)[..., 0]

    if inverse:
        # analytic inversion (quadratic)
        a = ((inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta) \
             + input_heights * (input_delta - input_derivatives))
        b = (input_heights * input_derivatives - (inputs - input_cumheights) * \
             (input_derivatives + input_derivatives_plus_one - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        # clip negative numerical jitter to zero
        discriminant = torch.clamp(discriminant, min=0.0)
        root = (2 * c) / (-b - torch.sqrt(discriminant + eps) + eps)

        outputs = root * input_bin_widths + input_cumwidths

        theta = root.clamp(0.0, 1.0)
        theta_one_minus_theta = theta * (1 - theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2) \
                                + 2 * input_delta * theta_one_minus_theta \
                                + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator.clamp(min=eps)) - 2 * torch.log(denominator.clamp(min=eps))
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / (input_bin_widths + eps)
        theta = theta.clamp(0.0, 1.0)
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta)
        outputs = input_cumheights + numerator / (denominator + eps)

        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2) \
                                + 2 * input_delta * theta_one_minus_theta \
                                + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator.clamp(min=eps)) - 2 * torch.log(denominator.clamp(min=eps))

        return outputs, logabsdet


# ----------------------------
# unconstrained_RQS (identity outside tail_bound)
# ----------------------------
def unconstrained_RQS(inputs, unnormalized_widths, unnormalized_heights,
                      unnormalized_derivatives, inverse=False,
                      tail_bound=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                      min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                      min_derivative=DEFAULT_MIN_DERIVATIVE):
    """
    Applies RQS inside [-tail_bound, tail_bound] and identity outside. Vectorized.
    """
    inside_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if inside_mask.any():
        inputs_in = inputs[inside_mask]
        w_in = unnormalized_widths[inside_mask, :]
        h_in = unnormalized_heights[inside_mask, :]
        d_in = unnormalized_derivatives[inside_mask, :]

        out_in, ld_in = RQS(inputs_in, w_in, h_in, d_in,
                            inverse=inverse,
                            left=-tail_bound, right=tail_bound,
                            bottom=-tail_bound, top=tail_bound,
                            min_bin_width=min_bin_width,
                            min_bin_height=min_bin_height,
                            min_derivative=min_derivative)
        outputs[inside_mask] = out_in
        logabsdet[inside_mask] = ld_in

    # outside interval -> identity
    outside_mask = ~inside_mask
    if outside_mask.any():
        outputs[outside_mask] = inputs[outside_mask]
        logabsdet[outside_mask] = 0.0

    return outputs, logabsdet


# ----------------------------
# ActNorm (data-dependent init)
# ----------------------------
class ActNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.initialized = False
        self.loc = nn.Parameter(torch.zeros(1, dim, dtype=torch.float64))
        self.scale = nn.Parameter(torch.ones(1, dim, dtype=torch.float64))

    def initialize(self, x):
        with torch.no_grad():
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, unbiased=False, keepdim=True).clamp(min=self.eps)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1.0 / std)
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.initialize(x)
        y = (x + self.loc) * self.scale
        logdet = torch.sum(torch.log(self.scale.abs())) * x.shape[0]
        return y, logdet

    def inverse(self, y):
        x = y / (self.scale + EPS) - self.loc
        logdet = -torch.sum(torch.log(self.scale.abs())) * y.shape[0]
        return x, logdet


# ----------------------------
# One-by-one invertible conv (LU param.)
# ----------------------------
class OneByOneConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        W = np.linalg.qr(np.random.randn(dim, dim))[0]
        P, L, U = sp.linalg.lu(W)
        self.register_buffer("P", torch.tensor(P, dtype=torch.float64))
        self.L = nn.Parameter(torch.tensor(L, dtype=torch.float64))
        self.S = nn.Parameter(torch.tensor(np.diag(U), dtype=torch.float64))
        self.U = nn.Parameter(torch.triu(torch.tensor(U, dtype=torch.float64), 1))
        self._W_inv = None

    def forward(self, x):
        L = torch.tril(self.L, -1) + torch.diag(torch.ones(self.L.shape[0]))
        U = torch.triu(self.U, 1) + torch.diag(self.S)
        W = self.P @ L @ U
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S))) * x.shape[0]
        return z, log_det

    def inverse(self, z):
        if self._W_inv is None:
            L = torch.tril(self.L, -1) + torch.diag(torch.ones(self.L.shape[0]))
            U = torch.triu(self.U, 1) + torch.diag(self.S)
            W = self.P @ L @ U
            self._W_inv = torch.inverse(W)
        x = z @ self._W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S))) * z.shape[0]
        return x, log_det


# ----------------------------
# Small FC to produce spline params
# ----------------------------
class FCNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )
        # zero-init last layer to start near-identity for splines
        last = self.network[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x):
        return self.network(x)


# ----------------------------
# NSF_CL coupling (modern Durkan RQS inside)
# Keeps split: first half conditions second half
# Allows optional internal ActNorm/OneByOneConv per block (L repetitions)
# ----------------------------
class NSF_CL(nn.Module):
    """
    Rational-Quadratic Coupling Layer supporting ANY input dimension.
    Asymmetric split:
        d1 = dim // 2
        d2 = dim - d1    (>= d1 when dim is odd)

    Two-step coupling:
        Step 1: lower(d1)  → transform upper(d2)
        Step 2: upper(d2) → transform lower(d1)
    """
    def __init__(self, dim, K=8, B=3.0, hidden_dim=128, base_network=FCNN,
                 L=1, use_actnorm=False, use_1x1conv=False,
                 x_min=None, x_max=None):
        super().__init__()

        self.dim = dim
        self.d1 = dim // 2
        self.d2 = dim - self.d1   # works for odd dims

        self.K = K
        self.B = B
        self.L = L

        # x_min / x_max buffers
        if x_min is None:
            xm1 = torch.full((self.d1,), -B, dtype=torch.float64)
            xm2 = torch.full((self.d2,), -B, dtype=torch.float64)
        else:
            xm = torch.as_tensor(x_min, dtype=torch.float64)
            if xm.ndim == 0:
                xm = xm.repeat(dim)
            xm1 = xm[:self.d1]
            xm2 = xm[self.d1:]

        if x_max is None:
            xM1 = torch.full((self.d1,), B, dtype=torch.float64)
            xM2 = torch.full((self.d2,), B, dtype=torch.float64)
        else:
            xM = torch.as_tensor(x_max, dtype=torch.float64)
            if xM.ndim == 0:
                xM = xM.repeat(dim)
            xM1 = xM[:self.d1]
            xM2 = xM[self.d1:]

        self.register_buffer("x_min_1", xm1.float())
        self.register_buffer("x_max_1", xM1.float())
        self.register_buffer("x_min_2", xm2.float())
        self.register_buffer("x_max_2", xM2.float())

        # Build blocks
        self.blocks = nn.ModuleList()
        for _ in range(L):
            block = nn.ModuleDict()

            if use_actnorm:
                block["actnorm"] = ActNorm(dim)
            if use_1x1conv:
                block["conv"] = OneByOneConv(dim)

            # Outputs:
            #   per dimension → (3K + 1) spline params
            out_d1 = self.d2 * (3*K + 1)   # params for upper(d2)
            out_d2 = self.d1 * (3*K + 1)   # params for lower(d1)

            block["f1"] = base_network(self.d1, out_d1, hidden_dim)
            block["f2"] = base_network(self.d2, out_d2, hidden_dim)

            self.blocks.append(block)

    # ------------------------------
    # helper to unpack spline params
    # ------------------------------
    def _unpack_params(self, net_out, out_dim):
        """net_out: [B, out_dim * (3K+1)] returns widths, heights, derivs"""
        B = net_out.shape[0]
        out = net_out.view(B, out_dim, 3*self.K + 1)
        widths = out[:, :, :self.K]
        heights = out[:, :, self.K:2*self.K]
        derivatives = out[:, :, 2*self.K:]
        return widths, heights, derivatives

    # ------------------------------
    # forward
    # ------------------------------
    def forward(self, x):
        B = x.shape[0]
        z = x
        total_logdet = torch.zeros(B, dtype=x.dtype, device=x.device)

        for block in self.blocks:
            # optional actnorm/conv
            if "actnorm" in block:
                z, ld = block["actnorm"](z)
                total_logdet += ld
            if "conv" in block:
                z, ld = block["conv"](z)
                total_logdet += ld

            lower = z[:, :self.d1]
            upper = z[:, self.d1:]

            # -------------
            # STEP 1: lower → params for upper
            # -------------
            p = block["f1"](lower)
            widths, heights, derivs = self._unpack_params(p, self.d2)

            y_upper = torch.zeros_like(upper)
            logdet_upper = torch.zeros_like(upper)

            for d in range(self.d2):
                w = widths[:, d, :]
                h = heights[:, d, :]
                derv = derivs[:, d, :]
                out_d, ld_d = unconstrained_RQS(upper[:, d], w, h, derv,
                                                inverse=False,
                                                tail_bound=self.B)
                y_upper[:, d] = out_d
                logdet_upper[:, d] = ld_d

            upper = y_upper
            total_logdet += logdet_upper.sum(dim=1)

            # -------------
            # STEP 2: upper → params for lower
            # -------------
            p2 = block["f2"](upper)
            widths2, heights2, derivs2 = self._unpack_params(p2, self.d1)

            y_lower = torch.zeros_like(lower)
            logdet_lower = torch.zeros_like(lower)

            for d in range(self.d1):
                w = widths2[:, d, :]
                h = heights2[:, d, :]
                derv = derivs2[:, d, :]
                out_d, ld_d = unconstrained_RQS(lower[:, d], w, h, derv,
                                                inverse=False,
                                                tail_bound=self.B)
                y_lower[:, d] = out_d
                logdet_lower[:, d] = ld_d

            lower = y_lower
            total_logdet += logdet_lower.sum(dim=1)

            z = torch.cat([lower, upper], dim=1)

        return z, total_logdet

    # ------------------------------
    # inverse
    # ------------------------------
    def inverse(self, z):
        B = z.shape[0]
        x = z
        total_logdet = torch.zeros(B, dtype=z.dtype, device=z.device)

        for block in reversed(self.blocks):

            lower = x[:, :self.d1]
            upper = x[:, self.d1:]

            # -------------
            # STEP 2 inverse: upper → params for lower
            # -------------
            p2 = block["f2"](upper)
            widths2, heights2, derivs2 = self._unpack_params(p2, self.d1)

            y_lower = torch.zeros_like(lower)
            logdet_lower = torch.zeros_like(lower)

            for d in range(self.d1):
                w = widths2[:, d, :]
                h = heights2[:, d, :]
                derv = derivs2[:, d, :]
                out_d, ld_d = unconstrained_RQS(lower[:, d], w, h, derv,
                                                inverse=True,
                                                tail_bound=self.B)
                y_lower[:, d] = out_d
                logdet_lower[:, d] = ld_d

            lower = y_lower
            total_logdet += logdet_lower.sum(dim=1)

            # -------------
            # STEP 1 inverse: lower → params for upper
            # -------------
            p1 = block["f1"](lower)
            widths, heights, derivs = self._unpack_params(p1, self.d2)

            y_upper = torch.zeros_like(upper)
            logdet_upper = torch.zeros_like(upper)

            for d in range(self.d2):
                w = widths[:, d, :]
                h = heights[:, d, :]
                derv = derivs[:, d, :]
                out_d, ld_d = unconstrained_RQS(upper[:, d], w, h, derv,
                                                inverse=True,
                                                tail_bound=self.B)
                y_upper[:, d] = out_d
                logdet_upper[:, d] = ld_d

            upper = y_upper
            total_logdet += logdet_upper.sum(dim=1)

            x = torch.cat([lower, upper], dim=1)

            # optional conv/actnorm inverses
            if "conv" in block:
                x, ld = block["conv"].inverse(x)
                total_logdet += ld
            if "actnorm" in block:
                x, ld = block["actnorm"].inverse(x)
                total_logdet += ld

        return x, total_logdet




# ----------------------------
# NormalizingFlowModel (wrap flows and select prior)
# ----------------------------
class NormalizingFlowModel(nn.Module):
    def __init__(self, dim, flows, prior_type="gaussian", cluster_params=None,fun=None,funA=None):
        super().__init__()
        self.dim = dim
        self.flows = nn.ModuleList(flows)
        self.prior_type = prior_type
        self.cluster_params = cluster_params
        self.CustomDistribution = fun
        self.fun = funA
        if prior_type == "gaussian":
            self.prior = MultivariateNormal(torch.zeros(dim, dtype=torch.float64), torch.eye(dim, dtype=torch.float64))
        elif prior_type == "cluster":
            # use your imported ClusterDistribution (CustomDistribution)
            if cluster_params is None:
                raise ValueError("cluster_params required for cluster prior")
            self.prior = self.CustomDistribution(self.fun,cluster_params)
        else:
            raise ValueError("Unknown prior_type")

    def forward(self, x):
        bsz, _ = x.shape
        log_det = torch.zeros(bsz, dtype=x.dtype, device=x.device)
        z = x
        for flow in self.flows:
            z, ld = flow.forward(z)
            log_det = log_det + ld
        lp = self.prior.log_prob(z)
        return z, lp, log_det

    def inverse(self, z):
        bsz, _ = z.shape
        log_det = torch.zeros(bsz, dtype=z.dtype, device=z.device)
        x = z
        # invert flows in reverse order
        for flow in reversed(self.flows):
            x, ld = flow.inverse(x)
            log_det = log_det + ld
        return x, log_det

    def sample(self, n_samples):
        if self.prior_type == "gaussian":
            z = self.prior.sample((n_samples,))
        else:
            z = self.prior.sample(n_samples)
            
        x, _ = self.inverse(z)
        return x

    def forward_with_logdet(self, z):
        bsz, _ = z.shape
        loget_total = torch.zeros(bsz, dtype=z.dtype, device=z.device)
        x = z
        for flow in self.flows:
            x, ld = flow.forward(x)
            loget_total = loget_total + ld
        lp = self.prior.log_prob(x)
        #logdet_total=torch.clamp(logdet_total, min=-20, max=20)
        return x, -loget_total
    
class CombinedFlow(nn.Module):
    def __init__(self, modelA, modelB):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        """
        x -> A -> z1 -> B -> z2
        """
        z1,_,logdet1 = self.modelA.forward(x)
        z2,_,logdet2 = self.modelB.forward(z1)
        return z2, logdet1 + logdet2

    def inverse(self, z):
        """
        z2 -> B^{-1} -> z1 -> A^{-1} -> x
        """
        z1, logdet2 = self.modelB.inverse(z)
        x, logdet1 = self.modelA.inverse(z1)
        return x, logdet1 + logdet2

    def forward_with_logdet(self, x):
        """
        x -> A -> z1 -> B -> z2
        """
        z1,logdet1 = self.modelA.forward_with_logdet(x)
        z2,logdet2 = self.modelB.forward_with_logdet(z1)
        return z2, logdet1 + logdet2        
       
def SG_sampler(
     gd, grid_vegas, fun_target, param, flow,
     device=None,
     n_vegas_proposals=100000,
     n_nf_proposals=100000,
     n_final=None,
     batch_proposal=20000,
     resample_method='systematic',
     jitter=0.0,
     ess_threshold=None):
     """
     Hybrid importance-resampling that merges proposals from
     (1) VEGAS grid sampler and (2) a trained Normalizing Flow (NF).

     Args:
       gd, grid_vegas, fun_target, param: same meanings as in VG_resampler_importance
       flow: trained NF model with .log_prob(x) implemented (returns log q_nf(x))
       device: torch.device or None (auto)
       n_vegas_proposals: number of proposals from VEGAS
       n_nf_proposals: number of proposals from Normalizing Flow
       n_final: number of final unweighted samples; default = n_vegas_proposals + n_nf_proposals
       batch_proposal: batch size for proposal drawing
       resample_method: 'systematic' or 'residual'
       jitter: optional small Gaussian jitter in unit space (applied after merge, before mapping to physical)
       ess_threshold: optional fractional threshold on ESS to warn (e.g. 0.2)

     Returns:
       x_resampled_phys: Tensor of shape (n_final, Nd) in physical domain
       stats: dict with ESS, totals, and breakdown
     """
     if device is None:
         device = device_info()
     dtype = torch.get_default_dtype()

     Nd = int(param['nd'])
     scl = param['scale']
     a = _to_tensor(param['xmin'][:Nd], dtype=dtype, device=device)
     b = _to_tensor(param['xmax'][:Nd], dtype=dtype, device=device)

     if n_final is None:
         n_final = n_vegas_proposals + n_nf_proposals

     # -----------------------------
     # 1) Collect VEGAS proposals
     # -----------------------------
     s_list_vg, w_list_vg,J_list_vg = [], [], []
     drawn = 0
     while drawn < n_vegas_proposals:
         this = min(batch_proposal, n_vegas_proposals - drawn)
         # s_batch in unit cube, fw_batch = f(s)*J_vg(s)
         s_batch, fw_batch, _, _, Jacobi_vg_batch = ais_vectorized_batch(
             gd, grid_vegas, fun_target, param, this, device
         )
         s_list_vg.append(s_batch)              # (this, Nd) in unit space
         w_list_vg.append(fw_batch.to(dtype))   # (this,)
         J_list_vg.append(Jacobi_vg_batch.to(dtype))   # (this,)
         drawn += this

     s_vg = torch.cat(s_list_vg, dim=0) if s_list_vg else torch.empty(0, Nd, dtype=dtype, device=device)
     f_vg = torch.cat(w_list_vg, dim=0) if w_list_vg else torch.empty(0, dtype=dtype, device=device)
     J_vg = torch.cat(J_list_vg, dim=0) if J_list_vg else torch.empty(0, dtype=dtype, device=device)
     # Map VEGAS samples to physical x
     x_vg = a + (b - a) * s_vg
     vol = torch.prod(b - a)
     
     
     # 2) NF refinement
     # -----------------------------
     flow.eval()
     with torch.no_grad():
         # Push VEGAS samples through NF to obtain refined samples x_nf
         # Assume flow.forward(x) refines toward higher-density regions
         #x_nf = flow.forward(x_vg)
         #log_q_nf = flow.log_prob(x_vg) # not x_nf
         
         x_nf, log_q_nf= flow.forward_with_logdet(x_vg)
         
     q_nf = torch.exp(torch.clamp(log_q_nf, min=-30, max=30))
     
     log_q_nf = torch.clamp(log_q_nf, min=-30, max=30)
     J_nf = torch.exp(-log_q_nf)

     
     # Convert x to unit space s = (x-a)/(b-a) to evaluate your fun_target (which expects unit coords)
     x_nf = x_nf.to(device=device, dtype=dtype)
     #s_nf = (x_nf-a)/(b-a)
     s_nf = torch.clamp((x_nf - a) / (b - a), 0.0, 1.0)
     s_nf = (x_nf - a) / (b - a)
     # Target (unnormalized) at s: same call used for VEGAS
     # fun_target(s, param) already returns f(s) * vol (like in your pipeline)
     f_nf,log_f = fun_target(s_nf, param)
     f_nf = f_nf.to(dtype)/vol

     # Importance weight for NF proposals: w = f(s) / q_nf(x)
     # (Same target as VEGAS branch; proposals differ.)
     #w_nf = f_nf / torch.clamp(q_nf, min=1e-2)

     # -----------------------------
     # 3) Merge proposals and resample
     # -----------------------------
     
     s_FuHsi = x_nf #s_nf
     w_FuHsi =vol/scl**Nd*f_nf*J_nf*J_vg
     w_vg = f_vg  #*J_vg 
     
     print('max of f_FuHsi')
     print(w_FuHsi.max())
     print(w_FuHsi.mean())
     print('max of f_flow')
     print(f_nf.max())
     print(f_nf.mean())
     print('max of J_flow')
     print(J_nf.max())
     print(J_nf.mean())
     print('max of J_vg')
     print(J_vg.max())
     print(J_vg.mean())
       
     fmax = w_FuHsi[w_FuHsi<1e4].max()
     
     w_FuHsi = torch.clamp(w_FuHsi, max=fmax)  # cut the abnormal samples

     #a = w_FuHsi < 1e4 #1e5
     #b = f_nf / torch.max(f_nf) > 1e-8
     #mask = a & b
     #w_FuHsi = w_FuHsi[b]
     print('size of wFuHsi:',w_FuHsi.size())
     print('max of f_FuHsi')
     print(w_FuHsi.max())
     print(w_FuHsi.mean())
     # Safety-normalize weights
     w_norm, total_w = _safe_norm_weights(w_FuHsi)  # returns weights summing to 1
     ESS = 1.0 / torch.sum(w_norm ** 2)
     ESS_value = float(ESS.item())
     n_prop = s_FuHsi.shape[0]
     n_prop
     if ess_threshold is not None:
         if ESS_value < ess_threshold * n_prop:
             print(f"WARNING: low ESS {ESS_value:.1f} < {ess_threshold * n_prop:.1f}")

     # Choose resampling indices
     if resample_method == 'systematic':
         idx = systematic_resample(w_norm, n_final, device=device)
     elif resample_method == 'residual':
         idx = residual_resample(w_norm, n_final, device=device)
     elif resample_method == 'metropolis':
         idx = metropolis_resample(
         w_norm, n_final,
         device=device,
         proposal_width=1000,
         thinning=100,        # "pick one index every 100 steps"
         warmup_steps=1000,)   # warming up for 1000 steps
    
     else:
         raise ValueError("resample_method must be 'systematic' or 'residual'")
         
     #idx = idx.to(torch.long)
     #s_resampled = s_FuHsi[idx, :]

     x_resampled = s_FuHsi[idx]

     if jitter > 0.0:
         noise = torch.randn_like(x_resampled) * float(jitter)
         # Convert to unit cube for clipping, then back to physical
         s_tmp = torch.clamp((x_resampled - a) / (b - a) + noise, 0.0, 1.0)
         x_resampled_2 = a + (b - a) * s_tmp

     # Map to physical space
     #x_resampled_phys = a + (b - a) * s_resampled

     stats = {
         'ESS': ESS_value,
         'total_weight': float(total_w.item()) if isinstance(total_w, torch.Tensor) else float(total_w),
         'n_proposals_total': int(n_prop),
         'n_proposals_vegas': int(s_vg.shape[0]),
         'n_proposals_nf': int(s_nf.shape[0]),
         'method': resample_method
     }
     return x_resampled, x_vg, x_nf,w_FuHsi,f_nf,J_nf,J_vg,idx, stats

def compute_ess(w):
    """
    Effective Sample Size for unnormalized weights.
    """
    w = torch.clamp(w, min=1e-30)
    return (w.sum() ** 2) / torch.sum(w ** 2)

def eval_in_chunks(fun, x, param, chunk_size=10000):
    outs = []
    outs_log=[]
    N = x.shape[0]
    for i in range(0, N, chunk_size):
        x_chunk = x[i:i+chunk_size]
        out_chunk,out_log_chunk = fun(x_chunk, param)
        outs.append(out_chunk)
        outs_log.append(out_log_chunk)
      
    f=torch.cat(outs, dim=0)
    log_f =     torch.cat(outs_log, dim=0)
    #print("eval in chunks")
    #print(f)
    #print(log_f)
    return f, log_f
    
def SG_sampler_combine(
     gd, grid_vegas, fun_target, param, flow,
     device=None,
     n_vegas_proposals=100000,
     n_nf_proposals=100000,   # kept for compatibility, not used separately
     n_final=None,
     batch_proposal=20000,
     resample_method='systematic',
     jitter=0.0,
     ess_threshold=None,
     M=30
):
    """
    Hybrid importance-resampling with M independent AIS+NF runs
    merged BEFORE resampling.

    Final output:
        x_resampled_extended: (n_final, Nd*M)
    """
    if device is None:
        device = device_info()
    dtype = torch.get_default_dtype()

    Nd = int(param['nd'])
    scl = param['scale']
    a = _to_tensor(param['xmin'][:Nd], dtype=dtype, device=device)
    b = _to_tensor(param['xmax'][:Nd], dtype=dtype, device=device)
    param_draw=param.copy()
    param_resample = param.copy()
    param_resample['Isrc']=0
    param_resample['nd'] = Nd*M
    param_resample['xmin'] = np.full(Nd*M,param['xmin'].min() )
    param_resample['xmax'] = np.full(Nd*M,param['xmax'].max() )
    if n_final is None:
        n_final = n_vegas_proposals + n_nf_proposals

    
    log_scl = torch.log(torch.tensor(scl, device=device, dtype=dtype, requires_grad=False))
    log_vol = torch.sum(torch.log(b - a)) 
    vol = torch.exp(log_vol)
    

    # ============================================================
    # Run AIS + NF M times
    # ============================================================
    x_nf_list = []
    x_vg_list = []
    J_vg_list = []
    log_q_nf_list = []

    for m in range(M):

        print(f'Running proposal batch {m+1}/{M}')

        # -----------------------------
        # 1) VEGAS proposals
        # -----------------------------
        s_list_vg, w_list_vg, J_list_vg = [], [], []
        drawn = 0
        while drawn < n_vegas_proposals:
            this = min(batch_proposal, n_vegas_proposals - drawn)
            s_batch, fw_batch, _, _, Jacobi_vg_batch = ais_vectorized_batch(
                gd, grid_vegas, fun_target, param_draw, this, device
            )
            s_list_vg.append(s_batch)
            w_list_vg.append(fw_batch.to(dtype))
            J_list_vg.append(Jacobi_vg_batch.to(dtype))
            drawn += this

        s_vg = torch.cat(s_list_vg, dim=0)
        f_vg = torch.cat(w_list_vg, dim=0)
        J_vg = torch.cat(J_list_vg, dim=0)

        # Map to physical
        x_vg = a + (b - a) * s_vg

        # -----------------------------
        # 2) NF refinement
        # -----------------------------
        flow.eval()
        with torch.no_grad():
            x_nf, log_q_nf = flow.forward_with_logdet(x_vg)
            
        x_nf_list.append(x_nf)
        x_vg_list.append(x_vg)
        J_vg_list.append(J_vg)
        log_q_nf_list.append(log_q_nf)

    # ============================================================
    # Merge all M runs
    # ============================================================
    if M>=2:
        x_nf = torch.cat(x_nf_list, dim=1)   # (n_prop, Nd*M)
        x_vg = torch.cat(x_vg_list, dim=1)
        #log_q_nf = torch.cat(log_q_nf_list, dim=0)
        log_J_vg_all = torch.log(torch.stack(J_vg_list, dim=0))
        log_J_vg = log_J_vg_all.sum(dim=0)   # shape: (N,)
        log_q_nf_all = torch.stack(log_q_nf_list, dim=0)
        log_q_nf = log_q_nf_all.sum(dim=0)   # shape: (N,)
    else:
        x_nf =  x_nf     # (n_prop, Nd*M)
        x_vg =  x_vg  
        #log_q_nf = torch.cat(log_q_nf_list, dim=0)
        log_q_nf  = log_q_nf 
        log_J_vg = torch.log(J_vg)            
    
    
    
    
    log_q_nf = torch.clamp(log_q_nf, min=-40, max=40)
    J_nf = torch.exp(-log_q_nf)

    a1 = _to_tensor(param_resample['xmin'][:Nd*M], dtype=dtype, device=device)
    b1 = _to_tensor(param_resample['xmax'][:Nd*M], dtype=dtype, device=device)
    
    
    log_volx = torch.sum(torch.log(b1 - a1)) #-Nd*M*log_scl
    volx = torch.exp(log_volx)

    #log_volxx = torch.sum(torch.log(b1 - a1)) 
    #volxx = torch.exp(log_volxx)
    
    # back to unit for target
    s_nf = (x_nf - a1) / (b1 - a1)
    print(param_resample)
    f_nf = eval_in_chunks(fun_target, s_nf, param_resample, chunk_size=10000).to(dtype)

    #f_nf = fun_target(s_nf, param_resample).to(dtype)  #/ volxx

 
    # final importance weight
    J_vg = torch.exp(log_J_vg)
    log_J_vg = torch.clamp(log_J_vg, min=-10000, max=10000)
    log_f_nf = torch.clamp(torch.log(f_nf), min=-4000, max=40000)
    log_FuHsi = -log_q_nf + log_J_vg+log_f_nf-Nd*M*log_scl
    
    log_FuHsi = torch.clamp(log_FuHsi, min=-40, max=40)
    #w_FuHsi = volx  * f_nf * J_nf * J_vg
    w_FuHsi = torch.exp(log_FuHsi)
    # clip extreme weights (your original logic)
    
    fmax = w_FuHsi[w_FuHsi/w_FuHsi.mean() < 1e3].max()
    print(fmax)
    w_FuHsi = torch.clamp(w_FuHsi, max=fmax) 
    print(w_FuHsi)
    print(w_FuHsi.mean())
    print(w_FuHsi.std())

    print('max of f_FuHsi')
    print(w_FuHsi.max() )
    print(w_FuHsi.mean())
    print('max of f_flow')
    print(f_nf.max())
    print(f_nf.mean())
    print('max of J_flow')
    print(J_nf.max())
    print(J_nf.mean())
    print('max of J_vg')
    print(J_vg.max())
    print(J_vg.mean())
    print((f_nf * J_nf * J_vg).mean()/scl**(Nd*M))

    print('max of log_FuHsi')
    print(log_FuHsi.max() )
    print(log_FuHsi.mean())  
    print('max of log_q_nf')
    print(log_q_nf.min() )
    print(log_q_nf.mean())    
    print('max of log_J_vg')
    print(log_J_vg.max() )
    print(log_J_vg.mean())
    print('max of log_f_nf')
    print(log_f_nf.max() )
    print(log_f_nf.mean())
    
    ESS = compute_ess(w_FuHsi)
    ESS_value = float(ESS.item())
    n_prop = w_FuHsi.shape[0]
    ESS_frac = ESS_value / n_prop
    
    print(f"ESS = {ESS_value:.1f} / {n_prop}  (ESS/N = {ESS_frac:.4f})") 
    s_FuHsi = x_nf
    
    
    
    #w_FuHsi = torch.cat(w_list, dim=0)     # (M*n_prop,)

    #print('Total proposals:', s_FuHsi.shape[0])

    # ============================================================
    # Normalize + ESS
    # ============================================================
    w_norm, total_w = _safe_norm_weights(w_FuHsi)
    ESS = 1.0 / torch.sum(w_norm ** 2)
    ESS_value = float(ESS.item())
    n_prop = s_FuHsi.shape[0]

    if ess_threshold is not None:
        if ESS_value < ess_threshold * n_prop:
            print(f"WARNING: low ESS {ESS_value:.1f} < {ess_threshold * n_prop:.1f}")

    # ============================================================
    # Resampling (ONCE, globally)
    # ============================================================
    if resample_method == 'systematic':
        idx = systematic_resample(w_norm, n_final, device=device)
    elif resample_method == 'residual':
        idx = residual_resample(w_norm, n_final, device=device)
    elif resample_method == 'metropolis':
        idx = metropolis_resample(
            w_norm, n_final,
            device=device,
            proposal_width=1000,
            thinning=100,
            warmup_steps=1000
        )
    else:
        raise ValueError("resample_method must be 'systematic' or 'residual'")

    x_resampled = s_FuHsi[idx]   # (n_final, Nd)

    # Optional jitter
    if jitter > 0.0:
        noise = torch.randn_like(x_resampled) * float(jitter)
        s_tmp = torch.clamp((x_resampled - a) / (b - a) + noise, 0.0, 1.0)
        x_resampled = a + (b - a) * s_tmp

    # ============================================================
    # Final reshape: (n_final, Nd*M)
    # ============================================================
    #x_resampled_extended = x_resampled.reshape(n_final, Nd * M)

    stats = {
        'ESS': ESS_value,
        'total_weight': float(total_w.item()) if isinstance(total_w, torch.Tensor) else float(total_w),
        'n_proposals_total': int(n_prop),
        'n_proposals_per_run': int(n_vegas_proposals),
        'M_runs': M,
        'method': resample_method
    }

    #return x_resampled_extended, idx, stats
    return x_resampled, x_vg, x_nf,w_FuHsi,f_nf,J_nf,J_vg,idx, stats


def SG_sampler_nucleus(
     gd, grid_vegas, fun_target, param,param_flow, flow,
     device=None,
     n_vegas_proposals=100000,
     n_nf_proposals=100000,   # kept for compatibility, not used separately
     n_final=None,
     batch_proposal=20000,
     resample_method='systematic',
     jitter=0.0,
     ess_threshold=None,
     M=30
):
    """
    Hybrid importance-resampling with M independent AIS+NF runs
    merged BEFORE resampling.

    Final output:
        x_resampled_extended: (n_final, Nd*M)
    """
    if device is None:
        device = device_info()
    dtype = torch.get_default_dtype()

    Nd = int(param['nd'])
    scl = param['scale']
    a = _to_tensor(param['xmin'][:Nd], dtype=dtype, device=device)
    b = _to_tensor(param['xmax'][:Nd], dtype=dtype, device=device)
    param_draw=param.copy()
    param_resample = param_flow.copy()
    if n_final is None:
        n_final = n_vegas_proposals + n_nf_proposals

    
    log_scl = torch.log(torch.tensor(scl, device=device, dtype=dtype, requires_grad=False))
    log_vol = torch.sum(torch.log(b - a)) 
    vol = torch.exp(log_vol)
    

    # ============================================================
    # Run AIS + NF M times
    # ============================================================
    x_nf_list = []
    x_vg_list = []
    J_vg_list = []
    log_q_nf_list = []

    for m in range(M):

        print(f'Running proposal batch {m+1}/{M}')

        # -----------------------------
        # 1) VEGAS proposals
        # -----------------------------
        s_list_vg, w_list_vg, J_list_vg = [], [], []
        drawn = 0
        while drawn < n_vegas_proposals:
            this = min(batch_proposal, n_vegas_proposals - drawn)
            s_batch, fw_batch, _, _, Jacobi_vg_batch = ais_vectorized_batch(
                gd, grid_vegas, fun_target, param_draw, this, device
            )
            s_list_vg.append(s_batch)
            w_list_vg.append(fw_batch.to(dtype))
            J_list_vg.append(Jacobi_vg_batch.to(dtype))
            drawn += this

        s_vg = torch.cat(s_list_vg, dim=0)
        f_vg = torch.cat(w_list_vg, dim=0)
        J_vg = torch.cat(J_list_vg, dim=0)

        # Map to physical
        x_vg = a + (b - a) * s_vg

       
            
        #x_nf_list.append(x_nf)
        x_vg_list.append(x_vg)
        J_vg_list.append(J_vg)
        #log_q_nf_list.append(log_q_nf)

    # ============================================================
    # Merge all M runs
    # ============================================================
    if M>=2:
        #x_nf = torch.cat(x_nf_list, dim=1)   # (n_prop, Nd*M)
        x_vg = torch.cat(x_vg_list, dim=1)
        #log_q_nf = torch.cat(log_q_nf_list, dim=0)
        log_J_vg_all = torch.log(torch.stack(J_vg_list, dim=0))
        log_J_vg = log_J_vg_all.sum(dim=0)   # shape: (N,)
        #log_q_nf_all = torch.stack(log_q_nf_list, dim=0)
        #log_q_nf = log_q_nf_all.sum(dim=0)   # shape: (N,)
    else:
        #x_nf =  x_nf     # (n_prop, Nd*M)
        x_vg =  x_vg  
        #log_q_nf = torch.cat(log_q_nf_list, dim=0)
        #log_q_nf  = log_q_nf 
        log_J_vg = torch.log(J_vg)            
    
 # -----------------------------
 # 2) NF refinement
 # -----------------------------
    flow.eval()
    n_chunks = 20000
    with torch.no_grad():
        x_nf, log_q_nf = eval_flow_forward_in_chunks(flow,x_vg,n_chunks)
        #x_nf, log_q_nf = flow.forward_with_logdet(x_vg)    
    
    
    log_q_nf = torch.clamp(log_q_nf, min=-40, max=40)
    J_nf = torch.exp(-log_q_nf)

    a1 = _to_tensor(param_resample['xmin'][:Nd*M], dtype=dtype, device=device)
    b1 = _to_tensor(param_resample['xmax'][:Nd*M], dtype=dtype, device=device)
    
    
    log_volx = torch.sum(torch.log(b1 - a1)) #-Nd*M*log_scl
    volx = torch.exp(log_volx)

    #log_volxx = torch.sum(torch.log(b1 - a1)) 
    #volxx = torch.exp(log_volxx)
    
    # back to unit for target
    s_nf = (x_nf - a1) / (b1 - a1)
    print(param_resample)
    f_nf,log_f_nf = eval_in_chunks(fun_target, s_nf, param_resample, chunk_size=20000)

    #f_nf = fun_target(s_nf, param_resample).to(dtype)  #/ volxx

 
    # final importance weight
    J_vg = torch.exp(log_J_vg)
    log_J_vg = torch.clamp(log_J_vg, min=-10000, max=10000)
    #log_f_nf = torch.clamp(torch.log(f_nf), min=-4000, max=40000)
    
    log_FuHsi = -log_q_nf + log_J_vg+log_f_nf-Nd*M*log_scl
    
    log_FuHsi = torch.clamp(log_FuHsi, min=-100, max=100)
    #w_FuHsi = volx  * f_nf * J_nf * J_vg
    w_FuHsi = torch.exp(log_FuHsi)
    # clip extreme weights (your original logic)
    
    fmax = w_FuHsi[w_FuHsi/w_FuHsi.mean() < 1e5].max()
    print(fmax)
    w_FuHsi = torch.clamp(w_FuHsi, max=fmax) 
    print(w_FuHsi)
    print(w_FuHsi.mean())
    print(w_FuHsi.std())

    print('max of f_FuHsi')
    print(w_FuHsi.max() )
    print(w_FuHsi.mean())
    print('max of f_flow')
    print(f_nf.max())
    print(f_nf.mean())
    print('max of J_flow')
    print(J_nf.max())
    print(J_nf.mean())
    print('max of J_vg')
    print(J_vg.max())
    print(J_vg.mean())
    print((f_nf * J_nf * J_vg).mean()/scl**(Nd*M))

    print('max of log_FuHsi')
    print(log_FuHsi.max() )
    print(log_FuHsi.mean())  
    print('max of log_q_nf')
    print(log_q_nf.min() )
    print(log_q_nf.mean())    
    print('max of log_J_vg')
    print(log_J_vg.max() )
    print(log_J_vg.mean())
    print('max of log_f_nf')
    print(log_f_nf.max() )
    print(log_f_nf.mean())
    
    ESS = compute_ess(w_FuHsi)
    ESS_value = float(ESS.item())
    n_prop = w_FuHsi.shape[0]
    ESS_frac = ESS_value / n_prop
    
    print(f"ESS = {ESS_value:.1f} / {n_prop}  (ESS/N = {ESS_frac:.4f})") 
    s_FuHsi = x_nf
    
    
    
    #w_FuHsi = torch.cat(w_list, dim=0)     # (M*n_prop,)

    #print('Total proposals:', s_FuHsi.shape[0])

    # ============================================================
    # Normalize + ESS
    # ============================================================
    w_norm, total_w = _safe_norm_weights(w_FuHsi)
    ESS = 1.0 / torch.sum(w_norm ** 2)
    ESS_value = float(ESS.item())
    n_prop = s_FuHsi.shape[0]

    if ess_threshold is not None:
        if ESS_value < ess_threshold * n_prop:
            print(f"WARNING: low ESS {ESS_value:.1f} < {ess_threshold * n_prop:.1f}")

    # ============================================================
    # Resampling (ONCE, globally)
    # ============================================================
    if resample_method == 'systematic':
        idx = systematic_resample(w_norm, n_final, device=device)
    elif resample_method == 'residual':
        idx = residual_resample(w_norm, n_final, device=device)
    elif resample_method == 'metropolis':
        idx = metropolis_resample(
            w_norm, n_final,
            device=device,
            proposal_width=1000,
            thinning=100,
            warmup_steps=1000
        )
    else:
        raise ValueError("resample_method must be 'systematic' or 'residual'")

    x_resampled = s_FuHsi[idx]   # (n_final, Nd)

    # Optional jitter
    if jitter > 0.0:
        noise = torch.randn_like(x_resampled) * float(jitter)
        s_tmp = torch.clamp((x_resampled - a) / (b - a) + noise, 0.0, 1.0)
        x_resampled = a + (b - a) * s_tmp

    # ============================================================
    # Final reshape: (n_final, Nd*M)
    # ============================================================
    #x_resampled_extended = x_resampled.reshape(n_final, Nd * M)

    stats = {
        'ESS': ESS_value,
        'total_weight': float(total_w.item()) if isinstance(total_w, torch.Tensor) else float(total_w),
        'n_proposals_total': int(n_prop),
        'n_proposals_per_run': int(n_vegas_proposals),
        'M_runs': M,
        'method': resample_method
    }

    #return x_resampled_extended, idx, stats
    return x_resampled, x_vg, x_nf,w_FuHsi,f_nf,J_nf,J_vg,idx, stats

def update_flow_boundaries(model, new_B=None, x_min=None, x_max=None):
    new_model = copy.deepcopy(model)

    for module in new_model.modules():
        if isinstance(module, NSF_CL):

            # ---- Update B ----
            if new_B is not None:
                module.B = new_B

            device = module.x_min_1.device

            # ---- Update x_min ----
            if x_min is not None:
                xm = torch.as_tensor(x_min, device=device)
                if xm.ndim == 0:
                    xm = xm.repeat(module.dim)

                module.x_min_1.copy_(xm[:module.d1])
                module.x_min_2.copy_(xm[module.d1:])

            # ---- Update x_max ----
            if x_max is not None:
                xM = torch.as_tensor(x_max, device=device)
                if xM.ndim == 0:
                    xM = xM.repeat(module.dim)

                module.x_max_1.copy_(xM[:module.d1])
                module.x_max_2.copy_(xM[module.d1:])

            # ---- If only B is given, use ±B ----
            if new_B is not None and x_min is None and x_max is None:
                module.x_min_1.fill_(-new_B)
                module.x_min_2.fill_(-new_B)
                module.x_max_1.fill_( new_B)
                module.x_max_2.fill_( new_B)

    return new_model
# ----------------------------
# create_flow_from_args: build flow list and model
# ----------------------------
def create_flow_from_args(args, dim,fun,funA, cluster_params=None, B=None,x_max=None,device="cuda"):
    flow_layers = []

    prior_type = "cluster" if args.use_mixture else "gaussian"

    for _ in range(args.flows):
        # optional top-level actnorm/conv (keeps your earlier style)
        if args.actnorm:
            flow_layers.append(ActNorm(dim))
        if args.convolve:
            flow_layers.append(OneByOneConv(dim))

        # coupling block (single NSF_CL block per 'flow' iteration)
        block = NSF_CL(dim=dim, K=args.num_bins, B=B, hidden_dim=args.hidden_dim,
                       base_network=FCNN, L=1, use_actnorm=False, use_1x1conv=False,
                       x_min=-x_max, x_max=x_max) # 5.0
        flow_layers.append(block)

    model = NormalizingFlowModel(dim=dim, flows=flow_layers, prior_type=prior_type, cluster_params=cluster_params,fun=fun,funA=funA)
    return model

# ----------------------------
# training function
# ----------------------------
def train_flow(model, train_loader, Nd, epochs=30, lr_max=5e-4,device="cuda",dirsave=None):
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    best_state = None
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for i, (x,) in enumerate(train_loader):
            x = x[:, :Nd].to(device)
            optimizer.zero_grad()
            try:
                z, lp, logdet = model.forward(x)
                total_lp = lp + logdet
                loss = -total_lp.mean()
                if not torch.isfinite(loss):
                    print(f"Skipping batch {i}, non-finite loss")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue

        if batches > 0:
            avg_loss = epoch_loss / batches
            loss_history.append(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(model.state_dict())
            scheduler.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} AvgLoss={avg_loss:.4f} Best={best_loss:.4f}")
                
        if  epoch % 10 == 0:
            # logdir
            cwd = os.getcwd()
            #dirsave = 'results_SG_cluster'
            logdir = os.path.join(cwd, dirsave)
            
            fig, ax = plt.subplots(1, 1)
            sample = z.clone().detach().cpu().numpy()
            #sample = z.clone().detach().to(device).numpy()
            ax.scatter( sample[:10000, 0], sample[:10000, 1], s = 2.0, alpha = 0.5)
            ax.set_aspect(1)
            fig.tight_layout()
            fig.savefig( os.path.join(logdir, 'sample{:d}.jpg'.format(epoch)), bbox_inches = 'tight', dpi = 512)
            plt.close(fig)                

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, loss_history


def train_flow_mask(model, train_loader, Nd, epochs=30, lr_max=5e-4,device="cuda",dirsave=None):
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    best_state = None
    loss_history = []
    total_batches = len(train_loader)
    fraction = 0.4
    batches_to_use = math.ceil(fraction * total_batches)
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for i, (x,) in enumerate(train_loader):
            if i >= batches_to_use:   # ✅ only use first half
                break
            
            x = x[:, :Nd].to(device)
            optimizer.zero_grad()
            try:
                z, lp, logdet = model.forward(x)
                total_lp = lp + logdet
                loss = -total_lp.mean()
                if not torch.isfinite(loss):
                    print(f"Skipping batch {i}, non-finite loss")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue

        if batches > 0:
            avg_loss = epoch_loss / batches
            loss_history.append(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(model.state_dict())
            scheduler.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} AvgLoss={avg_loss:.4f} Best={best_loss:.4f}")
                
        if  epoch % 10 == 0:
            # logdir
            cwd = os.getcwd()
            #dirsave = 'results_SG_cluster'
            logdir = os.path.join(cwd, dirsave)
            
            fig, ax = plt.subplots(1, 1)
            sample = z.clone().detach().cpu().numpy()
            #sample = z.clone().detach().to(device).numpy()
            ax.scatter( sample[:10000, 0], sample[:10000, 1], s = 2.0, alpha = 0.5)
            ax.set_aspect(1)
            fig.tight_layout()
            fig.savefig( os.path.join(logdir, 'sample{:d}.jpg'.format(epoch)), bbox_inches = 'tight', dpi = 512)
            plt.close(fig)                

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, loss_history


def eval_flow_in_chunks(flow, x, chunk_size=10000):
    outs = []
    outs_lp=[]
    outs_logdet=[]
    N = x.shape[0]
    #flow.eval()
    for i in range(0, N, chunk_size):
        x_chunk = x[i:i+chunk_size]
        #with torch.no_grad(): 
        out_chunk,out_lp_chunk,out_logdet_chunk = flow(x_chunk)
        
        outs.append(out_chunk)
        outs_lp.append(out_lp_chunk)
        outs_logdet.append(out_logdet_chunk)
      
    x_out=torch.cat(outs, dim=0)
    lp = torch.cat(outs_lp, dim=0) 
    logdet= torch.cat(outs_logdet, dim=0) 
    return x_out,lp, logdet

def eval_flow_forward_in_chunks(flow, x, chunk_size=10000):
    outs = []
    outs_log_q_nf=[] 
    N = x.shape[0]
    #flow.eval()
    for i in range(0, N, chunk_size):
        x_chunk = x[i:i+chunk_size]
        #with torch.no_grad(): 
        #out_chunk,out_lp_chunk,out_logdet_chunk = flow(x_chunk)
        out_chunk, out_log_q_nf_chunk = flow.forward_with_logdet(x_chunk) 
        
        outs.append(out_chunk)
        outs_log_q_nf.append(out_log_q_nf_chunk ) 
      
    x_out=torch.cat(outs, dim=0)
    log_q_nf = torch.cat(outs_log_q_nf, dim=0) 
    return x_out,log_q_nf




def SG_sampler_ais_nucleus(
     gd, grid_vegas, fun_target, param,param_resample,  
     device=None,
     n_vegas_proposals=100000, 
     n_final=None,
     batch_proposal=20000,
     resample_method='systematic',
     jitter=0.0,
     ess_threshold=None,
     M=30,
     Iprint=1,
     sfactor=1
):
    """
    Hybrid importance-resampling with M independent AIS+NF runs
    merged BEFORE resampling.

    Final output:
        x_resampled_extended: (n_final, Nd*M)
    """
    if device is None:
        device = device_info()
    dtype = torch.get_default_dtype()

    Nd = int(param['nd'])
    scl = param['scale']
    a = _to_tensor(param['xmin'][:Nd], dtype=dtype, device=device)
    b = _to_tensor(param['xmax'][:Nd], dtype=dtype, device=device)
    param_draw=param.copy() 
    if n_final is None:
        n_final = n_vegas_proposals + n_nf_proposals

    
    log_scl = torch.log(torch.tensor(scl, device=device, dtype=dtype, requires_grad=False))
    log_vol = torch.sum(torch.log(b - a)) 
    vol = torch.exp(log_vol)
    

    # ============================================================
    # Run AIS + NF M times
    # ============================================================
    x_nf_list = []
    x_vg_list = []
    J_vg_list = []
    log_q_nf_list = []

    for m in range(M):
        if Iprint==1:
            print(f'Running proposal batch {m+1}/{M}')

        # -----------------------------
        # 1) VEGAS proposals
        # -----------------------------
        s_list_vg, w_list_vg, J_list_vg = [], [], []
        drawn = 0
        while drawn < n_vegas_proposals:
            this = min(batch_proposal, n_vegas_proposals - drawn)
            s_batch, fw_batch, _, _, Jacobi_vg_batch = ais_vectorized_batch(
                gd, grid_vegas, fun_target, param_draw, this, device
            )
            s_list_vg.append(s_batch)
            w_list_vg.append(fw_batch.to(dtype))
            J_list_vg.append(Jacobi_vg_batch.to(dtype))
            drawn += this

        s_vg = torch.cat(s_list_vg, dim=0)
        f_vg = torch.cat(w_list_vg, dim=0)
        J_vg = torch.cat(J_list_vg, dim=0)

        # Map to physical
        x_vg = a + (b - a) * s_vg

       
            
        #x_nf_list.append(x_nf)
        x_vg_list.append(x_vg)
        J_vg_list.append(J_vg)
        #log_q_nf_list.append(log_q_nf)

    # ============================================================
    # Merge all M runs
    # ============================================================
    if M>=2:
        #x_nf = torch.cat(x_nf_list, dim=1)   # (n_prop, Nd*M)
        x_vg = torch.cat(x_vg_list, dim=1)
        #log_q_nf = torch.cat(log_q_nf_list, dim=0)
        log_J_vg_all = torch.log(torch.stack(J_vg_list, dim=0))
        log_J_vg = log_J_vg_all.sum(dim=0)   # shape: (N,)
        #log_q_nf_all = torch.stack(log_q_nf_list, dim=0)
        #log_q_nf = log_q_nf_all.sum(dim=0)   # shape: (N,)
    else:
        #x_nf =  x_nf     # (n_prop, Nd*M)
        x_vg =  x_vg  
        #log_q_nf = torch.cat(log_q_nf_list, dim=0)
        #log_q_nf  = log_q_nf 
        log_J_vg = torch.log(J_vg)            
    
 # -----------------------------
 # 2) NF refinement
 # -----------------------------
    #flow.eval()
    #n_chunks = 1000
    #with torch.no_grad():
    #    x_nf, log_q_nf = eval_flow_forward_in_chunks(flow,x_vg,n_chunks)
        #x_nf, log_q_nf = flow.forward_with_logdet(x_vg)    
    
    
    #log_q_nf = torch.clamp(log_q_nf, min=-40, max=40)
    #J_nf = torch.exp(-log_q_nf)

    a1 = _to_tensor(param_resample['xmin'][:Nd*M], dtype=dtype, device=device)
    b1 = _to_tensor(param_resample['xmax'][:Nd*M], dtype=dtype, device=device)
    
    
    log_volx = torch.sum(torch.log(b1 - a1)) #-Nd*M*log_scl
    volx = torch.exp(log_volx)

    #log_volxx = torch.sum(torch.log(b1 - a1)) 
    #volxx = torch.exp(log_volxx)
    
    # back to unit for target
    s_vg = (x_vg - a1) / (b1 - a1)
    if Iprint==1:
        print(param_resample)
    f_vg_all,log_f_vg_all = eval_in_chunks(fun_target, s_vg, param_resample, chunk_size=10000)

    #f_nf = fun_target(s_nf, param_resample).to(dtype)  #/ volxx

 
    # final importance weight
    J_vg = torch.exp(log_J_vg)
    log_J_vg = torch.clamp(log_J_vg, min=-10000, max=10000)
    #log_f_nf = torch.clamp(torch.log(f_nf), min=-4000, max=40000)
    
    log_ais =  log_J_vg + log_f_vg_all - Nd*M*log_scl+torch.log(sfactor)
    
    log_ais = torch.clamp(log_ais, min=-100, max=100)
    #w_FuHsi = volx  * f_nf * J_nf * J_vg
    w_ais = torch.exp(log_ais)
    # clip extreme weights (your original logic)
    
    fmax = w_ais[w_ais/w_ais.mean() < 1e6].max()
    
    w_ais = torch.clamp(w_ais, max=fmax) 
    ESS = compute_ess(w_ais)
    ESS_value = float(ESS.item())
    n_prop = w_ais.shape[0]
    ESS_frac = ESS_value / n_prop
    #print('max of w_ais')
    #print(w_ais.max() )
    #print(w_ais.mean())
    if Iprint==1:
        print(fmax)
        print(w_ais)
        print(w_ais.mean())
        print(w_ais.std())

        print('max of w_ais')
        print(w_ais.max() )
        print(w_ais.mean())
        print('max of log_w_ais')
        print(log_ais.max() )
        print(log_ais.mean()) 
        print('max of f_flow')
        print(f_vg_all.max())
        print(f_vg_all.mean()) 
        print('max of J_vg')
        print(J_vg.max())
        print(J_vg.mean())
        print((f_vg_all *  J_vg).mean()/scl**(Nd*M))


        print('max of log_J_vg')
        print(log_J_vg.max() )
        print(log_J_vg.mean())
        print('max of log_f_vg')
        print(log_f_vg_all.max() )
        print(log_f_vg_all.mean())
        print(f"ESS = {ESS_value:.1f} / {n_prop}  (ESS/N = {ESS_frac:.4f})") 
    s_vg = x_vg
    
    
    
    #w_FuHsi = torch.cat(w_list, dim=0)     # (M*n_prop,)

    #print('Total proposals:', s_FuHsi.shape[0])

    # ============================================================
    # Normalize + ESS
    # ============================================================
    w_norm, total_w = _safe_norm_weights(w_ais)
    ESS = 1.0 / torch.sum(w_norm ** 2)
    ESS_value = float(ESS.item())
    n_prop = s_vg.shape[0]

    if ess_threshold is not None:
        if ESS_value < ess_threshold * n_prop:
            print(f"WARNING: low ESS {ESS_value:.1f} < {ess_threshold * n_prop:.1f}")

    # ============================================================
    # Resampling (ONCE, globally)
    # ============================================================
    if resample_method == 'systematic':
        idx = systematic_resample(w_norm, n_final, device=device)
    elif resample_method == 'residual':
        idx = residual_resample(w_norm, n_final, device=device)
    elif resample_method == 'metropolis':
        idx = metropolis_resample(
            w_norm, n_final,
            device=device,
            proposal_width=1000,
            thinning=100,
            warmup_steps=1000
        )
    else:
        raise ValueError("resample_method must be 'systematic' or 'residual'")

    x_resampled = s_vg[idx]   # (n_final, Nd)

    # Optional jitter
    if jitter > 0.0:
        noise = torch.randn_like(x_resampled) * float(jitter)
        s_tmp = torch.clamp((x_resampled - a) / (b - a) + noise, 0.0, 1.0)
        x_resampled = a + (b - a) * s_tmp

    # ============================================================
    # Final reshape: (n_final, Nd*M)
    # ============================================================
    #x_resampled_extended = x_resampled.reshape(n_final, Nd * M)

    stats = {
        'ESS': ESS_value,
        'total_weight': float(total_w.item()) if isinstance(total_w, torch.Tensor) else float(total_w),
        'n_proposals_total': int(n_prop),
        'n_proposals_per_run': int(n_vegas_proposals),
        'M_runs': M,
        'method': resample_method
    }

    #return x_resampled_extended, idx, stats
    return x_resampled, x_vg,w_ais,J_vg,idx, stats

def SG_sampler_nucleus(
     gd, grid_vegas, fun_target, param,param_flow, flow,
     device=None,
     n_vegas_proposals=100000,
     n_nf_proposals=100000,   # kept for compatibility, not used separately
     n_final=None,
     batch_proposal=20000,
     resample_method='systematic',
     jitter=0.0,
     ess_threshold=None,
     M=30,
     Iprint=1,
     sfactor=1    
):
    """
    Hybrid importance-resampling with M independent AIS+NF runs
    merged BEFORE resampling.

    Final output:
        x_resampled_extended: (n_final, Nd*M)
    """
    if device is None:
        device = device_info()
    dtype = torch.get_default_dtype()

    Nd = int(param['nd'])
    scl = param['scale']
    a = _to_tensor(param['xmin'][:Nd], dtype=dtype, device=device)
    b = _to_tensor(param['xmax'][:Nd], dtype=dtype, device=device)
    param_draw=param.copy()
    param_resample = param_flow.copy()
    if n_final is None:
        n_final = n_vegas_proposals + n_nf_proposals

    
    log_scl = torch.log(torch.tensor(scl, device=device, dtype=dtype, requires_grad=False))
    log_vol = torch.sum(torch.log(b - a)) 
    vol = torch.exp(log_vol)
    

    # ============================================================
    # Run AIS + NF M times
    # ============================================================
    x_nf_list = []
    x_vg_list = []
    J_vg_list = []
    log_q_nf_list = []

    for m in range(M):
        if Iprint==1:
            print(f'Running proposal batch {m+1}/{M}')

        # -----------------------------
        # 1) VEGAS proposals
        # -----------------------------
        s_list_vg, w_list_vg, J_list_vg = [], [], []
        drawn = 0
        while drawn < n_vegas_proposals:
            this = min(batch_proposal, n_vegas_proposals - drawn)
            s_batch, fw_batch, _, _, Jacobi_vg_batch = ais_vectorized_batch(
                gd, grid_vegas, fun_target, param_draw, this, device
            )
            s_list_vg.append(s_batch)
            w_list_vg.append(fw_batch.to(dtype))
            J_list_vg.append(Jacobi_vg_batch.to(dtype))
            drawn += this

        s_vg = torch.cat(s_list_vg, dim=0)
        f_vg = torch.cat(w_list_vg, dim=0)
        J_vg = torch.cat(J_list_vg, dim=0)

        # Map to physical
        x_vg = a + (b - a) * s_vg

       
            
        #x_nf_list.append(x_nf)
        x_vg_list.append(x_vg)
        J_vg_list.append(J_vg)
        #log_q_nf_list.append(log_q_nf)

    # ============================================================
    # Merge all M runs
    # ============================================================
    if M>=2:
        #x_nf = torch.cat(x_nf_list, dim=1)   # (n_prop, Nd*M)
        x_vg = torch.cat(x_vg_list, dim=1)
        #log_q_nf = torch.cat(log_q_nf_list, dim=0)
        log_J_vg_all = torch.log(torch.stack(J_vg_list, dim=0))
        log_J_vg = log_J_vg_all.sum(dim=0)   # shape: (N,)
        #log_q_nf_all = torch.stack(log_q_nf_list, dim=0)
        #log_q_nf = log_q_nf_all.sum(dim=0)   # shape: (N,)
    else:
        #x_nf =  x_nf     # (n_prop, Nd*M)
        x_vg =  x_vg  
        #log_q_nf = torch.cat(log_q_nf_list, dim=0)
        #log_q_nf  = log_q_nf 
        log_J_vg = torch.log(J_vg)            
    
 # -----------------------------
 # 2) NF refinement
 # -----------------------------
    flow.eval()
    n_chunks = 20000
    with torch.no_grad():
        x_nf, log_q_nf = eval_flow_forward_in_chunks(flow,x_vg,n_chunks)
        #x_nf, log_q_nf = flow.forward_with_logdet(x_vg)    
    
    
    log_q_nf = torch.clamp(log_q_nf, min=-50, max=50)
    J_nf = torch.exp(-log_q_nf)

    a1 = _to_tensor(param_resample['xmin'], dtype=dtype, device=device)
    b1 = _to_tensor(param_resample['xmax'], dtype=dtype, device=device)
    
    
    log_volx = torch.sum(torch.log(b1 - a1)) #-Nd*M*log_scl
    volx = torch.exp(log_volx)

    #log_volxx = torch.sum(torch.log(b1 - a1)) 
    #volxx = torch.exp(log_volxx)
    
    # back to unit for target
    s_nf = (x_nf - a1) / (b1 - a1)
    #print(param_resample)
    f_nf,log_f_nf = eval_in_chunks(fun_target, s_nf, param_resample, chunk_size=20000)

    #f_nf = fun_target(s_nf, param_resample).to(dtype)  #/ volxx

 
    # final importance weight
    J_vg = torch.exp(log_J_vg)
    log_J_vg = torch.clamp(log_J_vg, min=-10000, max=10000)
    #log_f_nf = torch.clamp(torch.log(f_nf), min=-4000, max=40000)
    
    log_FuHsi = -log_q_nf + log_J_vg+log_f_nf-Nd*M*log_scl+torch.log(sfactor)
    
    log_FuHsi = torch.clamp(log_FuHsi, min=-100, max=100)
    #w_FuHsi = volx  * f_nf * J_nf * J_vg
    w_FuHsi = torch.exp(log_FuHsi)
    # clip extreme weights (your original logic)
    
    fmax = w_FuHsi[w_FuHsi/w_FuHsi.mean() < 1e5].max()
    
    
    mask = (w_FuHsi > w_FuHsi.max() / 1e5) & (w_FuHsi < w_FuHsi.max() / 2)
    
    
    #print(fmax)
    w_FuHsi = torch.clamp(w_FuHsi, max=fmax) 
    #w_FuHsi = w_FuHsi[mask]
    
    if Iprint==1:
        print(w_FuHsi)
        print(w_FuHsi.mean())
        print(w_FuHsi.std())

        print('max of f_FuHsi')
        print(w_FuHsi.max() )
        print(w_FuHsi.mean())
        print('max of f_flow')
        print(f_nf.max())
        print(f_nf.mean())
        print('max of J_flow')
        print(J_nf.max())
        print(J_nf.mean())
        print('max of J_vg')
        print(J_vg.max())
        print(J_vg.mean())
        print((f_nf * J_nf * J_vg).mean()/scl**(Nd*M))

        print('max of log_FuHsi')
        print(log_FuHsi.max() )
        print(log_FuHsi.mean())  
        print('max of log_q_nf')
        print(log_q_nf.min() )
        print(log_q_nf.mean())    
        print('max of log_J_vg')
        print(log_J_vg.max() )
        print(log_J_vg.mean())
        print('max of log_f_nf')
        print(log_f_nf.max() )
        print(log_f_nf.mean())
    
    ESS = compute_ess(w_FuHsi)
    ESS_value = float(ESS.item())
    n_prop = w_FuHsi.shape[0]
    ESS_frac = ESS_value / n_prop
    if Iprint==1:    
        print(f"ESS = {ESS_value:.1f} / {n_prop}  (ESS/N = {ESS_frac:.4f})") 
    s_FuHsi = x_nf
    
    
    
    #w_FuHsi = torch.cat(w_list, dim=0)     # (M*n_prop,)

    #print('Total proposals:', s_FuHsi.shape[0])

    # ============================================================
    # Normalize + ESS
    # ============================================================
    w_norm, total_w = _safe_norm_weights(w_FuHsi)
    ESS = 1.0 / torch.sum(w_norm ** 2)
    ESS_value = float(ESS.item())
    n_prop = s_FuHsi.shape[0]
    if Iprint==1:    
        print('shape of w_FuHsi',w_FuHsi.size())

    if ess_threshold is not None:
        if ESS_value < ess_threshold * n_prop:
            print(f"WARNING: low ESS {ESS_value:.1f} < {ess_threshold * n_prop:.1f}")

    # ============================================================
    # Resampling (ONCE, globally)
    # ============================================================
    if resample_method == 'systematic':
        idx = systematic_resample(w_norm, n_final, device=device)
    elif resample_method == 'residual':
        idx = residual_resample(w_norm, n_final, device=device)
    elif resample_method == 'metropolis':
        idx = metropolis_resample(
            w_norm, n_final,
            device=device,
            proposal_width=1000,
            thinning=100,
            warmup_steps=1000
        )
    else:
        raise ValueError("resample_method must be 'systematic' or 'residual'")

    x_resampled = s_FuHsi[idx]   # (n_final, Nd)

    # Optional jitter
    if jitter > 0.0:
        noise = torch.randn_like(x_resampled) * float(jitter)
        s_tmp = torch.clamp((x_resampled - a) / (b - a) + noise, 0.0, 1.0)
        x_resampled = a + (b - a) * s_tmp

    # ============================================================
    # Final reshape: (n_final, Nd*M)
    # ============================================================
    #x_resampled_extended = x_resampled.reshape(n_final, Nd * M)

    stats = {
        'ESS': ESS_value,
        'total_weight': float(total_w.item()) if isinstance(total_w, torch.Tensor) else float(total_w),
        'n_proposals_total': int(n_prop),
        'n_proposals_per_run': int(n_vegas_proposals),
        'M_runs': M,
        'method': resample_method
    }

    #return x_resampled_extended, idx, stats
    return x_resampled, x_vg, x_nf,w_FuHsi,f_nf,J_nf,J_vg,idx, stats



def SG_nucleus(
     fun_integrand,CustomDistribution,Nd=12, M=1,I_ais_only=1,seed=12345, Nit=251,trained_model=None, 
     device=None,
     dtype=None):
    # ============================================================
    # 0. Set AIS parameters
    # ============================================================
    # 
    #Nd = 12
    #M=52

    #MA=26
    #MB=M-MA

    parser = ArgumentParser()
    #parser.add_argument("--nd", type=int, required=True, help="dimension")
    #parser.add_argument("--M", type=int, required=True, help="dimension")
    #args = parser.parse_args()
    #Nd=args.nd
    #M=args.M

    scl=1.5    # very important
    Iscal = scl**Nd   # scale factor
    Isrc=0     # 0: no src, >0 with src
    Isrc_target=1
    Inucleus = 208
    gdv = [150 + k for k in range(Nd)]   # adjusted for GPU run
    Nsample = 4000000
    Nstep = 20
    xmin=-11
    xmax=11
    batch_size = 20000  

    param_ais = {
        'nd': Nd, 
        'xmin': np.full(Nd, xmin),
        'xmax': np.full(Nd, xmax),    
        'scale':scl,
        'Isrc':Isrc,
        'Inucleus':Inucleus
    }

    param_ais_resample = param_ais.copy() # For resampling
    param_ais_resample['Isrc']=Isrc_target


    # ============================================================
    # 0. Set Flow parameters
    # ============================================================
    #Nit=251
    if I_ais_only==1:
        Nit = 1
    # ----------------------------
    # argparse + main entry (keeps your original CLI)
    # ----------------------------
    #import logging 
    #parser = ArgumentParser()
    parser.add_argument("--batch", default=512, type=int)
    parser.add_argument("--flows", default=4, type=int)
    parser.add_argument("--num-bins", dest="num_bins", default=8, type=int)
    parser.add_argument("--hidden-dim", dest="hidden_dim", default=256, type=int)
    parser.add_argument("--iterations", default=Nit, type=int)
    parser.add_argument("--use-mixture", action="store_true", default=True) 
    #parser.add_argument("--use-mixture", action="store_true", default=False)
    parser.add_argument("--convolve", action="store_true")
    parser.add_argument("--actnorm", action="store_true")
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    cluster_params = param_ais.copy()  # your earlier param dictionary
    cluster_params['Isrc']=Isrc_target
    cluster_params['nd']=Nd*M
    cluster_params['xmin'] = np.full(Nd*M, xmin)
    cluster_params['xmax'] = np.full(Nd*M, xmax)



    a = _to_tensor(param_ais['xmin'], dtype=dtype, device=device) 
    b = _to_tensor(param_ais['xmax'], dtype=dtype, device=device) 

    vol = torch.prod(b - a)


    # ============================================================
    # 1. Run AIS
    # ============================================================
    print(f"Running on device: {device}, dtype: {torch.get_default_dtype()}")
    I, Ierr, I_MC, grid_ais = SG_AIS_integrator(fun_integrand, Nd, gdv, Nsample, Nstep, param_ais,
                                            device=device, batch_size=batch_size, seed=seed, stratified=False)

    print(f"\nFinal result: {I.item()/Iscal} ± {Ierr.item()/Iscal}")

    # Final sampling for visualization
    Ns = 500000
    x_ais, Jacobi, fw = SG_AIS_sampler(Ns, gdv, grid_ais, Nd, param_ais, fun_integrand,
                                      device=device, batch_size=10000, stratified=False)
    x0 = SG_AIS_inverse_mapping(x_ais, param_ais,Nd,gdv,grid_ais)
    x_ais_resample=[]

    for k in range(2):
        print(k)
        x_ais_resample_, status = SG_AIS_resampler(gdv, grid_ais, fun_integrand, param_ais_resample,device=device,total_proposals=10000000,n_final=Ns,batch_proposal=20000)
        x_ais_resample.append(x_ais_resample_)

    x_ais_resample = torch.cat(x_ais_resample,dim=0)

    x0_resample = SG_AIS_inverse_mapping(x_ais_resample, param_ais,Nd,gdv,grid_ais)
    if device.type=='cpu':
        plot_ais(x0,x_ais*0.5,x0_resample,x_ais_resample*0.5) 
        plot_correlation_function_mixed_event_fast_in_chunks(x_ais,1000,1000000,device=device) 
        plot_correlation_function_mixed_event_fast_in_chunks(x_ais_resample,1000,10000,device=device)
        
        
# ============================================================
# # training dataset Run AIS M times
# ============================================================

    lr_max=3e-5
    Ns = 200000
    batch_size = 20000
    n_proposals = 1000_000
    n_repeat = 10
    n_final = 10000      
    B=11.0
    x_max =10.0
    dirsave = 'results_SG_cluster' 
    if M==1:
        Ns = 200000
        batch_size = 20000
    elif M==2:
        Ns = 200000
        batch_size = 20000        
    elif M==5:
        Ns = 200000
        batch_size = 10000  
    elif M==12:
        Ns = 200000
        batch_size = 20000    
    elif M==16:
        Ns = 200000
        batch_size = 20000  
    elif M==18:
        Ns = 200000
        batch_size = 20000          
    elif M==26:
        Ns = 200000
        batch_size = 10000      
    elif M==52:
        Ns=100000
        batch_size = 2000


    x_train_list = [] 
    x_test_list = [] 
    for m in range(M):
        print(f'Running AIS {m+1}/{M}')
        # -----------------------------
        # 1) AIS proposals
        x_train_, J_vg, f_vg = SG_AIS_sampler(Ns, gdv, grid_ais, Nd,param_ais, fun_integrand,
                                          device=device, batch_size=batch_size, stratified=False) 
        x_train_list.append(x_train_)
        x_test_, _, _ = SG_AIS_sampler(Ns, gdv, grid_ais, Nd,param_ais, fun_integrand,
                                          device=device, batch_size=batch_size, stratified=False) 
        x_test_list.append(x_test_)    
    if M>=2:
        x_train = torch.cat(x_train_list, dim=1) 
        x_test = torch.cat(x_test_list, dim=1) 
    else:
        x_train = x_train_
        x_test = x_test_

    train_loader, train_dataset = prepare_nf_training_data(x_train, batch_size=batch_size)   
    test_loader, test_dataset = prepare_nf_training_data(x_test, batch_size=batch_size)

    # ============================================================
    # 2. Schrodinger generator-flow phase
    # ============================================================ 
    torch.set_grad_enabled(True)
    print(f'Running Flow, Nd={Nd}, M= {M}, Ns = {Ns}')
    # Build model and train (assumes you have train_loader defined earlier in file)
    if trained_model is None:
        modelA = create_flow_from_args(args=args, dim=Nd*M, cluster_params=cluster_params,funA=fun_integrand, device=device,fun=CustomDistribution,B=B,x_max=x_max)
        model0 = copy.deepcopy(modelA).to(device)
        trained_modelA, loss_hist = train_flow(model=modelA, train_loader=train_loader, Nd=Nd*M, epochs=args.iterations, lr_max=lr_max,device=device,dirsave=dirsave)
        if device.type=='cpu':
            plt.figure(figsize=(10, 8))
            plt.plot(loss_hist)
            plt.title('Loss_{flow}')
            plt.show()        
    else:
        trained_modelA=copy.deepcopy(trained_model).to(device) 
    

    x_ais=x_ais.to(device)
    x_train=x_train.to(device)
    x_test=x_test.to(device)

    trained_modelA.eval()
    with torch.no_grad(): 

        x_test_B,lp_test, logdet_test=eval_flow_in_chunks(trained_modelA,x_test)  
        x_train_B,lp_train, logdet_train=eval_flow_in_chunks(trained_modelA,x_train)
        loss_train = -lp_train.mean()-logdet_train.mean()
        loss_test = -lp_test.mean()-logdet_test.mean()

    #print(lp_train.mean(), lp_test.mean())
    #print(lp_train.std(), lp_test.std())  
    #print(loss_train, loss_test)    
    
    
    # ============================================================
    # 2. Integration evaluation
    # ============================================================ 
    param_flow=cluster_params.copy()
    trained_model=trained_modelA
    #I_ini,err_ini,I_ais, err_ais, I_SG, err_SG,Loss_KL_ini,Loss_KL_ais,Loss_KL_sg=SG_Ana_nucleus(trained_model,fun_integrand,scl, 1000000, Nd, M, param_flow, param_ais,gdv,grid_ais,device=device,dtype=dtype)

    
    
# ============================================================
# 3. Resampling
# ============================================================ 
    n_proposals = 1000_000
    if M==1:
        n_repeat = 10
    if M==2:
        n_repeat = 10        
    if M==12:
        n_repeat = 20     
    if M==16:
        n_repeat = 20    
    if M==18:
        n_repeat = 20         
    if M==26:
        n_repeat = 20

    n_final = 10000 
    #M=30
    cluster_params = param_ais.copy()  # your earlier param dictionary
    cluster_params['Isrc']=Isrc_target
    cluster_params['nd']=Nd*M
    cluster_params['xmin'] = np.full(Nd*M, xmin)
    cluster_params['xmax'] = np.full(Nd*M, xmax)
    SG_cut = 1.0
    sfactor = torch.tensor(1e2, dtype=dtype, device=device)  
    if M==1:
        SG_cut = 1.0
        sfactor = torch.tensor(1e3, dtype=dtype, device=device) 

    if M==2:
        SG_cut = 1.0
        sfactor = torch.tensor(1e3, dtype=dtype, device=device) 
        
    if M==52:
        SG_cut = 1.0
        sfactor = torch.tensor(1e15, dtype=dtype, device=device) 
    if M == 30:
        SG_cut = 1.0
        sfactor = torch.tensor(1e2, dtype=dtype, device=device)

    if M == 26:
        SG_cut = 1.0
        sfactor = torch.tensor(1e2, dtype=dtype, device=device)    
    if M == 12:
        SG_cut = 1.0
        sfactor = torch.tensor(1e2, dtype=dtype, device=device)  
        
    x_SG = []
    x_SG_R = []
    x_SG2 = []
    x_SG_nf=[]
    x_ais_B=[]
    w_SG=[]
    SG_A_max=[]
    #I_ais_only=0
    t0=time.time()

    for k in range(n_repeat):
        if k % 2 == 1:
            print(k)
    
        n_vegas_proposals=n_proposals
        n_nf_proposals=n_proposals
        if I_ais_only==1:
            x_SG_, x_ais_B_,   w_SG_,   J_ais, idx, status = SG_sampler_ais_nucleus(
                gdv,
                grid_ais,
                fun_integrand,
                param_ais,
                cluster_params, 
                device=device,

                n_vegas_proposals=n_vegas_proposals, 

                n_final=n_final,
                batch_proposal=20000,
                M=M,
                resample_method='systematic',
                jitter=0.0,
                ess_threshold=None,
                Iprint=0,
                sfactor = sfactor
            )
        else:
            x_SG_, x_ais_B_, x_SG_nf_, w_SG_, f_nf, J_nf, J_ais, idx, status = SG_sampler_nucleus(
                gdv,
                grid_ais,
                fun_integrand,
                param_ais,
                cluster_params,
                trained_model,
                device=device,

                n_vegas_proposals=n_vegas_proposals,
                n_nf_proposals=n_nf_proposals,

                n_final=n_final,
                batch_proposal=20000,
                M=M,
                resample_method='systematic',
                jitter=0.0,
                ess_threshold=None,
                Iprint=0,
                sfactor = sfactor            
            )        

        mask = (w_SG_ >w_SG_.max() / 1e8) # & (w_SG_ < w_SG_.max() / 2)
        mask = (w_SG_ >w_SG_.max() / 1e7)
        #mask = w_SG_ > SG_cut

        w_SG_ = w_SG_[mask]
        x_ais_B_=x_ais_B_[mask]
        SG_A_max.append(w_SG_.max())

        x_SG_R.append(x_SG_.cpu())
        w_SG.append(w_SG_.cpu())
        x_ais_B.append(x_ais_B_.cpu())
        
        print("integrand time:",time.time()-t0)
        
        if I_ais_only==0:
            x_SG_nf_=x_SG_nf_[mask]
            x_SG_nf.append(x_SG_nf_.cpu())
        #x_ais_B.append(x_ais_B_)

    # concatenate into a single tensor
    x_SG_R = torch.cat(x_SG_R, dim=0)
    #SG_max = torch.cat(SG_max, dim=0)
    w_SG_A = torch.cat(w_SG, dim=0)
    if I_ais_only==1:
        x_SG_A = torch.cat(x_ais_B, dim=0)
    else:
        x_SG_A = torch.cat(x_SG_nf, dim=0)
        
    SG_A_max=torch.stack(SG_A_max)
    w_SG_A=w_SG_A.to(device)
    x_SG_A=x_SG_A.to(device)    
    x_SG_R=x_SG_R.to(device) 

    
    w_SG = w_SG_A
    s_SG = x_SG_A
    print("size of w_SG = ",w_SG.size())
    print("mean of w_SG = ",w_SG.mean())
    n_final = 500000
    resample_method='systematic'
    ESS = compute_ess(w_SG)
    ESS_value = float(ESS.item())
    n_prop = w_SG.shape[0]
    ESS_frac = ESS_value / n_prop

    print(f"ESS = {ESS_value:.1f} / {n_prop}  (ESS/N = {ESS_frac:.4f})") 




        #w_FuHsi = torch.cat(w_list, dim=0)     # (M*n_prop,)

        #print('Total proposals:', s_FuHsi.shape[0])

        # ============================================================
        # Normalize + ESS
        # ============================================================
    w_norm, total_w = _safe_norm_weights(w_SG)
    ESS = 1.0 / torch.sum(w_norm ** 2)
    ESS_value = float(ESS.item())
    n_prop = s_SG.shape[0]
    #print('shape of w_FuHsi',w_SG.size())

    #if ess_threshold is not None:
    #    if ESS_value < ess_threshold * n_prop:
    #        print(f"WARNING: low ESS {ESS_value:.1f} < {ess_threshold * n_prop:.1f}")

        # ============================================================
        # Resampling (ONCE, globally)
        # ============================================================
    if resample_method == 'systematic':
        idx = systematic_resample(w_norm, n_final, device=device)
    elif resample_method == 'residual':
        idx = residual_resample(w_norm, n_final, device=device)
    elif resample_method == 'metropolis':
        idx = metropolis_resample(
            w_norm, n_final,
            device=device,
             proposal_width=1000,
             thinning=100,
             warmup_steps=1000
        )
    else:
        raise ValueError("resample_method must be 'systematic' or 'residual'")

    x_SG_resample_A = s_SG[idx]   # (n_final, Nd)    
    
    return  param_ais,x_ais_resample,cluster_params,w_SG_A, x_SG_A,x_SG_R,trained_model,x_SG_resample_A




def SG_Ana_nucleus(flow,fun_integrand,scl, Ns, Nd,M,param_flow, param_ais,gdv,grid_ais,  device="cuda",dtype=None,dirsave=None):
    
    
    n_ais_proposals = Ns
    batch_proposal = 2000
    log_scl = torch.log(torch.tensor(scl, device=device, dtype=dtype, requires_grad=False))
    
    x_nf_list = []
    x_vg_list = []
    J_vg_list = []
    xini_list = []
    log_q_nf_list = []
    a = _to_tensor(param_ais['xmin'], dtype=dtype, device=device)[:Nd]
    b = _to_tensor(param_ais['xmax'], dtype=dtype, device=device)[:Nd]
    for m in range(M):

        print(f'Running proposal batch {m+1}/{M}')

        # -----------------------------
        # 1) VEGAS proposals
        # -----------------------------
        s_list_vg, w_list_vg, J_list_vg,xini_list2 = [], [], [],[]
        drawn = 0
        while drawn < n_ais_proposals:
            this = min(batch_proposal, n_ais_proposals - drawn)
            
            X= torch.rand((batch_proposal, Nd), dtype=dtype, device=device)
            
            s_batch,Jacobi_vg_batch = SG_AIS_mapping(X, param_ais,Nd,gdv,grid_ais, device=device,return_jacobian=True)
            
 
            s_list_vg.append(s_batch)
            xini_list2.append(X)
            #w_list_vg.append(fw_batch.to(dtype))
            J_list_vg.append(Jacobi_vg_batch.to(dtype))
            drawn += this

        s_vg = torch.cat(s_list_vg, dim=0)
        #f_vg = torch.cat(w_list_vg, dim=0)
        J_vg = torch.cat(J_list_vg, dim=0)
        xini = torch.cat(xini_list2, dim=0)

        # Map to physical
        #x_vg = a + (b - a) * s_vg
        x_vg = s_vg

        
            
        #x_nf_list.append(x_nf)
        x_vg_list.append(x_vg)
        J_vg_list.append(J_vg) 
        xini_list.append(xini)
        
    # ============================================================
    # Merge all M runs
    # ============================================================
    if M>=2:
        #x_nf = torch.cat(x_nf_list, dim=1)   # (n_prop, Nd*M)
        x_vg = torch.cat(x_vg_list, dim=1)
        #log_q_nf = torch.cat(log_q_nf_list, dim=0)
        log_J_vg_all = torch.log(torch.stack(J_vg_list, dim=0))
        log_J_vg = log_J_vg_all.sum(dim=0)   # shape: (N,)
        #log_q_nf_all = torch.stack(log_q_nf_list, dim=0)
        #log_q_nf = log_q_nf_all.sum(dim=0)   # shape: (N,)
        xini =  torch.cat(xini_list, dim=1)
    else:
        #x_nf =  x_nf     # (n_prop, Nd*M)
        x_vg =  x_vg  
        #log_q_nf = torch.cat(log_q_nf_list, dim=0)
        #log_q_nf  = log_q_nf 
        log_J_vg = torch.log(J_vg) 
        xini = xini            
        
 # -----------------------------
 # 2) NF refinement
 # -----------------------------
    flow.eval()
    n_chunks = 10000
    with torch.no_grad():
        x_nf, log_q_nf = eval_flow_forward_in_chunks(flow,x_vg,n_chunks)
        #x_nf, log_q_nf = flow.forward_with_logdet(x_vg)    
    
    
    #log_q_nf = torch.clamp(log_q_nf, min=-40, max=40)
    J_nf = torch.exp(-log_q_nf)

    a1 = _to_tensor(param_flow['xmin'][:Nd*M], dtype=dtype, device=device)
    b1 = _to_tensor(param_flow['xmax'][:Nd*M], dtype=dtype, device=device)
    
    
    log_volx = torch.sum(torch.log(b1 - a1)) #-Nd*M*log_scl
    #volx = torch.exp(log_volx)

    #log_volxx = torch.sum(torch.log(b1 - a1)) 
    #volxx = torch.exp(log_volxx)
    
    # back to unit for target
    s_nf = (x_nf - a1) / (b1 - a1) 
    s_ais = (x_vg - a1) / (b1 - a1) 
    
    f_nf,log_f_nf = eval_in_chunks(fun_integrand, s_nf, param_flow, chunk_size=10000)
    w_ais,log_w_ais = eval_in_chunks(fun_integrand, s_ais, param_flow, chunk_size=10000)
    w_ini,log_w_ini = eval_in_chunks(fun_integrand, xini, param_flow, chunk_size=10000)

    #f_nf = fun_target(s_nf, param_resample).to(dtype)  #/ volxx

 
    # final importance weight
    #J_vg = torch.exp(log_J_vg)
    #log_J_vg = torch.clamp(log_J_vg, min=-10000, max=10000)
    #log_f_nf = torch.clamp(torch.log(f_nf), min=-4000, max=40000)
    
    log_FuHsi = log_f_nf-log_volx+log_J_vg-Nd*M*log_scl-log_q_nf 
    
    #log_FuHsi = torch.clamp(log_FuHsi, min=-100, max=100)
    #w_FuHsi = volx  * f_nf * J_nf * J_vg
    f_SG = torch.exp(log_FuHsi)
    f_ini = torch.exp(log_w_ini-Nd*M*log_scl)
    f_ais = torch.exp(log_w_ais-log_volx+log_J_vg-Nd*M*log_scl)
    #f_SG = (f_nf/vol*J_vg/Iscal)*J_nf 
    
    I_SG = f_SG.mean()
    err_SG = f_SG.std()/np.sqrt(Ns)  
    
    I_ini = f_ini.mean()
    err_ini = f_ini.std()/np.sqrt(Ns)
    
    I_ais = f_ais.mean()
    err_ais = f_ais.std()/np.sqrt(Ns)
    
     
    print("Dimension = ",Nd*M)
    print("Monte Carlo 积分近似结果:", I_ini,err_ini, I_ais, err_ais,  I_SG, err_SG)
    
    fmax,idx = torch.max(f_SG, dim=0)
    print(f_nf[idx])
    print(J_vg[idx])
    print(J_nf[idx])
    
    
    
    Loss_KL_ini = -torch.log(f_ini).mean()
    #Loss_KL_ini_std = torch.log(f_ini).std()
    Loss_KL_ais = -torch.log(f_ais).mean()
    Loss_KL_sg = -torch.log(f_SG).mean() 
    print("SG-Uniform: Loss = ",  Loss_KL_ini)
    print("SG-vg: Loss = ",  Loss_KL_ais) 
    print("SG-NF: Loss = ",  Loss_KL_sg) 
    
    return   I_ini,err_ini,I_ais, err_ais,   I_SG, err_SG,Loss_KL_ini,Loss_KL_ais, Loss_KL_sg

def SG_Ana(trained_combine, trained_modelA,fun_integrand,Iscal, Ns, Nd, param_ais,gdv,grid_ais,  device="cuda",dtype=None,dirsave=None):
    
    #Ns = 2000000
    X= torch.rand((Ns, Nd), dtype=dtype, device=device)
    
    Y,J_vg = SG_AIS_mapping(X, param_ais,Nd,gdv,grid_ais, device=device,return_jacobian=True)
    
    xini = SG_AIS_inverse_mapping(Y, param_ais,Nd,gdv,grid_ais)
    
    with torch.no_grad():
        #Z, log_q_nf= trained_modelA.forward_with_logdet(Y)
        Z, log_q_nf= trained_combine.forward_with_logdet(Y)
        Z_A, log_q_nf_A= trained_modelA.forward_with_logdet(Y)
    
    a = _to_tensor(param_ais['xmin'], dtype=dtype, device=device)[:Nd]
    b = _to_tensor(param_ais['xmax'], dtype=dtype, device=device)[:Nd]
    J_nf = torch.exp(-log_q_nf)
    J_nf_A = torch.exp(-log_q_nf_A)
    s_nf = (Z - a) / (b - a)
    s_nf_A = (Z_A - a) / (b - a)
    s_ais = (Y - a) / (b - a)
    vol = torch.prod(b - a)
    f_ais0 = fun_integrand(s_ais,param_ais).to(dtype)
    f_ais = f_ais0/vol*J_vg/Iscal
    I_ais = f_ais.mean()
    err_ais = f_ais.std()/np.sqrt(Ns)
    
    f_nf = fun_integrand(s_nf, param_ais).to(dtype)
    f_nf_A = fun_integrand(s_nf_A, param_ais).to(dtype)
    f_SG = (f_nf/vol*J_vg/Iscal)*J_nf
    f_SG_A = (f_nf_A/vol*J_vg/Iscal)*J_nf_A
    #I_FuHsi = f_FuHsi[f_FuHsi<300].mean()
    #err_FuHsi = f_FuHsi[f_FuHsi<300].std()/np.sqrt(Ns)
    
    I_SG = f_SG.mean()
    err_SG = f_SG.std()/np.sqrt(Ns)
    I_SG_A = f_SG_A.mean()
    err_SG_A = f_SG_A.std()/np.sqrt(Ns)
    print("Nd = ",Nd)
    print("Monte Carlo 积分近似结果:",  I_ais, err_ais,  I_SG_A, err_SG_A, I_SG, err_SG) 
    
    fmax,idx = torch.max(f_SG, dim=0)
    print(f_nf[idx])
    print(J_vg[idx])
    print(J_nf[idx])
    
    #xini = SG_AIS_inverse_mapping(x_ais_C, param_ais,Nd,gdv,grid_ais)
    f_ini = fun_integrand(xini, param_ais)
    #print(f_ini)
    #print(Iscal)
    f_ini = f_ini/Iscal
    #print(f_ini.mean())
    #print(-torch.log(f_ini).mean())
    
    Loss_KL_ini = -torch.log(f_ini).mean()
    Loss_KL_ini_std = torch.log(f_ini).std()
    Loss_KL_ais = -torch.log(f_ais).mean()
    Loss_KL_sg = -torch.log(f_SG).mean()
    Loss_KL_sg_A = -torch.log(f_SG_A).mean()
    print("SG-Uniform: Loss = ",  Loss_KL_ini)
    print("SG-vg: Loss = ",  Loss_KL_ais)
    print("SG-NF: LossA = ",  Loss_KL_sg_A)
    print("SG-NF: Loss = ",  Loss_KL_sg) 
    
    return I_ais, err_ais,  I_SG_A, err_SG_A, I_SG, err_SG,Loss_KL_ini,Loss_KL_ais,Loss_KL_sg_A,Loss_KL_sg
    
def prepare_nf_training_data(
    x_Vegas_MC: torch.Tensor,
    fw: torch.Tensor = None,
    batch_size: int = 1024,
    shuffle: bool = True,
    device=None
):
    """
    Convert VEGAS resampled data into training-ready dataset for Normalizing Flow.
    
    Args:
        x_Vegas_MC: torch.Tensor of shape (N, Nd), resampled samples in physical domain
        fw: optional torch.Tensor of shape (N,), importance weights (before resampling).
            If provided, will normalize and return alongside data.
        batch_size: minibatch size for DataLoader
        shuffle: whether to shuffle dataset
        device: torch.device (default: x_Vegas_MC.device)
    
    Returns:
        loader: DataLoader yielding (x_batch,) or (x_batch, logw_batch)
        dataset: TensorDataset object (kept for flexibility)
    """
    if device is None:
        device = x_Vegas_MC.device
    
    # Ensure double → float32 for NF training (usually needed)
    #x = x_Vegas_MC.to(dtype=torch.float32, device=device)
    x = x_Vegas_MC.to(device=device)

    if fw is not None:
        # normalize weights and return log-weights
        #fw = fw.to(dtype=torch.float32, device=device)
        fw = fw.to(device=device)
        w_norm = fw / fw.sum()
        logw = torch.log(torch.clamp(w_norm, min=1e-12))
        dataset = TensorDataset(x, logw)
    else:
        dataset = TensorDataset(x)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset





