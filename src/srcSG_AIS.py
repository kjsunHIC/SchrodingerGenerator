import torch,math

from srcSG_utilities import  _to_tensor,check_device_and_seed,device_info


# -------------------------
# Grid update (device-aware)
# -------------------------
def grid_update(gdv, gd, d, alpha: float):
    """
    gdv: list[int]
    gd: list[1D tensors length gdv[k]+1] on some device
    d: tensor (Nd, gd_max) on same device
    returns new list of grids (same device as input)
    """
    Nd = len(gdv)
    gd_new = list(gd)
    gd_max = d.shape[1]
    device = d.device
    dtype = d.dtype

    for k in range(Nd):
        nb = gdv[k]
        s = d[k, :nb].clone()

        if nb > 1:
            s2 = s.clone()
            s2[0] = (s[0] * 7 + s[1]) / 8
            s2[-1] = (s[-1] * 7 + s[-2]) / 8
            if nb > 2:
                s2[1:-1] = (s[:-2] + 6 * s[1:-1] + s[2:]) / 8
            s = s2

        total = s.sum()
        if total <= 0:
            s.fill_(1.0 / nb)
        else:
            s = s / total

        eps = torch.finfo(s.dtype).eps
        s_clamped = torch.clamp(s, min=eps)
        denom = -torch.log(s_clamped)
        g = ((1.0 - s_clamped) / denom) ** alpha

        grd = gd[k]
        grdnew = torch.zeros_like(grd)
        grdnew[0] = grd[0]
        grdnew[-1] = grd[-1]

        g_mean = g.mean()
        Sd = 0.0
        j = 0
        i = 0
        while i < nb - 1: # and j < nb:
            if Sd < g_mean:
                j += 1
                Sd += g[j - 1]
            else:
                Sd -= g_mean
                i += 1
                denom_val = g[j - 1] if g[j - 1] > 0 else g_mean
                frac = Sd / denom_val
                jm1 = max(0, j - 1)
                grd_left = grd[jm1]
                grd_right = grd[min(jm1 + 1, nb)]
                grdnew[i] = grd_left + (grd_right - grd_left) * (1 - frac)

        gd_new[k] = grdnew
    return gd_new

def _safe_norm_weights(fw, eps=0.0):
    """Return normalized non-negative weights and sum. Avoid NaNs/Infs."""
    # ensure non-negative (if integrand can be negative, you may take abs or shift — here we assume fw>=0)
    fw_pos = torch.clamp(fw, min=0.0)
    total = fw_pos.sum()
    if total == 0 or not torch.isfinite(total):
        # fallback: uniform weights
        n = fw_pos.numel()
        return torch.full_like(fw_pos, 1.0 / n), total
    return fw_pos / total, total

def metropolis_resample(
    w_norm,
    n_samples,
    device=None,
    proposal_width=1,
    thinning=100,
    warmup_steps=1000,
):
    """
    Metropolis-Hastings resampler with warm-up and thinning.

    Behavior:
      - Perform `warmup_steps` MH transitions (not recorded).
      - Then for each requested sample:
          run `thinning` MH transitions, and record the current index once.
      - This yields one recorded index every `thinning` transitions.

    Args:
        w_norm (torch.Tensor): normalized weights (sum ~ 1), shape (N,).
        n_samples (int): number of indices to return (i.e., number of recorded states).
        device (torch.device or None): where tensors live; defaults to w_norm.device.
        proposal_width (int): symmetric random-walk step range: [-proposal_width, +proposal_width].
        thinning (int): number of MH steps between recorded samples (set to 100 per request).
        warmup_steps (int): initial MH transitions before recording (set to 1000 per request).

    Returns:
        torch.LongTensor: indices of length n_samples (on `device`).
    """
    import torch

    if device is None:
        device = w_norm.device
    N = int(w_norm.shape[0]) if w_norm.numel() else 0
    if N == 0 or n_samples <= 0:
        return torch.empty(0, dtype=torch.long, device=device)

    # Safety epsilon and CDF for possible independent draws
    eps = 1e-300
    cdf = torch.cumsum(w_norm, dim=0).to(device)

    def direct_draw():
        # independent draw according to w_norm using the CDF
        r = torch.rand(1, device=device)
        # searchsorted returns a tensor; extract int
        return int(torch.searchsorted(cdf, r).item())

    # Start from an independent draw
    idx = direct_draw()

    # Warm-up: run MH transitions without recording
    for _ in range(warmup_steps):
        step = int(torch.randint(-proposal_width, proposal_width + 1, (1,), device=device).item())
        cand = (idx + step) % N
        cand = direct_draw()
        num = float(w_norm[cand].item())
        den = float(w_norm[idx].item()) + eps
        alpha = 1.0 if num >= den else (num / den)
        if torch.rand(1, device=device).item() < alpha:
            idx = cand

    # Sampling with thinning: record one index after `thinning` transitions
    indices = torch.empty(n_samples, dtype=torch.long, device=device)
    for i in range(n_samples):
        for _ in range(thinning):
            step = int(torch.randint(-proposal_width, proposal_width + 1, (1,), device=device).item())
            cand = (idx + step) % N
            cand = direct_draw()
            num = float(w_norm[cand].item())
            den = float(w_norm[idx].item()) + eps
            alpha = 1.0 if num >= den else (num / den)
            if torch.rand(1, device=device).item() < alpha:
                idx = cand
        # After `thinning` transitions, record current index (one index per thinning block)
        indices[i] = idx

    return indices

def systematic_resample(weights, N_resample, device=None):
    """
    Systematic resampling (vectorized).
    weights: 1D tensor summing to 1 on device
    returns: indices of selected particles (length N_resample)
    """
    if device is None:
        device = weights.device
    # cumulative sum
    cdf = torch.cumsum(weights, dim=0)
    # generate N_resample positions uniformly spaced with a random offset
    u0 = torch.rand(1, device=device) / N_resample
    positions = u0 + (torch.arange(N_resample, device=device, dtype=weights.dtype) / N_resample)
    # searchsorted (vectorized): returns indices i s.t. cdf[i-1] < positions <= cdf[i]
    idx = torch.searchsorted(cdf, positions)
    return idx

def residual_resample(weights, N_resample, device=None):
    """
    Residual resampling (vectorized).
    weights: normalized weights sum to 1.
    Mix deterministic and random part.
    """
    if device is None:
        device = weights.device
    N = weights.shape[0]
    Ns = torch.floor(weights * N_resample).to(torch.long)   # deterministic counts
    deterministic_count = Ns.sum().item()
    residual_N = N_resample - deterministic_count
    # deterministic indices
    det_idx = torch.repeat_interleave(torch.arange(N, device=device), Ns.to(device=device))
    if residual_N <= 0:
        return det_idx  # already filled
    # residual weights
    w_res = (weights * N_resample) - Ns.to(weights.dtype)
    w_res = torch.clamp(w_res, min=0.0)
    if w_res.sum() <= 0:
        # fallback: sample uniformly among indices
        rand_idx = torch.randint(0, N, (residual_N,), device=device)
    else:
        w_res = w_res / w_res.sum()
        rand_idx = systematic_resample(w_res, residual_N, device=device)
    # combine
    if det_idx.numel() == 0:
        return rand_idx
    return torch.cat([det_idx, rand_idx], dim=0)
# -------------------------
# GPU ais driver (single process)
# -------------------------
def SG_AIS_integrator(fun_ais, Nd, gdv, Nsample, Nstep, param,
                 device=None, batch_size=20000, seed=12345, stratified=False, alpha=1.6):
    """
    Runs ais on the given device (GPU if available). Single-process.
    batch_size: number of samples to draw per batch on GPU.
    """
    device = check_device_and_seed(device, seed)
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    gd_max = max(gdv)
    # initialize grid on device
    gd_ais = [torch.linspace(0.0, 1.0, gdv[k] + 1, dtype=dtype, device=device) for k in range(Nd)]

    n_batches = math.ceil(Nsample / batch_size)
    I_MC = torch.zeros(Nstep, dtype=dtype, device=device)
    Ierr_MC = torch.zeros(Nstep, dtype=dtype, device=device)

    for Loop in range(1, Nstep + 1):
        d = torch.zeros(Nd, gd_max, dtype=dtype, device=device)
        fw2_list = []
        I_list = []

        # iterate batches sequentially on GPU
        for bidx in range(n_batches):
            this = batch_size if (bidx < n_batches - 1) else (Nsample - batch_size * (n_batches - 1))
            if this <= 0:
                break
            s_batch, fw_batch, s1_batch, bin_idx_batch, J_batch = \
                ais_vectorized_batch(gdv, gd_ais, fun_ais, param, this, device,
                                                     stratified=stratified)
            # accumulate per-dim bin counts and f^2 using bincount (works on GPU)
            for k in range(Nd):
                idxs = bin_idx_batch[:, k]
                # counts per bin
                counts = torch.bincount(idxs, minlength=gdv[k]).to(dtype=dtype)
                weighted = torch.bincount(idxs, weights=(fw_batch ** 2).to(dtype=dtype), minlength=gdv[k])
                d[k, :gdv[k]] += weighted  # we'll divide by counts later; accumulate f^2 sum
                # store counts separately by adding to a count array? Simpler: accumulate counts to separate container
                # We'll track counts in a temporary array
                if bidx == 0:
                    # initialize a counts container on first batch
                    if 'counts_all' not in locals():
                        counts_all = torch.zeros((Nd, gd_max), dtype=dtype, device=device)
                    # (counts_all already exists after first)
                counts_all[k, :gdv[k]] += counts.to(dtype=dtype)

            fw2_list.append(torch.sum(fw_batch ** 2))
            I_list.append(torch.mean(fw_batch))

        # convert counts_all and d to usable forms
        # d currently holds accumulated f^2 sums; convert to average <f^2>_bin = d / counts_all (where counts>0)
        mask = counts_all > 0
        d_avg = torch.zeros_like(d)
        d_avg[mask] = d[mask] / counts_all[mask]
        # reset counts_all for next iteration
        counts_all = torch.zeros((Nd, gd_max), dtype=dtype, device=device)

        fw2_batch = torch.stack(fw2_list)
        I_batch = torch.stack(I_list)

        I_MC[Loop - 1] = I_batch.mean()
        Ierr_MC[Loop - 1] = torch.sqrt((fw2_batch.sum() / Nsample - I_MC[Loop - 1] ** 2) / max(1, Nsample - 1))

        Nc = 1 if Loop <= 10 else 8
        w = 1.0 / (Ierr_MC[Nc - 1:Loop] ** 2 + 1e-30)
        I = torch.sum(I_MC[Nc - 1:Loop] * w) / torch.sum(w)
        Ierr = 1.0 / torch.sqrt(torch.sum(w))
        chi2 = torch.sum(((I_MC[Nc - 1:Loop] - I) ** 2) * w) / max(1, Loop - (Nc - 1))
        print(f"{Loop:02d}  I = {I.item():.8e}  +/- {Ierr.item():.2e}   chi2/dof ~ {chi2.item():.3f}")

        # update grid: need to pass d_avg to grid_update (same device)
        gd_ais = grid_update(gdv, gd_ais, d_avg, alpha)

    return I, Ierr, I_MC.cpu(), [g.cpu() for g in gd_ais]

def SG_AIS_inverse_mapping(s,param, Nd,gdv,grid, device=None):
    """
    Inverse mapping of ais sampler output.
    
    Args:
        s (Tensor): ais-sampled points in unit cube, shape (N, d).
        grid (Tensor): ais grid edges, shape (d, nbins+1).
        device: torch.device (optional).

    Returns:
        u (Tensor): Uniform [0,1]^d samples that would generate s.
    """
    if device is None:
        device = s.device
    dtype = s.dtype

    a = _to_tensor(param['xmin'][:Nd], dtype=dtype, device=device)
    b = _to_tensor(param['xmax'][:Nd], dtype=dtype, device=device)

    #d, nbins_plus1 = grid.shape
    #nbins = nbins_plus1 - 1
    N = s.shape[0]

    u = torch.empty_like(s)

    for dim in range(Nd):
        edges = grid[dim]  # (nbins+1,)
        si = (s[:, dim]-a[dim])/(b[dim]-a[dim])     # (N,)
        #print(edges)
        #print(si[0])
        # Find bin index: k s.t. edges[k] <= si < edges[k+1]
        edges = edges.to(si.device)
        k = torch.searchsorted(edges, si, right=True)   - 1
        nbins = gdv[dim]
        #print(k)
        k = torch.clamp(k, 0, nbins - 1)
        #print(k)

        left = edges[k]
        right = edges[k + 1]
        frac = (si - left) / (right - left + 1e-15)

        u[:, dim] = (k + frac) / nbins

    return u
def SG_AIS_resampler(
    gd, grid_ais, fun_ais, param,
    device=None,
    total_proposals=200000,
    n_final=None,
    batch_proposal=20000,
    resample_method='systematic',
    jitter=0.0,
    ess_threshold=None
):
   # """
   # Importance-resampling VG_resampler (GPU-parallel, vectorized).

    #Args:
    #  gd, grid_ais, fun_ais, param: as before
    #  device: torch.device or None (auto)
    #  total_proposals: total number of ais proposals to draw (large number for stable resampling)
    #  n_final: number of final unweighted samples to return; default = total_proposals
    #  batch_proposal: number of proposals per call to ais_vectorized_batch
    #  resample_method: 'systematic' or 'residual' (both vectorized)
    #  jitter: float (0..1) amount of Gaussian jitter in unit-cube space after resampling (small, e.g. 1e-3)
    #  ess_threshold: if provided (0..1) as fraction of total_proposals, compute ESS and warn if low
    #Returns:
    #  x_resampled_phys: Tensor shape (n_final, Nd) in physical domain [a,b]
    #  stats: dict with 'ESS', 'total_weight', 'n_proposals', 'method'
    #"""
    if device is None:
        device = device_info()
    dtype = torch.get_default_dtype()

    Nd = int(param['nd'])
    a = _to_tensor(param['xmin'][:Nd], dtype=dtype, device=device)
    b = _to_tensor(param['xmax'][:Nd], dtype=dtype, device=device)

    if n_final is None:
        n_final = total_proposals

    # accumulate proposals
    proposals = []   # list of (s_batch, fw_batch) on device
    fw_list = []
    s_list = []

    drawn = 0
    while drawn < total_proposals:
        this = min(batch_proposal, total_proposals - drawn)
        s_batch, fw_batch, _, _, _ = ais_vectorized_batch(gd, grid_ais, fun_ais, param, this, device)
        # s_batch shape (this, Nd), fw_batch shape (this,)
        s_list.append(s_batch)          # unit-cube proposals
        fw_list.append(fw_batch)        # weighted f = f(x)*Jacobian
        drawn += this

    s_all = torch.cat(s_list, dim=0)   # (total_proposals, Nd)
    fw_all = torch.cat(fw_list, dim=0).to(dtype)  # (total_proposals,)

    # normalize weights safely
    w_norm, total_w = _safe_norm_weights(fw_all)
    # ESS = 1 / sum(w^2)
    ESS = 1.0 / torch.sum(w_norm ** 2)
    ESS_value = float(ESS.item())

    if ess_threshold is not None:
        if ESS_value < ess_threshold * total_proposals:
            print(f"WARNING: low ESS {ESS_value:.1f} < {ess_threshold*total_proposals:.1f}")

    # choose resampling indices
    if resample_method == 'systematic':
        idx = systematic_resample(w_norm, n_final, device=device)
    elif resample_method == 'residual':
        idx = residual_resample(w_norm, n_final, device=device)
    else:
        raise ValueError("resample_method must be 'systematic' or 'residual'")

    # idx length n_final (may be unordered)
    # gather unit-cube samples
    # torch.searchsorted returns Long but ensure idx in valid range
    idx = idx.to(torch.long)
    s_resampled = s_all[idx, :]  # (n_final, Nd)

    # Optional small jitter in unit space to add diversity:
    if jitter is not None and jitter > 0.0:
        # Gaussian jitter scaled by local bin width estimate
        # Compute per-dim local width by looking up bin index for each sample
        # We'll approximate by using neighbouring grid spacing: use grid_ais arrays
        # Simpler: jitter absolute in unit space
        noise = torch.randn_like(s_resampled) * float(jitter)
        s_resampled = s_resampled + noise
        s_resampled = torch.clamp(s_resampled, 0.0, 1.0)

    # Map to physical domain:
    x_resampled_phys = a + (b - a) * s_resampled

    stats = {
        'ESS': ESS_value,
        'total_weight': float(total_w.item()) if isinstance(total_w, torch.Tensor) else float(total_w),
        'n_proposals': int(total_proposals),
        'method': resample_method
    }

    return x_resampled_phys, stats
def SG_AIS_sampler(Ns, gdv, grid_ais, Nd, param, funa, device=None, batch_size=20000, stratified=False):
    device = check_device_and_seed(device)
    dtype = torch.get_default_dtype()
    a = _to_tensor(param['xmin'][:Nd], dtype=dtype, device=device)
    b = _to_tensor(param['xmax'][:Nd], dtype=dtype, device=device)

    x_MC = torch.empty((Ns, Nd), dtype=dtype, device=device)
    Jacobi = torch.empty((Ns,), dtype=dtype, device=device)
    fw = torch.empty((Ns,), dtype=dtype, device=device)

    i = 0
    while i < Ns:
        this = min(batch_size, Ns - i)
        s_batch, fw_batch, _, _, J_batch = ais_vectorized_batch(
            gdv, grid_ais, funa, param, this, device, stratified=stratified
        )
        x_MC[i:i + this] = (b - a) * s_batch + a
        Jacobi[i:i + this] = J_batch
        fw[i:i + this] = fw_batch
        i += this

    return x_MC.cpu(), Jacobi.cpu(), fw.cpu()

def SG_AIS_mapping(u, param, Nd, gdv, grid, device=None, return_jacobian=False):
    """
    Forward mapping of uniform [0,1]^Nd -> VEGAS sample in physical space.

    Args:
        u (Tensor): Uniform samples in [0,1]^Nd, shape (N, Nd).
        param (dict): parameter dict containing 'xmin' and 'xmax' arrays.
        Nd (int): number of active dimensions.
        gdv (iterable): number of bins per dimension (length Nd).
        grid (Tensor): VEGAS grid edges, shape (Nd, nbins+1), edges in [0,1].
        device (torch.device, optional)
        return_jacobian (bool): if True return (x_phys, J) where J is det(dx/du) shape (N,)

    Returns:
        x_phys (Tensor): points in physical domain, shape (N, Nd)
        (optional) J (Tensor): Jacobian determinant of mapping u->x, shape (N,)
    """
    if device is None:
        device = u.device
    dtype = u.dtype

    # ensure u on device/dtype
    u = u.to(device=device, dtype=dtype)
    a = _to_tensor(param['xmin'][:Nd], dtype=dtype, device=device)
    b = _to_tensor(param['xmax'][:Nd], dtype=dtype, device=device)

    N = u.shape[0]
    s_unit = torch.empty_like(u)   # si in [0,1] before scaling to physical domain
    # per-dim factor for Jacobian: dx/du for each sample and dimension
    if return_jacobian:
        per_dim_factor = torch.empty((N, Nd), dtype=dtype, device=device)

    eps = 1e-15
    # loop over dimensions (vectorized per-dim inside)
    for dim in range(Nd):
        nbins = int(gdv[dim])
        edges = grid[dim].to(device=device, dtype=dtype)   # (nbins+1,)

        # clamp u to [0,1)
        ui = torch.clamp(u[:, dim], min=0.0, max=1.0 - 1e-16)

        # compute bin index k = floor(ui * nbins)
        uf = ui * nbins
        k = torch.floor(uf).long()       # (N,)
        k = torch.clamp(k, 0, nbins - 1)

        frac = uf - k.float()            # fractional position inside bin [0,1)

        # gather left/right edges for each sample
        left = edges[k]                  # (N,)
        right = edges[k + 1]             # (N,)
        width = (right - left).clamp(min=eps)

        # unit-space si inside [0,1]
        si = left + frac * width         # (N,)
        s_unit[:, dim] = si

        if return_jacobian:
            # dsi/du = nbins * width
            dsi_du = (nbins * width)
            # dx/du = (b-a) * dsi_du
            per_dim_factor[:, dim] = (b[dim] - a[dim]) * dsi_du

    # Map unit-space to physical space
    x_phys = a + (b - a) * s_unit

    if return_jacobian:
        # Jacobian determinant is product over dims of dx/du
        J = torch.prod(per_dim_factor, dim=1)  # (N,)
        return x_phys, J

    return x_phys
# -------------------------
# Vectorized batch sampler on device
# -------------------------
def ais_vectorized_batch(gdv, gdais, funa, param, batch_N, device, stratified=False):
    """
    Draw `batch_N` samples on `device`.
    Returns s_batch (batch_N, Nd), fw_batch (batch_N,), s1 (batch_N,Nd), bin_idx (batch_N,Nd), J_batch (batch_N,)
    """
    dtype = torch.get_default_dtype()
    Nd = len(gdv)
    # per-sample bin indices
    if stratified:
        # simple per-dim stratification (shuffle of repeated indices)
        bin_idx = torch.empty((batch_N, Nd), dtype=torch.long, device=device)
        for k in range(Nd):
            bins = gdv[k]
            reps = (batch_N + bins - 1) // bins
            idxs = torch.arange(bins, device=device, dtype=torch.long).repeat_interleave(reps)[:batch_N]
            idxs = idxs[torch.randperm(batch_N, device=device)]
            bin_idx[:, k] = idxs
    else:
        bin_idx = torch.randint(0, gdv[0], (batch_N, Nd), dtype=torch.long, device=device)
        # Note: previous line uses gdv[0] by mistake if gdv vary; do per-dim
        # fix per-dim:
        for k in range(Nd):
            bin_idx[:, k] = torch.randint(0, gdv[k], (batch_N,), dtype=torch.long, device=device)

    u_inside = torch.rand((batch_N, Nd), dtype=dtype, device=device)
    s = torch.empty((batch_N, Nd), dtype=dtype, device=device)
    s1 = torch.empty((batch_N, Nd), dtype=dtype, device=device)

    for k in range(Nd):
        grd = gdais[k]  # tensor on device
        grd = grd.to(bin_idx.device)
        left = grd[bin_idx[:, k]]
        right = grd[bin_idx[:, k] + 1]
        widths = right - left
        s1[:, k] = widths
        s[:, k] = left + u_inside[:, k] * widths

    # Jacobian: product over dims (width * gdv[k])
    gdv_tensor = torch.tensor(gdv, dtype=dtype, device=device)
    J_batch = torch.prod(s1 * gdv_tensor.unsqueeze(0), dim=1)  # (batch_N,)
    fvals,_ = funa(s, param)  # vectorized evaluation on device 
    fw_batch = fvals * J_batch
    return s, fw_batch, s1, bin_idx, J_batch
