import torch
import numpy as np
import matplotlib.pyplot as plt



def plot_ais(x0,x_MC, x1,
              x_ais_MC):
    x0 = x0.detach().cpu()
    x1 = x1.detach().cpu()
    x_MC = x_MC.detach().cpu()
    x_ais_MC = x_ais_MC.detach().cpu()

    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1) 
    plt.hist2d(x0[:, 0].numpy(), x0[:, 1].numpy(), bins=50, range=[[-0.0, 1.0], [-0.0, 1.0]], cmap='viridis')
    plt.colorbar()
    plt.title('Uniform Sampling (base grid)')
    plt.subplot(2, 2, 2) 
    plt.hist2d(x_MC[:, 0].numpy(), x_MC[:, 1].numpy(), bins=50, range=[[-5, 5], [-5, 5]], cmap='viridis')
    plt.colorbar()
    plt.title('Direct Sampling (ais grid)')
    plt.subplot(2, 2, 3) 
    plt.hist2d(x1[:, 0].numpy(), x1[:, 1].numpy(), bins=50, range=[[-0.0, 1.0], [-0.0, 1.0]], cmap='viridis')
    plt.colorbar()
    plt.title('ReSampling (base grid)')
    plt.subplot(2, 2, 4)
    plt.hist2d(x_ais_MC[:, 0].numpy(), x_ais_MC[:, 1].numpy(), bins=50, range=[[-5, 5], [-5, 5]], cmap='viridis')
    plt.colorbar()
    plt.title('ReSampling (ais grid)')
    #plt.subplot(1, 3, 3)
    #plt.plot(I_MC.numpy(), marker='o', lw=1)
    #plt.title('Iteration estimates $I_k$')
    plt.tight_layout()
    plt.show()  
    
def plot_training_data_2d(train_dataset, 
                          dim1: int = 0, dim2: int = 1, 
                          bins: int = 200, 
                          scatter: bool = True, 
                          hist2d: bool = True,
                          alpha: float = 0.5,
                          figsize=(6,6),
                          save_path: str = None):
    """
    Make a 2D plot of training dataset (scatter + histogram).

    Args:
        train_dataset: torch.Tensor, TensorDataset, or custom dataset
        dim1: index of x-axis dimension
        dim2: index of y-axis dimension
        bins: number of bins for 2D histogram
        scatter: if True, show scatter plot
        hist2d: if True, show 2D histogram / density
        alpha: transparency for scatter points
        figsize: figure size
        save_path: optional file path to save the figure
    """
    # Handle different dataset types
    if isinstance(train_dataset, torch.utils.data.TensorDataset):
        data = train_dataset.tensors[0]
    elif isinstance(train_dataset, torch.Tensor):
        data = train_dataset
    else:
        # Assume custom dataset → stack samples
        data = torch.stack([train_dataset[i] for i in range(len(train_dataset))])

    # Move to CPU numpy
    data = data.detach().cpu().numpy()
    #print(data[0,:])
    x = data[:, dim1]
    y = data[:, dim2]

    plt.figure(figsize=figsize)

    if hist2d:
        plt.hist2d(x, y, bins=bins, cmap="inferno", density=True)
        plt.colorbar(label="Density")

    if scatter:
        plt.scatter(x, y, s=2, alpha=alpha, c="cyan", edgecolors="none")

    plt.xlabel(f"Dimension {dim1}")
    plt.ylabel(f"Dimension {dim2}")
    plt.title("Training Data 2D Distribution")

    #if save_path:
     #   plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    
def two2threeD(x,y,xmin,xmax):
    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=50,
        range=[[xmin, xmax], [xmin, xmax]]
    )
    
    # Bin centers
    X = 0.5 * (xedges[:-1] + xedges[1:])
    Y = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(X, Y)
    
    Z = H.T   # transpose for correct orientation
    return X, Y, Z
def plot_SG_full(x0,x_vg,x_nf,x_Vegas_MC,x_resample_ini,x_resample_vg,x_FuHsi):
    device = x0.device
    x0=x0.detach().cpu()
    x_vg=x_vg.detach().cpu()
    x_nf=x_nf.detach().cpu()
    x_Vegas_MC=x_Vegas_MC.detach().cpu()
    x_resample_ini = x_resample_ini.detach().cpu()
    x_resample_vg = x_resample_vg.detach().cpu()
    x_FuHsi = x_FuHsi.detach().cpu()
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 3, 1)
    plt.hist2d(x0[:, 0].numpy(), x0[:, 1].numpy(), bins=50, range=[[0, 1], [0, 1]], cmap='viridis')
    plt.colorbar()
    plt.title('Uniform Sampling')
    
    plt.subplot(3, 3, 2)
    plt.hist2d(x_vg[:, 0].numpy(), x_vg[:, 1].numpy(), bins=50, range=[[-5, 5], [-5, 5]], cmap='viridis')
    plt.colorbar()
    plt.title('Direct Sampling (VEGAS grid)')
    
    plt.subplot(3, 3, 3)
    plt.hist2d(x_nf[:, 0].numpy(), x_nf[:, 1].numpy(), bins=50, range=[[-5, 5], [-5, 5]], cmap='viridis')
    plt.colorbar()
    plt.title('Sampling (SG grid)')   
    
    
    plt.subplot(3, 3, 4)
    plt.hist2d(x0[:, 0].numpy(), x0[:, 1].numpy(), bins=50, range=[[0, 1], [0, 1]], cmap='viridis')
    plt.colorbar()
    plt.title('Uniform Sampling')
    
    plt.subplot(3, 3, 5)
    plt.hist2d(x_vg[:, 0].numpy(), x_vg[:, 1].numpy(), bins=50, range=[[-5, 5], [-5, 5]], cmap='viridis')
    plt.colorbar()
    plt.title('Direct Sampling (VEGAS grid)')
    
    plt.subplot(3, 3, 6)
    plt.hist2d(x_Vegas_MC[:, 0].numpy(), x_Vegas_MC[:, 1].numpy(), bins=50, range=[[-5, 5], [-5, 5]], cmap='viridis')
    plt.colorbar()
    plt.title('ReSampling (VEGAS grid)')
    
    plt.subplot(3, 3, 4+3)
    plt.hist2d(x_resample_ini[:, 0].numpy(), x_resample_ini[:, 1].numpy(), bins=50, range=[[-0, 1.], [-0., 1.]], cmap='viridis')
    plt.colorbar()
    plt.title('Sampling (initial grid)')
    
    plt.subplot(3, 3, 5+3)
    plt.hist2d(x_resample_vg[:, 0].numpy(), x_resample_vg[:, 1].numpy(), bins=50, range=[[-5, 5], [-5, 5]], cmap='viridis')
    plt.colorbar()
    plt.title('Sampling (Vegas grid)')   
    
    
    plt.subplot(3, 3, 6+3)
    plt.hist2d(x_FuHsi[:, 0].numpy(), x_FuHsi[:, 1].numpy(), bins=50, range=[[-5, 5], [-5, 5]], cmap='viridis')
    plt.colorbar()
    plt.title('ReSampling (SG grid)')
    
    #plt.subplot(1, 3, 3)
    #plt.plot(I_MC.numpy(), marker='o', lw=1)
    #plt.title('Iteration estimates $I_k$')
    plt.tight_layout()
    plt.show()  
    
    plt.figure(figsize=(10, 2.2))
    plt.subplot(1, 4, 1)
    plt.hist2d(x0[:, 0].numpy(), x0[:, 1].numpy(), bins=50, range=[[0, 1], [0, 1]], cmap='viridis')
    plt.colorbar()
    plt.title('Uniform Sampling')
    
    plt.subplot(1, 4, 2)
    plt.hist2d(x_vg[:, 0].numpy(), x_vg[:, 1].numpy(), bins=50, range=[[-5, 5], [-5, 5]], cmap='viridis')
    plt.colorbar()
    plt.title('Direct Sampling (VEGAS grid)')
    
    plt.subplot(1, 4, 3)
    plt.hist2d(x_nf[:, 0].numpy(), x_nf[:, 1].numpy(), bins=50, range=[[-5, 5], [-5, 5]], cmap='viridis')
    plt.colorbar()
    plt.title('Sampling (SG grid)')   
    
    
    mask_a = (x_vg[:, 0] > 1.5) & (x_vg[:, 1] > 1.5) & (x_vg[:, 1] < 2.2) &  (x_vg[:, 0] < 2.2)

    mask_b = (x_vg[:, 0] > 1.5) & (x_vg[:, 1] > -2.2) & (x_vg[:, 0] < 2.2) &  (x_vg[:, 1] <-1.5)
    
    indices_a = torch.nonzero(mask_a, as_tuple=True)[0]
    indices_b = torch.nonzero(mask_b, as_tuple=True)[0]
    
    # or equivalently:
    # indices = mask.nonzero(as_tuple=True)[0]
    
    #print(indices)
    x_vg_cuta=x_vg[indices_a,:]
    x_nf_cuta = x_nf[indices_a,:]
    
    x_vg_cutb=x_vg[indices_b,:]
    x_nf_cutb = x_nf[indices_b,:]
    
    x0_cuta=x0[indices_a,:]
    x0_cutb=x0[indices_b,:]
    
    plt.figure(figsize=(10, 2.2))
    plt.subplot(1, 4, 1)
    plt.hist2d(x0[:, 0].numpy(), x0[:, 1].numpy(), bins=50, range=[[0, 1], [0, 1]], cmap='viridis')
    plt.scatter(x0_cuta[1:10,0].numpy(),x0_cuta[1:10,1].numpy(), c='red',marker='*',edgecolors='red',alpha=0.8)
    plt.scatter(x0_cutb[1:10,0].numpy(),x0_cutb[1:10,1].numpy(), c='blue',marker='*',edgecolors='blue',alpha=0.8)
    plt.colorbar()
    plt.title('Uniform Sampling')
    
    plt.subplot(1, 4, 2)
    plt.hist2d(x_vg[:, 0].numpy(), x_vg[:, 1].numpy(), bins=50, range=[[-5, 5], [-5, 5]], cmap='viridis')
    plt.colorbar()
    
    plt.scatter(x_vg_cuta[1:10,0].numpy(),x_vg_cuta[1:10,1].numpy(), c='red',marker='*',edgecolors='red',alpha=0.8)
    plt.scatter(x_vg_cutb[1:10,0].numpy(),x_vg_cutb[1:10,1].numpy(), c='blue',marker='*',edgecolors='blue',alpha=0.8)
    
    plt.title('Direct Sampling (VEGAS grid)')
    
    plt.subplot(1, 4, 3)
    plt.hist2d(x_nf[:, 0].numpy(), x_nf[:, 1].numpy(), bins=50, range=[[-5, 5], [-5, 5]], cmap='viridis')
    plt.colorbar()
    plt.scatter(x_nf_cuta[1:10,0].numpy(),x_nf_cuta[1:10,1].numpy(), c='red',marker='.',edgecolors='red',alpha=0.8)
    plt.scatter(x_nf_cutb[1:10,0].numpy(),x_nf_cutb[1:10,1].numpy(), c='blue',marker='.',edgecolors='blue',alpha=0.8)
    plt.title('Sampling (SG grid)')
    
    plt.subplot(1, 4, 4)
    plt.hist2d(x_FuHsi[:, 0].numpy(), x_FuHsi[:, 1].numpy(), bins=50, range=[[-5, 5], [-5, 5]], cmap='viridis')
    plt.colorbar()
    plt.title('ReSampling (SG grid)')
    
    #plt.subplot(1, 3, 3)
    #plt.plot(I_MC.numpy(), marker='o', lw=1)
    #plt.title('Iteration estimates $I_k$')
    plt.tight_layout()
    plt.show() 
    
    xlim = (-5, 5)
    ylim = (-5, 5)
    xlim1 = (-0.5, 1.5)
    ylim1 = (-0.5, 1.5)
    fig=plt.figure(figsize=(10, 2.2)) 
    ax = fig.add_subplot(1,4,1, projection='3d')
    x = x0[:, 0].numpy()
    y = x0[:, 1].numpy()
    mask = (x >= 0.0) & (x <= 1.0)
    mask1 = (y >= 0.0) & (y <= 1.0)
    x = x[mask]
    y = y[mask1]

    X,Y,Z=two2threeD(x,y,-0.2,1.2) 
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True)
    #ax.set_xlim(*xlim1)
    #ax.set_zlim(*(0,800))
    ax.set_zlim(*(0,1600))
    ax.view_init(elev=45, azim=135)
    #ax.set_xlabel(r'$x_1$')
    #ax.set_ylabel(r'$x_2$')
    #ax.set_zlabel('Counts')
    
    ax = fig.add_subplot(1,4,2, projection='3d')
    X,Y,Z=two2threeD(x_vg[:, 0].numpy(),x_vg[:, 1].numpy(),-5,5) 
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.view_init(elev=45, azim=135)
    
    ax = fig.add_subplot(1,4,3, projection='3d')
    X,Y,Z=two2threeD(x_nf[:, 0].numpy(),x_nf[:, 1].numpy(),-5,5) 
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.view_init(elev=45, azim=135)
    
    ax = fig.add_subplot(1,4,4, projection='3d')
    X,Y,Z=two2threeD(x_FuHsi[:, 0].numpy(),x_FuHsi[:, 1].numpy(),-5,5) 
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.view_init(elev=45, azim=135)
    
    
    #ax.contour3D(X, Y, Z, 50,  cmap='viridis')
    #ax.contourf(X, Y, Z, 50, cmap='viridis')
    plt.tight_layout()
    plt.show() 

def plot_SG(x0,x_MC, x1,
              x_Vegas_MC,x_sg):
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1) 
    plt.hist2d(x0[:, 0].numpy(), x0[:, 1].numpy(), bins=50, range=[[-0.0, 1.0], [-0.0, 1.0]], cmap='viridis')
    plt.colorbar()
    plt.title('Uniform Sampling (base grid)')
    plt.subplot(2, 2, 2) 
    plt.hist2d(x_MC[:, 0].numpy(), x_MC[:, 1].numpy(), bins=50, range=[[-5, 5], [-5, 5]], cmap='viridis')
    plt.colorbar()
    plt.title('Direct Sampling (VEGAS grid)')
    #plt.subplot(1, 4, 3) 
    #plt.hist2d(x1[:, 0].numpy(), x1[:, 1].numpy(), bins=50, range=[[-0.0, 1.0], [-0.0, 1.0]], cmap='viridis')
    #plt.colorbar()
    #plt.title('ReSampling (base grid)')
    plt.subplot(2, 2, 3)
    plt.hist2d(x_Vegas_MC[:, 0].numpy(), x_Vegas_MC[:, 1].numpy(), bins=50, range=[[-5, 5], [-5, 5]], cmap='viridis')
    plt.colorbar()
    plt.title('ReSampling (VEGAS grid)')
    
    plt.subplot(2, 2, 4)
    plt.hist2d(x_sg[:, 0].numpy(), x_sg[:, 1].numpy(), bins=50, range=[[-5, 5], [-5, 5]], cmap='viridis')
    plt.colorbar()
    plt.title('SG mapping')
    #plt.subplot(1, 3, 3)
    #plt.plot(I_MC.numpy(), marker='o', lw=1)
    #plt.title('Iteration estimates $I_k$')
    plt.tight_layout()
    plt.show()    
def plot_SG2(x0,x_MC, x1,
              x_Vegas_MC,x_sg):
    
    x0=x0.detach().cpu()
    x1=x1.detach().cpu()
    x_MC=x_MC.detach().cpu()
    x_sg=x_sg.detach().cpu()
    x_Vegas_MC=x_Vegas_MC.detach().cpu()
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1) 
    plt.hist2d(x0[:, 0].numpy(), x0[:, 1].numpy(), bins=50, range=[[-0.0, 1.0], [-0.0, 1.0]], cmap='viridis')
    plt.colorbar()
    plt.title('Uniform Sampling (base grid)')
    plt.subplot(2, 2, 2) 
    plt.hist2d(x_MC[:, 0].numpy(), x_MC[:, 1].numpy(), bins=50, range=[[-8, 8], [-8, 8]], cmap='viridis')
    plt.colorbar()
    plt.title('Direct Sampling (VEGAS grid)')
    plt.subplot(2, 2, 3)
    plt.hist2d(x_Vegas_MC[:, 0].numpy(), x_Vegas_MC[:, 1].numpy(), bins=50, range=[[-8, 8], [-8, 8]], cmap='viridis')
    plt.colorbar()
    plt.title('ReSampling (VEGAS grid)')
    
    plt.subplot(2, 2, 4)
    plt.hist2d(x_sg[:, 0].numpy(), x_sg[:, 1].numpy(), bins=50, range=[[-8, 8], [-8, 8]], cmap='viridis')
    plt.colorbar()
    plt.title('SG mapping') 
    plt.tight_layout()
    plt.show()
    
    
def plot_correlation_function_mixed_event_fast(
    samples_data: torch.Tensor,
    num_mixed_event_pairs: int,
    ipic=1,
    device="cpu"
):
    samples_data = samples_data.to(device)
    Nsample = samples_data.shape[0]
    num_particles = samples_data.shape[1] // 3

    X = samples_data.view(Nsample, num_particles, 3)

    #Y = samples_data.reshape(Nsample, 3, num_particles)
    #X = Y.permute(0, 2, 1)
    # === 1. Real pairs (vectorized) ===
    # pairwise distances within each event
    diff = X[:, :, None, :] - X[:, None, :, :]   # (N, P, P, 3)
    dist = torch.norm(diff, dim=-1)              # (N, P, P)

    # take upper triangle only
    iu = torch.triu_indices(num_particles, num_particles, offset=1)
    all_pairwise_distances_real = dist[:, iu[0], iu[1]].reshape(-1)

    bin_edges = torch.linspace(0, 4, 21, device=device)
    bin_width = bin_edges[1] - bin_edges[0]

    counts_real = torch.histc(
        all_pairwise_distances_real,
        bins=20, min=0.0, max=4.0
    )
    #counts_real, _ = torch.histogram(
    #    all_pairwise_distances_real,
    #    bins=bin_edges
    #)    
    rho_real = counts_real / (counts_real.sum() * bin_width)

    # === 2. Mixed events (vectorized) ===
    idx1 = torch.randint(0, Nsample, (num_mixed_event_pairs,), device=device)
    idx2 = (idx1 + torch.randint(1, Nsample, (num_mixed_event_pairs,), device=device)) % Nsample
    #idx2 = torch.randint(0, Nsample, (num_mixed_event_pairs,), device=device)

    p1 = torch.randint(0, num_particles, (num_mixed_event_pairs,), device=device)
    p2 = torch.randint(0, num_particles, (num_mixed_event_pairs,), device=device)

    r1 = X[idx1, p1]
    r2 = X[idx2, p2]

    all_pairwise_distances_mixed = torch.norm(r1 - r2, dim=1)

     
    #counts_mixed, _ = torch.histogram(
    #    all_pairwise_distances_mixed,
    #    bins=bin_edges
    #)

    #counts_mixed = torch.histogram(
    #    all_pairwise_distances_mixed,
    #    bins=20, min=0.0, max=4.0
    #)
    
    counts_mixed = torch.histc(
        all_pairwise_distances_mixed,
        bins=20, min=0.0, max=4.0
    )
    rho_mixed = counts_mixed / (counts_mixed.sum() * bin_width)

    # === 3. Correlation ===
    eps = 1e-12
    C_dr = torch.where(
        rho_mixed > eps,
        rho_real / rho_mixed,
        torch.tensor(float("nan"), device=device)
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # === 4. Plot ===
    if ipic==1:
        plt.figure()
        plt.bar(bin_centers.cpu(), C_dr.cpu(), width=bin_width.cpu())
        plt.axhline(1.0, color='r', linestyle='--')
        plt.xlim(0, 4)
        plt.ylim(0, 1.6)
        plt.xlabel(r'$\Delta r$')
        plt.ylabel(r'$C(\Delta r)$')
        plt.title(r'Two-particle correlation function $C(\Delta r)$')
        plt.grid(True)
        plt.show()

    return bin_centers, C_dr,counts_real,counts_mixed, bin_width   

def plot_correlation_function_mixed_event_fast_in_chunks(
    samples_data: torch.Tensor,
    chunk_size=100,
    num_mixed_event_pairs=100, 
    device="cpu"
):
    outs_rho_real = []
    outs_rho_mixed=[] 
    N = samples_data.shape[0]
    for i in range(0, N, chunk_size):
        #print(i,N)
        x_chunk = samples_data[i:i+chunk_size,:]
        bin_centers, C_dr,counts_real,counts_mixed,bin_width = plot_correlation_function_mixed_event_fast(
            x_chunk,num_mixed_event_pairs,0,device=device)
        
        outs_rho_real.append(counts_real)
        outs_rho_mixed.append(counts_mixed) 
      
    outs_rho_real=torch.stack(outs_rho_real) 
    outs_rho_mixed=torch.stack(outs_rho_mixed)
    
    rho_real = torch.mean(outs_rho_real,dim=0)
    rho_mixed = torch.mean(outs_rho_mixed,dim=0) 
    
    rho_mixed = rho_mixed / (rho_mixed.sum() * bin_width)
    
    rho_real = rho_real / (rho_real.sum() * bin_width)
    
    eps = 1e-12
    C_dr_new = torch.where(
        rho_mixed > eps,
        rho_real / rho_mixed,
        torch.tensor(float("nan"), device=device)
    )
    plt.figure()
    plt.plot(bin_centers.cpu(), C_dr_new.cpu(),'-b.') 
    plt.axhline(1.0, color='r', linestyle='--')
    plt.xlim(0, 4)
    plt.ylim(0, 1.6)
    plt.show()
    print('counts real',torch.sum(outs_rho_real,dim=0))
    print('counts mix',torch.sum(outs_rho_mixed,dim=0))
    
    return bin_centers,C_dr_new

 