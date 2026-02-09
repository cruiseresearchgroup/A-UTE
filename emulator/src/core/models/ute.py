import numpy as np
import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
from omegaconf import DictConfig
from emulator.src.core.models.basemodel import BaseModel
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
import torchsde


class LatentSDEVector(torchsde.SDEIto):
    def __init__(self, latent_dim):
        super().__init__(noise_type='diagonal')
        self.dim = latent_dim*12*18
        self.hidden_channels = 32
        self.latent_dim = latent_dim
        
        # ========== ENCODER ==========
        self.enc1 = nn.Sequential(
            nn.Conv2d(latent_dim, 64, 3, padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)  # 96x144 → 48x72

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(32, 128), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(32, 128), 
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(1)  # 3x24x36 → 3x12x18

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(32, 256), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256), 
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=1, stride=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(32, 128), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(32, 128), 
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(32, 64), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(32, 64), 
            nn.ReLU(inplace=True)
        )

        # ========== FINAL CONV1x1 ==========
        self.out_conv = nn.Conv2d(64, latent_dim, kernel_size=1)
        
        
        self.g_net = nn.Sequential(
            nn.Conv2d(latent_dim, self.hidden_channels, kernel_size=3, stride=1, padding=1),  # Conv layer
            nn.GroupNorm(latent_dim,self.hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.gdecoder = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_channels, latent_dim, kernel_size=3, stride=1, padding=1)  # Upsample
        )

    def f(self, t, y):
        y = y.view(y.size(0),self.latent_dim,10,20)
        # Encoder
        x1 = self.enc1(y)
        p1 = self.pool1(x1)
        x2 = self.enc2(p1)
        p2 = self.pool2(x2)
        x3 = self.enc3(p2)

        u2 = self.up2(x3)
        d2 = self.dec2(torch.cat([u2, x2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, x1], dim=1))

        out = self.out_conv(d1)

        return out.view(out.size(0), -1)

    def g(self, t, y):
        y = y.view(y.size(0),self.latent_dim,10,20)
        y = self.g_net(y)
        y = self.gdecoder(y)
        y = y.view(y.size(0), -1)  
        return y

class LatentDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        self.up1 = nn.ConvTranspose3d(latent_dim, latent_dim, kernel_size=(1,1,1), stride=(1,1,1))
        self.decode1 = nn.Sequential(
            nn.Conv3d(latent_dim, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=(3,3,3), stride=(3,3,3))        # for roll-out 12
        # self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=(1,2,2), stride=(1,2,2))          # for roll-out 6,4

        self.decode2 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True)
        )
        
        # self.up3 = nn.ConvTranspose3d(32, 32, kernel_size=(3,2,2), stride=(3,2,2))        # for roll-out 6
        self.up3 = nn.ConvTranspose3d(32, 32, kernel_size=(2,3,3), stride=(2,3,3))          # for roll-out 4

        self.decode3 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, z, x2):
        # Upsample → Conv → Norm → ReLU
        z = self.up1(z)                                                                     # 1x10x20 → 1x10x20
        z = F.interpolate(z, scale_factor=2, mode='trilinear', align_corners=False)         # 1x10x20 → 2x20x40
        z = self.decode1(z)

        d2 = self.up2(z)                                                                     # 2x20x40 → 6x60x120
        z = self.decode2(torch.cat([d2, x2], dim=1))

        z = self.up3(z)                                                                      # 6x60x120 → 12x180x360     for roll-out 12, 6, 4
        z = self.decode3(z)

        out = self.out_conv(z)
        return out

class LatentEncoder(nn.Module):
    def __init__(self, output_channels, latent_dim):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv3d(output_channels, 16, 3, padding=1), 
            nn.ReLU()
            )
        self.pool1 = nn.MaxPool3d(kernel_size=(2,3,3), stride=(2,3,3))          # 12x180x360 → 6x60x120        for roll-out 12
        # self.pool1 = nn.MaxPool3d(kernel_size=(3,2,2), stride=(3,2,2))        # 6 → 2                      for roll-out 6
        # self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))        # 4 → 2                      for roll-out 4

        self.encoder2 = nn.Sequential(
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU()
            )
        self.pool2 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(3,3,3))      # 6x60x120 → 2x20x40        for roll-out 12
        # self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))    # 2 → 1                      for roll-out 6,4

        self.encoder3 =nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU()
            ) 
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))      # 2x20x40 → 1x10x20        for roll-out 12
        # self.pool3 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)) # 1 → 1                      for roll-out 6,4

        self.encoder4 = nn.Sequential(      
            nn.Conv3d(64, latent_dim, 3, padding=1),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool3d(1)
        
    def forward(self, z):
        z = z.permute(1,2,0,3,4) # Shape (B, V, T, Lat, Lon)

        x1 = self.encoder1(z)
        m1 = self.pool1(x1)
        x2 = self.encoder2(m1)
        m2 = self.pool2(x2)
        x3 = self.encoder3(m2)
        m3 = self.pool3(x3)
        x4 = self.encoder4(m3)
        m4 = self.pool4(x4)
        return x4, x2

def compute_initial_velocity(u_past, smoothing_alpha=1e-7):
    batch_size, T, channels, lat, lon = u_past.shape

    # Generate time indices
    time = torch.arange(T, dtype=torch.float32, device=u_past.device)

    # Ensure time dimension matches T
    if time.shape[0] != T:
        raise ValueError(f"Time shape {time.shape} must match u_past time dimension {T}.")

    # Initialize velocity tensor
    v_x = torch.zeros(batch_size, lat, lon, device=u_past.device)
    v_y = torch.zeros(batch_size, lat, lon, device=u_past.device)

    # Loop over each variable separately
    for i in range(channels):  
        # Flatten spatial dimensions (batch, T, lat*lon) for each variable
        u_past_flat = u_past[:, :, i, :, :].reshape(batch_size, T, lat * lon)  

        # Compute cubic spline coefficients
        coeffs = natural_cubic_spline_coeffs(time, u_past_flat)

        # Fit cubic spline model
        spline = NaturalCubicSpline(coeffs)

        # Compute numerical time derivative at t0
        u_dot_flat = spline.derivative(time[0])  # (batch, lat*lon)

        # Reshape back to (batch, lat, lon)
        u_dot = u_dot_flat.reshape(batch_size, lat, lon)

        # Compute spatial gradients ∇u using PyTorch's `torch.gradient`
        grad_x, grad_y = torch.gradient(u_past[:, 0, i, :, :], dim=(-1, -2))

        # Normalize gradients
        grad_x = grad_x.clamp(min=-1.0, max=1.0)
        grad_y = grad_y.clamp(min=-1.0, max=1.0)

        # Compute divergence term ∇ · v
        div_v = torch.gradient(grad_x, dim=-1)[0] + torch.gradient(grad_y, dim=-2)[0]

        # Solve for velocity using least squares with smoothing
        v_x += (-u_dot - u_past[:, 0, i, :, :] * div_v) / (grad_x + smoothing_alpha)
        v_y += (-u_dot - u_past[:, 0, i, :, :] * div_v) / (grad_y + smoothing_alpha)

    # Normalize velocities
    v_x = v_x.clamp(min=-1.0, max=1.0)
    v_y = v_y.clamp(min=-1.0, max=1.0)

    # Stack velocity components
    v_init = torch.stack([v_x, v_y], dim=1)  # (batch, 2, lat, lon)
    return v_init

class VelocityRefiner(nn.Module):
    """
    Small MLP to refine velocity estimates dynamically.
    """
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1)  # Output vx, vy
        )

    def forward(self, u, v_prev):
        return v_prev + self.mlp(u)

class AdvectionDiffusionODEfunc(nn.Module):
    """
    Defines advection-diffusion for tas & pr, using learned velocity fields.
    """
    def __init__(self, lat, lon, diffusion_tas=0.1, diffusion_pr=0.05):
        super().__init__()
        self.dx = torch.tensor(360 / lon, dtype=torch.float32)
        self.dy = torch.tensor(180 / lat, dtype=torch.float32)

        # Diffusion coefficients
        self.D_tas = nn.Parameter(torch.tensor(diffusion_tas), requires_grad=False)
        self.D_pr = nn.Parameter(torch.tensor(diffusion_pr), requires_grad=False)
        
        # Velocity refiner
        self.velocity_refiner = VelocityRefiner(input_dim=2, hidden_dim=32)
        
    def compute_forcing(self, ghgs):
        alpha_temp = 0.01  # Scaling factor for temperature increase
        alpha_pr = 0.005  # Scaling factor for precipitation impact

        ghg_effect = ghgs.sum(dim=1, keepdim=True)  # Sum across all GHG species
        temp_forcing = alpha_temp * ghg_effect  # Increase in temperature
        pr_forcing = alpha_pr * ghg_effect  # Change in precipitation

        forcing_terms = torch.cat([temp_forcing, pr_forcing], dim=1)  # Shape (batch, 2, lat, lon)
        return forcing_terms
    
    def set_ghgs(self, ghgs):
        self.ghgs_full = ghgs  # ghgs: (B, T, 4, H, W)
        

    def get_ghgs_at_t(self, t):
        T = self.ghgs_full.size(1)  # total time steps, say 120 months
        t_idx = t * T  # scale t in [0,1] to [0,T]
        t0 = torch.floor(t_idx).long().clamp(0, T - 2)
        t1 = t0 + 1
        alpha = (t_idx - t0.float()).view(-1, 1)  # Ensures it's always 2D

        ghgs_t0 = self.ghgs_full[:, t0, :, :, :]
        ghgs_t1 = self.ghgs_full[:, t1, :, :, :]
        return (1 - alpha) * ghgs_t0 + alpha * ghgs_t1


    def forward(self, t, u_v):
        u, v = u_v  # Unpack state and velocity
        batch_size, channels, lat, lon = u.shape
        gas_sources = self.get_ghgs_at_t(t)
        tas_pr = u
        vx, vy = v[:, 0:1, :, :], v[:, 1:2, :, :]
        v_refined = self.velocity_refiner(tas_pr, v)
        
        vx, vy = v_refined[:, 0:1, :, :], v_refined[:, 1:2, :, :]
        vx_sliced = vx[:, :, :, 1:-1]  
        vy_sliced = vy[:, :, 1:-1, :]  

        # Compute advection terms with corrected slicing
        adv_x = -vx_sliced * (tas_pr[:, :, :, 2:] - tas_pr[:, :, :, :-2]) / (2 * self.dx)
        adv_y = -vy_sliced * (tas_pr[:, :, 2:, :] - tas_pr[:, :, :-2, :]) / (2 * self.dy)

        diff_x = self.D_tas * (tas_pr[:, :, :, 2:] - 2 * tas_pr[:, :, :, 1:-1] + tas_pr[:, :, :, :-2]) / (self.dx ** 2)
        diff_y = self.D_pr * (tas_pr[:, :, 2:, :] - 2 * tas_pr[:, :, 1:-1, :] + tas_pr[:, :, :-2, :]) / (self.dy ** 2)
        
        # Align dimensions
        adv_x = adv_x[:, :, 1:-1, :]  # Align for latitude
        adv_y = adv_y[:, :, :, 1:-1]  # Align for longitude
        diff_x = diff_x[:, :, 1:-1, :]
        diff_y = diff_y[:, :, :, 1:-1]
        
        du_dt = torch.zeros_like(u)
        forcings = self.compute_forcing(gas_sources)
        du_dt = F.pad(adv_x + adv_y + diff_x + diff_y, (1, 1, 1, 1), mode='circular') + forcings

        du_dt_full = torch.cat([gas_sources, du_dt], dim=1)     
        dv_dt = v_refined - v  # Velocity change over time
        return du_dt, dv_dt

class ODEBlock(nn.Module):
    """
    Wraps the ODE function into an integration block.
    """
    def __init__(self, lat, lon, solver='dopri5'):
        super().__init__()
        self.odefunc = AdvectionDiffusionODEfunc(lat,lon)
        self.solver = solver

    def forward(self, x, v_init, ghgs, ts):
        device = x.device  # Ensure both are on the same device
        v_init = v_init.to(device)  # Move velocity field to the same device as x        
                
        if x.shape[0] != v_init.shape[0]:
            min_batch = min(x.shape[0], v_init.shape[0])
            x = x[:min_batch]  # Trim to match
            v_init = v_init[:min_batch]

        self.odefunc.set_ghgs(ghgs)  # Store GHGs inside the func

        x_out, v_out = odeint(self.odefunc, (x, v_init), ts, method=self.solver)
        return x_out,v_out
    
class NeuralODEModel(BaseModel):
    def __init__(
    self,
    in_var_ids: List[str],
    out_var_ids: List[str],
    longitude: int = 144,
    latitude: int = 96,
    datamodule_config: DictConfig = None,
    input_channels: int = 6,
    hidden_channels: int = 64,
    output_channels: int = 2,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    adjoint: bool = False,
    solver_method: str = 'dopri5',
    latent_dim=16,
    *args,
    **kwargs,
):
        super(NeuralODEModel, self).__init__(datamodule_config=datamodule_config, *args, **kwargs)
        self.save_hyperparameters()
        self.ode_block = ODEBlock(lat=latitude, lon=longitude, solver=solver_method)
        self.latent_dim = latent_dim
        self.decoder = LatentDecoder(latent_dim, output_channels)     
        self.encoder = LatentEncoder(output_channels, latent_dim)
        self.sde_vec = LatentSDEVector(latent_dim)
        
    def on_train_start(self):
        """
        Compute initial velocities before training begins.
        """
        self.log_text.info("Computing Initial Velocity Fields...")
        train_loader = self.trainer.datamodule.train_dataloader()
        for batch in train_loader:
            u_past, _ = batch  
            break 
        self.v_init = compute_initial_velocity(u_past)
        self.log_text.info("Initial Velocity Computed!")

    def forward(self, x):
        v_current = getattr(self, "v_init", None)
        if v_current is None:
            v_current = self.compute_initial_velocity(x)   # <- new
        v_current = v_current.to(x.device, dtype=x.dtype)
        
        batch_size, Horizon, _, lat, lon = x.shape
        ghgs = x[:, :, :4, :, :]  # Shape: (batch, horizon, 4, lat, lon)
        x_current = x[:, 0, -2:, :, :]  # Shape: (batch, vars, lat, lon)
        v_current = self.v_init  # Use initial velocity
        predictions, means, variances = [], [], []
        
        roll_out = 12
        # roll_out = 6
        # roll_out = 4
        
        # Auto-regressive prediction loop
        for t in range(1, Horizon, roll_out):  
            ghgs_t = ghgs[:, t:(t+roll_out), :, :, :] 
            x_step = x_current
            init_time = t
            final_time = (t+roll_out)-1
            steps_val = final_time - init_time
            new_time_steps = torch.linspace(init_time,final_time,steps=int(steps_val)+1).to(x.device)
            ts = 0.01*new_time_steps.float().to(x.device).flatten().float()
            x_next, v_next = self.ode_block(x_step, v_current, ghgs_t, ts) # Shape [12, 2, 6, 96, 144] (roll-out, batch, vars, lat, lon)
        
            time_steps = torch.linspace(init_time,final_time,steps=2).to(x.device)
            tss = 0.01*time_steps.float().to(x.device).flatten().float()
            z0, x2 = self.encoder(x_next)
            z0_flat = z0.view(z0.size(0), -1)

            preds_realizations = []

            for _ in range(5):
                bm = torchsde.BrownianInterval(t0=ts[0], t1=ts[-1], size=z0_flat.shape, device=z0_flat.device, levy_area_approximation='space-time')
                zt_flat = torchsde.sdeint(self.sde_vec, z0_flat, tss, bm=bm, method='srk')[-1]
                zt = zt_flat.view_as(z0)
                decoded = self.decoder(zt, x2)
                
                decoded = decoded.permute(0,2,1,3,4) 
                preds_realizations.append(decoded) 

            preds_stack = torch.stack(preds_realizations, dim=0)
            mean_t = preds_stack.mean(dim=0)
            var_t = preds_stack.var(dim=0)

            predictions.append(preds_stack)
            means.append(mean_t)
            variances.append(var_t)

            x_current = mean_t[:,-1,:,:,:]
            v_current = v_next[-1]
            
        means = torch.stack(means, dim=1) 
        variances = torch.stack(variances, dim=1) 
        B, S, T, C, H, W = means.shape
        means = means.view(B, S*T, C, H, W)
        variances  = variances.view(B, S*T, C, H, W)
        print(means.shape, variances.shape)

        return means, means, variances




