import torch
from functools import partial
from torchdiffeq import odeint
import numpy as np

class OTFlow:
    def __init__(self,
                 model,
                 ema,
                 # data parameters
                 image_size = 32,
                 in_channel = 3,
                 # time parameters
                 p_std = 1.0,
                 p_mean = -0.4,
                 # loss parameters
                 loss_type = "l2",
                 class_dropout = 0.1,
                 # cfg parameters
                 num_classes = 1000,
                 time_sampler = "lognormal",
                 reweight = "uniform"):
        self.p_std = p_std
        self.p_mean = p_mean
        self.loss_type = loss_type
        self.model = model
        self.ema = ema
        self.class_dropout = class_dropout
        self.image_size = image_size
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.time_sampler = time_sampler
        self.reweight = reweight

        
    def get_xt(self, x, n, t):
        t = t.view(-1, 1, 1, 1)
        return (1 - t) * x + t * n

    def velocity(self, x, n, t):
        t = t.view(-1, 1, 1, 1)
        return (n - x)

    def mean_flatten(self, x):
        return x.mean(dim=(1, 2, 3))

    
    def sample_t_uniform(self, bs, device):
        t = torch.rand(bs, device=device)# * (1 - 1e-6) + 1e-6  # avoid t=0
        return t
    
    def sample_t_lognormal(self, bs, device):
        normal_samples = torch.randn(bs, device=device) * self.p_std + self.p_mean
        t = 1 / (1 + torch.exp(-normal_samples))
        return t
    
    def get_loss(self, error, v, reweight):
        if self.loss_type == "adaptive_loss_weight":
            sq_norm_error = torch.mean(error**2, dim=(1, 2, 3), keepdim=False)
            weight = 1.0/(sq_norm_error + 1e-7).pow(self.p)
            loss = weight.detach() * sq_norm_error
            loss = loss * reweight
            return loss
        if self.loss_type == "velocity_loss_weight":
            sq_norm_error = torch.mean(error**2, dim=(1, 2, 3), keepdim=False)
            velocity_weight = 1.0/torch.mean(v**2, dim=(1, 2, 3), keepdim=False)
            weight = 1.0/(sq_norm_error + 1e-7).pow(self.p)
            loss = sq_norm_error * weight.detach() * velocity_weight
            loss = loss * reweight
            return loss
        elif self.loss_type == "l2":
            loss = torch.mean(error**2, dim=(1,2,3)) * reweight
            return loss
        else:
            raise ValueError(f"Loss type {self.loss_type} not supported")

    def get_reweight(self, t, r):
        if self.reweight == "uniform":
            return torch.ones_like(t)
        elif self.reweight == "snr":
            return t/((1-t)+1e-8)
        elif self.reweight == "min_snr":
            return torch.minimum(t/((1-t)+1e-6), 3.0*torch.ones_like(t))
        elif self.reweight == "ict":
            return 1/(t-r+1e-4)

    def loss(self, x, c):
        self.model.train()
        bs = x.shape[0]
        # sample t and r from lognormal distribution (r < t)
        if self.time_sampler == "uniform":
            t = self.sample_t_uniform(bs, x.device)
        elif self.time_sampler == "lognormal":
            t = self.sample_t_lognormal(bs, x.device)
        else:
            raise ValueError(f"Time sampler {self.time_sampler} not supported")
        # sample n from normal distribution
        n = torch.randn_like(x, device=x.device)
        xt = self.get_xt(x, n, t)
        uncond = torch.ones_like(c, device=x.device)*self.num_classes
        # loss computation
        v = self.velocity(x, n, t)
        # auto mask 10% condition class
        class_dropout_mask = torch.rand(c.shape[0], device=x.device) < self.class_dropout
        c = torch.where(class_dropout_mask, uncond, c)

        # wrap model prediction
        model_wrapper = partial(self.model, y=c)

        # compute loss - handle tuple return if classifier head exists
        model_output = model_wrapper(xt, t)
        if isinstance(model_output, tuple):
            vt = model_output[0]  # Only use velocity for standard loss
        else:
            vt = model_output

        # flow matching loss with correlation weighting
        error = vt - v
        reweight = self.get_reweight(t, r=None)
        loss = self.get_loss(error, v, reweight)
        # reconstruction loss
        return loss.mean()

    def sample_euler(self, model, x=None, class_idx=[207, 360, 387, 974, 88, 979, 417, 279], device="cuda", cfg=1.0, num_steps=250):
        model.eval()
        n_samples = len(class_idx)
        y = torch.tensor(class_idx, device=device)

        # Create initial noise
        gen = torch.Generator(device=device).manual_seed(0)
        x = torch.randn(n_samples, self.in_channel, self.image_size, self.image_size, device=device, generator=gen) if x is None else x

        # Time schedule: t from 1.0 -> 0.0 with `num_steps` linearly spaced steps
        timesteps = torch.linspace(1.0, 0.0, num_steps, device=device)

        for i in range(num_steps - 1):
            t_cur = timesteps[i].expand(n_samples)
            t_next = timesteps[i + 1].expand(n_samples)

            # Predict noise with classifier-free guidance
            if cfg > 1.0:
                y_full = torch.cat([y, torch.tensor([1000] * n_samples, device=device)])
                x_in = torch.cat([x, x], 0)
                t_in = torch.cat([t_cur, t_cur], 0)
                vec = model.forward_with_cfg(x_in, t_in, y=y_full, cfg_scale=cfg)
                vec, _ = vec.chunk(2, dim=0)
            else:
                vec = model(x, t_cur, y=y)
                if isinstance(vec, tuple):
                    vec = vec[0]

            # Euler update step
            dt = (t_next - t_cur).view(-1, 1, 1, 1)
            x = x + dt * vec
        return x

    def sample_ode(self, model, x=None, class_idx=[207, 360, 387, 974, 88, 979, 417, 279], device="cuda", cfg=1.0, num_steps=250, sampler_type="dopri5", atol=1e-5, rtol=1e-5):
        model.eval()
        def _fn(t, x):
            t = torch.ones(class_idx.shape[0], device=device) * t
            if cfg > 1.0:
                y_full = torch.cat([class_idx, torch.tensor([1000] * class_idx.shape[0], device=device)])
                x_in = torch.cat([x, x], 0)
                t_in = torch.cat([t, t], 0)
                vec = model.forward_with_cfg(x_in, t_in, y=y_full, cfg_scale=cfg)
                vec, _ = vec.chunk(2, dim=0)
            else:
                vec = model(x, t, y=class_idx)
            return vec
        t = torch.linspace(1, 0, num_steps)
        atol = [atol] * x.shape[1]
        rtol = [rtol] * x.shape[1]
        x = torch.randn(len(class_idx), self.in_channel, self.image_size, self.image_size, device=device) if x is None else x
        samples = odeint(
            _fn,
            x,
            t,
            method=sampler_type,
            atol=atol,
            rtol=rtol,
        )
        return samples

    def sample_euler_maruyama(self, 
                              model, 
                              x=None, 
                              class_idx=[207, 360, 387, 974, 88, 979, 417, 279], 
                              device="cuda", 
                              cfg=1.0, 
                              num_steps=250, 
                              last_h = 0.04, 
                              diffusion_form="SBDM", 
                              diffusion_norm=1.0):
        model.eval()

        n_samples = len(class_idx)
        y = torch.tensor(class_idx, device=device)

        # Create initial noise
        gen = torch.Generator(device=device).manual_seed(0)
        x = torch.randn(n_samples, self.in_channel, self.image_size, self.image_size, device=device, generator=gen) if x is None else x

        # velocity function
        def v_func(x, t, y):
            if cfg > 1.0:
                y_full = torch.cat([y, torch.tensor([1000] * x.shape[0], device=device)])
                x_in = torch.cat([x, x], 0)
                t_in = torch.cat([t, t], 0)
                vec = model.forward_with_cfg(x_in, t_in, y=y_full, cfg_scale=cfg)
                vec, _ = vec.chunk(2, dim=0)
            else:
                vec = model(x, t, y=y)
                if isinstance(vec, tuple):
                    vec = vec[0]
            return vec
        
        def score_from_v_func(x, v, t):
            return -((1-t)*v + x)/t

        def drift_fn(x, t, y):
            v = v_func(x, t, y)
            t = t.view(-1, 1, 1, 1)
            s = score_from_v_func(x, v, t)
            w = compute_diffusion(t, form=diffusion_form, norm=diffusion_norm)
            return v - w * s, w
        
        def compute_diffusion(t, form="constant", norm=1.0):
            diffusion = t/(1-t)  # avoid div zero
            choices = {
                "constant": norm,   
                "SBDM": norm * diffusion,
                "sigma": norm * t,
                "decreasing": 0.25 * (norm * torch.cos(np.pi * (1-t)) + 1) ** 2,
                "inccreasing-decreasing": norm * torch.sin(np.pi * (1-t)) ** 2,
            }
            try:
                diffusion = choices[form]
            except KeyError:
                raise NotImplementedError(f"Diffusion form {form} not implemented")
            return diffusion

        # Time schedule: t from 1.0 -> 0.0 with `num_steps` linearly spaced steps. Using last step ode

        timesteps = torch.linspace(1-4e-2, last_h, num_steps, device=device)

        for i in range(num_steps - 1):
            t_cur = timesteps[i].expand(n_samples)
            t_next = timesteps[i + 1].expand(n_samples)

            # Euler-Maruyama update step
            dt = (t_next - t_cur).view(-1, 1, 1, 1)
            # For flow matching: dx = v(x,t) dt + sqrt(2*t) dW
            # The diffusion coefficient is sqrt(2*t) and dW ~ N(0, |dt|)
            drift, diff = drift_fn(x, t_cur, y)
            noise = torch.randn_like(x) * torch.sqrt(torch.abs(dt)) * torch.sqrt(2 * diff)
            x = x + dt * drift + noise
        # last step ode
        t_cur = timesteps[-1].expand(n_samples)
        t_next = torch.zeros_like(t_cur)
        drift, _ = drift_fn(x, t_cur, y)
        dt = (t_next - t_cur).view(-1, 1, 1, 1)
        x = x + dt * drift
        return x