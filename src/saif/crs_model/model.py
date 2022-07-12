import torch
import torch.nn as nn


def _check_input(arr):
    if arr.ndim == 1:
        return arr[None, :]

    return arr

class CRSModel(nn.Module):
    def __init__(self, site_info):
        super().__init__()
        #TODO:
        #Resolve: should site_info be passed to forward?

        self.tnsr = site_info['tectonic_shear_stressing_rate'] # Pa/s
        self.tssr = site_info['tectonic_normal_stressing_rate'] # Pa/s

        self.sigma = site_info['sigma'] # Pa
        self.biot = site_info['biot'] # dimensionless

        self.R0 = site_info['background_rate']
        
        self.N0 = self.R0 * site_info['init_delta_t']

    def forward(self, params, p, dpdt, delta_t):
        """
        params: [nbatch, 3]
        [mu_minus_alpha, rate_coeff, rate_factor]

        The following variables are either 1D series, or a batch.
        They will be reshaped to match the size of the params vector

        p: [nbatch, nsteps] or [nsteps]
        dpdt: [nbatch, nsteps] or [nsteps]
        delta_t: [nbatch, nsteps] or [nsteps]
        """

        p = _check_input(p)
        dpdt = _check_input(dpdt)
        delta_t = _check_input(delta_t)
        

        B, T = params.shape[0], p.shape[-1]

        mu_minus_alpha = params[:, 0, None]
        rate_coeff = params[:, 1, None]
        rate_factor = params[:, 2, None]
        eta = 1 / rate_factor

        # TODO: Should we also broadcast along the site?
        # TODO: check when s_dot (CSR) is equal to 0

        ## Compute the stress inputs from pressure data
        # Coulomb stressing rate
        s_dot = self.tssr - mu_minus_alpha * (self.tnsr - dpdt)
        # Scaled sigma effective
        asigma = rate_coeff * (self.sigma - self.biot * p)
        

        # TODO: check that these time-steps align 
        # (ie make sure R(t) aligns with s_dot(t) and not s_dot(t + 1))
        exp_term = torch.exp(s_dot * delta_t / asigma)

        # TODO: Consider different signals to forecast:
        # - predict rate? Would require pre-processing on the event
        #   to obtain targets for rate
        # - predict rate, but cumsum to get number? A compromise.
        # - predict number directly? Potentially unbounded.
        #   Large, unnormalized outputs.
        Rt = []
        R = self.R0 * torch.ones(B, 1).to(p.device)
        Rt.append(R)
        
        Nt = []
        N = self.N0 * torch.ones(B, 1).to(p.device)
        Nt.append(N)

        for i in range(T):
            scaled_R = eta * R / s_dot[:, i, None]
            denom = 1 - scaled_R * (1 - exp_term[:, i, None])
            
            R = R * exp_term[:, i, None] / denom
            N = asigma[:, i, None] / eta * torch.log(denom)

            Rt.append(R)
            Nt.append(N)
            
        Rt = torch.cat(Rt, dim=-1)
        Nt = torch.cat(Nt, dim=-1).cumsum(dim=-1)

        return Rt, Nt