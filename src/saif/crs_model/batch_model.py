import torch
import torch.nn as nn

def masked_collect(output, mask):
    n_site, n_batch, n_steps = output.shape
    
    masked_output = mask * output
    masked_output = masked_output.reshape(n_site, -1)
    masked_output = masked_output.cumsum(-1)
    masked_output = masked_output.reshape(n_site, n_batch, n_steps)
    masked_output = mask * masked_output
    
    return masked_output

class BatchCRSModel(nn.Module):
    def __init__(self, site_info):
        super().__init__()
        #TODO:
        #Resolve: should site_info be passed to forward?
        ##Yes...match size of the batch

        self.tssr = site_info['tectonic_shear_stressing_rate'] # Pa/s
        self.tnsr = site_info['tectonic_normal_stressing_rate'] # Pa/s

        self.sigma = site_info['sigma'] # Pa
        self.biot = site_info['biot'] # dimensionless

    def run_recurrence(self, params, p, dpdt, delta_t, R0):
        """
        params: [nsites, 3]
        [mu_minus_alpha, rate_coeff, rate_factor]

        The following variables are either 1D series, or a batch.
        They will be reshaped to match the size of the params vector

        p: [nsites, nbatch, nsteps]
        dpdt: [nsites, nbatch, nsteps]
        delta_t: [nsites, nbatch, nsteps]
        
        R0: [nsites, nbatch, 1]
        N0: [nsites, nbatch, 1]
        """
        #print('params', params)
        #p = _check_input(p)
        #dpdt = _check_input(dpdt)
        #delta_t = _check_input(delta_t)

        n_site, n_batch, n_steps = p.shape

        
        mu_minus_alpha = params[:, 0, None, None]
        rate_coeff = params[:, 1, None, None]
        rate_factor = params[:, 2, None, None]
        #TODO: it will be better to just parameterize eta
        eta = 1 / rate_factor
        
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
        Nt = []
        
        #Delay time already taken into account with pre-processing
        R = R0[:, :, None]
        for i in range(n_steps):
            scaled_R = eta * R / s_dot[:, :, i, None]
            denom = 1 - scaled_R * (1 - exp_term[:, :, i, None]) + 1e-5
            R = R * exp_term[:, :, i, None] / denom
            N = asigma[:, :, i, None] / eta * torch.log(denom)

            Rt.append(R)
            Nt.append(N)
            
        Rt = torch.cat(Rt, dim=-1)
        Nt = torch.cat(Nt, dim=-1)
        
        return Rt, Nt

    def forward(self, params, p, dpdt, delta_t, R0, N0):
        """
        Batches are treated independently of one another.
        """
        Rt, Nt = self.run_recurrence(params, p, dpdt, delta_t, R0)
        Nt = Nt.cumsum(dim=-1) + N0[:, :, None]

        return Rt, Nt

    def masked_forward(self, params, p, dpdt, delta_t, R0, mask):
        """
        Batches depend on one another (via cumsum)
        """
        Rt, Nt = self.run_recurrence(params, p, dpdt, delta_t, R0)
        Nt = masked_collect(Nt, mask)

        return Rt, Nt
