import torch
from basicsr.models.image_restoration_model import ImageRestorationModel


class Speckle2VoidModel(ImageRestorationModel):
    def __init__(self, opt):
        super(Speckle2VoidModel, self).__init__(opt)
        self.L_noise = opt.get('L_noise', 1)
        self.k_penalty_tv = opt.get('k_penalty_tv', 5e-5)
        
        self.X_prior = None
        self.X_posterior = None
        self.alpha = None
        self.beta = None
        self.mask = None
        
        self.loss = None

    def _deduct_clean_out(self, out):
        self.alpha = out[:, 0:1, :, :] + 1
        self.beta = out[:, 1:2, :, :]
        self.X_prior = self.beta / (self.alpha - 1 + 1e-19)
        # self.X_posterior = (self.beta + (self.L_noise * self.lq)) / (self.L_noise + self.alpha - 1)
        self.X_posterior = (self.beta + (self.L_noise * self.lq))
        return self.X_posterior

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred = self.net_g(self.lq[i:j])
                pred = self._deduct_clean_out(pred)
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def calculate_loss(self):
        L_replicated = torch.ones_like(self.alpha) * self.L_noise
        
        # Concatenate alpha and L_replicated along the last dimension
        alpha_L = torch.cat([self.alpha, L_replicated], dim=1)
        
        # Compute the logarithm of beta with small value added for numerical stability
        log_beta = torch.log(self.beta + 1e-19)
        
        # Compute alpha * log(beta)
        alpha_log_beta_complete = - self.alpha * log_beta
        
        # Compute the log of the beta function using the PyTorch equivalent
        beta_func_complete = torch.special.gammaln(alpha_L)
        
        # Compute (L + alpha) * log(beta + L * X_noisy)
        alpha_log_beta_noisy_complete = (self.L_noise + self.alpha) * torch.log(self.beta + self.L_noise * self.lq + 1e-19)
        
        # Compute the log probability of the noisy pixel y_i
        # log_p_y = (self.L_noise * torch.log(self.L_noise + 1e-19)) \
        #           + ((self.L_noise - 1) * torch.log(self.lq + 1e-19)) \
        #           - alpha_log_beta_complete \
        #           - alpha_log_beta_noisy_complete \
        #           - beta_func_complete
        log_p_y = - alpha_log_beta_complete \
                  - alpha_log_beta_noisy_complete \
                  - beta_func_complete
        
        
        # Apply mask to exclude pixels with the median from the loss computation
        log_p_y = log_p_y # * self.mask
        
        # Compute the total variation (TV) regularization term
        tot_var = torch.sum(torch.abs(self.X_posterior[:, :, :-1] - self.X_posterior[:, :, 1:])) + \
                  torch.sum(torch.abs(self.X_posterior[:, :-1, :] - self.X_posterior[:, 1:, :]))
        
        # Calculate total variation penalty
        self.total_variation = self.k_penalty_tv * torch.mean(tot_var)
        
        # From log likelihood to loss
        # self.loss = - torch.sum(log_p_y) / torch.sum(self.mask)
        self.loss = - torch.mean(log_p_y)
        
        # Adding total variation regularizer
        self.loss += self.total_variation
        return self.loss

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            self.lq = torch.pow(self.lq, 2)  # amp -> intensity!
            out = self.net_g(self.lq)
            
            self.output = self._deduct_clean_out(out)  # intensity
            self.output = torch.pow(self.output, 0.5)  # self.output is amplitude!
            # also update alpha, beta, prior, posterior
            
            l_total = self.calculate_loss()
            self.lq = torch.pow(self.lq, 0.5)  # make lq into amplitude
            loss_dict = {'l_loss': l_total}

        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            if use_grad_clip:
                self.scaler.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            l_total.backward()
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
