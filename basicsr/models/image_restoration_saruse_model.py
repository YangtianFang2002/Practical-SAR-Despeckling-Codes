import torch
from basicsr.models.image_restoration_model import ImageRestorationModel
import random
from collections import OrderedDict


class SarUseModel(ImageRestorationModel):
    def __init__(self, opt):
        super(SarUseModel, self).__init__(opt)

        self.nrd_lq = None

    def _noise_correction(self, noise):

        # mean scaling
        mean = torch.mean(noise)

        noise = noise / mean

        # shuffling
        B, C, H, W = noise.shape

        k = random.randint(1, 4)

        # unfold: (B, C, H/k, W/k, k, k)
        patches = noise.unfold(2, k, k).unfold(3, k, k)
        
        # patches (B, C, H/k, W/k, k, k)
        patches = patches.contiguous().view(B, C, -1, k, k)
        
        indices = torch.randperm(patches.shape[2])
        patches = patches[:, :, indices, :, :]
        
        #  (B, C, H/k, W/k, k, k)
        patches = patches.view(B, C, H // k, W // k, k, k)
        
        noise = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        noise = noise.view(B, C, H, W)

        return noise

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.scaler is not None):

            # extract noise
            with torch.no_grad():
                nrd_pred = self.net_g(self.nrd_lq)
                nrd_noise = self.nrd_lq / (nrd_pred + 1e-8)

            nrd_noise = self._noise_correction(nrd_noise)

            self.lq = self.gt * nrd_noise

            # predict
            preds = self.net_g(self.lq)
            if not isinstance(preds, list):
                preds = [preds]

            self.output = preds[-1]

            # make lq, gt, pred the same resolution
            self.sync_data_size()

            l_total = 0
            loss_dict = OrderedDict()
            # pixel loss
            if self.cri_pix:
                l_pix = 0.
                for cri_name, cri_pix in [('l_pix', self.cri_pix),
                                          ('l_pix_1', self.cri_pix_1),
                                          ('l_pix_2', self.cri_pix_2)]:
                    if cri_pix:
                        l_pix = 0.
                        for pred in preds:
                            _l_pix = cri_pix(pred, self.gt)
                            l_pix += _l_pix

                        l_total += l_pix
                        loss_dict[cri_name] = l_pix

            if self.cri_pix_3i:
                l_pix = 0.
                cri_name = 'l_pix'

                l_pix = self.cri_pix_3i(self.lq, self.output, self.gt)

                l_total += l_pix
                loss_dict[cri_name] = l_pix

            # perceptual loss
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            #
                if l_percep is not None:
                    l_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_total += l_style
                    loss_dict['l_style'] = l_style

            # l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

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

    def feed_data(self, data, is_val=False):
        if 'nrd_lq_i' in data:
            self.nrd_lq = data['nrd_lq_i'].to(self.device)
            # self.lq = data['sd_lq_i'].to(self.device)  # lq generated on the fly
            self.gt = data['sd_gt_i'].to(self.device)
        else:  # eval
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
