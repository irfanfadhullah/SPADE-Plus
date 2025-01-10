import torch
import torch.nn as nn
import models.networks as networks
import util.util as util

class CascadedPix2PixModel(nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        """
        Initialize 3 Generators (G1, G2, G3) and 3 Discriminators (D1, D2, D3), 
        similar to Pix2PixModel, but for a cascaded pipeline.
        """
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        # 1. Define and (optionally) load all networks
        self.netG1, self.netG2, self.netG3, \
        self.netD1, self.netD2, self.netD3, \
        self.netE = self.initialize_networks(opt)

        # 2. Define Losses if training
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=opt)
            self.criterionFeat = nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    def forward(self, data, mode):
        """
        Forward entry point, branching by mode:
        - mode == 'generator': returns G losses + generated fakes
        - mode == 'discriminator': returns D losses
        - mode == 'inference': returns just the generated images
        """
        # Preprocess or parse the data in a similar pattern to Pix2PixModel
        real_A, real_A_mask, real_B, real_B_mask = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(real_A, real_A_mask, real_B, real_B_mask)
            return g_loss, generated

        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(real_A, real_A_mask, real_B, real_B_mask)
            return d_loss

        elif mode == 'inference':
            with torch.no_grad():
                fake_dict = self.generate_fake(real_A)
            # Return just the final stage or all intermediate fakes
            return fake_dict['fake_B']  # or return the entire dictionary if needed

        else:
            raise ValueError(f"Mode {mode} not recognized.")

    def preprocess_input(self, data):
        """
        Move data to GPU if necessary, parse out relevant keys
        """
        real_A = data['A'].cuda() if self.use_gpu() else data['A']
        real_A_mask = data['A_mask'].cuda() if self.use_gpu() else data['A_mask']
        real_B = data['B'].cuda() if 'B' in data and data['B'] is not None else None
        real_B_mask = data['B_mask'].cuda() if 'B_mask' in data and data['B_mask'] is not None else None

        return real_A, real_A_mask, real_B, real_B_mask

    def compute_generator_loss(self, real_A, real_A_mask, real_B, real_B_mask):
        """
        Forward pass through G1, G2, G3. 
        Compute adversarial + optional feature matching + VGG losses.
        """
        G_losses = {}
        # 1. Generate fake images (3-stage pipeline)
        fake_dict = self.generate_fake(real_A)
        fake_A_mask = fake_dict['fake_A_mask']
        fake_B_mask = fake_dict['fake_B_mask']
        fake_B      = fake_dict['fake_B']

        # 2. Discriminator predictions on fakes vs. real
        pred_fake_A_mask, pred_real_A_mask = self.discriminate(
            self.netD1, real_A, fake_A_mask, real_A_mask, stage='A_mask')
        pred_fake_B_mask, pred_real_B_mask = self.discriminate(
            self.netD2, fake_A_mask, fake_B_mask, real_B_mask, stage='B_mask')
        pred_fake_B, pred_real_B = self.discriminate(
            self.netD3, fake_B_mask, fake_B, real_B, stage='B')

        # 3. Adversarial losses
        # G1
        G_losses['G1_GAN'] = self.criterionGAN(pred_fake_A_mask, True, for_discriminator=False)
        # G2
        G_losses['G2_GAN'] = self.criterionGAN(pred_fake_B_mask, True, for_discriminator=False)
        # G3
        G_losses['G3_GAN'] = self.criterionGAN(pred_fake_B, True, for_discriminator=False)

        # 4. Optional Feature Matching losses
        if not self.opt.no_ganFeat_loss:
            G_losses['G1_GAN_Feat'] = self.compute_gan_feat_loss(pred_fake_A_mask, pred_real_A_mask)
            G_losses['G2_GAN_Feat'] = self.compute_gan_feat_loss(pred_fake_B_mask, pred_real_B_mask)
            G_losses['G3_GAN_Feat'] = self.compute_gan_feat_loss(pred_fake_B, pred_real_B)

        # 5. Optional VGG losses (on final output or on intermediate masks)
        if not self.opt.no_vgg_loss:
            if real_A_mask is not None:
                G_losses['G1_VGG'] = self.criterionVGG(fake_A_mask, real_A_mask) * self.opt.lambda_vgg
            if real_B_mask is not None:
                G_losses['G2_VGG'] = self.criterionVGG(fake_B_mask, real_B_mask) * self.opt.lambda_vgg
            if real_B is not None:
                G_losses['G3_VGG'] = self.criterionVGG(fake_B, real_B) * self.opt.lambda_vgg

        # 6. (Optional) KLD if using VAE
        if self.opt.use_vae:
            # You would define a method encode_z() and incorporate it into the pipeline
            # G_losses['KLD'] = ...
            pass

        # Return dictionary of losses + dict of generated outputs
        generated = {
            'fake_A_mask': fake_A_mask,
            'fake_B_mask': fake_B_mask,
            'fake_B': fake_B
        }
        return G_losses, generated

    def compute_discriminator_loss(self, real_A, real_A_mask, real_B, real_B_mask):
        """
        1. Generate fake images with no grad (detach).
        2. Compute D losses (D1, D2, D3).
        """
        D_losses = {}
        with torch.no_grad():
            fake_dict = self.generate_fake(real_A)
            fake_A_mask = fake_dict['fake_A_mask'].detach()
            fake_B_mask = fake_dict['fake_B_mask'].detach()
            fake_B      = fake_dict['fake_B'].detach()

        # D1: compares (real_A, real_A_mask) vs. (real_A, fake_A_mask)
        pred_fake_A_mask, pred_real_A_mask = self.discriminate(
            self.netD1, real_A, fake_A_mask, real_A_mask, stage='A_mask')
        D_losses['D1_fake'] = self.criterionGAN(pred_fake_A_mask, False, for_discriminator=True)
        D_losses['D1_real'] = self.criterionGAN(pred_real_A_mask, True, for_discriminator=True)

        # D2: compares (fake_A_mask, real_B_mask) vs. (fake_A_mask, fake_B_mask)
        pred_fake_B_mask, pred_real_B_mask = self.discriminate(
            self.netD2, fake_A_mask, fake_B_mask, real_B_mask, stage='B_mask')
        D_losses['D2_fake'] = self.criterionGAN(pred_fake_B_mask, False, for_discriminator=True)
        D_losses['D2_real'] = self.criterionGAN(pred_real_B_mask, True, for_discriminator=True)

        # D3: compares (fake_B_mask, real_B) vs. (fake_B_mask, fake_B)
        pred_fake_B, pred_real_B = self.discriminate(
            self.netD3, fake_B_mask, fake_B, real_B, stage='B')
        D_losses['D3_fake'] = self.criterionGAN(pred_fake_B, False, for_discriminator=True)
        D_losses['D3_real'] = self.criterionGAN(pred_real_B, True, for_discriminator=True)

        return D_losses

    def generate_fake(self, real_A):
        """
        Run the 3-stage cascade:
          1) A -> A_mask
          2) A_mask -> B_mask
          3) B_mask -> B
        Returns dictionary of {fake_A_mask, fake_B_mask, fake_B}
        """
        fake_A_mask = self.netG1(real_A)
        fake_B_mask = self.netG2(fake_A_mask)
        fake_B = self.netG3(fake_B_mask)
        return {
            'fake_A_mask': fake_A_mask,
            'fake_B_mask': fake_B_mask,
            'fake_B': fake_B
        }

    def discriminate(self, netD, cond, fake, real, stage=''):
        """
        Similar to Pix2PixModel's 'discriminate' method, 
        but for each stage we define the input pairs:
        For example, if cond=fake_A_mask (stage2), netD sees (fake_A_mask, fake_B_mask).
        """
        if real is None:
            # e.g. if the stage doesn't exist in the data
            return None, None

        # Concatenate condition and image
        fake_concat = torch.cat([cond, fake], dim=1)
        real_concat = torch.cat([cond, real], dim=1)

        # Forward pass both
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        out = netD(fake_and_real)

        # Split the output back into pred_fake / pred_real
        pred_fake, pred_real = self.divide_pred(out)
        return pred_fake, pred_real

    def divide_pred(self, pred):
        """
        For multiscale or multi-layer discriminators, 
        split the combined predictions back into fake/real parts.
        """
        if isinstance(pred, list):
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0)//2] for tensor in p])
                real.append([tensor[tensor.size(0)//2:] for tensor in p])
        else:
            fake = pred[:pred.size(0)//2]
            real = pred[pred.size(0)//2:]
        return fake, real

    def compute_gan_feat_loss(self, pred_fake, pred_real):
        """
        Optional feature matching loss used in pix2pixHD.
        """
        if pred_fake is None or pred_real is None:
            return 0.0
        num_D = len(pred_fake)
        GAN_Feat_loss = self.FloatTensor(1).fill_(0)
        for i in range(num_D):
            num_intermediate = len(pred_fake[i]) - 1
            for j in range(num_intermediate):
                unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
        return GAN_Feat_loss

    def create_optimizers(self, opt):
        """
        Create separate optimizers for G (G1, G2, G3, optionally E) and D (D1, D2, D3).
        """
        G_params = list(self.netG1.parameters()) + list(self.netG2.parameters()) + list(self.netG3.parameters())
        if opt.use_vae and self.netE is not None:
            G_params += list(self.netE.parameters())

        if opt.isTrain:
            D_params = list(self.netD1.parameters()) + list(self.netD2.parameters()) + list(self.netD3.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        """
        Save networks as in Pix2PixModel.
        """
        util.save_network(self.netG1, 'G1', epoch, self.opt)
        util.save_network(self.netG2, 'G2', epoch, self.opt)
        util.save_network(self.netG3, 'G3', epoch, self.opt)

        if self.opt.isTrain:
            util.save_network(self.netD1, 'D1', epoch, self.opt)
            util.save_network(self.netD2, 'D2', epoch, self.opt)
            util.save_network(self.netD3, 'D3', epoch, self.opt)
        if self.opt.use_vae and self.netE is not None:
            util.save_network(self.netE, 'E', epoch, self.opt)

    def initialize_networks(self, opt):
        """
        Create the 3 Generator and 3 Discriminator networks (plus Encoder if use_vae).
        Load checkpoints if needed.
        """
        # Generators
        netG1 = networks.define_G(opt)
        netG2 = networks.define_G(opt)
        netG3 = networks.define_G(opt)

        # Discriminators
        netD1 = networks.define_D(opt) if opt.isTrain else None
        netD2 = networks.define_D(opt) if opt.isTrain else None
        netD3 = networks.define_D(opt) if opt.isTrain else None

        # Encoder (for VAE)
        netE = networks.define_E(opt) if opt.use_vae else None

        # Load existing weights if not isTrain or if continue_train
        if not opt.isTrain or opt.continue_train:
            netG1 = util.load_network(netG1, 'G1', opt.which_epoch, opt)
            netG2 = util.load_network(netG2, 'G2', opt.which_epoch, opt)
            netG3 = util.load_network(netG3, 'G3', opt.which_epoch, opt)
            if opt.isTrain:
                netD1 = util.load_network(netD1, 'D1', opt.which_epoch, opt)
                netD2 = util.load_network(netD2, 'D2', opt.which_epoch, opt)
                netD3 = util.load_network(netD3, 'D3', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG1, netG2, netG3, netD1, netD2, netD3, netE

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
