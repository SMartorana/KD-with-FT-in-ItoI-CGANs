from model import Generator
from model import Discriminator
from model import Conv_reshape
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import time
import datetime
import gc

# A simple hook class that returns the input and output of a layer during forward/backward pass
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, input, output):
        self.input = input
        self.output = output
        #print("hooked.\n")

    def close(self):
        self.hook.remove()

class Solver(object):
    """Solver for training and testing StarGAN."""
    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        #self.tipo_flusso = 'res[5]'      #conv2d, res[5], res[1,3,5], out, res[5]+out
        self.tipo_flusso = config.tipo_flusso
        self.tipo_flusso_D = config.tipo_flusso_D

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        torch.cuda.empty_cache()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator STUDENT."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.tipo_flusso, self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.tipo_flusso_D, self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        elif self.dataset in ['Both']:
            self.G = Generator(self.tipo_flusso, self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.tipo_flusso_D, self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G_student')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

        """Restore the trained generator TEACHER (CelebA) ."""
        if self.dataset in ['CelebA', 'RaFD']:
            if(self.tipo_flusso != None):
                self.G_teacher = Generator(self.tipo_flusso, 64, self.c_dim, 6)
            if(self.tipo_flusso_D != None):
                self.D_teacher = Discriminator(self.tipo_flusso_D, self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        elif self.dataset in ['Both']:
            if(self.tipo_flusso != None):
                self.G_teacher = Generator(self.tipo_flusso, 64, self.c_dim+self.c2_dim+2, 6)
            if(self.tipo_flusso_D != None):
                self.D_teacher = Discriminator(self.tipo_flusso_D, self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

#        self.g_teacher_optimizer = torch.optim.Adam(self.G_teacher.parameters(), self.g_lr, [self.beta1, self.beta2])
        if(self.tipo_flusso != None):
            self.print_network(self.G_teacher, 'G_teacher')
            iters = self.test_iters
            self.G_teacher.to(self.device)
            G_teacher_path = os.path.join(self.model_save_dir, '{}-G_teacher.ckpt'.format(iters))
            self.G_teacher.load_state_dict(torch.load(G_teacher_path, map_location=lambda storage, loc: storage)) 
        if(self.tipo_flusso_D != None):
            self.print_network(self.D_teacher, 'D_teacher')
            iters = self.test_iters
            self.D_teacher.to(self.device)
            D_teacher_path = os.path.join(self.model_save_dir, '{}-D_teacher.ckpt'.format(iters))
            self.D_teacher.load_state_dict(torch.load(D_teacher_path, map_location=lambda storage, loc: storage)) 

        """Create a conv_reshape."""
        factor = 64 // self.g_conv_dim
        if (self.tipo_flusso == 'conv2d' or self.tipo_flusso == 'res[5]' or self.tipo_flusso == 'res[5]+out'):
            self.Conv_reshape = Conv_reshape(self.g_conv_dim*4*factor, factor)
            
            self.conv_reshape_optimizer = torch.optim.Adam(self.Conv_reshape.parameters(), self.g_lr, [self.beta1, self.beta2])
            self.print_network(self.Conv_reshape, 'Conv_reshape')

            self.Conv_reshape.to(self.device)
        elif (self.tipo_flusso == 'res[1,3,5]'):
            self.Conv_reshape = []
            self.conv_reshape_optimizer = []
            for j in range(2 + 1):
                self.Conv_reshape.append(Conv_reshape(self.g_conv_dim*4*factor, factor))
            
                self.conv_reshape_optimizer.append(torch.optim.Adam(self.Conv_reshape[j].parameters(), self.g_lr, [self.beta1, self.beta2]))
                self.print_network(self.Conv_reshape[j], 'Conv_reshape ' + str(j))

                self.Conv_reshape[j].to(self.device)
        print("Tipo flusso:", self.tipo_flusso)
        print("Tipo flusso D:", self.tipo_flusso_D)
        print("torch.__version__:", torch.__version__)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        wait = True
        while(wait and resume_iters > 0):
            print(os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters)))
            if(not os.path.isfile(os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters)))):
                self.resume_iters -= 10000
                resume_iters = self.resume_iters
                print("resume_iters: ", resume_iters)
            else:
                wait = False
        if(resume_iters == 0):
            return
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        G_teacher_path = os.path.join(self.model_save_dir, '{}-G_teacher.ckpt'.format(self.test_iters))
        D_teacher_path = os.path.join(self.model_save_dir, '{}-D_teacher.ckpt'.format(self.test_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        if(self.tipo_flusso != None):
            self.G_teacher.load_state_dict(torch.load(G_teacher_path, map_location=lambda storage, loc: storage))
        if(self.tipo_flusso_D != None):
            self.D_teacher.load_state_dict(torch.load(D_teacher_path, map_location=lambda storage, loc: storage))
        if (self.tipo_flusso == 'conv2d' or self.tipo_flusso == 'res[5]' or self.tipo_flusso == 'res[5]+out'):
            C_reshape = os.path.join(self.model_save_dir, '{}-CR.ckpt'.format(resume_iters))
            self.Conv_reshape.load_state_dict(torch.load(C_reshape, map_location=lambda storage, loc: storage))
        elif (self.tipo_flusso == 'res[1,3,5]'):
            for j in range(2 + 1):
                C_reshape = os.path.join(self.model_save_dir, '{}-CR{}.ckpt'.format(resume_iters,j))
                self.Conv_reshape[j].load_state_dict(torch.load(C_reshape, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        if (self.tipo_flusso == 'conv2d' or self.tipo_flusso == 'res[5]' or self.tipo_flusso == 'res[5]+out'):
            self.conv_reshape_optimizer.zero_grad()
        elif (self.tipo_flusso == 'res[1,3,5]'):
            for j in range(2 + 1):
                self.conv_reshape_optimizer[j].zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    
    def threshold(self, x):
        x = x.clone()
        x = (x >= 0.5).float()
        return x

    def compute_accuracy(self, x, y, dataset):
        if dataset == 'CelebA':
            x = F.sigmoid(x)
            predicted = self.threshold(x)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct, dim=0) * 100.0
        else:
            _, predicted = torch.max(x, dim=1)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct) * 100.0
        return accuracy

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        torch.autograd.set_detect_anomaly(True)
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        #MSE loss
        loss_mse = nn.MSELoss()

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            self.restore_model(self.resume_iters)
            start_iters = self.resume_iters
            

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            """ # Compute classification accuracy of the discriminator
            if (i+1) % self.log_step == 0:
                accuracies = self.compute_accuracy(out_cls, label_org, self.dataset)
                log = ["{:.3f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                if self.dataset == 'CelebA':
                    print('Classification D_Acc (Black/Blond/Brown/Gender/Aged): ')
                else:
                    print('Classification D_Acc (8 emotional expressions): ')
                print(log) """

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            #feature_loss
            loss_feature_D = 0
            if(self.tipo_flusso_D):
                out_src_D, out_cls_D = self.D_teacher(x_real)
                features_teacher_D = self.D_teacher.hook_layer
                features_student_D = self.D.hook_layer
                loss_feature_D = loss_mse(features_teacher_D, features_student_D)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp + loss_feature_D*10
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            #if(self.tipo_flusso_D):
            loss['D/loss_feature'] = loss_feature_D.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                """ # Compute classification accuracy of the discriminator
                if (i+1) % self.log_step == 0:
                    accuracies = self.compute_accuracy(out_cls, label_trg, self.dataset)
                    log = ["{:.3f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                    if self.dataset == 'CelebA':
                        print('Classification G_Acc (Black/Blond/Brown/Gender/Aged): ')
                    else:
                        print('Classification G_Acc (8 emotional expressions): ')
                    print(log) """

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Distillation loss su hook layer
               
                #.output prende il valore in uscita al layer selezionato
                #.input prende il valore in ingresso al layer selezionato
                ##features_teacher = Hook(self.G_teacher.conv2[1])
                ##features_student = Hook(self.G.conv2[1])
                #loss_features = torch.mean(torch.pow(reshape_teacher - features_student, 2)) # loss giusta = MSE LOSS
                if(self.tipo_flusso != None):
                    x_teacher = self.G_teacher(x_real, c_trg)
                loss_feature = 0
                if (self.tipo_flusso == 'conv2d' or self.tipo_flusso == 'res[5]'):
                    features_teacher = self.G_teacher.hook_layer
                    reshape_teacher = self.Conv_reshape(features_teacher)
                    features_student = self.G.hook_layer
                    loss_feature = loss_mse(reshape_teacher, features_student)
                elif (self.tipo_flusso == 'res[1,3,5]'):
                    features_teacher = []
                    reshape_teacher = []
                    features_student = []
                    loss_features = []
                    loss_feature = 0
                    for j, hook in enumerate(self.G_teacher.hook_layer):
                        features_teacher.append(hook)
                        reshape_teacher.append(self.Conv_reshape[j](features_teacher[j]))
                        features_student.append(self.G.hook_layer[j])
                        loss_features.append(loss_mse(reshape_teacher[j], features_student[j]))
                        loss_feature += loss_features[j]
                elif (self.tipo_flusso == 'out'):
                    loss_feature = loss_mse(x_teacher, x_fake)
                elif (self.tipo_flusso == 'res[5]+out'):
                    features_teacher = self.G_teacher.hook_layer
                    reshape_teacher = self.Conv_reshape(features_teacher)
                    features_student = self.G.hook_layer
                    loss_feature = loss_mse(reshape_teacher, features_student) + loss_mse(x_teacher, x_fake)
                    # print(loss_mse(reshape_teacher, features_student).item())
                    # loss_feature += loss_mse(x_teacher, x_fake)
                    # print(loss_mse(x_teacher, x_fake).item())
                    # print()
                
                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + 10*loss_feature          #dare piu' peso -> 10* o 100*
                self.reset_grad()
#               gc.collect()
                g_loss.backward()
                self.g_optimizer.step()
                if (self.tipo_flusso == 'conv2d' or self.tipo_flusso == 'res[5]' or self.tipo_flusso == 'res[5]+out'):
                    self.conv_reshape_optimizer.step()
                elif (self.tipo_flusso == 'res[1,3,5]'):
                    for j in range(2 + 1):
                        self.conv_reshape_optimizer[j].step()
                

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                # if(self.tipo_flusso):
                loss['G/loss_features'] = loss_feature.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                if (self.tipo_flusso == 'conv2d' or self.tipo_flusso == 'res[5]' or self.tipo_flusso == 'res[5]+out'):
                    C_reshape = os.path.join(self.model_save_dir, '{}-CR.ckpt'.format(i+1))
                    torch.save(self.Conv_reshape.state_dict(), C_reshape)
                elif (self.tipo_flusso == 'res[1,3,5]'):
                    for j in range(2 + 1):
                        C_reshape = os.path.join(self.model_save_dir, '{}-CR{}.ckpt'.format(i+1,j))
                        torch.save(self.Conv_reshape[j].state_dict(), C_reshape)

                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def train_multi(self):
        """Train StarGAN with multiple datasets."""        
        # Data iterators.
        celeba_iter = iter(self.celeba_loader)
        rafd_iter = iter(self.rafd_loader)

        # Fetch fixed inputs for debugging.
        x_fixed, c_org = next(celeba_iter)
        x_fixed = x_fixed.to(self.device)
        c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
        c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
        zero_celeba = torch.zeros(x_fixed.size(0), self.c_dim).to(self.device)           # Zero vector for CelebA.
        zero_rafd = torch.zeros(x_fixed.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
        mask_celeba = self.label2onehot(torch.zeros(x_fixed.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
        mask_rafd = self.label2onehot(torch.ones(x_fixed.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            for dataset in ['CelebA', 'RaFD']:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                
                # Fetch real images and labels.
                data_iter = celeba_iter if dataset == 'CelebA' else rafd_iter
                
                try:
                    x_real, label_org = next(data_iter)
                except:
                    if dataset == 'CelebA':
                        celeba_iter = iter(self.celeba_loader)
                        x_real, label_org = next(celeba_iter)
                    elif dataset == 'RaFD':
                        rafd_iter = iter(self.rafd_loader)
                        x_real, label_org = next(rafd_iter)

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                if dataset == 'CelebA':
                    c_org = label_org.clone()
                    c_trg = label_trg.clone()
                    zero = torch.zeros(x_real.size(0), self.c2_dim)
                    mask = self.label2onehot(torch.zeros(x_real.size(0)), 2)
                    c_org = torch.cat([c_org, zero, mask], dim=1)
                    c_trg = torch.cat([c_trg, zero, mask], dim=1)
                elif dataset == 'RaFD':
                    c_org = self.label2onehot(label_org, self.c2_dim)
                    c_trg = self.label2onehot(label_trg, self.c2_dim)
                    zero = torch.zeros(x_real.size(0), self.c_dim)
                    mask = self.label2onehot(torch.ones(x_real.size(0)), 2)
                    c_org = torch.cat([zero, c_org, mask], dim=1)
                    c_trg = torch.cat([zero, c_trg, mask], dim=1)

                x_real = x_real.to(self.device)             # Input images.
                c_org = c_org.to(self.device)               # Original domain labels.
                c_trg = c_trg.to(self.device)               # Target domain labels.
                label_org = label_org.to(self.device)       # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)       # Labels for computing classification loss.

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src, out_cls = self.D(x_real)
                out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org, dataset)

                # Compute loss with fake images.
                x_fake = self.G(x_real, c_trg)
                out_src, _ = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()
            
                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i+1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake = self.G(x_real, c_trg)
                    out_src, out_cls = self.D(x_fake)
                    out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, label_trg, dataset)

                    # Target-to-original domain.
                    x_reconst = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training info.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(et, i+1, self.num_iters, dataset)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_celeba_list:
                        c_trg = torch.cat([c_fixed, zero_rafd, mask_celeba], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    for c_fixed in c_rafd_list:
                        c_trg = torch.cat([zero_celeba, c_fixed, mask_rafd], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        
        with torch.no_grad():
            for i, (x_real, c_org, filename) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                filename = str(filename).split('\'')[1]

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, filename)
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(self.celeba_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
                c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
                zero_celeba = torch.zeros(x_real.size(0), self.c_dim).to(self.device)            # Zero vector for CelebA.
                zero_rafd = torch.zeros(x_real.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
                mask_celeba = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
                mask_rafd = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

                # Translate images.
                x_fake_list = [x_real]
                for c_celeba in c_celeba_list:
                    c_trg = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))
                for c_rafd in c_rafd_list:
                    c_trg = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))