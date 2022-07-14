import argparse
import os

import torch
from tqdm import tqdm
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import matplotlib.pyplot as plt
from torch.utils.tensorboard import writer, SummaryWriter
from torch.utils.data import DataLoader
from math import pi
import numpy as np
import time

from datasets import *
from models.wgangp import Generator, Critic
from time_series_analysis import ts_analyser

class Trainer:
    

    def __init__(self, generator, critic, gen_optimizer, critic_optimizer, dataset,
                 gp_weight=10, critic_iterations=5, print_every=200, sample_count=2,checkpoint_frequency=200):
        self.dataset = dataset
        self.NOISE_LENGTH = self.dataset.dataset.shape[1]
        self.g = generator
        self.g_opt = gen_optimizer
        self.c = critic
        self.c_opt = critic_optimizer
        self.losses = {'g': [], 'c': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.checkpoint_frequency = checkpoint_frequency
        self.sample_count = sample_count
        self.stats = {'k_stat':[],
                        'k_p':[],
                        'g_stat':[],
                        'g_p':[],
                        'g_loss':[],
                        'c_loss':[],
                        'r_score':[]}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.g = self.g.to(self.device)
        self.c = self.c.to(self.device)
        print('CUDA Device used: ', self.device)
        self.dir_setup() 
        
    def dir_setup(self):
        PATH = os.getcwd()
        folder_list = ['Output/training_samples/dynamic_latents', 'Output/training_samples/fixed_latents', 'Output/ks_statistics', 'Output/checkpoints']
        for folder in folder_list:
            new_path = os.path.join(PATH,folder)
            if not os.path.isdir(new_path):
                os.makedirs(new_path)

    def _critic_train_iteration(self, real_data):

        batch_size = real_data.size()[0]
        noise_shape = (batch_size, self.NOISE_LENGTH)
        generated_data = self.sample_generator(noise_shape)
        real_data = Variable(real_data)
        real_data = real_data.to(self.device)

        # Pass data through the Critic
        c_real = self.c(real_data)
        c_generated = self.c(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(real_data, generated_data)
        self.losses['GP'].append(gradient_penalty.data.item())

        # Create total loss and optimize
        self.c_opt.zero_grad()
        d_loss = c_generated.mean() - c_real.mean() + gradient_penalty
        d_loss.backward()
        self.c_opt.step()

        self.losses['c'].append(d_loss.data.item())

    def _generator_train_iteration(self, data):
        self.g_opt.zero_grad()
        batch_size = data.size()[0]
        latent_shape = (batch_size, self.NOISE_LENGTH)

        generated_data = self.sample_generator(latent_shape)
        # Calculate loss and optimize
        d_generated = self.c(generated_data)
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.g_opt.step()
        self.losses['g'].append(g_loss.data.item())

    def _gradient_penalty(self, real_data, generated_data):

        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.to(self.device)
        
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.to(self.device)

        # Pass interpolated data through Critic
        prob_interpolated = self.c(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]
        # Gradients have shape (batch_size, num_channels, series length),
        # here we flatten to take the norm per example for every batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data.item())

        # Derivatives of the gradient close to 0 can cause problems because of the
        # square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader, epoch):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self._critic_train_iteration(data.float())
            # Only update generator every critic_iterations iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data)

    def train(self, data_loader, epochs, plot_training_samples=True, checkpoint=None):
        if checkpoint:
            path = os.path.join('Output/checkpoints', checkpoint)
            state_dicts = torch.load(path, map_location=torch.device('cpu'))
            self.g.load_state_dict(state_dicts['g_state_dict'])
            self.c.load_state_dict(state_dicts['d_state_dict'])
            self.g_opt.load_state_dict(state_dicts['g_opt_state_dict'])
            self.c_opt.load_state_dict(state_dicts['d_opt_state_dict'])

        # Define noise_shape
        noise_shape = (self.dataset.dataset.shape[0], self.NOISE_LENGTH)


        # Fix latents to see how series generation improves during training
        self.fixed_latents = Variable(self.sample_latent(noise_shape)).to(self.device)
        t_0 = time.time()
        print('Training beginning')
        for epoch in range(epochs):
            self._train_epoch(data_loader, epoch + 1)
    
            # Save checkpoint
            if epoch % self.checkpoint_frequency == 0:
                torch.save({
                    'epoch': epoch,
                    'd_state_dict': self.c.state_dict(),
                    'g_state_dict': self.g.state_dict(),
                    'd_opt_state_dict': self.c_opt.state_dict(),
                    'g_opt_state_dict': self.g_opt.state_dict(),
                }, 'Output/checkpoints/epoch_{}.pkl'.format(epoch))

            if plot_training_samples and (epoch % self.print_every == 0):
                self.validate(epoch, noise_shape)
        print('Duration: {}'.format(time.time() - t_0))
    def validate(self, epoch, noise_shape):
        
        self.g.eval()
        # Sample a different region of the latent distribution to check for mode collapse
        ''' 
        dynamic_latents_list = []
        for i in range(self.sample_count):
            dynamic_latents = Variable(self.sample_latent(noise_shape))
            dynamic_latents = dynamic_latents.to(self.device)
            fake_data_dynamic_latents = self.g(dynamic_latents).cpu().data
            dynamic_latents_list.append(fake_data_dynamic_latents)'''
        dynamic_latents = Variable(self.sample_latent(noise_shape))
        dynamic_latents = dynamic_latents.to(self.device)
        fake_data_dynamic_latents = self.g(dynamic_latents).cpu().data.numpy()
        # Generate fake data using both fixed and dynamic latents
        fake_data_fixed_latents = self.g(self.fixed_latents).cpu().data.numpy()
        
        if epoch > 0:
            self.stat_comparison(np.array(fake_data_fixed_latents), epoch)

        plt.figure()
        for i in range(self.sample_count):
            plt.plot(fake_data_dynamic_latents[i].T)
        plt.title('dynamic latents, epoch {}'.format(epoch))
        plt.savefig('Output/training_samples/dynamic_latents/series_epoch_{}.png'.format(epoch))
        plt.show()
        plt.close()

        plt.figure()
        plt.plot(fake_data_fixed_latents[0].T)
        plt.title('fixed latents, epoch {}'.format(epoch))
        plt.savefig('Output/training_samples/fixed_latents/series_epoch_{}.png'.format(epoch))
        plt.close()
        self.g.train()

    def stat_comparison(self, fake_data_fixed_latents, epoch):
        metric = ts_analyser(fake_data_fixed_latents[0], epoch)
        metric.comparison(self.dataset.dataset[0, :], title='synthetic vs train set'.format(epoch), label=['Synthetic', 'Training'], show=True)
        self.stats['k_stat'].append(metric.KS['k_stat'])
        self.stats['k_p'].append(metric.KS['k_p'])
        self.stats['g_stat'].append(metric.granger['g_stat'])
        self.stats['g_p'].append(metric.granger['g_p'])
        self.stats['r_score'].append(metric.r_score)
        self.stats['c_loss'].append(self.losses['c'][-1])
        self.stats['g_loss'].append(self.losses['g'][-1])
        
        f, ax = plt.subplots(2,2, figsize=(25,10))
        ax1 = ax[0][0].twinx()
        ax[0][0].plot(self.stats['k_stat'], color='b',label = 'k_stat')
        ax1.semilogy(self.stats['k_p'], color='orange',label = 'k_p')
        ax[0][0].set_title('KS Test')
        ax[0][0].set_xlabel('KS stat: {:.5f}, KS p:{:.5f}'.format(self.stats['k_stat'][-1],self.stats['k_p'][-1]))

        ax2 = ax[0][1].twinx()
        ax[0][1].plot(self.stats['g_stat'], color='b',label = 'g_stat')
        ax2.semilogy(self.stats['g_p'], color='orange',label = 'g_p')
        ax[0][1].set_title('Granger Causality')
        ax[0][1].set_xlabel('G stat: {:.5f}, G p:{:.5f}'.format(self.stats['g_stat'][-1],self.stats['g_p'][-1]))

        ax[1][1].plot(self.stats['r_score'], color='b',label = 'g_stat')
        ax[1][1].set_title('R score')
        ax[1][1].set_xlabel('R_score: {}'.format(self.stats['r_score'][-1]))
        ax[1][1].set_ylim((-2, 1))

        ax[1][0].plot(self.stats['g_loss'], color='black',label = 'generator')
        ax[1][0].plot(self.stats['c_loss'], color='red',label = 'critic')
        ax[1][0].set_title('G/C Loss')
        ax[1][0].set_xlabel('G loss: {:.5f}, C loss:{:.5f}'.format(self.stats['g_loss'][-1],self.stats['c_loss'][-1]))


        f.legend()
        f.suptitle('series_epoch_{}'.format(epoch))
        f.savefig('Output/ks_statistics/series_epoch_{}_{:.5f}.png'.format(epoch, time.time()))
        plt.show()
        plt.close('all') 
        
    def sample_generator(self, latent_shape):
        latent_samples = Variable(self.sample_latent(latent_shape))
        latent_samples = latent_samples.to(self.device)
        return self.g(latent_samples)

    @staticmethod
    def sample_latent(shape):
        return torch.randn(shape)

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        return generated_data.data.cpu().numpy()


