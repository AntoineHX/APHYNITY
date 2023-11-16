import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics

from utils import fix_seed, make_basedir, convert_tensor
from utils import CalculateNorm, Logger

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg') # Generate images without having a window appear
matplotlib.rcParams["figure.dpi"] = 100

_EPSILON = 1e-5

import os, time, io, logging, sys
from functools import partial

from torchvision.utils import make_grid

import collections

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class BaseExperiment(object):
    def __init__(self, device, path='./exp', seed=None):
        self.device = device
        self.path = make_basedir(path)
            
        if seed is not None:
            fix_seed(seed)

    def training(self, mode=True):
        for m in self.modules():
            m.train(mode)

    def evaluating(self):
        self.training(mode=False)

    def zero_grad(self):
        for optimizer in self.optimizers():
            optimizer.zero_grad()        

    def to(self, device):
        for m in self.modules():
            m.to(device)
        return self

    def modules(self):
        for name, module in self.named_modules():
            yield module

    def named_modules(self):
        for name, module in self._modules.items():
            yield name, module

    def datasets(self):
        for name, dataset in self.named_datasets():
            yield dataset

    def named_datasets(self):
        for name, dataset in self._datasets.items():
            yield name, dataset

    def optimizers(self):
        for name, optimizer in self.named_optimizers():
            yield optimizer

    def named_optimizers(self):
        for name, optimizer in self._optimizers.items():
            yield name, optimizer

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            if not hasattr(self, '_modules'):
                self._modules = collections.OrderedDict()
            self._modules[name] = value
        elif isinstance(value, DataLoader):
            if not hasattr(self, '_datasets'):
                self._datasets = collections.OrderedDict()
            self._datasets[name] = value
        elif isinstance(value, Optimizer):
            if not hasattr(self, '_optimizers'):
                self._optimizers = collections.OrderedDict()
            self._optimizers[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        if '_datasets' in self.__dict__:
            datasets = self.__dict__['_datasets']
            if name in datasets:
                return datasets[name]
        if '_optimizers' in self.__dict__:
            optimizers = self.__dict__['_optimizers']
            if name in optimizers:
                return optimizers[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __delattr__(self, name):
        if name in self._modules:
            del self._modules[name]
        elif name in self._datasets:
            del self._datasets[name]
        elif name in self._optimizers:
            del self._optimizers[name]
        else:
            object.__delattr__(self, name)

def show(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

class LoopExperiment(BaseExperiment):
    def __init__(
        self, train, test=None, root=None, nepoch=10, **kwargs):
        super().__init__(**kwargs)
        self.train = train
        self.test = test
        self.nepoch = nepoch
        self.logger = Logger(filename=os.path.join(self.path, 'log.txt'))
        print(' '.join(sys.argv))

    def train_step(self, batch, val=False):
        self.training()
        batch = convert_tensor(batch, self.device)
        loss, output = self.step(batch)
        with torch.no_grad():
            metric = self.metric(**output, **batch)

        return batch, output, loss, metric

    def val_step(self, batch, val=False):
        self.evaluating()
        with torch.no_grad():
            batch = convert_tensor(batch, self.device)
            loss, output = self.step(batch, backward=False)
            metric = self.metric(**output, **batch)

        return batch, output, loss, metric

    def log(self, epoch, iteration, metrics):
        message = '[{step}][{epoch}/{max_epoch}][{i}/{max_i}]'.format(
            step=epoch *len(self.train)+ iteration+1,
            epoch=epoch+1,
            max_epoch=self.nepoch,
            i=iteration+1,
            max_i=len(self.train)
        )
        for name, value in metrics.items():
            message += ' | {name}: {value}'.format(name=name, value=value.item() if isinstance(value, torch.Tensor) else value)
            
        print(message)

    def step(self, **kwargs):
        raise NotImplementedError


class APHYNITYExperiment(LoopExperiment):
    def __init__(self, net, optimizer, min_op, lambda_0, tau_2, niter=1, nupdate=100, nlog=10, display=True, **kwargs):
        super().__init__(**kwargs)

        self.traj_loss = nn.MSELoss()
        self.net = net.to(self.device)
        self.optimizer = optimizer

        self.min_op = min_op
        self.tau_2 = tau_2
        self.niter = niter
        self._lambda = lambda_0

        self.nlog = nlog
        self.nupdate = nupdate
        self.train_history = {'loss_train': np.empty((1,2)), 'loss_test':np.empty((1,2))}
        self.display = display
        if self.display:
            os.makedirs(self.path +'/png')
            self.fig = plt.figure(figsize=(12, 4), facecolor='white')
            self.ax_traj = self.fig.add_subplot(131)
            self.ax_spd = self.fig.add_subplot(132)
            self.ax_loss = self.fig.add_subplot(133)
            plt.show(block=False)
    
    def lambda_update(self, loss):
        self._lambda = self._lambda + self.tau_2 * loss
    
    def _forward(self, states, t, backward):
        target = states
        y0 = states[:, :, 0]
        pred = self.net(y0, t)
        loss = self.traj_loss(pred, target)
        aug_deriv = self.net.model_aug.get_derivatives(states)

        if self.min_op == 'l2_normalized':
            loss_op = ((aug_deriv.norm(p=2, dim=1) / (states.norm(p=2, dim=1) + _EPSILON)) ** 2).mean()
        elif self.min_op == 'l2':
            loss_op = (aug_deriv.norm(p=2, dim=1) ** 2).mean()

        if backward:
            loss_total = loss * self._lambda + loss_op

            loss_total.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        loss = {
            'loss': loss,
            'loss_op': loss_op,
        }

        output = {
            'states_pred'     : pred,
        }
        return loss, output

    def step(self, batch, backward=True):
        states = batch['states']
        t = batch['t'][0]
        loss, output = self._forward(states, t, backward)
        return loss, output

    def metric(self, states, states_pred, **kwargs):
        metrics = {}
        metrics['param_error'] = statistics.mean(abs(v1-float(v2))/v1 for v1, v2 in zip(self.train.dataset.params.values(), self.net.get_pde_params().values()))
        metrics.update(self.net.get_pde_params())
        metrics.update({f'{k}_real': v for k, v in self.train.dataset.params.items() if k in metrics})
        return metrics

    def run(self):
        # #Load checkpoint
        # epoch, loss_test_min = self.load('./exp/pendulum/2023-11-15 11:50:44.826525/model_1.002e-04.pt')

        # self.visualize()

        # return
        loss_test_min = None
        for epoch in range(self.nepoch): 
            for iteration, data in enumerate(self.train, 0):
                for _ in range(self.niter):
                    _, _, loss, metric = self.train_step(data)

                total_iteration = epoch * (len(self.train)) + (iteration + 1)
                loss_train = loss['loss'].item()
                self.lambda_update(loss_train)

                self.train_history['loss_train']= np.append(self.train_history['loss_train'], np.array([[epoch,loss_train]]), axis=0)

                if total_iteration % self.nlog == 0:
                    self.log(epoch, iteration, loss | metric)
                    if self.display:
                        self.visualize(title=f'epoch: {epoch}/{self.nepoch}, loss: {loss["loss"].item():.3e}')

                if total_iteration % self.nupdate == 0:
                    with torch.no_grad():
                        loss_test = 0.
                        for j, data_test in enumerate(self.test, 0):
                            _, _, loss, metric = self.val_step(data_test)
                            loss_test += loss['loss'].item()
                            
                        loss_test /= j + 1

                        self.train_history['loss_test']=np.append(self.train_history['loss_test'], np.array([[epoch,loss_test]]), axis=0)

                        if loss_test_min == None or loss_test_min > loss_test:
                            loss_test_min = loss_test
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.net.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': loss_test_min, 
                            }, self.path + f'/model_{epoch}_{loss_test_min:.3e}.pt')

                            if self.display:
                                self.visualize(title=f'epoch: {epoch}/{self.nepoch}, loss: {loss_test_min:.3e}',
                                    save_name=f'model_{epoch}_{loss_test_min:.3e}.png')
                                
                        loss_test = {
                            'loss_test': loss_test,
                        }
                        print('#' * 80)
                        self.log(epoch, iteration, loss_test | metric)
                        print(f'lambda: {self._lambda}')
                        print('#' * 80)

    def load(self, path):
        #Load checkpoint
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_test_min = checkpoint['loss'] 
        return epoch, loss_test_min
    
    def visualize(self, title='', save_name=None):
        with torch.no_grad():
            self.evaluating()

            data = next(iter(self.test)) 
            # convert_tensor(data, self.device)
            target = data['states'][0, :, :].unsqueeze(0) # [batch_size, state, time]
            t = data['t'][0]
            y0 = target[:, :, 0]
            # print(f't:{t.device}, y0:{y0.device}')

            pred = self.net(convert_tensor(y0, self.device), convert_tensor(t, self.device))

            #Title of the plot
            if title:
                self.fig.suptitle(title)

            self.ax_traj.cla()
            self.ax_traj.set_title('Trajectories')
            self.ax_traj.set_xlabel('t')
            self.ax_traj.set_ylabel('theta')
            self.ax_traj.plot(t.cpu().numpy(), target.cpu().numpy()[0, 0, :], 'g-', label='target')
            self.ax_traj.plot(t.cpu().numpy(), pred.cpu().numpy()[0, 0, :], 'b--', label='pred')
            self.ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
            self.ax_traj.legend()

            self.ax_spd.cla()
            self.ax_spd.set_title('Velocities')
            self.ax_spd.set_xlabel('t')
            self.ax_spd.set_ylabel('omega')
            self.ax_spd.plot(t.cpu().numpy(), target.cpu().numpy()[0, 1, :], 'g-', label='target')
            self.ax_spd.plot(t.cpu().numpy(), pred.cpu().numpy()[0, 1, :], 'b--', label='pred')
            self.ax_spd.set_xlim(t.cpu().min(), t.cpu().max())
            self.ax_spd.legend()

            self.ax_loss.cla()
            self.ax_loss.set_title('Losses')
            self.ax_loss.set_xlabel('epoch')
            self.ax_loss.set_ylabel('loss')
            self.ax_loss.plot(self.train_history['loss_train'][:,0], self.train_history['loss_train'][:,1], 'g-', label='train')
            self.ax_loss.plot(self.train_history['loss_test'][:,0], self.train_history['loss_test'][:,1], 'r-', label='test')
            self.ax_loss.set_xlim(np.min(self.train_history['loss_train'][:,0]), np.max(self.train_history['loss_train'][:,0]))
            self.ax_loss.set_ylim(0,1e-1)
            self.ax_loss.legend()
 
            if save_name:
                plt.savefig(self.path +f'/png/{save_name}')
            plt.draw()
            plt.pause(0.001)