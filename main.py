import os
import argparse
import time
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
import os
import argparse
import time
import numpy as np


import torch.nn as nn
import torch.optim as optim
from kode.adjoint import _shape_to_flat
from kode.misc import _flat_to_shape

from torchdiffeq import odeint as node_odeint

from kode.adjoint import odeint_adjoint 

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR

   
import logging


parser = argparse.ArgumentParser('KoNODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--niters', type=int, default=200)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default= 1e-4)
parser.add_argument('--delta_t', type=float, default= 0.1)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--w_size', type=int, default=10)
parser.add_argument('--solver', type=str, default='euler')
parser.add_argument('--gamma', type=float, default=0.95, help='Learning rate decay rate for ExponentialLR')
args = parser.parse_args()

device_id = args.gpu
torch.cuda.set_device(device_id)
  

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

torch.set_default_dtype(torch.float64)

np.random.seed(args.seed)
torch.manual_seed(args.seed)


class TrajectoryDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)
        
        self.num_points, self.num_trajectories,  self.dim = self.data.shape

    def __len__(self):
        return self.num_trajectories

    def __getitem__(self, idx):
        
        trajectory = torch.tensor(self.data[:, idx], dtype=torch.float64)
        return trajectory[0].unsqueeze(-2), trajectory.unsqueeze(-2)


train_dataset = TrajectoryDataset('train_data.npy')
test_dataset = TrajectoryDataset('test_data.npy')
val_dataset = TrajectoryDataset('val_data.npy')



train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


def augmented_func(t, y_aug):
    y_tuple = func(t, _shape_to_flat((), *y_aug))
    return y_tuple

    

class ODEFunc(nn.Module):

    def __init__(self, w_size, num_layers=4, hidden_dim=80, input_dim=2, output_dim=2,intermediate_dim=1000):
        super(ODEFunc, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.intermediate_dim = intermediate_dim
        self.theta = None

        self.nfe_forward = 0
        self.nfe_backward = 0
        
        
        theta_size = self.input_dim * self.hidden_dim + self.hidden_dim  
        theta_size_0 = theta_size
        
        for i in range(num_layers - 2):
            theta_size += self.hidden_dim * self.hidden_dim + self.hidden_dim
            if i==0:
                theta_size_1  = self.hidden_dim * self.hidden_dim + self.hidden_dim
            if i==1:
                theta_size_2  = self.hidden_dim * self.hidden_dim + self.hidden_dim
            if i==2:
                theta_size_3  = self.hidden_dim * self.hidden_dim + self.hidden_dim
            if i==3:
                theta_size_4  = self.hidden_dim * self.hidden_dim + self.hidden_dim
            

        theta_size += self.hidden_dim * self.output_dim + self.output_dim
        theta_size_5 = self.hidden_dim * self.output_dim + self.output_dim
       
        self.encoder = nn.Sequential(
            nn.Linear(w_size, 50),
            nn.SiLU(),
            nn.Linear(50,num_layers*50),
            
        )
        
        
        
        self.encoder_0 = nn.Sequential(
            nn.Linear(50, theta_size_0),
            nn.SiLU(),
           
        )
        
        self.encoder_1 = nn.Sequential(
            nn.Linear(50, 50),
            nn.SiLU(),    
            nn.Linear(50,theta_size_1),
           
        )
        
        self.encoder_2 = nn.Sequential(
            nn.Linear(50, 50),
            nn.SiLU(),    
            nn.Linear(50,theta_size_1),
           
        )
        
        
        self.encoder_5 = nn.Sequential(
            nn.Linear(50, theta_size_5),
            nn.SiLU(),
           
        )
        
        
      
        self.ReLU = nn.ReLU()
        self.w_size = w_size

    def update_w(self, w, a):
        n_batch, _, w_size = w.shape 
        
        re = a[...,:w_size//2]
        im = a[...,w_size//2:]
        
        
        w_ = torch.zeros_like(w)
        w_[..., 0::2] = w[..., 1::2]  
        w_[..., 1::2] = w[..., 0::2]
        w_[...,1::2] *= -1
       
        im_result = w_ * torch.repeat_interleave(im, 2 ,dim=-1)
        re_result = w * torch.repeat_interleave(re, 2 ,dim=-1)
        dw_dt = re_result + im_result
        
        
        return dw_dt

    def assign_weights(self, theta):

        start_idx = 0
        weights, biases = [], []
        batch_size = theta.size(0)
        
        weight_numel = self.input_dim * self.hidden_dim
        
        weight = theta[:,:, start_idx:start_idx + weight_numel].view(batch_size,self.input_dim,self.hidden_dim)
        start_idx += weight_numel

        bias_numel = self.hidden_dim
        bias = theta[:, :,start_idx:start_idx + bias_numel].view(batch_size,1,self.hidden_dim)
        start_idx += bias_numel

        weights.append(weight)
        biases.append(bias)

        for _ in range(self.num_layers - 2):
            weight_numel = self.hidden_dim * self.hidden_dim
            weight = theta[:, :,start_idx:start_idx + weight_numel].view(batch_size,self.hidden_dim, self.hidden_dim)
            start_idx += weight_numel

            bias_numel = self.hidden_dim
            bias = theta[:, :,start_idx:start_idx + bias_numel].view(batch_size,1,self.hidden_dim)
            start_idx += bias_numel

            weights.append(weight)
            biases.append(bias)

        weight_numel = self.hidden_dim * self.output_dim
        weight = theta[:,:, start_idx:start_idx + weight_numel].view(batch_size,self.hidden_dim,self.output_dim)
        start_idx += weight_numel

        bias_numel = self.output_dim
        bias = theta[:, :,start_idx:start_idx + bias_numel].view(batch_size,1,self.output_dim)
        start_idx += bias_numel

        weights.append(weight)
        biases.append(bias)

        return weights, biases

    def reset_nfe(self):
        
        self.nfe_forward = 0
        self.nfe_backward = 0
        
    def forward(self, t, y_aug):

        
        self.nfe_forward += 1
        y, w, a = y_aug
        
        dw_dt = self.update_w(w, a)
       
        w_expanded = self.encoder(w.squeeze(-2))
        w_0 = w_expanded[...,:50]
        w_1 = w_expanded[...,50:100]
        w_2 = w_expanded[...,100:150]
        w_5 = w_expanded[...,150:200]
        theta_0 = self.encoder_0(w_0)
        theta_1 = self.encoder_1(w_1)
        theta_2 = self.encoder_2(w_2)
        theta_5 = self.encoder_5(w_5)
        
        theta = torch.cat([theta_0,theta_1,theta_2,theta_5],dim=-1).unsqueeze(-2)
       
        self.theta = theta
        
        weights, biases = self.assign_weights(theta)
        y = torch.matmul(y, weights[0]) + biases[0]
        
        y = F.relu(y)
        
        for i in range(1, self.num_layers - 1):
            y = torch.matmul(y, weights[i]) + biases[i]
            y = F.relu(y)
        
        dy_dt = torch.matmul(y, weights[-1]) + biases[-1]
       
        return (dy_dt, dw_dt, torch.zeros_like(a))

def learning_rate_with_decay(batch_size, batch_denom, niters, boundary_iters, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom
    
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundary_iters] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn

def init_weights(m):
    if isinstance(m, nn.Linear):  
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias) 

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':
    
    ii = 0

    w_size = args.w_size
    
    func = ODEFunc(w_size = w_size).to(device)
    
    
    a0 = torch.zeros((1,1, w_size)).to(device) 
    a = a0.requires_grad_(True)
    
    iter_loss_records = [] 
    forward_times = []
    backward_times = []
    

    optimizer = optim.Adam(func.parameters(),  args.lr)
    optimizer_a = optim.Adam([a], args.lr)
    
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    scheduler_a = ExponentialLR(optimizer_a, gamma=args.gamma)
   
    
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    
    
    best_loss = float('inf')
    


    min_lr = 1e-6
    kwargs = dict(rtol=1e-7, atol=1e-9)
   
    
    for epoch in range(args.niters):
        scheduler.step()
        scheduler_a.step()

        for param_group in optimizer.param_groups:
            if param_group['lr'] < min_lr:
                
                param_group['lr'] = min_lr
                print(f"Epoch {epoch+1}: Learning rate is {min_lr:.6e}")
            else:
                print(f"Epoch {epoch+1}: Learning rate is {param_group['lr']:.6e}")
                
        for param_group in optimizer_a.param_groups:
            if param_group['lr'] < min_lr:
                
                param_group['lr'] = min_lr
                print(f"Epoch {epoch+1}: Learning rate is {min_lr:.6e}")
            else:
                print(f"Epoch {epoch+1}: Learning rate is {param_group['lr']:.6e}")
                
        total_loss = 0
        total_val_loss = 0
       
        epoch_start_time = time.time() 
        
        for batch_idx, (y0, y) in enumerate(train_loader):
            
            func.train()
            
            func.reset_nfe()
           
            
            y0, y = y0.to(device), y.transpose(0, 1).to(device)
           
            t = torch.linspace(0, y.size(0) * 0.1, y.size(0)).to(device)
            
            start_time = time.time()
                
            w0 = torch.randn(args.batch_size, 1, w_size).to(device) * 0.01
                
            pred_y, _, _ = odeint_adjoint(func, (y0, w0, a), t, method=args.solver, **kwargs)
            loss = torch.mean((pred_y - y)**2)
               
            
            forward_time = time.time() - start_time
            
            forward_times.append(forward_time)
            
           
            global_iter = epoch * len(train_loader) + batch_idx  
            
            
           
            print(f"Epoch {epoch+1} | Batch_idx {batch_idx+1}/{len(train_loader)} | Train Loss: {loss.item()}")
            
            start_time = time.time()
            func.reset_nfe() 
           
            optimizer.zero_grad()
            optimizer_a.zero_grad()
                
            loss.backward()
            
            backward_time = time.time() - start_time
            
            backward_times.append(backward_time)
           

            optimizer.step()
            optimizer_a.step()
            
            total_loss += loss.item()
            iter_loss_records.append((global_iter, loss.item()))

        print(f"Epoch {epoch+1}/{args.niters} | Train Loss: {total_loss / len(train_loader):.6f}")
        
    
        func.eval()
       
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (y0, y) in enumerate(val_loader):
                y0, y = y0.to(device), y.transpose(0, 1).to(device)
                t = torch.linspace(0, y.size(0) * args.delta_t, y.size(0)).to(device)
            
                w0 = torch.randn(args.batch_size, 1, w_size).to(device) * 0.01
                pred_y, _, _ = odeint_adjoint(func, (y0, w0, a), t, method=args.solver,**kwargs)
                val_loss = torch.mean((pred_y - y)**2)
                
                global_iter = epoch * len(val_loader) + batch_idx 
                total_val_loss += val_loss.item()
               

        total_val_loss /= len(val_loader)
        
        print(f"Validation Loss: {total_val_loss:.6f}")

        
        if total_val_loss < best_loss:
            best_loss = total_val_loss
            
            torch.save({'func': func.state_dict(), 'a': a.detach().cpu().numpy()}, f'knode_best_model_{args.seed}_{args.w_size}_{args.solver}.pth')

    
    forward_mean = np.mean(forward_times)
    forward_std = np.std(forward_times)
    backward_mean = np.mean(backward_times)
    backward_std = np.std(backward_times)
    
    print(f"  Forward Time - Mean: {forward_mean:.4f}, Std: {forward_std:.4f}")
    print(f"  Backward Time - Mean: {backward_mean:.4f}, Std: {backward_std:.4f}")

    

