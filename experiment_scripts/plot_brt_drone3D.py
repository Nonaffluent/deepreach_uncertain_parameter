# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, training, loss_functions, modules

import torch
import numpy as np
import math
from torch.utils.data import DataLoader
import configargparse

checkpoint_toload=119000
# Load the model
model = modules.SingleBVPNet(in_features=4, out_features=1, type='sine', mode='mlp', final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
model.cuda()
root_path = os.path.join('./deepreach_uncertain_parameter/experiment_scripts/logs', 'Drone3D_no_x_dbar')
ckpt_dir = os.path.join(root_path, 'checkpoints')
ckpt_path = os.path.join(ckpt_dir, 'model_epoch_%04d.pth' % checkpoint_toload)
checkpoint = torch.load(ckpt_path)
# checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
try:
  model_weights = checkpoint['model']
except:
  model_weights = checkpoint
model.load_state_dict(model_weights)
model.eval()

angle_alpha=1.2
epoch=checkpoint_toload
tMax=0.5
omega_max=80.0

# times to plot
times = [0., 0.5*tMax, tMax]
num_times = len(times)

# Theta slices to be plotted
thetas = [-math.pi, -0.5*math.pi, 0., 0.5*math.pi, math.pi]
thetas = np.array(thetas) 
num_thetas = len(thetas)
# Create a figure
fig = plt.figure(figsize=(5*num_thetas, 5*num_times))

# Get the meshgrid in the (x, y) coordinate
sidelen = 200
mgrid_coords = dataio.get_mgrid(sidelen)

#dbar=6.0
# Start plotting the results
for i in range(num_times):
    time_coords = torch.ones(mgrid_coords.shape[0], 1) * times[i]

    for j in range(num_thetas):
      theta_coords = torch.ones(mgrid_coords.shape[0], 1) * thetas[j]
      theta_coords = theta_coords / (1* math.pi)

      coords = torch.cat((time_coords, mgrid_coords, theta_coords), dim=1) 
      model_in = {'coords': coords.cuda()}
      model_out = model(model_in)['model_out']

      # Detatch model ouput and reshape
      model_out = model_out.detach().cpu().numpy()
      model_out = model_out.reshape((sidelen, sidelen))

      # Unnormalize the value function
      norm_to = 0.02
      mean = 0.7
      var = 0.9
      model_out = (model_out*var/norm_to) + mean 

      # Plot the zero level sets
      model_out = (model_out <= 0.001)*1.      

      # Plot the actual data
      ax = fig.add_subplot(num_times, num_thetas, (j+1) + i*num_thetas)
      ax.set_title('t = %0.2f, theta = %0.2f' % (times[i], thetas[j]))
      s = ax.imshow(model_out.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
      fig.colorbar(s) 

#summ_dir = os.path.join(root_path, 'summaries')
fig.savefig(os.path.join('./deepreach_uncertain_parameter/experiment_scripts/logs', 'Drone3D_no_x_dbar.png'))