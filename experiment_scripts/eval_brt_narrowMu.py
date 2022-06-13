# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, training, loss_functions, modules

import torch
import numpy as np
import math
from torch.utils.data import DataLoader

checkpoint_toload=179000
# Load the model
model = modules.SingleBVPNet(in_features=12, out_features=1, type='sine', mode='mlp', final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
model.cuda()
root_path = os.path.join('./deepreach_uncertain_parameter/experiment_scripts/logs', 'Narrow_mu')
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


tMax=4.0
dataset = dataio.ReachabilityNarrowPassageSource(numpoints=65000, tMax=tMax,
                                                 gx_factor=6.0, norm_scheme='hack1',
                                                 speed_setting='medium_v2', 
                                                 env_setting='v2', 
                                                 HJIVI_smoothing_setting='v2', 
                                                 smoothing_exponent=10.0)


def plot_brt_val_point(time, unnormalized_state, dataset):
  
  sidelen = 200
  mgrid_coords = dataio.get_mgrid(sidelen)

  alpha = dataset.alpha
  beta = dataset.beta

  time_coords = torch.ones(mgrid_coords.shape[0], 1) * time/alpha['time']

  x_R1_coords = torch.ones(mgrid_coords.shape[0], 1) *  (unnormalized_state[0] - beta['x'])/ alpha['x']
  y_R1_coords = torch.ones(mgrid_coords.shape[0], 1) *  (unnormalized_state[1] - beta['y'])/ alpha['y']
  th_R1_coords = torch.ones(mgrid_coords.shape[0], 1) * (unnormalized_state[2] - beta['th'])/ alpha['th']
  v_R1_coords = torch.ones(mgrid_coords.shape[0], 1) *  (unnormalized_state[3] - beta['v'])/ alpha['v']
  phi_R1_coords = torch.ones(mgrid_coords.shape[0], 1) *(unnormalized_state[4] - beta['phi'])/ alpha['phi']

  x_R2_coords = torch.ones(mgrid_coords.shape[0], 1) *  (unnormalized_state[5] - beta['x'])/ alpha['x']
  y_R2_coords = torch.ones(mgrid_coords.shape[0], 1) *  (unnormalized_state[6] - beta['y'])/ alpha['y']
  th_R2_coords = torch.ones(mgrid_coords.shape[0], 1) * (unnormalized_state[7] - beta['th'])/ alpha['th']
  v_R2_coords = torch.ones(mgrid_coords.shape[0], 1) *  (unnormalized_state[8] - beta['v'])/ alpha['v']
  phi_R2_coords = torch.ones(mgrid_coords.shape[0], 1) *(unnormalized_state[9] - beta['phi'])/ alpha['phi']

  mu_coords = torch.ones(mgrid_coords.shape[0], 1) * (unnormalized_state[10])

  coords_plot = torch.cat((time_coords,mgrid_coords,th_R1_coords,v_R1_coords,phi_R1_coords,x_R2_coords,y_R2_coords,th_R2_coords,v_R2_coords,phi_R2_coords,mu_coords), dim=1) 
  coords_eval = torch.cat((time_coords,x_R1_coords,y_R1_coords,th_R1_coords,v_R1_coords,phi_R1_coords,x_R2_coords,y_R2_coords,th_R2_coords,v_R2_coords,phi_R2_coords,mu_coords), dim=1) 
  
  model_in_plot = {'coords': coords_plot.cuda()}
  model_out_plot = model(model_in_plot)['model_out']

  model_in_eval = {'coords': coords_eval.cuda()}
  model_out_eval = model(model_in_eval)['model_out']

  # Detatch model ouput and reshape
  model_out_plot = model_out_plot.detach().cpu().numpy()
  model_out_plot = model_out_plot.reshape((sidelen, sidelen))

  model_out_eval = model_out_eval.detach().cpu().numpy()
  model_out_eval = model_out_eval.reshape((sidelen, sidelen))

  # Unnormalize the value function
  valfunc = (model_out_plot*dataset.var/dataset.norm_to) + dataset.mean  #dataset multiplied mean and var by x_alpha
  brt = (valfunc <= 0.001)*1.

  valfunc_eval= (model_out_eval*dataset.var/dataset.norm_to) + dataset.mean

  # R position
  R1_pos = [unnormalized_state[0], unnormalized_state[1]]
  R2_pos = [unnormalized_state[5], unnormalized_state[6]]
  mu = unnormalized_state[10]

  fig = plt.figure(figsize=(8, 2*3.8))
  
  ### Plot the zero level sets
  ax = fig.add_subplot(2, 1, 1)
  ax.set_title('t = %0.2f' % (time))
  s = ax.imshow(brt.T, cmap='bwr', origin='lower', extent=(-alpha['x'], alpha['x'], -alpha['y'], alpha['y']), aspect=(alpha['x']/alpha['y']), vmin=-1., vmax=1.)
  ax.set_aspect('equal')
  #fig.colorbar(s) 
  ax = add_environment_stuff(ax, R2_pos, mu, alpha)
  ax.plot(R1_pos[0], R1_pos[1], marker="x", markersize=8, markeredgecolor="black", markerfacecolor="black")
  ax.text(R1_pos[0], R1_pos[1]-0.5, '%.2f' % valfunc_eval[0,0] , ha="center")


  ### Plot the actual value function
  ax_valfunc = fig.add_subplot(2, 1, 2)
  sV1 = ax_valfunc.imshow(valfunc.T, cmap='bwr_r', alpha=0.8, origin='lower', extent=(-alpha['x'], alpha['x'], -alpha['y'], alpha['y']), aspect=(alpha['x']/alpha['y']))
  sV2 = ax_valfunc.contour(valfunc.T, cmap='bwr_r', alpha=0.5, origin='lower', levels=10, extent=(-alpha['x'], alpha['x'], -alpha['y'], alpha['y']))
  plt.clabel(sV2, levels=10, colors='k')
  ax_valfunc.set_aspect('equal')
  #fig_valfunc.colorbar(sV1)
  ax_valfunc = add_environment_stuff(ax_valfunc, R2_pos, mu, alpha)

  return fig

def add_environment_stuff(ax, R2_pos=None, mu=None, alpha=None):
  ## Plot obstacles
  # Stranded vehicle
  diam_le = 4.0 + 2.0*mu #vehicle length as diameter
  diam_wi = 1.5 + 0.5*mu #vehicle width as diameter
  obs_ellipse = patches.Ellipse((dataset.stranded_car_pos[0], dataset.stranded_car_pos[1]), diam_le, diam_wi, color='gray', alpha=0.5)
  ax.add_artist(obs_ellipse)
  # Outside radius
  obs_ellipse = patches.Ellipse((dataset.stranded_car_pos[0], dataset.stranded_car_pos[1]), diam_le + dataset.L, diam_wi + dataset.L, color='gray', alpha=0.5)
  ax.add_artist(obs_ellipse)

  # Second vehicle
  if R2_pos is not None:
    R2_circle = plt.Circle((R2_pos[0], R2_pos[1]), 0.5*dataset.L, color='k', alpha=0.5)
    ax.add_artist(R2_circle)

  if dataset.collision_setting in ['v1', 'v2']:
    # Lower and upper curbs
    rect_lower = patches.Rectangle((-alpha['x'], -alpha['y']), 2*alpha['x'], alpha['y'] - dataset.curb_positions[1], facecolor='gray', alpha=0.5)
    rect_upper = patches.Rectangle((-alpha['x'], dataset.curb_positions[1]), 2*alpha['x'], alpha['y'] - dataset.curb_positions[1], facecolor='gray', alpha=0.5)
    ax.add_patch(rect_lower)
    ax.add_patch(rect_upper)

    # Lower and upper curbs (outside boundary)
    rect_lower = patches.Rectangle((-alpha['x'], -alpha['y']), 2*alpha['x'], alpha['y'] - dataset.curb_positions[1]+0.5*dataset.L, facecolor='gray', alpha=0.5)
    rect_upper = patches.Rectangle((-alpha['x'], dataset.curb_positions[1] - 0.5*dataset.L), 2*alpha['x'], alpha['y'] - dataset.curb_positions[1]+0.5*dataset.L, facecolor='gray', alpha=0.5)
    ax.add_patch(rect_lower)
    ax.add_patch(rect_upper)

  # Lane boundary and axis limits
  ax.axhline(y=0.0, color='k', linestyle='--')
  ax.set_xlim(-alpha['x'], alpha['x'])
  ax.set_ylim(-alpha['y'], alpha['y'])
  
  return ax


#         [x1 , y1  , th1, v1 ,phi1,   x2,  y2,   th2   , v2 ,phi2, mu]
nat_state=[-6.0, -1.0, 0.0, 3.0, 0.0, -6.0, 1.4, -math.pi, 2.0, 0.0, 1.0]
t_plot=4.0
fig2 = plot_brt_val_point(t_plot, nat_state, dataset)
fig2.savefig(os.path.join('./deepreach_uncertain_parameter/experiment_scripts/logs', 'Narrow_brt_.png'))