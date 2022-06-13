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
import configargparse

p = configargparse.ArgumentParser()

p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=False,help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')
p.add_argument('--num_epochs', type=int, default=180000, help='Number of epochs to train for.')
p.add_argument('--epochs_til_ckpt', type=int, default=1000, help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100, help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'], help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'], help='Whether to use uniform velocity parameter')
p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--pretrain_iters', type=int, default=100000, required=False, help='Number of pretrain iterations')
p.add_argument('--counter_start', type=int, default=-1, required=False, help='Defines the initial time for the curriculul training')
p.add_argument('--counter_end', type=int, default=-1, required=False, help='Defines the linear step for curriculum training starting from the initial time')
p.add_argument('--num_src_samples', type=int, default=10000, required=False, help='Number of source samples at each time step')
p.add_argument('--sampling_bias_ratio', type=float, default=0.0, required=False, help='Sampling bias ratio in the non obstacle region')

p.add_argument('--norm_scheme', type=str, default='hack1', required=False, choices=['hack1', 'hack2'], help='Normalization scheme to be used')
p.add_argument('--speed_setting', type=str, default='medium_v2', required=False, help='The speed setting for the simulation.')
p.add_argument('--env_setting', type=str, default='v2', required=False, help='The environment setting for the simulation.')
p.add_argument('--ham_version', type=str, default='v1', required=False, help='The Hamiltonian version.')
p.add_argument('--collision_setting', type=str, default='v1', required=False, help='The Hamiltonian version.')
p.add_argument('--target_setting', type=str, default='v1', required=False, help='The Hamiltonian version.')
p.add_argument('--clip_value_gradients', action='store_true', default=False, required=False, help='Clip dVdX and dVdT.')
p.add_argument('--curriculum_version', type=str, default='v1', required=False, help='The curriculum training version.')

p.add_argument('--HJIVI_smoothing_setting', type=str, default='v2', required=False, help='Smoothing setting for the HJIVI loss')
p.add_argument('--smoothing_exponent', type=float, default=2.0, required=False, help='Smoothing exponent')
p.add_argument('--diffModel', action='store_true', default=False, required=False, help='Should we train the difference model instead.')
p.add_argument('--minWith', type=str, default='target', required=False, choices=['none', 'zero', 'target'], help='BRS vs BRT computation')
p.add_argument('--gx_factor', type=float, default=1.0, required=False, help='Multiply g(x) by a number to make system care about value in those states.')
p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')
p.add_argument('--pretrain', action='store_true', default=True, required=False, help='Pretrain dirichlet conditions')
p.add_argument('--adjust_relative_grads', action='store_true', default=False, required=False, help='Adjust the relative gradient values.')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--checkpoint_toload', type=int, default=0, help='Checkpoint from which to restart the training.')

p.add_argument('--tMin', type=float, default=0.0, required=False, help='Start time of the simulation')
p.add_argument('--tMax', type=float, default=1.0, required=False, help='End time of simulation')
opt = p.parse_args()


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


dataset = dataio.ReachabilityNarrowPassageSource(numpoints=65000, pretrain=opt.pretrain, tMax=opt.tMax, tMin=opt.tMin,
                                                  counter_start=opt.counter_start, counter_end=opt.counter_end, 
                                                  pretrain_iters=opt.pretrain_iters, norm_scheme=opt.norm_scheme, gx_factor=opt.gx_factor, 
                                                  speed_setting=opt.speed_setting, env_setting=opt.env_setting, target_setting=opt.target_setting,
                                                  collision_setting=opt.collision_setting, clip_value_gradients=opt.clip_value_gradients,
                                                  ham_version=opt.ham_version, sampling_bias_ratio=opt.sampling_bias_ratio, 
                                                  curriculum_version=opt.curriculum_version, HJIVI_smoothing_setting=opt.HJIVI_smoothing_setting, 
                                                  smoothing_exponent=opt.smoothing_exponent, num_src_samples=opt.num_src_samples, diffModel=opt.diffModel)



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