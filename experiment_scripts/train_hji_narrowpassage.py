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
p.add_argument('--experiment_name', type=str, required=True,help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')
p.add_argument('--num_epochs', type=int, default=180000, help='Number of epochs to train for.')
p.add_argument('--epochs_til_ckpt', type=int, default=1000, help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100, help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'], help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'], help='Whether to use uniform velocity parameter')
p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--pretrain_iters', type=int, default=10000, required=False, help='Number of pretrain iterations')
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

# Set the source coordinates for the target set and the obstacle sets
source_coords = [0., 0., 0.]
if opt.counter_start == -1:
  opt.counter_start = opt.checkpoint_toload

if opt.counter_end == -1:
  opt.counter_end = opt.num_epochs

dataset = dataio.ReachabilityNarrowPassageSource(numpoints=65000, pretrain=opt.pretrain, tMax=opt.tMax, tMin=opt.tMin,
                                                  counter_start=opt.counter_start, counter_end=opt.counter_end, 
                                                  pretrain_iters=opt.pretrain_iters, norm_scheme=opt.norm_scheme, gx_factor=opt.gx_factor, 
                                                  speed_setting=opt.speed_setting, env_setting=opt.env_setting, target_setting=opt.target_setting,
                                                  collision_setting=opt.collision_setting, clip_value_gradients=opt.clip_value_gradients,
                                                  ham_version=opt.ham_version, sampling_bias_ratio=opt.sampling_bias_ratio, 
                                                  curriculum_version=opt.curriculum_version, HJIVI_smoothing_setting=opt.HJIVI_smoothing_setting, 
                                                  smoothing_exponent=opt.smoothing_exponent, num_src_samples=opt.num_src_samples, diffModel=opt.diffModel)
dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

model = modules.SingleBVPNet(in_features=11, out_features=1, type=opt.model, mode=opt.mode,
                                final_layer_factor=1., hidden_features=512, num_hidden_layers=opt.num_hl)
model.cuda()

# Define the loss
loss_fn = loss_functions.initialize_hji_narrowpassage(dataset, opt.minWith, opt.diffModel)

root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Alphas and betas
alpha = dataset.alpha
beta = dataset.beta

# Level to plot
level = 0.001

# Define the validation function
def val_fn_BRS(model):
  # Time values at which the function needs to be plotted
  times = np.array([0., 0.25, 0.50, 0.75, 1.0])*opt.tMax
  num_times = np.shape(times)[0]

  # Slices to be plotted (x-y of the first vehicle)
  slices = np.array([[0.0, 0.0, 0.0, -6.0, 1.4, -math.pi, 0.0, 0.0], 
                     [0.0, 3.0, 0.0, -6.0, 1.4, -math.pi, 2.0, 0.0], 
                     [0.0, 6.5, 0.0, -6.0, 1.4, -math.pi, 4.0, 0.0], 
                     [0.0, 4.0, 0.0, 6.0, 1.4, -math.pi, 3.0, 0.0],
                     [0.0, 1.0, 0.0, 6.0, 1.4, -math.pi, 3.0, 0.0]])
  num_slices = slices.shape[0]

  # Create figures
  fig = plt.figure(figsize=(5*num_slices, 5*num_times))
  fig_valfunc = plt.figure(figsize=(5*num_slices, 5*num_times))

  # Get the meshgrid in the (x, y) coordinate
  sidelen = 200
  mgrid_coords = dataio.get_mgrid(sidelen)

  # Start plotting the results
  for i in range(num_times):
    time_coords = torch.ones(mgrid_coords.shape[0], 1) * times[i]/alpha['time']

    for j in range(num_slices):
      th_R1_coords = torch.ones(mgrid_coords.shape[0], 1) * (slices[j, 0] - beta['th'])/ alpha['th']
      v_R1_coords = torch.ones(mgrid_coords.shape[0], 1) * (slices[j, 1] - beta['v'])/ alpha['v']
      phi_R1_coords = torch.ones(mgrid_coords.shape[0], 1) * (slices[j, 2] - beta['phi'])/ alpha['phi']

      x_R2_coords = torch.ones(mgrid_coords.shape[0], 1) * (slices[j, 3] - beta['x'])/ alpha['x']
      y_R2_coords = torch.ones(mgrid_coords.shape[0], 1) * (slices[j, 4] - beta['y'])/ alpha['y']
      th_R2_coords = torch.ones(mgrid_coords.shape[0], 1) * (slices[j, 5] - beta['th'])/ alpha['th']
      v_R2_coords = torch.ones(mgrid_coords.shape[0], 1) * (slices[j, 6] - beta['v'])/ alpha['v']
      phi_R2_coords = torch.ones(mgrid_coords.shape[0], 1) * (slices[j, 7] - beta['phi'])/ alpha['phi']

      coords = torch.cat((time_coords, mgrid_coords, th_R1_coords, v_R1_coords, phi_R1_coords, x_R2_coords, y_R2_coords, th_R2_coords, v_R2_coords, phi_R2_coords), dim=1) 
      model_in = {'coords': coords.cuda()}
      model_out = model(model_in)['model_out']

      # Detatch model ouput and reshape
      model_out = model_out.detach().cpu().numpy()
      model_out = model_out.reshape((sidelen, sidelen))

      # Unnormalize the value function
      valfunc = (model_out*dataset.var/dataset.norm_to) + dataset.mean 

      # Account for the diff model
      if opt.diffModel:
        lx = dataset.compute_IC(coords[..., 1:])[2]
        lx = lx.detach().cpu().numpy()
        lx = lx.reshape((sidelen, sidelen))
        valfunc = valfunc + lx

      brt = (valfunc <= level)*1.

      # R2 position
      R2_pos = [slices[j, 3], slices[j, 4]]

      ### Plot the zero level sets
      ax = fig.add_subplot(num_times, num_slices, (j+1) + i*num_slices)
      ax.set_title('t = %0.2f, slice = %i' % (times[i], j+1))
      s = ax.imshow(brt.T, cmap='bwr', origin='lower', extent=(-alpha['x'], alpha['x'], -alpha['y'], alpha['y']), aspect=(alpha['x']/alpha['y']), vmin=-1., vmax=1.)
      fig.colorbar(s) 
      ax = add_environment_stuff(ax, R2_pos)

      ### Plot the actual value function
      ax_valfunc = fig_valfunc.add_subplot(num_times, num_slices, (j+1) + i*num_slices)
      sV1 = ax_valfunc.imshow(valfunc.T, cmap='bwr_r', alpha=0.8, origin='lower', extent=(-alpha['x'], alpha['x'], -alpha['y'], alpha['y']), aspect=(alpha['x']/alpha['y']))
      sV2 = ax_valfunc.contour(valfunc.T, cmap='bwr_r', alpha=0.5, origin='lower', levels=10, extent=(-alpha['x'], alpha['x'], -alpha['y'], alpha['y']))
      plt.clabel(sV2, levels=10, colors='k')
      fig_valfunc.colorbar(sV1)
      ax_valfunc = add_environment_stuff(ax_valfunc, R2_pos)

  return fig, fig_valfunc


def add_environment_stuff(ax, R2_pos=None):
  ## Plot obstacles
  # Stranded vehicle
  obs_ellipse = patches.Ellipse((dataset.stranded_car_pos[0], dataset.stranded_car_pos[1]), dataset.le, dataset.wi, color='gray', alpha=0.5)
  ax.add_artist(obs_ellipse)

  # Outside radius
  obs_ellipse = patches.Ellipse((dataset.stranded_car_pos[0], dataset.stranded_car_pos[1]), dataset.le + 0.5*dataset.L, dataset.wi + 0.5*dataset.L, color='gray', alpha=0.5)
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


def val_fn(model, ckpt_dir, epoch):
  # Run the validation of sets
  fig, fig_valfunc = val_fn_BRS(model)
  fig.savefig(os.path.join(ckpt_dir, 'BRS_validation_plot_epoch_%04d.png' % epoch))
  fig_valfunc.savefig(os.path.join(ckpt_dir, 'Value_function_validation_plot_epoch_%04d.png' % epoch))

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, clip_grad=opt.clip_grad,
               use_lbfgs=opt.use_lbfgs, validation_fn=val_fn, start_epoch=opt.checkpoint_toload, 
               adjust_relative_grads=opt.adjust_relative_grads)