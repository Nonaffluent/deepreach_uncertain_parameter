# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, training, loss_functions, modules, diff_operators
# import ilqg
# import dynamics
# import quadratic_cost_with_wrapping

import torch
import numpy as np
import scipy
from scipy import linalg
import math
from torch.utils.data import DataLoader
import configargparse
import scipy.io as spio
from scipy.interpolate import RegularGridInterpolator as rgi

# Basic parameters
logging_root = './logs'
fig_suffix = 'narrowMu_diff_false'

experiment = {'name': './deepreach_uncertain_parameter/experiment_scripts/logs/Narrow_mu',
              'env_setting': 'v2', 'speed_setting': 'medium_v2', 'diffModel': False}########***********
checkpoint = 179000

# Scenarios to run for each experiment
R1_speed = 4.0
R2_speed = 3.0
#                 [ x1 ,  y1 , th1, v1      ,phi1,  x2,  y2,   th2   , v2      ,phi2, mu]
xinit = np.array([[-6.0, -1.4, 0.0, R1_speed, 0.0, 6.0, 1.4, -math.pi, R2_speed, 0.0, 1.0]])

# tMax BRS for each scenario
BRS_time = 4.0

# Time horizon for simulation
tMax = 4.0  # Absolute time coordinates
dt = 0.0025 # Absolute time coordinates

# Alpha[x] for normalization
alpha_x = 8.0

# Load the dataset
dataset = dataio.ReachabilityNarrowPassageSource(numpoints=65000, tMax=tMax,
                                                 gx_factor=6.0, 
                                                 speed_setting=experiment['speed_setting'], 
                                                 env_setting=experiment['env_setting'], 
                                                 HJIVI_smoothing_setting='v2', 
                                                 smoothing_exponent=10.0, 
                                                 diffModel=experiment['diffModel'])

# Initialize the model
model = modules.SingleBVPNet(in_features=12, out_features=1, type='sine', mode='mlp', final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
model.cuda()

# Traj colors 
traj_colors = ['r', 'k', 'b']
# Number of slices
num_slices = 1

def angle_normalize(x):
  return (((x + np.pi) % (2 * np.pi)) - np.pi)

def unnormalize_valfunc(valfunc):
  # Unnormalize the value function
  norm_to = 0.02
  mean = 0.25 * alpha_x
  var = 0.5 * alpha_x
  return (valfunc*var/norm_to) + mean 


def add_environment_stuff(ax, dataset, mu):
  alpha = dataset.alpha
  beta = dataset.beta

  ## Plot obstacles
  # Stranded vehicle
  diam_le = 4.0 + 2.0*mu #vehicle length as diameter
  diam_wi = 1.5 + 0.5*mu   #vehicle width as diameter
  obs_ellipse = patches.Ellipse((dataset.stranded_car_pos[0], dataset.stranded_car_pos[1]), diam_le, diam_wi, color='gray', alpha=0.5)
  ax.add_artist(obs_ellipse)
  # Outside radius
  obs_ellipse = patches.Ellipse((dataset.stranded_car_pos[0], dataset.stranded_car_pos[1]), diam_le + dataset.L, diam_wi + dataset.L, color='gray', alpha=0.5)
  ax.add_artist(obs_ellipse)

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

  # Plot the goal region
  goal_circle1 = plt.Circle((dataset.goalX[0], dataset.goalY[0]), dataset.L, color=traj_colors[0], alpha=0.5)
  goal_circle2 = plt.Circle((dataset.goalX[1], dataset.goalY[1]), dataset.L, color=traj_colors[1], alpha=0.5)
  ax.add_artist(goal_circle1)
  ax.add_artist(goal_circle2)

  return ax

def initialize_dataset(experiment):
  dataset = dataio.ReachabilityNarrowPassageSource(numpoints=65000, speed_setting=experiment['speed_setting'], env_setting=experiment['env_setting'], 
                                                    tMax=4.0, ham_version='v1')
  return dataset

def load_model(experiment, checkpoint):
  try:
    root_path = os.path.join(logging_root, experiment['name'])
    ckpt_dir = os.path.join(root_path, 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, 'model_epoch_%04d.pth' % checkpoint)
    checkpoint = torch.load(ckpt_path)
    model_weights = checkpoint['model']
    model.load_state_dict(model_weights)
    model.eval()
    return model, True
  except:
    return model, False


# Dataset and model
dataset = initialize_dataset(experiment)
model, status = load_model(experiment, checkpoint)

# Alphas and betas
alpha = dataset.alpha
beta = dataset.beta

# Create a figure
fig = plt.figure(figsize=(15, 5))

# Time vector
tau = np.arange(0., tMax, dt)
num_timesteps = np.shape(tau)[0]

# Setup the state and control arrays
states = np.zeros((11, num_timesteps))
controls = np.zeros((4, num_timesteps-1))

# Store the values
values = np.zeros((1, num_timesteps-1))

# Initialize the actual trajectories
states[:, 0] = xinit

# Start the trajectory iteration
for k in range(num_timesteps-1):

  print('Time step %i of %i' %(k+1, num_timesteps))

  # Setup the input vector
  coords = torch.ones(1, 12)
  coords[:, 0] = coords[:, 0] * (tMax - tau[k])

  coords[:, 1] = coords[:, 1] * (states[0, k] - dataset.beta['x']) / dataset.alpha['x']
  coords[:, 2] = coords[:, 2] * (states[1, k] - dataset.beta['y']) / dataset.alpha['y']
  coords[:, 3] = coords[:, 3] * (states[2, k] - dataset.beta['th']) / dataset.alpha['th']
  coords[:, 4] = coords[:, 4] * (states[3, k] - dataset.beta['v']) / dataset.alpha['v']
  coords[:, 5] = coords[:, 5] * (states[4, k] - dataset.beta['phi']) / dataset.alpha['phi']

  coords[:, 6] = coords[:, 6] * (states[5, k] - dataset.beta['x']) / dataset.alpha['x']
  coords[:, 7] = coords[:, 7] * (states[6, k] - dataset.beta['y']) / dataset.alpha['y']
  coords[:, 8] = coords[:, 8] * (states[7, k] - dataset.beta['th']) / dataset.alpha['th']
  coords[:, 9] = coords[:, 9] * (states[8, k] - dataset.beta['v']) / dataset.alpha['v']
  coords[:, 10] = coords[:, 10] * (states[9, k] - dataset.beta['phi']) / dataset.alpha['phi']

  coords[:, 11] = coords[:, 11] * states[10, k]

  coords_unnormalized = torch.tensor(states[:, k:k+1]).cuda()

  model_in = {'coords': coords.cuda()}
  model_out = model(model_in)

  # Compute the spatial derivative
  du, status = diff_operators.jacobian(model_out['model_out'], model_out['model_in'])
  dudx_normalized = du[0, :, 0, 1:].detach().cpu().numpy()

  # Store the values
  value = model_out['model_out'].detach().cpu().numpy()
  value = unnormalize_valfunc(value)[0, 0]

  # Account for the diff model
  if experiment['diffModel']:
    coords_var = torch.tensor(coords.clone(), requires_grad=True)
    lx = dataset.compute_IC(coords_var[:, 1:])[2]
    lx_normalized = (lx - dataset.mean)*dataset.norm_to/dataset.var
    lx_grads = diff_operators.gradient(lx_normalized, coords_var)[..., 1:]

    # Add l(x) to the value function
    lx = lx.detach().cpu().numpy()
    value = value + lx[0, 0]

    # Add l(x) gradients to the dudx
    # import ipdb; ipdb.set_trace()
    lx_grads = lx_grads.detach().cpu().numpy()
    dudx_normalized = dudx_normalized + lx_grads

  values[0, k] = value
  dudx_normalized = dudx_normalized[0]
  print(value)

  ## Propagate the state
  # Optimal control computation
  aMin1 = dataset.aMin * (states[3, k] > dataset.vMin)
  aMax1 = dataset.aMax * (states[3, k] < dataset.vMax)
  psiMin1 = dataset.psiMin * (states[4, k] > dataset.phiMin)
  psiMax1 = dataset.psiMax * (states[4, k] < dataset.phiMax)

  aMin2 = dataset.aMin * (states[8, k] > dataset.vMin)
  aMax2 = dataset.aMax * (states[8, k] < dataset.vMax)
  psiMin2 = dataset.psiMin * (states[9, k] > dataset.phiMin)
  psiMax2 = dataset.psiMax * (states[9, k] < dataset.phiMax)

  aOpt1 = (dudx_normalized[..., 3] > 0) * aMin1 + (dudx_normalized[..., 3] <= 0) * aMax1
  psiOpt1 = (dudx_normalized[..., 4] > 0) * psiMin1 + (dudx_normalized[..., 4] <= 0) * psiMax1
  aOpt2 = (dudx_normalized[..., 8] > 0) * aMin2 + (dudx_normalized[..., 8] <= 0) * aMax2
  psiOpt2 = (dudx_normalized[..., 9] > 0) * psiMin2 + (dudx_normalized[..., 9] <= 0) * psiMax2

  opt_ctrl = [aOpt1, psiOpt1, aOpt2, psiOpt2] 

  # Dynamics propagation
  next_state = dataset.propagate_state(coords_unnormalized[:, 0], opt_ctrl, dt)
  states[:, k+1] = next_state.detach().cpu().numpy()
  states[2, k+1] = angle_normalize(states[2, k+1])
  states[4, k+1] = angle_normalize(states[4, k+1])
  states[7, k+1] = angle_normalize(states[7, k+1])
  states[9, k+1] = angle_normalize(states[9, k+1])

# Store the nominal and optimal trajectories 
scenario1_data = {}
scenario1_data['Q1_traj_actual'] = states[:5, :]
scenario1_data['Q2_traj_actual'] = states[5:, :]

# Plot the safe trajectories
ax = fig.add_subplot(1, 2, 1)
alphaMin = 0.0
alphaMax = 1.0
alphas = np.linspace(alphaMax, alphaMin, num_timesteps)
for l in range(0, num_timesteps-1, 2):
  ax.plot(states[0, l:l+2], states[1, l:l+2], traj_colors[0], alpha=alphas[l], linewidth=1.5)
  ax.plot(states[5, l:l+2], states[6, l:l+2], traj_colors[1], alpha=alphas[l], linewidth=1.5)
ax = add_environment_stuff(ax, dataset, xinit[0,10])
ax.set_aspect('equal')
# Plot the distance between the two cars
ax = fig.add_subplot(1, 2, 2)
dist_R1_R2 = np.linalg.norm(states[:2, :] - states[5:7, :], axis=0)
ax.plot(tau, dist_R1_R2 - dataset.L, traj_colors[0])
ax.legend(['d(Q1, Q2)'])

fig.savefig(os.path.join( experiment['name'], 'Traj_plot_' + fig_suffix +'.png'))
# spio.savemat(os.path.join(logging_root, 'Paper_plots', 'NarrowPassage', 'trajectory_data_' + fig_suffix +'.mat'), scenario1_data)