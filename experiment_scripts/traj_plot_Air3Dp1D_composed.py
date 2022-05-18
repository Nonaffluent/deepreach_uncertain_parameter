# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, training, loss_functions, modules, diff_operators

import torch
import numpy as np
import scipy
import math
from torch.utils.data import DataLoader
import configargparse
import scipy.io as spio
from scipy.interpolate import RegularGridInterpolator as rgi
from matplotlib.gridspec import GridSpec

# Basic parameters
logging_root = './deepreach_uncertain_parameter/experiment_scripts/logs'
angle_alpha = 1.2
speed = 0.75
omega_b = 3.0
collisionR = 0.25

# Checkpoint to load for the trajectory plots
experiment_name = '3Dp1D_u0'
# checkpoints_toload = [90000, 100000, 110000, 119000]
# query_times = [0.8, 0.9, 1.0, 1.1]

checkpoints_toload = [119000]
query_times = [1.0]

# Initial state
xinit = np.array([0.5, -0.25, 0.5*math.pi, 5.0])#[x_r,y_r,theta_r,w_a]
xinit_a = np.array([0, 0, 0])#x_a,y_a,theta_a
xinit_b = xinit_a + xinit[0:3] #x_b,y_b,theta_b

# Simulation time
tMax = 1.0
dt = 0.0025

# Normalization parameters
norm_to = 0.02
mean = 0.25
var = 0.5

# Time vector
num_timesteps = int(tMax/dt)+1
t_omega_change=np.array([0, 0.1, 0.25])#times when omega_a changes
omega_t=np.array([xinit[3], 2.5, xinit[3]])#omega_a values
k_t=np.zeros(len(omega_t)) #used to store the iteration of the change

# Number of cases to simulate
num_ckpts = len(checkpoints_toload)
num_query_times = len(query_times)

# Normalize the angle between [-pi, pi]
def angle_normalize(x):
  return (((x + np.pi) % (2 * np.pi)) - np.pi)

def omega_normalize(x):
  return ((x-4.0)/1.5)

# Unnormalize the value function
def unnormalize_valfunc(valfunc):
  return (valfunc*var/norm_to) + mean

# Compute the next state
def propagate_state(state, control, disturbance):
  state_next = state + dt*np.array([-speed + speed*np.cos(state[2]) + control[0]*state[1], speed*np.sin(state[2]) - control[0]*state[0], disturbance[0] - control[0], 0])
  state_next[2] = angle_normalize(state_next[2])
  return state_next

def propagate_state_a(state, control):
  state_next = state + dt*np.array( [speed*np.cos(state[2]), speed*np.sin(state[2]), control[0]])
  state_next[2] = angle_normalize(state_next[2])
  return state_next

def propagate_state_b(state, disturbance):
  state_next = state + dt*np.array( [speed*np.cos(state[2]), speed*np.sin(state[2]), disturbance[0]])
  state_next[2] = angle_normalize(state_next[2])
  return state_next


#evaluate val fcn at coords and time
def compute_v(coords, time):
  coords_local = coords * 1.0
  coords_local[..., 0] = coords_local[..., 0] * time

  # Compute the value function
  model_in = {'coords': coords_local.cuda()}
  model_out = model(model_in)['model_out'].detach().cpu().numpy()
  model_out = unnormalize_valfunc(model_out)
  return model_out[0, 0] 


def compute_tEarliest(coords,query_time):
  eps = 0.0001
  upper = query_time
  lower = 0.0

  while (upper - lower) > dt:
    tEarliest = 0.5*(upper + lower)
    valueAtX = compute_v(coords, tEarliest)

    if valueAtX < eps:
  		# Point is in reachable set; eliminate all upper indices
      upper = tEarliest	
    else:
  		# too late
      lower = tEarliest + dt

  tEarliest = lower
  return tEarliest


# Load the model
model = modules.SingleBVPNet(in_features=5, out_features=1, type='sine', mode='mlp', final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
model.cuda()
root_path = os.path.join(logging_root, experiment_name)
ckpt_dir = os.path.join(root_path, 'checkpoints')
ckpt_path = os.path.join(ckpt_dir, 'model_epoch_%04d.pth' % checkpoints_toload[0])
checkpoint = torch.load(ckpt_path)
try:
  model_weights = checkpoint['model']
except:
  model_weights = checkpoint  
model.load_state_dict(model_weights)
model.eval()

# Initialize the trajectory
state_traj = np.zeros((num_timesteps, 4))
state_traj_a = np.zeros((num_timesteps, 3))
state_traj_b = np.zeros((num_timesteps, 3))
dVdX = np.zeros((num_timesteps, 4))
ctrls = np.zeros((num_timesteps, 1))
dstbs = np.zeros((num_timesteps, 1))
ctrl_dets = np.zeros((num_timesteps, 1))
dstb_dets = np.zeros((num_timesteps, 1))
state_traj[0] = xinit
state_traj_a[0] = xinit_a
state_traj_b[0] = xinit_b

# Start the trajectory iteration
for k in range(num_timesteps-1):
  # Setup the input vector
  coords = torch.ones(1, 5)#(t,x,y,th,wa)
  coords[0, 1] = coords[0, 1] * state_traj[k, 0]
  coords[0, 2] = coords[0, 2] * state_traj[k, 1]
  coords[0, 3] = coords[0, 3] * state_traj[k, 2]/(math.pi * angle_alpha)
  coords[0, 4] = coords[0, 4] * omega_normalize(state_traj[k, 3]) 

  tEarliest = compute_tEarliest(coords,query_times[0])
  coords[0, 0] = coords[0, 0] * tEarliest   

  # Compute the value function
  model_in = {'coords': coords.cuda()}
  # model_in = {'coords': coords}
  model_out = model(model_in)

  # Compute the spatial derivative of the value function
  du, status = diff_operators.jacobian(model_out['model_out'], model_out['model_in'])
  dudx = du[0, :, 0, 1:]
  dudx = (var/norm_to)*dudx.detach().cpu().numpy()
  dudx[..., 2] = dudx[..., 2] / (angle_alpha * math.pi)
  dVdX[k] = dudx[0]

  # Compute the optimal control
  det = dudx[..., 0] * state_traj[k, 1] - dudx[..., 1] * state_traj[k, 0] - dudx[..., 2]#p1*y-p2*x-p3
  ctrls[k, 0] = state_traj[k, 3] * np.sign(det)
  ctrl_dets[k, 0] = det

  # Compute the optimal disturbance
  dstbs[k, 0] = -omega_b * np.sign(dudx[..., 2])
  dstb_dets[k, 0] = dudx[..., 2]

  # Propagate the state
  state_traj[k+1] = propagate_state(state_traj[k], ctrls[k], dstbs[k])
  state_traj_a[k+1] = propagate_state_a(state_traj_a[k], ctrls[k])
  state_traj_b[k+1] = propagate_state_b(state_traj_b[k], dstbs[k])

  n = np.argwhere(t_omega_change==k*dt)#if this k corresponds to an omega change instant 
  if n.size: #n is non empty
    state_traj[k+1,3]=omega_t[n[0]]#change the omega_a in the state vector
    k_t[n[0]]= k #save the iteration that corresponds to this time

# Setting up the plot surface
fig = plt.figure(figsize=(10, 12))
gs = GridSpec(nrows=6, ncols=5)

#Plot relative position state
ax = fig.add_subplot(gs[0:3,0:3])
s = ax.plot(state_traj[:, 0], state_traj[:, 1], 'r')
#Plot start and omega change points
for m in range(len(k_t)):
  ax.plot(state_traj[k_t[m].astype(int), 0], state_traj[k_t[m].astype(int), 1], marker="o", markersize=5, markeredgecolor="k", markerfacecolor="r")
# Plot the target set
circle = plt.Circle(np.array([0, 0]), collisionR, color='k', fill=False, linestyle='--')
ax.add_artist(circle)
# Set the axes limits and title
ax.set_title('ckpt = %iK, t = %0.2f' % (checkpoints_toload[0]/1000 - 10, query_times[0]))
ax.set_xlim(-1., 1.)
ax.set_ylim(-1., 1.)


#Plot absolute position states
ax1 = fig.add_subplot(gs[3:6,0:3])
s1 = ax1.plot(state_traj_a[:, 0], state_traj_a[:, 1], 'b', zorder=1)
s1 = ax1.plot(state_traj_b[:, 0], state_traj_b[:, 1], 'r', zorder=1)
ax1.set_xlim(-1., 1.)
ax1.set_ylim(-1., 1.)

n_data_points=len(state_traj[:, 0])
arrow_steps=5
step_size=round(n_data_points/arrow_steps)
for l in range(arrow_steps):
  c1=step_size*l
  c2=step_size*l + 20
  ac=(0.1+0.05*l, 0.6-0.10*l, 0.1+0.05*l)#arrowcolor
  plt.arrow(state_traj_a[c1, 0], state_traj_a[c1, 1], state_traj_a[c2, 0]-state_traj_a[c1, 0], state_traj_a[c2, 1]-state_traj_a[c1, 1],
  ec=ac, fc=ac, alpha=1.0, width=.02,head_width=.04, head_length=.05,length_includes_head=True,zorder=2)
  plt.arrow(state_traj_b[c1, 0], state_traj_b[c1, 1], state_traj_b[c2, 0]-state_traj_b[c1, 0], state_traj_b[c2, 1]-state_traj_b[c1, 1],
  ec=ac, fc=ac, alpha=1.0, width=.02,head_width=.04, head_length=.05,length_includes_head=True,zorder=2)
  # Plot the target set
  circle = plt.Circle(np.array([state_traj_a[c1, 0], state_traj_a[c1, 1]]), collisionR, color=ac, fill=False, linestyle='--', alpha=.7)
  ax1.add_artist(circle)



#plot BRT when omega_a changes
for i in range(len(k_t)):
  # Get the meshgrid in the (x, y) coordinate
  sidelen = 200
  proj_over=np.array([state_traj[(k_t[i]+1).astype(int), 3] ,state_traj[k_t[i].astype(int), 2], (tMax-t_omega_change[i])] )#(wa,th,t)
  mgrid_coords = dataio.get_mgrid(sidelen)
  omega_a_coords = torch.ones(mgrid_coords.shape[0], 1) * proj_over[0] 
  omega_a_coords = (omega_a_coords - 4) / 1.5        #scale back to (-1,+1)
  theta_coords = torch.ones(mgrid_coords.shape[0], 1) * proj_over[1]  
  theta_coords = theta_coords / (angle_alpha * math.pi)
  time_coords = torch.ones(mgrid_coords.shape[0], 1) * proj_over[2]  
  coords = torch.cat((time_coords, mgrid_coords, theta_coords, omega_a_coords), dim=1) 
  model_in = {'coords': coords.cuda()}
  model_out = model(model_in)['model_out']
  # Detatch model ouput and reshape
  model_out = model_out.detach().cpu().numpy()
  model_out = model_out.reshape((sidelen, sidelen))
  # Unnormalize the value function
  norm_to = 0.02
  mean = 0.25
  var = 0.5
  model_out = (model_out*var/norm_to) + mean
  # Plot the zero level sets
  model_out = (model_out <= 0.001)*1.
  # Plot the actual data
  ax0 = fig.add_subplot(gs[2*i:2*(i+1),3:5])
  ax0.set_title('Wa=%0.1f, th=%0.2f, t=%0.1f ' % (proj_over[0] , proj_over[1] ,proj_over[2] ), fontsize=7)
  ax0.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    labelbottom=False,labelleft=False)
  s = ax0.imshow(model_out.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
  ax0.plot(state_traj[k_t[i].astype(int), 0], state_traj[k_t[i].astype(int), 1], marker="o", markersize=5, markeredgecolor="k", markerfacecolor="w")
  x_grid_pos=(state_traj[k_t[i].astype(int), 0] + 1)*sidelen/2
  y_grid_pos=(state_traj[k_t[i].astype(int), 1] + 1)*sidelen/2
  print(x_grid_pos.astype(int),y_grid_pos.astype(int),model_out[x_grid_pos.astype(int),y_grid_pos.astype(int)])#xy coord in 200x200 grid and val fcn there
  fig.colorbar(s) 

# Run the program
fig.savefig(os.path.join('./deepreach_uncertain_parameter/', 'Traj_plots', 'Traj_comp.png'))
# Save the trajectory data
data_dict ={}
data_dict['state_traj'] = state_traj
data_dict['dVdX'] = dVdX
data_dict['ctrls'] = ctrls
data_dict['dstbs'] = dstbs
data_dict['ctrl_dets'] = ctrl_dets
data_dict['dstb_dets'] = dstb_dets
spio.savemat(os.path.join('./deepreach_uncertain_parameter/', 'Traj_plots', 'traj_data_deepreach_Air3Dp1d.mat'), data_dict)