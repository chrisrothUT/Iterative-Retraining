# Code from paper FIX WHEN YOU HAVE ARCHIVE LINK

import numpy as np
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import torch.multiprocessing as mp
import math as m
import random
import os
import sys
import time
import psutil
from itertools import product, permutations, combinations

num_sites = int(sys.argv[1]) #number of lattice sites
batch_size = int(sys.argv[2]) #batch size
learning_rate = float(sys.argv[3]) #learning rate
J2 = float(sys.argv[4]) # J2/J1 

num_batches = 1000 #total number of minibatches for training
max_norm = 1.0 #gradient clipping
hidden_nodes = 128 #number of hidden units

if torch.cuda.is_available(): #run on GPU if available
	device = torch.device("cuda:0")
else:
	device = torch.device("cpu")

class GRU(nn.Module):
	#uses GRU to process information over array [-1,in_channels,num_sites]
	#returns array [-1,out_channels,num_sites,num_sites]

	def __init__(self,in_channels,out_channels,num_sites):
		super(GRU,self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.num_sites = num_sites

		self.input_input = nn.Linear(self.in_channels,self.out_channels,1)
		self.input_hidden = nn.Linear(self.out_channels,self.out_channels,1)
		self.update_input = nn.Linear(self.in_channels,self.out_channels,1)
		self.update_hidden = nn.Linear(self.out_channels,self.out_channels,1)

		self.layer_norm = nn.LayerNorm([self.out_channels]) #Layer norm over hidden state (only along hidden node dimension)
	
	def first_GRU_step(self,input):
		#takes input drive and outputs hidden state
		
		input_gate = torch.tanh(self.input_input(input))
		update_gate = torch.sigmoid(self.update_input(input))

		hidden = input_gate*update_gate
		
		hidden = self.layer_norm(hidden.permute(0,1)).permute(0,1)
		
		return hidden

	def GRU_step(self,hidden,input):
		#takes previous hidden state and input drive and outputs current_hidden state
		
		input_gate = torch.tanh(self.input_input(input) + self.input_hidden(hidden))
		update_gate = torch.sigmoid(self.update_input(input) + self.update_hidden(hidden))

		hidden = hidden*(torch.ones_like(update_gate) - update_gate) + input_gate*update_gate
		
		hidden = self.layer_norm(hidden.permute(0,1)).permute(0,1)
	
		return hidden

	def forward(self,x):
		for site in np.arange(self.num_sites):
			if site == 0:
				hidden = self.first_GRU_step(x[:,:,site])
				full_hidden = hidden.clone().unsqueeze(-1)
			else:
				hidden = self.GRU_step(hidden,x[:,:,site])
				full_hidden = torch.cat((full_hidden,hidden.unsqueeze(-1)),2)

		return full_hidden

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()

		self.gru = GRU(2,hidden_nodes,num_sites)

		#two layer readout for probability and phase of conditional wavefunction 
	
		self.probs_hid1 = nn.Conv1d(hidden_nodes,hidden_nodes,1)
		self.ang_hid1 = nn.Conv1d(hidden_nodes,hidden_nodes,1)
		self.probs_hid2 = nn.Conv1d(hidden_nodes,hidden_nodes,1)
		self.ang_hid2 = nn.Conv1d(hidden_nodes,hidden_nodes,1)

		self.probs = nn.Conv1d(hidden_nodes,2,1) 
		self.sin = nn.Conv1d(hidden_nodes,2,1) 
		self.cos = nn.Conv1d(hidden_nodes,2,1) 

		#after rolling the input backwards make sure the first electron doesn't see the last one 

		self.mask = torch.ones([num_sites]).to(device)
		self.mask[0] = 0
		self.mask = self.mask.unsqueeze(0).unsqueeze(0)

	def forward(self,inp):

		#input is [batch_size, 2, num_sites, num_sites] 
		
		#symmetrize the model by reversing the direction of the input
		inp = torch.cat((inp,inp.flip([2])),0)

		hidden = self.gru(inp)

		hidden = hidden.roll([1],[2])*self.mask

		probs_hidden = F.relu(self.probs_hid1(hidden))
		probs_hidden = F.relu(self.probs_hid2(probs_hidden))
		ang_hidden = F.relu(self.ang_hid1(hidden))
		ang_hidden = F.relu(self.ang_hid2(ang_hidden))

		probs = F.softmax(self.probs(probs_hidden),1)
		sin = self.sin(ang_hidden)		
		cos = self.cos(ang_hidden)	

		phase = torch.atan2(sin,cos)

		prob_wf = torch.sum(torch.log(torch.sum(probs*inp,1)),1)
		phase_wf = torch.sum(torch.sum(phase*inp,1),1)

		phase_symwf = torch.reshape(phase_wf,[2,-1])
		log_sq_symwf = torch.reshape(prob_wf,[2,-1])
	
		log_wf = torch.squeeze(0.5*torch.log(torch.mean(torch.exp(log_sq_symwf-torch.max(log_sq_symwf,0,True)[0]),0)) + 0.5*torch.max(log_sq_symwf,0,True)[0],0)  
		phase_wf = torch.atan2(torch.sum(torch.exp(log_sq_symwf-torch.max(log_sq_symwf,0,True)[0])*torch.sin(phase_symwf),0),torch.sum(torch.exp(log_sq_symwf-torch.max(log_sq_symwf,0,True)[0])*torch.cos(phase_symwf),0))

		return log_wf, phase_wf

	def sample(self):

		inp = torch.ones([batch_size,2,num_sites]).to(device)
		hidden = torch.zeros([batch_size,hidden_nodes,num_sites]).to(device)

		for i in np.arange(num_sites):	
			
			if i > 0:
				if i == 1:
					hidden[:,:,i] = GRU.first_GRU_step(self.gru,inp[:,:,i-1])
				else:
					hidden[:,:,i] = GRU.GRU_step(self.gru,hidden[:,:,i-1],inp[:,:,i-1])

			probs_hidden = F.relu(self.probs_hid1(hidden))
			probs_hidden = F.relu(self.probs_hid2(probs_hidden))

			probs = F.softmax(self.probs(probs_hidden),1)

			thresh = torch.rand(len(inp)).to(device) 
			is_one = (1. + torch.sign(probs[:,1,i] - thresh))/2.
			inp[:,:,i] = torch.cat((torch.unsqueeze(1.-is_one.clone(),1),torch.unsqueeze(is_one.clone(),1)),1)			

		return inp

def local_energy(state,NN,device):

	# returns diagonal contribution to local energy, AFM = \sum_i s^z_i s^z_{i+1}, M = \sum_i s^z_i, as well as neighbors and next neighbors that have matrix elements with state 0.5 and 0.5*J2 respectively

	#state is [num_sites,2] 

	neighbors = []
	next_neighbors = []
	diag_energy = 0

	neighbors.append(state)

	for i in np.arange(num_sites):  
		if i+1 < num_sites:
			if state[1,i] == 1 and state[0,i+1] == 1:
				final_state = state.copy()
				final_state[1,i] = 0
				final_state[0,i+1] = 0
				final_state[0,i] = 1
				final_state[1,i+1] = 1 
				neighbors.append(final_state)	
			if state[0,i] == 1 and state[1,i+1] == 1:
				final_state = state.copy()
				final_state[0,i] = 0
				final_state[1,i+1] = 0
				final_state[1,i] = 1
				final_state[0,i+1] = 1 
				neighbors.append(final_state)	
		if i+2 < num_sites:
			if state[1,i] == 1 and state[0,i+2] == 1:
				final_state = state.copy()
				final_state[1,i] = 0
				final_state[0,i+2] = 0
				final_state[0,i] = 1
				final_state[1,i+2] = 1 
				next_neighbors.append(final_state)	
			if state[0,i] == 1 and state[1,i+2] == 1:
				final_state = state.copy()
				final_state[0,i] = 0
				final_state[1,i+2] = 0
				final_state[1,i] = 1
				final_state[0,i+2] = 1 
				next_neighbors.append(final_state)	

	state = np.sum(state*np.expand_dims(np.asarray([-1,1]),1),0)	
	right_shifted_state = np.roll(state,1)	
	right_shifted_state[0] = 0
	j1_mag = np.sum(right_shifted_state*state) 
	two_right_shifted_state = np.roll(state,2)	
	two_right_shifted_state[:2] = 0
	j2_mag = np.sum(two_right_shifted_state*state) 
	diag_energy = 0.25/num_sites*(j1_mag + J2*j2_mag) #This is the energy contribution from H_ss	

	afm = j1_mag/num_sites
	mag = np.sum(state)/num_sites
	
	return diag_energy, neighbors, next_neighbors, afm, mag

def train_network():
	
	file_ext = str(num_sites) + '_' + str(batch_size) +  '_' + str(learning_rate) + '_' + str(J2) + '.txt'
	loss_file = 'loss_heisenberg_' + file_ext
	info_file = 'info_heisenberg_' + file_ext
	f = open(loss_file,'w')
	f2 = open(info_file,'w')

	NN = Net().to(device)

	# Unactivate this hashtag to load a saved model. As you would do for iterative retraining	
#	NN.load_state_dict(torch.load("model_80_100_0.001_0.0.txt"))

	optimizer = torch.optim.Adam(NN.parameters(), lr=learning_rate)

	# Set the decay of the learning rate
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1/np.sqrt(0.001*epoch + 1))
	
	avg_energy = 0
	avg_magmo = 0
	avg_afm = 0

	for batch in np.arange(num_batches):
	
		with torch.no_grad():	
			batch_states = NN.sample()

		batch_states = batch_states.cpu().detach().numpy()
		
		neighbors = []
		next_neighbors = []
		neighbor_len = []
		next_neighbor_len = []
		mag = []
		afm = []
		diag_energy = []

		for state in batch_states:
			de,nb,nnb,af,mg = local_energy(state,NN,device)
			neighbors.extend(nb)
			next_neighbors.extend(nnb)
			neighbor_len.append(len(nb))
			next_neighbor_len.append(len(nnb))
			mag.append(mg)
			afm.append(af)
			diag_energy.append(de)

		# This parameter chunks up the data so everything fits into memory. Set this as large as possible as you can get away with!
		partition_size = 8000
		num_partitions = (len(neighbors) - 1)// partition_size + 1
		num_n_partitions = (len(next_neighbors) - 1)// partition_size + 1

		log_wfs = np.zeros([0])
		phase_wfs = np.zeros([0])

		for part in np.arange(num_partitions):
			with torch.no_grad():
				lwf, pwf = NN.forward(torch.FloatTensor(np.asarray(neighbors)[int(partition_size*part):int(partition_size*(part+1))]).to(device))
				lwf = lwf.cpu().detach().numpy()
				pwf = pwf.cpu().detach().numpy()
				log_wfs = np.concatenate((log_wfs,lwf),0)
				phase_wfs = np.concatenate((phase_wfs,pwf),0)

		if J2 > 0:
			nlog_wfs = np.zeros([0])
			nphase_wfs = np.zeros([0])
			for part in np.arange(num_n_partitions):
				with torch.no_grad():
					nlwf, npwf = NN.forward(torch.FloatTensor(np.asarray(next_neighbors)[int(partition_size*part):int(partition_size*(part+1))]).to(device))
					nlwf = nlwf.cpu().detach().numpy()
					npwf = npwf.cpu().detach().numpy()
					nlog_wfs = np.concatenate((nlog_wfs,nlwf),0)
					nphase_wfs = np.concatenate((nphase_wfs,npwf),0)

		position = 0
		energies = np.zeros([batch_size],dtype='complex')
		log_wf = np.zeros([batch_size])
		phase_wf = np.zeros([batch_size])

		if J2 > 0:
			nposition = 0 
			for i in np.arange(batch_size):
				log_wf[i] = log_wfs[position]
				phase_wf[i] = phase_wfs[position]
				energies[i] = diag_energy[i] + 0.5*np.sum(np.exp(log_wfs[int(position + 1):int(position + neighbor_len[i])] + 1j*phase_wfs[int(position + 1):int(position + neighbor_len[i])] - log_wf[i] - 1j*phase_wf[i]))/num_sites + 0.5*J2*np.sum(np.exp(nlog_wfs[int(nposition):int(nposition + next_neighbor_len[i])] + 1j*nphase_wfs[int(nposition):int(nposition + next_neighbor_len[i])] - log_wf[i] - 1j*phase_wf[i]))/num_sites  
				position = position + neighbor_len[i]
				nposition = nposition + next_neighbor_len[i]
		else:
			for i in np.arange(batch_size):
				log_wf[i] = log_wfs[position]
				phase_wf[i] = phase_wfs[position]
				energies[i] = diag_energy[i] + 0.5*np.sum(np.exp(log_wfs[int(position + 1):int(position + neighbor_len[i])] + 1j*phase_wfs[int(position + 1):int(position + neighbor_len[i])] - log_wf[i] - 1j*phase_wf[i]))/num_sites 
				position = position + neighbor_len[i]

		energies = np.nan_to_num(np.asarray(energies))
		log_wf = np.nan_to_num(np.asarray(log_wf))
		phase_wf = np.nan_to_num(np.asarray(phase_wf))
		afm = np.nan_to_num(np.asarray(afm))
		mag = np.nan_to_num(np.asarray(mag))
		mean_energies = np.mean(energies)
		mean_entropies = np.mean(2*log_wf)/num_sites
		mean_afm = np.mean(afm)
		mean_magmo = np.mean(np.abs(mag))

		residuals = np.conj(energies - mean_energies)

		real_residuals = np.real(residuals)			
		imag_residuals = np.imag(residuals)			
		entropy_residuals = (2*log_wf/(num_sites) + np.log(2))
		mag_residuals = np.square(mag)

		optimizer.zero_grad()
		log_wf,phase_wf = NN.forward(torch.FloatTensor(batch_states).to(device))
		# Set T = 0 if you just want to learn a small model fast
#		T = 0
		T = 1./(1 + 0.001*batch)		
		C = 100.

		loss = torch.sum(log_wf*torch.FloatTensor(real_residuals + T*entropy_residuals + C*mag_residuals).to(device)-phase_wf*torch.FloatTensor(imag_residuals).to(device))
		loss.backward()
		for param in NN.parameters():
			param.grad[torch.isnan(param.grad)] = 0
		norm = torch.nn.utils.clip_grad_norm_(NN.parameters(),max_norm)
		optimizer.step()
		scheduler.step()

		#info file keeps track of gradient norm, M, AFM, pseudo-entropy, total cost function, energy
		f2.write(str(norm))
		f2.write('\t')
		f2.write(str(mean_magmo))
		f2.write('\t')
		f2.write(str(mean_afm))
		f2.write('\t')
		f2.write(str(mean_entropies))
		f2.write('\t')
		f2.write(str(mean_energies + T*mean_entropies + C*np.mean(np.square(mag))))
		f2.write('\t')
		f2.write(str(mean_energies))
		f2.write('\n')
		f2.flush()

		avg_energy = avg_energy + mean_energies
		avg_afm = avg_afm + mean_afm
		avg_magmo = avg_magmo + mean_magmo

		if not (batch + 1) % 100:
			f.write(str(avg_energy/100.))
			f.write('\t')
			f.write(str(avg_magmo/100.))
			f.write('\t')
			f.write(str(avg_afm/100.))
			f.write('\n')
			f.flush()
			avg_energy = 0 
			avg_magmo = 0 
			avg_afm = 0 
		
			# save the model every 100 batches
			torch.save(NN.state_dict(), 'model_load_' + file_ext)

train_network()
