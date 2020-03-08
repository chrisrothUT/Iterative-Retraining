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

num_sites = int(sys.argv[1]) #length of lattice (number of electrons is num_sites*num_sites)
batch_size = int(sys.argv[2]) #batch size
learning_rate = float(sys.argv[3]) 
J2 = float(sys.argv[4]) #J2/J1

num_batches = 10000
max_norm = 1.0 #gradient clipping
hidden_nodes = 128
num_layers = 5 #number of layers 

if torch.cuda.is_available():
	device = torch.device("cuda:0")
else:
	device = torch.device("cpu")

class LSTM(nn.Module):
	#process information over array [-1,in_channels,num_sites,num_sites] from left to right (over last index)
	#returns array [-1,out_channels,num_sites,num_sites]

	def __init__(self,in_channels,out_channels,num_sites):
		super(LSTM,self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.num_sites = num_sites

		self.input_input = nn.Conv1d(self.in_channels,self.out_channels,1)
		self.input_hidden = nn.Conv1d(self.out_channels,self.out_channels,1)
		self.forget_input = nn.Conv1d(self.in_channels,self.out_channels,1)
		self.forget_hidden = nn.Conv1d(self.out_channels,self.out_channels,1)
		self.cell_input = nn.Conv1d(self.in_channels,self.out_channels,1)
		self.cell_hidden = nn.Conv1d(self.out_channels,self.out_channels,1)
		self.output_input = nn.Conv1d(self.in_channels,self.out_channels,1)
		self.output_hidden = nn.Conv1d(self.out_channels,self.out_channels,1)

		self.layer_norm = nn.LayerNorm([self.out_channels])
		self.layer_norm_cell = nn.LayerNorm([self.out_channels])

	def first_LSTM_step(self,input):
		#takes input drive and outputs hidden state
		
		input_gate = torch.sigmoid(self.input_input(input))
		cell_gate = torch.tanh(self.cell_input(input))
		output_gate = torch.sigmoid(self.output_input(input))

		cell = input_gate*cell_gate
		hidden = torch.tanh(cell)*output_gate 

		hidden = self.layer_norm(hidden.permute(0,2,1)).permute(0,2,1)
		cell = self.layer_norm_cell(cell.permute(0,2,1)).permute(0,2,1)
		
		return hidden, cell

	def LSTM_step(self,hidden,cell,input):
		#takes previous hidden state and input drive and outputs current_hidden state
		
		input_gate = torch.sigmoid(self.input_input(input) + self.input_hidden(hidden))
		forget_gate = torch.sigmoid(self.forget_input(input) + self.forget_hidden(hidden))
		cell_gate = torch.tanh(self.cell_input(input) + self.cell_hidden(hidden))
		output_gate = torch.sigmoid(self.output_input(input) + self.output_hidden(hidden))
		
		cell = cell*forget_gate + input_gate*cell_gate
		hidden = torch.tanh(cell)*output_gate 
	
		hidden = self.layer_norm(hidden.permute(0,2,1)).permute(0,2,1)
		cell = self.layer_norm_cell(cell.permute(0,2,1)).permute(0,2,1)
	
		return hidden,cell

	def forward(self,x):
		for site in np.arange(self.num_sites):
			if site == 0:
				hidden, cell = self.first_LSTM_step(x[:,:,:,site])
				full_hidden = hidden.clone().unsqueeze(-1)
			else:
				hidden, cell = self.LSTM_step(hidden,cell,x[:,:,:,site])
				full_hidden = torch.cat((full_hidden,hidden.unsqueeze(-1)),3)

		return full_hidden

class Layer(nn.Module):
	def __init__(self,layer_num):
		super(Layer,self).__init__()

		self.layer_num = layer_num

		if self.layer_num == 0:
			self.top_down = LSTM(2,hidden_nodes,num_sites)
		else:
			self.top_down = LSTM(hidden_nodes,hidden_nodes,num_sites)

		self.left_right = LSTM(hidden_nodes,hidden_nodes,num_sites)
		self.right_left = LSTM(hidden_nodes,hidden_nodes,num_sites)

		self.W1 = nn.Conv2d(2*hidden_nodes,4*hidden_nodes,1)
		self.W2 = nn.Conv2d(4*hidden_nodes,hidden_nodes,1)

		self.top_mask = torch.ones([num_sites]).to(device)
		self.top_mask[0] = 0
		self.top_mask = self.top_mask.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
		
		self.left_mask = torch.ones([num_sites]).to(device)
		self.left_mask[0] = 0
		self.left_mask = self.left_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

		self.layer_norm = nn.LayerNorm([hidden_nodes])

	def forward(self,x):

		hidden = self.top_down(x.permute(0,1,3,2)).permute(0,1,3,2)
		hidden_lr = self.left_right(hidden)
		hidden_rl = self.right_left(hidden.flip([3])).flip([3])
		
		hidden_rl = hidden_rl.roll([1],[2])*self.top_mask
		hidden_lr = hidden_lr.roll([1],[3])*self.left_mask
		hidden = torch.cat((hidden_lr,hidden_rl),1)

		hidden = self.W2(F.relu(self.W1(hidden)))
		
		if self.layer_num == 0:
			return self.layer_norm((hidden).permute(0,2,3,1)).permute(0,3,1,2)
		else:
			return self.layer_norm((hidden + x).permute(0,2,3,1)).permute(0,3,1,2)

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()

		#lstms for rest of layers (4 per layer)	
		self.layers = nn.ModuleList([Layer(layer) for layer in np.arange(num_layers)])		

		self.probs_hid = nn.Conv2d(hidden_nodes,hidden_nodes,1)
		self.ang_hid = nn.Conv2d(hidden_nodes,hidden_nodes,1)
		self.probs_hid2 = nn.Conv2d(hidden_nodes,hidden_nodes,1)
		self.ang_hid2 = nn.Conv2d(hidden_nodes,hidden_nodes,1)

		self.probs = nn.Conv2d(hidden_nodes,2,1) 
		self.sin = nn.Conv2d(hidden_nodes,2,1) 
		self.cos = nn.Conv2d(hidden_nodes,2,1) 

	def forward(self,inp):

		#input is [batch_size, 2, num_sites, num_sites] 

		for layer in np.arange(num_layers):

			if layer == 0:
				hidden = self.layers[layer](inp)
			else:	
				hidden = self.layers[layer](hidden)

		probs_hidden = F.relu(self.probs_hid(hidden))
		ang_hidden = F.relu(self.ang_hid(hidden))
		probs_hidden = F.relu(self.probs_hid2(probs_hidden))
		ang_hidden = F.relu(self.ang_hid2(ang_hidden))

		probs = F.softmax(self.probs(probs_hidden),1)
		sin = self.sin(ang_hidden)		
		cos = self.cos(ang_hidden)	

		phase = torch.atan2(sin,cos)

		log_wf = 0.5*torch.sum(torch.sum(torch.log(torch.sum(probs*inp,1)),1),1)
		phase_wf = torch.sum(torch.sum(torch.sum(phase*inp,1),1),1)

		return log_wf, phase_wf

	def sample(self):

		inp = torch.ones([batch_size,2,num_sites,num_sites]).to(device)

		for i in np.arange(num_sites):	
			for j in np.arange(num_sites):
				for layer in np.arange(num_layers):

					if layer == 0:
						hidden = self.layers[layer](inp)
					else:	
						hidden = self.layers[layer](hidden)

				probs_hidden = F.relu(self.probs_hid(hidden))
				probs_hidden = F.relu(self.probs_hid2(probs_hidden))

				probs = F.softmax(self.probs(probs_hidden),1)

				thresh = torch.rand(len(inp)).to(device) 
				is_one = (1 + torch.sign(probs[:,1,i,j] - thresh))/2
				inp[:,:,i,j] = torch.cat((torch.unsqueeze(1-is_one.clone(),1),torch.unsqueeze(is_one.clone(),1)),1)			
		return inp

def local_energy(state,NN,device):

	#state is [num_sites,num_sites,2] 

	neighbors = []
	next_neighbors = []
	diag_energy = 0

	neighbors.append(state)

	for i in np.arange(num_sites):  
		for j in np.arange(num_sites):
			if i+1 < num_sites:
				if state[1,i,j] == 1 and state[0,i+1,j] == 1:
					final_state = state.copy()
					final_state[1,i,j] = 0
					final_state[0,i+1,j] = 0
					final_state[0,i,j] = 1
					final_state[1,i+1,j] = 1 
					neighbors.append(final_state)	
				if state[0,i,j] == 1 and state[1,i+1,j] == 1:
					final_state = state.copy()
					final_state[0,i,j] = 0
					final_state[1,i+1,j] = 0
					final_state[1,i,j] = 1
					final_state[0,i+1,j] = 1 
					neighbors.append(final_state)	
				if j + 1 < num_sites:
					if state[1,i,j] == 1 and state[0,i+1,j+1] == 1:
						final_state = state.copy()
						final_state[1,i,j] = 0
						final_state[0,i+1,j+1] = 0
						final_state[0,i,j] = 1
						final_state[1,i+1,j+1] = 1 
						next_neighbors.append(final_state)	
					if state[0,i,j] == 1 and state[1,i+1,j+1] == 1:
						final_state = state.copy()
						final_state[0,i,j] = 0
						final_state[1,i+1,j+1] = 0
						final_state[1,i,j] = 1
						final_state[0,i+1,j+1] = 1 
						next_neighbors.append(final_state)	
				if j > 0:
					if state[1,i,j] == 1 and state[0,i+1,j-1] == 1:
						final_state = state.copy()
						final_state[1,i,j] = 0
						final_state[0,i+1,j-1] = 0
						final_state[0,i,j] = 1
						final_state[1,i+1,j-1] = 1 
						next_neighbors.append(final_state)	
					if state[0,i,j] == 1 and state[1,i+1,j-1] == 1:
						final_state = state.copy()
						final_state[0,i,j] = 0
						final_state[1,i+1,j-1] = 0
						final_state[1,i,j] = 1
						final_state[0,i+1,j-1] = 1 
						next_neighbors.append(final_state)	
			if j+1<num_sites:
				if state[1,i,j] == 1 and state[0,i,j+1] == 1:
					final_state = state.copy()
					final_state[1,i,j] = 0
					final_state[0,i,j+1] = 0
					final_state[0,i,j] = 1
					final_state[1,i,j+1] = 1 
					neighbors.append(final_state)	
				if state[0,i,j] == 1 and state[1,i,j+1] == 1:
					final_state = state.copy()
					final_state[0,i,j] = 0
					final_state[1,i,j+1] = 0
					final_state[1,i,j] = 1
					final_state[0,i,j+1] = 1 
					neighbors.append(final_state)	

	state = np.sum(state*np.expand_dims(np.expand_dims(np.asarray([-1,1]),1),2),0)	
	right_shifted_state = np.roll(state,1,1)	
	right_shifted_state[:,0] = 0
	horizontal_mag = np.sum(right_shifted_state*state) 
	up_shifted_state = np.roll(state,1,0)	
	up_shifted_state[0] = 0
	vertical_mag = np.sum(up_shifted_state*state) 
	up_right_shifted_state  = np.roll(up_shifted_state,1,1)
	up_right_shifted_state[:,0] = 0 
	up_left_shifted_state  = np.roll(up_shifted_state,-1,1)
	up_left_shifted_state[:,-1] = 0 
	diag_mag = np.sum(state*up_right_shifted_state) + np.sum(state*up_left_shifted_state) 
	diag_energy = (horizontal_mag + vertical_mag + J2*diag_mag)*0.25/(num_sites*num_sites)
	afm = (horizontal_mag + vertical_mag)/(num_sites*num_sites)
	mag = np.sum(state)/(num_sites*num_sites)
	
	return diag_energy, neighbors, next_neighbors, afm, mag

def train_network():
	
	file_ext = str(num_sites) + '_' + str(batch_size) +  '_' + str(learning_rate) + '_' + str(J2) + '.txt'
	loss_file = 'loss_heisenberg_' + file_ext
	info_file = 'info_heisenberg_' + file_ext
	f = open(loss_file,'w')
	f2 = open(info_file,'w')

	NN = Net().to(device)

	#Uncomment to load previous model as done in iterative retraining
#	NN.load_state_dict(torch.load("model_38_100_1e-05_1.0_0.0.txt"))
	optimizer = torch.optim.Adam(NN.parameters(), lr=learning_rate)
	
	#Uncomment to have decreasing learning rate
#	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1/(0.001*epoch + 1))
	
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

		partition_size = 100
		part_size = 50
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
				energies[i] = diag_energy[i] + 0.5*np.sum(np.exp(log_wfs[int(position + 1):int(position + neighbor_len[i])] + 1j*phase_wfs[int(position + 1):int(position + neighbor_len[i])] - log_wf[i] - 1j*phase_wf[i]))/(num_sites*num_sites) + 0.5*J2*np.sum(np.exp(nlog_wfs[int(nposition):int(nposition + next_neighbor_len[i])] + 1j*nphase_wfs[int(nposition):int(nposition + next_neighbor_len[i])] - log_wf[i] - 1j*phase_wf[i]))/(num_sites*num_sites)  
				position = position + neighbor_len[i]
				nposition = nposition + next_neighbor_len[i]
		else:
			for i in np.arange(batch_size):
				log_wf[i] = log_wfs[position]
				phase_wf[i] = phase_wfs[position]
				energies[i] = diag_energy[i] + 0.5*np.sum(np.exp(log_wfs[int(position + 1):int(position + neighbor_len[i])] + 1j*phase_wfs[int(position + 1):int(position + neighbor_len[i])] - log_wf[i] - 1j*phase_wf[i]))/(num_sites*num_sites)  

				position = position + neighbor_len[i]

		energies = np.nan_to_num(np.asarray(energies))
		log_wf = np.nan_to_num(np.asarray(log_wf))
		phase_wf = np.nan_to_num(np.asarray(phase_wf))
		afm = np.nan_to_num(np.asarray(afm))
		mag = np.nan_to_num(np.asarray(mag))
		mean_energies = np.mean(energies)
		mean_entropies = np.mean(2*log_wf)/(num_sites*num_sites)
		mean_afm = np.mean(afm)
		mean_magmo = np.mean(np.abs(mag))

		residuals = np.conj(energies - mean_energies)

		real_residuals = np.real(residuals)			
		imag_residuals = np.imag(residuals)			
		entropy_residuals = (2*log_wf/(num_sites*num_sites) + np.log(2))
		mag_residuals = np.square(mag)
		
		optimizer.zero_grad()
		#Set T = 0 to train a small model quickly
#		T = 0
		T = 1./(1 + 0.001*batch)
		C = 10.
		
		num_partitions = (batch_size - 1)// part_size + 1
		for partition in np.arange(num_partitions):
			log_wf,phase_wf = NN.forward(torch.FloatTensor(batch_states[int(partition*part_size):int((partition+1)*part_size)]).to(device))
			loss = torch.sum(log_wf*torch.FloatTensor((real_residuals + T*entropy_residuals + C*mag_residuals)[int(partition*part_size):int((partition+1)*part_size)]).to(device)-phase_wf*torch.FloatTensor(imag_residuals[int(partition*part_size):int((partition+1)*part_size)]).to(device))
			loss.backward()
			for param in NN.parameters():
				param.grad[torch.isnan(param.grad)] = 0

		norm = torch.nn.utils.clip_grad_norm_(NN.parameters(),max_norm)
		optimizer.step()

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
			f.write(str(avg_energy/100))
			f.write('\t')
			f.write(str(avg_magmo/100))
			f.write('\t')
			f.write(str(avg_afm/100))
			f.write('\n')
			f.flush()
			avg_energy = 0 
			avg_magmo = 0 
			avg_afm = 0 

			#save model every 100 minibatches
			torch.save(NN.state_dict(), 'model_heisenberg_' + file_ext)

#		scheduler.step()
	
	
train_network()
