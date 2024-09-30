import torch 
from torch.utils.data import Dataset
from tqdm import tqdm 
import numpy as np
from .solution import calculate_offline_optimal

class TrajectCR_Dataset(Dataset):
    
    def __init__(self, train_seq, switch_cost, mute=False):

        num_seq = train_seq.shape[0]
        optimal_cost_array = np.zeros(num_seq)
        
        print("Calculating Offline Optimal values ...")
        if mute:
            seq_list = range(num_seq)
        else:
            seq_list = tqdm(range(num_seq))
        for i in seq_list:
            sample_seq = train_seq[i,:,:]
            _, optimal_cost = calculate_offline_optimal(sample_seq[1:], sample_seq[0], switch_weight=switch_cost)
            optimal_cost_array[i] = optimal_cost

        self.X_dataset = torch.from_numpy(train_seq).float()
        #print(f'train seq is {train_seq.size}')
        #print(f'X.shape is {self.X_dataset.size()}')
        self.input_dim = self.X_dataset.size(2)
        self.sequence_length = self.X_dataset.size(1)
        #self.optimal_cost_array = optimal_cost_array   #changed this to torch to try to fix error in dynamic code
        self.optimal_cost_array = torch.from_numpy(optimal_cost_array).float()

    def __len__(self):
        return np.shape(self.X_dataset)[0]

    def __getitem__(self, idx):  
        original_data = self.X_dataset[idx, :, :]
        optimal_cost = self.optimal_cost_array[idx]
        
        return original_data, optimal_cost
  



class TrajectCR_Dataset_Dynamic(Dataset):
    
    def __init__(self, train_seq, switch_cost, mute=False):

        num_seq = train_seq.shape[0]
        optimal_cost_array = np.zeros(num_seq)
        optimal_action_array = np.zeros(num_seq,train_seq.shape[1],1)
        
        print("Calculating Offline Optimal values ...")
        if mute:
            seq_list = range(num_seq)
        else:
            seq_list = tqdm(range(num_seq))
        for i in seq_list:
            sample_seq = train_seq[i,:,:]
            optimal_action, optimal_cost = calculate_offline_optimal(sample_seq[1:], sample_seq[0], switch_weight=switch_cost)
            optimal_cost_array[i] = optimal_cost
            optimal_action = optimal_action.reshape([-1,1])
            optimal_action_array[i,:,:] = optimal_action

        self.X_dataset = torch.from_numpy(train_seq).float()
        self.input_dim = self.X_dataset.size(2)
        self.sequence_length = self.X_dataset.size(1)
        #self.optimal_cost_array = optimal_cost_array   #changed this to torch to try to fix error in dynamic code
        self.optimal_cost_array = torch.from_numpy(optimal_cost_array).float()
        self.optimal_action_array = torch.from_numpy(optimal_action_array).float()

    def __len__(self):
        return np.shape(self.X_dataset)[0]

    def __getitem__(self, idx):  
        original_data = self.X_dataset[idx, :, :]
        optimal_cost = self.optimal_cost_array[idx]
        optimal_action = self.optimal_action_array[idx,:,:]
        
        return original_data, optimal_cost, optimal_action
  