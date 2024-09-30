import torch
import torch.nn as nn
import numpy as np
from .solution import calculate_offline_optimal, calculate_offline_optimal_dynamic

## ML model
class LSTM_unroll(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, 
                 seq_length, w, l_1, l_2, l_3):
        
        super(LSTM_unroll, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        self.w = w
        self.l_1 = l_1
        self.l_2 = l_2
        self.l_3 = l_3

    def forward(self, x, mode="train", calib = True):
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device = x.device)
        
        c_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device = x.device)
        
        action_0 = torch.zeros(x.size(0), 1, x.size(2), device = x.device)
        action_sequence = []
        
        seq_length = x.size(1)
        
        
        if (mode not in ["train", "val"]):
            raise NotImplementedError
        
        if (seq_length != self.seq_length and mode == "train"):
            print("The length of sequence is changed, be careful!!!!")
            assert 0
            
        for i in range(seq_length):
            x_i = x[:,i,:].unsqueeze(1)
            input_i = torch.cat([x_i, action_0], 2)
            ula, (h_out, c_out) = self.lstm(input_i, (h_0, c_0))
            
            ula = ula.view(-1, self.hidden_size)
            out = self.fc(ula).unsqueeze(1)
            
            h_0 = h_out
            c_0 = c_out
            
            if i < 1:
                # The initial action directly follow the demand
                action_0 = x_i
            else:
                action_prev = action_sequence[-1]
                if calib:
                    action_0 = (1 + self.w*self.l_2)*x_i + self.w*self.l_1*action_prev + self.w*self.l_3*out
                    action_0 /= 1 + self.w*(self.l_1 + self.l_2 + self.l_3)
                else:
                    action_0 = out
            
            action_sequence.append(action_0)

        final_out = torch.cat(action_sequence,1)
        
        return final_out
    



class LSTM_unroll_dynamic(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, 
                 seq_length, w, l_1, l_2, l_3):
        
        super(LSTM_unroll_dynamic, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        #self.p_1 = p_1
        #torch.set_default_dtype(torch.torch.float64)
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        self.w = w
        self.l_1 = l_1
        self.l_2 = l_2
        self.l_3 = l_3

    def forward(self, x, p_1, mode="train", calib = True):
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device = x.device)
        
        c_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device = x.device)
        
        action_0 = torch.zeros(x.size(0), 1, x.size(2), device = x.device)
        xtilde = torch.zeros(x.size(0), 1, x.size(2), device = x.device) #for calculating rho
        action_sequence = []
        #input_sequence = []
        p_2 = 0
        
        seq_length = x.size(1)
        #print('input x')
        #print(x.shape)
        #print(p_1.shape)
        
        #self.l_3 = 1.0/(torch.exp(2*p_1)) 
        #self.l_2 = (2*self.l_1/(2*10))*(torch.sqrt((1+(10/2)*(self.l_3/self.l_1))**2 + 4*10**2/(10*2)) + 1 - 2/self.l_1 - (10/2)*(self.l_3/self.l_1))
        if (mode not in ["train", "val"]):
            raise NotImplementedError
        
        if (seq_length != self.seq_length and mode == "train"):
            print("The length of sequence is changed, be careful!!!!")
            assert 0
        #print(x[:,:,0].unsqueeze(1).shape)   
        for i in range(seq_length):
            x_i = x[:,i,:].unsqueeze(1)
            #print(i)
            #print(x_i.shape)
            #print(type(x_i))
            #print(action_0.shape)
            input_i = torch.cat([x_i, action_0], 2)
            ula, (h_out, c_out) = self.lstm(input_i, (h_0, c_0))
            #input_sequence = torch.cat(input_sequence, x_i)
            #input_array = np.array(input_sequence)
            #input_array = input_array.reshape(-1,1)
            
            ula = ula.view(-1, self.hidden_size)
            out = self.fc(ula).unsqueeze(1)
            #print(f'{out.shape} outshape')
            #p_2 = np.expand_dims(p_1, axis=1)
            #p_2 = p_1.unsqueeze(1)
            
            h_0 = h_out
            c_0 = c_out
            #self.l_3 = 1.0/(torch.exp((2.5/40)*p_2 + 4*2.5*p_2**2 - (1/2))) 
            #print(f'l_3 is {self.l_3.shape}')
            #self.l_2 = (2*self.l_1/(2*10))*(torch.sqrt((1+(10/2)*(self.l_3/self.l_1))**2 + 4*10**2/(10*2)) + 1 - 2/self.l_1 - (10/2)*(self.l_3/self.l_1))
            #print(self.l_3.shape)
            #print(self.l_2.shape)
            if i < 1:
                # The initial action directly follow the demand
                action_0 = x_i
                xtilde = x_i.detach()
                #print('action0')
                #print(action_0.shape)
            else:
                action_prev = action_sequence[-1]
                if calib:
                    input_array = x[:,:i+1,:].detach()
                    #print('inputarray')
                    #print( i)
                    #print(f'input array shape is {input_array.shape}')
                    xtilde = torch.cat([xtilde,out.detach()],1)
                    #print(f'xtilde is {xtilde.shape}')
                    #print(f'input is {input_i}')
                    #print(f'output is {out}')
                    #print(f'xtilde is {xtilde}')
                    #print(f'action_sequence is {action_sequence}')
                    optimal_action, optimal_cost = calculate_offline_optimal_dynamic(input_array[:,1:,:], input_array[:,0,:], switch_weight=10)
                    #print(f'optimal_action.shape is {optimal_action.shape}')
                    #print(f'input array is {input_array}')
                    #print(f'optimal action is {optimal_action}')
                    #print(f'optimal_cost is {optimal_cost}')
                    p_2 = torch.norm(xtilde-optimal_action, dim=1)/optimal_cost
                    p_2 = p_2.unsqueeze(1)
                    #print(f'p2 shape is {p_2.shape}')
                    #print(f'p2 is {p_2}')
                    #self.l_3 = 1.0/(torch.exp((2.5/40)*p_2 + 4*2.5*p_2**2 - (1/2)))
                    #self.l_3 = 1.0/(torch.exp(p_2**2 - (1/2))) #second exponential check
                    self.l_3 = 1.0/(torch.exp(p_2)) #exponential without the square
                    #print(f'self3 is {self.l_3}')
                    self.l_2 = (2*self.l_1/(2*10))*(torch.sqrt((1+(10/2)*(self.l_3/self.l_1))**2 + 4*10**2/(10*2)) + 1 - 2/self.l_1 - (10/2)*(self.l_3/self.l_1))
                    #print(f'self.l_2 is {self.l_2}')
                #if calib:
                    action_0 = (1 + self.w*self.l_2)*x_i + self.w*self.l_1*action_prev + self.w*self.l_3*out
                    action_0 /= 1 + self.w*(self.l_1 + self.l_2 + self.l_3)
                    #print(f'for iter {i} action_o is {action_0}')
                else:
                    action_0 = out
                    #print(f'action_o is {action_0[1:5,0]}')
            #print(action_0.shape)
            action_sequence.append(action_0)

        final_out = torch.cat(action_sequence,1)
        
        return final_out, p_2
