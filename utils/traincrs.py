
import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

#from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn



from utils.preprocess import sliding_windows, load_power_shortage
from utils.loss import object_loss_cost, object_loss_cr
from utils.model import LSTM_unroll
from utils.dataset import TrajectCR_Dataset
from tqdm import tqdm
import json 
import argparse

n_iter = 0
n_iter_val = 0
use_cuda = False

def train_cr_vals(ml_model, optimizer, train_dataloader, val_dataloader, demand_validation, 
            num_epoch, switch_weight, min_cr, temp_seq, input_dict, op_temp_cost_array, df_train, df_test,
            df_val, mtl_weight = 0.5, mute=True, l_1=1, l_2=1, l_3=1, calib=True):
    from utils.loss import object_loss_cost as objl
    from utils.loss import object_loss_cr as objlr
    global n_iter, n_iter_val, use_cuda

    #random.seed(10)
    #np.random.seed(10)
    #torch.manual_seed(10)
    
    if not mute:
        epoch_iter = tqdm(range(num_epoch))
    else:
        epoch_iter = range(num_epoch)
    
    for epoch in epoch_iter:
        ml_model.train()
        for _, (demand,opt_cost) in enumerate(train_dataloader):
            demand = demand.float()
            if use_cuda: 
                demand = demand.cuda()
                opt_cost = opt_cost.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            action_ml = ml_model(demand, calib = calib)
            
            
            if mtl_weight == 1.0:
                loss_calib = torch.zeros((1,1))
                
                loss_ml = object_loss_cost(demand, action_ml, c=switch_weight)
                # loss_ml = object_loss_cr(demand, action_ml, opt_cost, min_cr = min_cr, c=switch_weight)
                loss = loss_ml

            else:
                loss_ml = objlr(demand, action_ml, opt_cost, min_cr = min_cr, c=switch_weight)
                
                action_calib = ml_model(demand, calib = calib)
                loss_calib = object_loss_cost(demand, action_calib, c = switch_weight)
                
                loss = mtl_weight*loss_ml + (1-mtl_weight)*loss_calib

            loss.backward()
            optimizer.step()
#updated names to include scalars for easy tracking
            #writer.add_scalar(f'Loss_train/no_calib_{l_1}_{l_2}_{l_3}', loss_ml.item(), n_iter)
            #writer.add_scalar(f'Loss_train/with_calib_{l_1}_{l_2}_{l_3}', loss_calib.item(), n_iter)
            #writer.add_scalar(f'Loss_train/overall_{l_1}_{l_2}_{l_3}', loss.item(), n_iter)
            n_iter += 1
        df_train.loc[len(df_train.index)] = [epoch, input_dict['name'], loss.item(), loss_ml.item(), loss_calib.item()]
        #writer.flush()
        
        # Calculate evaluation cost
        ml_model.eval()

        with torch.no_grad():
            val_iter = 0
            loss_val_calib = 0
            for _, (demand,opt_cost) in enumerate(val_dataloader):
                demand = demand.float()
                val_iter += 1
                opt_cost = torch.reshape(opt_cost, (opt_cost.shape[0], 1))
                #opt_cost = np.expand_dims(opt_cost, axis=1)
                #opt_cost = opt_cost.float()
                if use_cuda: 
                    demand = demand.cuda()
                    opt_cost = opt_cost.cuda()
                    #print(demand.shape)
                    #print(opt_cost.shape)
                    # zero the parameter gradients
                    #optimizer.zero_grad()
                    #demand = demand.type('torch.FloatTensor')
                    #opt_cost = opt_cost.type('torch.FloatTensor')
                    #action_ml = ml_model(demand, torch.from_numpy(opt_cost).float(), calib = calib)
                action_ml = ml_model(demand, mode="val", calib = False)
                action_val_calib = ml_model(demand, mode="val", calib = input_dict["testcalib"])
                loss_val_calib += object_loss_cost(demand, action_val_calib, c = switch_weight)
                #print(f'demand validation dynamic = {demand.shape}')
            #action_val_ml    = ml_model(demand_validation, opt_cost_dynamic, mode="val", calib=False)
            #action_val_calib = ml_model(demand_validation, opt_cost_dynamic, mode="val", calib=True)
            #print(f'val iteration # is {val_iter}') 
            #print(f'demand validation dynamic = {demand.shape}')   
            loss_val_calib = loss_val_calib/val_iter
            df_val.loc[len(df_val.index)] = [ epoch,input_dict["name"], loss_val_calib.item()]

            #action_val_ml    = ml_model(demand_validation, mode="val", calib=False)
            #action_val_calib = ml_model(demand_validation, mode="val", calib=input_dict["testcalib"])
            #df_test.loc[len(df_test.index)] = [ epoch,input_dict["name"], average_cost_value, cr_value]

            #loss_val_ml = object_loss_cost(demand_validation, action_val_ml, c = switch_weight)
            #for _, (demand,opt_cost) in enumerate(val_dataloader):
            #    demand = demand.float()
            #    loss_val_calib = object_loss_cost(demand, action_val_calib, c = switch_weight)
            #df_val.loc[len(df_val.index)] = [ epoch,input_dict["name"], loss_val_calib.item()]
            #print(f'demand validation = {demand_validation.shape}')
#updated names with lambdas for ease of tracking
        #writer.add_scalar(f'Loss_val/no_calib_{l_1}_{l_2}_{l_3}', loss_val_ml.item()/100, n_iter_val)
        #writer.add_scalar(f'Loss_val/with_calib_{l_1}_{l_2}_{l_3}', loss_val_calib.item()/100, n_iter_val)
        n_iter_val += 1

        if epoch%5 == 0:
            for i in range(1,3): #journal change originally 5
                #temp_seq = sliding_windows(test_raw, seq_length)
                #temp_seq.shape
                #temp_action_seq = torch.zeros(temp_seq[i].shape)
                #temp_action_seq.shape
                #ml_model.eval()
                #print(f'epoch is {epoch}')
                tmval = ml_model(torch.from_numpy(temp_seq[i]).float(), mode="val", calib=input_dict["testcalib"])
                #print(tmval.shape)
                #print(torch.max(tmval))
                hit_cost = torch.norm(tmval - torch.from_numpy(temp_seq[i]).float(), dim=1)
                hit_cost = (hit_cost)**2 #remove 1/2
                #print(torch.max(hit_cost))

                #switch_diff = action[:,1:,:] - action[:,:-1,:]
                switch_cost = torch.norm(tmval[:,1:,:] - tmval[:,:-1,:], dim=1)
                switch_cost = (10)*(switch_cost)**2 #remove division by 2
                #print(switch_cost.shape)
                val_cost_norm = hit_cost + switch_cost
                norm_cost = torch.div(val_cost_norm, torch.from_numpy(op_temp_cost_array[i]).unsqueeze(1))
                #print(torch.max(norm_cost))
                average_cost_value = torch.mean(norm_cost).item()
                cr_value = torch.max(norm_cost).item()
                df_test.loc[len(df_test.index)] = [ epoch,input_dict["name"], i+1, average_cost_value, cr_value]
                #return [average_cost_value, cr_value]


    #writer.close()



def train_cr_vals_dynamic(ml_model, optimizer, train_dataloader, val_dataloader,temp_dataloader,
                          temp_seq, opt_cost_dynamic, num_epoch, switch_weight, min_cr, 
                          input_dict, op_temp_cost_array, 
            df_train, df_test, df_val, df_testp, mtl_weight = 0.5, mute=True, l_1=1, l_2=1, l_3=1, calib=True):
    from utils.loss import object_loss_cost as objl
    from utils.loss import object_loss_cr as objlr
    global n_iter, n_iter_val, use_cuda

    #random.seed(10)
    #np.random.seed(10)
    #torch.manual_seed(10)
    
    if not mute:
        epoch_iter = tqdm(range(num_epoch))
    else:
        epoch_iter = range(num_epoch)
    
    for epoch in epoch_iter:
        ml_model.train()
        for k, (demand,opt_cost) in enumerate(train_dataloader):
            demand = demand.float()
            opt_cost = torch.reshape(opt_cost, (opt_cost.shape[0], 1))
            
            if use_cuda: 
                demand = demand.cuda()
                opt_cost = opt_cost.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            
            action_ml, p2 = ml_model(demand, opt_cost, calib = calib)
            
            
            if mtl_weight == 1.0:
                loss_calib = torch.zeros((1,1))
                
                loss_ml = object_loss_cost(demand, action_ml, c=switch_weight)
                # loss_ml = object_loss_cr(demand, action_ml, opt_cost, min_cr = min_cr, c=switch_weight)
                loss = loss_ml
                #print(f'loss is {loss}')

            else:
                loss_ml = objlr(demand, action_ml, opt_cost, min_cr = min_cr, c=switch_weight)
                
                action_calib, p2_calib  = ml_model(demand, opt_cost, calib = calib)
                loss_calib = object_loss_cost(demand, action_calib, c = switch_weight)
                
                loss = mtl_weight*loss_ml + (1-mtl_weight)*loss_calib
                #print(f'final loss is {loss}')

            loss.backward()
            optimizer.step()
#updated names to include scalars for easy tracking
            #writer.add_scalar(f'Loss_train/no_calib_{l_1}_{l_2}_{l_3}', loss_ml.item(), n_iter)
            #writer.add_scalar(f'Loss_train/with_calib_{l_1}_{l_2}_{l_3}', loss_calib.item(), n_iter)
            #writer.add_scalar(f'Loss_train/overall_{l_1}_{l_2}_{l_3}', loss.item(), n_iter)
            n_iter += 1
        df_train.loc[len(df_train.index)] = [epoch, input_dict['name'], loss.item(), loss_ml.item(), loss_calib.item()] 
        #writer.flush()
        
        # Calculate evaluation cost
        ml_model.eval()

        with torch.no_grad():
            val_iter = 0
            loss_val_calib = 0
            for _, (demand,opt_cost) in enumerate(val_dataloader):
                demand = demand.float()
                val_iter += 1
                opt_cost = torch.reshape(opt_cost, (opt_cost.shape[0], 1))
                #opt_cost = np.expand_dims(opt_cost, axis=1)
                #opt_cost = opt_cost.float()
                if use_cuda: 
                    demand = demand.cuda()
                    opt_cost = opt_cost.cuda()
                    #print(demand.shape)
                    #print(opt_cost.shape)
                    # zero the parameter gradients
                    #optimizer.zero_grad()
                    #demand = demand.type('torch.FloatTensor')
                    #opt_cost = opt_cost.type('torch.FloatTensor')
                    #action_ml = ml_model(demand, torch.from_numpy(opt_cost).float(), calib = calib)
                action_ml, p2_val = ml_model(demand, opt_cost, mode="val", calib = False)
                action_val_calib, p2_val_calib  = ml_model(demand, opt_cost, mode="val", calib = input_dict["testcalib"])
                loss_val_calib += object_loss_cost(demand, action_val_calib, c = switch_weight)
                #print(f'demand validation dynamic = {demand.shape}')
            #action_val_ml    = ml_model(demand_validation, opt_cost_dynamic, mode="val", calib=False)
            #action_val_calib = ml_model(demand_validation, opt_cost_dynamic, mode="val", calib=True)
            #print(f'val iteration # is {val_iter}') 
            #print(f'demand validation dynamic = {demand.shape}')   
            loss_val_calib = loss_val_calib/val_iter
            df_val.loc[len(df_val.index)] = [ epoch,input_dict["name"], loss_val_calib.item()]

            #loss_val_ml = object_loss_cost(demand_validation, action_val_ml, c = switch_weight)
            #loss_val_calib = object_loss_cost(demand_validation, action_val_calib, c = switch_weight)
#updated names with lambdas for ease of tracking
        #writer.add_scalar(f'Loss_val/no_calib_{l_1}_{l_2}_{l_3}', loss_val_ml.item()/100, n_iter_val)
        #writer.add_scalar(f'Loss_val/with_calib_{l_1}_{l_2}_{l_3}', loss_val_calib.item()/100, n_iter_val)
        n_iter_val += 1
        if epoch > num_epoch-3:
            for i in range(1,4): #journal change originally 5 changed to 4 to exclude all quarters
                temp_action_seq = torch.zeros(temp_seq[i].shape)
                temp_action_seq.shape
                ml_model.eval()
                for _, (demand,opt_cost) in enumerate(temp_dataloader[i]):
                        demand = demand.float()
                        
                        opt_cost = torch.reshape(opt_cost, (opt_cost.shape[0], 1))
                        #opt_cost = np.expand_dims(opt_cost, axis=1)
                        #opt_cost = opt_cost.float()
                        if use_cuda: 
                            demand = demand.cuda()
                            opt_cost = opt_cost.cuda()
                        #print(demand.shape)
                        #print(opt_cost.shape)
                        # zero the parameter gradients
                        #optimizer.zero_grad()
                        #demand = demand.type('torch.FloatTensor')
                        #opt_cost = opt_cost.type('torch.FloatTensor')
                        #action_ml = ml_model(demand, torch.from_numpy(opt_cost).float(), calib = calib)
                        tmval, p2_test = ml_model(demand, opt_cost, mode="val", calib = input_dict['testcalib'])
                #tmval = lstm(torch.from_numpy(temp_seq).float(), torch.from_numpy(op_val_cost_array).float(),mode="val", calib=input_dict["testcalib"])
                #tmval = lstm(temp_dataloader,mode="val", calib=input_dict["testcalib"])
                #print(tmval.shape)
                #print(torch.max(tmval))
                        #print(f'{tmval}')        
                hit_cost = torch.norm(tmval - torch.from_numpy(temp_seq[i]).float(), dim=1)
                hit_cost = (hit_cost)**2 #remove division by 2
                #print(torch.max(hit_cost))

                #switch_diff = action[:,1:,:] - action[:,:-1,:]
                switch_cost = torch.norm(tmval[:,1:,:] - tmval[:,:-1,:], dim=1)
                switch_cost = (10)*(switch_cost)**2 #remove division by 2
                #print(switch_cost.shape)
                val_cost_norm = hit_cost + switch_cost
                norm_cost = torch.div(val_cost_norm, torch.from_numpy(op_temp_cost_array[i]).unsqueeze(1))
                #print(torch.max(norm_cost))
                #print(f'norm cost {norm_cost.shape}, rho {p2_test.shape}') #journal change
                average_cost_value = torch.mean(norm_cost).item()
                cr_value = torch.max(norm_cost).item()
                #df_testp.loc[len(df_test.index)] = [epoch, input_dict["name"], i+1]
                df_test.loc[len(df_test.index)] = [ epoch,input_dict["name"], i+1, average_cost_value, cr_value]
                #return [average_cost_value, cr_value]
                if epoch == (num_epoch-1):
                     dict_list = [input_dict["name"]]*p2_test.shape[0]
                     quarter_list = [i+1]*norm_cost.shape[0]
                     df_temp = pd.DataFrame({list(df_testp.columns.values)[0]:dict_list, list(df_testp.columns.values)[1]:quarter_list, list(df_testp.columns.values)[2]:p2_test.flatten().tolist(), list(df_testp.columns.values)[3]:norm_cost.flatten().tolist()}) 
                     df_testp.loc[range((i-1)*p2_test.shape[0],(i)*p2_test.shape[0])] = df_temp.values
            #print(f'norm cost {norm_cost.shape}, rho {p2_test.shape}') #journal change
        if epoch%5 == 0:
           print(epoch)        
            #dict_list = [input_dict["name"]]*p2_test.shape[0]
            #quarter_list = [i+1]*norm_cost.shape[0]
            #df_temp = pd.DataFrame({list(df_testp.columns.values)[0]:dict_list, list(df_testp.columns.values)[1]:quarter_list, list(df_testp.columns.values)[2]:p2_test.flatten().tolist(), list(df_testp.columns.values)[3]:norm_cost.flatten().tolist()}) 
            #df_testp.loc[range(p2_test.shape[0])] = df_temp.values

    #writer.close()    