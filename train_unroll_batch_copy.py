import pandas as pd
import numpy as np
import random
import math
import os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from utils.preprocess import sliding_windows, load_power_shortage
from utils.loss import object_loss_cost, object_loss_cr
from utils.model import LSTM_unroll, LSTM_unroll_dynamic
from utils.dataset import TrajectCR_Dataset
from tqdm import tqdm
import json 
import argparse

from utils.solution import calculate_offline_optimal
#from train_unroll_batch import single_experiment, train_cr
from utils.traincrs import train_cr_vals_dynamic, train_cr_vals

n_iter = 0
n_iter_val = 0
use_cuda = False

def train_cr(ml_model, optimizer, writer, train_dataloader, demand_validation, 
            num_epoch, switch_weight, min_cr, mtl_weight = 0.5, mute=True, l_1=1, l_2=1, l_3=1):

    global n_iter, n_iter_val, use_cuda
    
    if not mute:
        epoch_iter = tqdm(range(num_epoch))
    else:
        epoch_iter = range(num_epoch)
    
    for _ in epoch_iter:
        ml_model.train()
        for _, (demand,opt_cost) in enumerate(train_dataloader):
            demand = demand.float()
            if use_cuda: 
                demand = demand.cuda()
                opt_cost = opt_cost.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            action_ml = ml_model(demand, calib = True)
            
            
            if mtl_weight == 1.0:
                loss_calib = torch.zeros((1,1))
                
                loss_ml = object_loss_cost(demand, action_ml, c=switch_weight)
                # loss_ml = object_loss_cr(demand, action_ml, opt_cost, min_cr = min_cr, c=switch_weight)
                loss = loss_ml

            else:
                loss_ml = object_loss_cr(demand, action_ml, opt_cost, min_cr = min_cr, c=switch_weight)
                
                action_calib = ml_model(demand, calib = True)
                loss_calib = object_loss_cost(demand, action_calib, c = switch_weight)
                
                loss = mtl_weight*loss_ml + (1-mtl_weight)*loss_calib

            loss.backward()
            optimizer.step()
#updated names to include scalars for easy tracking
            writer.add_scalar(f'Loss_train/no_calib_{l_1}_{l_2}_{l_3}', loss_ml.item(), n_iter)
            writer.add_scalar(f'Loss_train/with_calib_{l_1}_{l_2}_{l_3}', loss_calib.item(), n_iter)
            writer.add_scalar(f'Loss_train/overall_{l_1}_{l_2}_{l_3}', loss.item(), n_iter)
            n_iter += 1

        writer.flush()
        
        # Calculate evaluation cost
        ml_model.eval()

        with torch.no_grad():
            action_val_ml    = ml_model(demand_validation, mode="val", calib=False)
            action_val_calib = ml_model(demand_validation, mode="val", calib=True)

            loss_val_ml = object_loss_cost(demand_validation, action_val_ml, c = switch_weight)
            loss_val_calib = object_loss_cost(demand_validation, action_val_calib, c = switch_weight)
#updated names with lambdas for ease of tracking
        writer.add_scalar(f'Loss_val/no_calib_{l_1}_{l_2}_{l_3}', loss_val_ml.item()/100, n_iter_val)
        writer.add_scalar(f'Loss_val/with_calib_{l_1}_{l_2}_{l_3}', loss_val_calib.item()/100, n_iter_val)
        n_iter_val += 1

    writer.close()

def single_experiment(writer, w, l_1, l_2, l_3, mtl_weight, min_cr, 
                        epoch_num, lr_list, batch_size, mute=True, 
                        csv_file = "data/solar_2015.csv"):
    
    global use_cuda, n_iter, n_iter_val
    
    n_iter = 0
    n_iter_val = 0
    print("Parameters")
    print("     w     l_1     l_2     l_3     mtl     ")
    print("  {:.3f}   {:.3f}   {:.3f}   {:.3f}   {:.3f}".format(w, l_1, l_2, l_3, mtl_weight))

    hidden_size = 10
    num_classes = 1

    input_size = 2 * num_classes
    seq_length = 25
    num_layers = 3
    
    # df_header = pd.read_csv(csv_file, nrows=1) ## general information (e.g. time zone, elevation)
    df= pd.read_csv(csv_file, header = 2)

    data_raw = load_power_shortage(df)

    n_trian_step=24*60
    n_val_step=24*30
    # n_test_step=24*60

    ## Splitting training and testing dataset
    data_raw = data_raw.reshape([-1,1])
    train_raw=data_raw[:n_trian_step, :]
    val_raw=data_raw[n_trian_step:n_trian_step+n_val_step, :]
    
    train_seq = sliding_windows(train_raw, seq_length)
    
    
    traject_dataset_train = TrajectCR_Dataset(train_seq, w, mute=mute)
    train_dataloader = DataLoader(traject_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

    val_seq = val_raw.reshape([1,-1,1])
    val_seq_tensor = torch.from_numpy(val_seq).float()
    if use_cuda: val_seq_tensor = val_seq_tensor.cuda()

    lstm = LSTM_unroll(num_classes, input_size, hidden_size, num_layers, 
                            seq_length, w, l_1, l_2, l_3)
        
    optimizer = optim.Adam(lstm.parameters(), lr=lr_list[0])
    if use_cuda: lstm = lstm.cuda()

    for lr in lr_list:
        train_cr(lstm, optimizer, writer, train_dataloader, val_seq_tensor, 
            epoch_num, w, min_cr, mtl_weight = mtl_weight, mute=mute, l_1=l_1, l_2=l_2, l_3=l_3) #added l_1, l_2, l_3 to call to track output
        optimizer.param_groups[0]["lr"] = lr
    
    pth_path = writer.get_logdir() + "lstm_unroll.pth"
    torch.save(lstm.state_dict(), pth_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a L2O Model')
    parser.add_argument('config', help='train config file path')
    
    args = parser.parse_args()
    return args
def ltwo(l1, l3,m,alp):
    l2 = (m*l1/(2*B))*(math.sqrt((1+(B/m)*(l3/l1))**2 + 4*B**2/(alp*m)) + 1 - 2/l1 - (B/m)*(l3/l1))
    return l2

def optimal_cost_calc(raw_data, seq_length):
    val_seq = sliding_windows(raw_data, seq_length)

    num_val_seq = val_seq.shape[0]
    optimal_val_cost_array = np.zeros(num_val_seq)
        
    #print("Calculating Offline Optimal values ...")
    #if mute:
    seq_val_list = range(num_val_seq)
    #else:
    #    seq_list = tqdm(range(num_seq))
    for i in seq_val_list:
        sample_seq = val_seq[i,:,:]
        _, optimal_cost = calculate_offline_optimal(sample_seq[1:], sample_seq[0], switch_weight=w)
        optimal_val_cost_array[i] = optimal_cost

    return optimal_val_cost_array

if __name__ == "__main__":
    '''
    args = parse_args()
    with  open(args.config, "r") as f:
        config_data = json.load(f)

    # Problem definition
    w = config_data["w"]
    l_1 = config_data["l_1"]
    l_2 = config_data["l_2"]
    l_3 = config_data["l_3"]
    min_cr = config_data["min_cr"]

    # Traing parameters
    epoch_num = config_data["epoch_num"]
    lr_list = config_data["lr_list"]
    batch_size = config_data["batch_size"]
    
    # Experiment parameters
    base_log_dir = config_data["base_log_dir"]
    mtl_list = np.array(config_data["mtl_list"])
    '''
    #for mtl_weight in mtl_list:
    #    writer_path = base_log_dir + "/mtl_{:.2f}/".format(mtl_weight)
    #    writer = SummaryWriter(writer_path)
    #    single_experiment(writer, w, l_1, l_2, l_3, mtl_weight, min_cr, 
    #                epoch_num, lr_list, batch_size, mute=False)
    
    #random.seed(10)
    hidden_size = 10
    num_classes = 1

    input_size = 2 * num_classes
    seq_length = 25
    num_layers = 3
    base_log_dir = "./ec_l2o_log/"

    csv_file = "data/solar_2015.csv"
    mute = True
    use_cuda = False
    w = 10
    min_cr = 2
    mtl_weigh = .6
    switch_weight = 1
    mtl_list = np.array([0.5])

    m = 2
    B = 10
    alp = 10

    df_train = pd.DataFrame(columns=[ 'epoch', 'type', 'loss', 'loss_ml', 'loss_calib'])
    df_val = pd.DataFrame(columns=[ 'epoch', 'type', 'loss'])
    df_test = pd.DataFrame(columns=[ 'epoch', 'type', 'quarter', 'average', 'cr'])
    #df_testp = pd.DataFrame(columns=[ 'epoch', 'type', 'quarter', 'p2', 'norm'])

    df_train_dy = pd.DataFrame(columns=[ 'epoch', 'type', 'loss', 'loss_ml', 'loss_calib'])
    df_val_dy = pd.DataFrame(columns=[ 'epoch', 'type', 'loss'])
    df_test_dy = pd.DataFrame(columns=[ 'epoch', 'type', 'quarter', 'average', 'cr'])
    new_index = pd.RangeIndex(20000)
    df_testp = pd.DataFrame(index=new_index,columns=['type', 'quarter', 'p2', 'norm'])



    '''
    def ltwo(l1, l3,m,alp):
        l2 = (m*l1/(2*B))*(math.sqrt((1+(B/m)*(l3/l1))**2 + 4*B**2/(alp*m)) + 1 - 2/l1 - (B/m)*(l3/l1))
        return l2

    def optimal_cost_calc(raw_data, seq_length):
        val_seq = sliding_windows(raw_data, seq_length)

        num_val_seq = val_seq.shape[0]
        optimal_val_cost_array = np.zeros(num_val_seq)
            
        #print("Calculating Offline Optimal values ...")
        #if mute:
        seq_val_list = range(num_val_seq)
        #else:
        #    seq_list = tqdm(range(num_seq))
        for i in seq_val_list:
            sample_seq = val_seq[i,:,:]
            _, optimal_cost = calculate_offline_optimal(sample_seq[1:], sample_seq[0], switch_weight=w)
            optimal_val_cost_array[i] = optimal_cost

        return optimal_val_cost_array
    '''


    pureml = {}
    pureml["l1"] = 1
    pureml["l2"] = 0
    pureml["l3"] = 0
    pureml["mtl_weight"] = 1
    pureml["c"] =1
    pureml["traincalib"] = False
    pureml["testcalib"] = False
    pureml["name"] = "pureml"

    robd = {}
    robd["l1"] = 1
    robd["l2"] = ltwo(1, 0, m, alp)
    robd["l3"] = 0
    robd["mtl_weight"] = .6
    robd["c"] =1
    robd["traincalib"] = True
    robd["testcalib"] = True
    robd["name"] = "robd"


    mlrobd = {}
    mlrobd["l1"] = 1
    mlrobd["l2"] = ltwo(1, 0, m, alp)  #changed l3 to 0 different from paper to check
    mlrobd["l3"] = 0   #should robd have non-zero l3
    mlrobd["mtl_weight"] = 1
    mlrobd["c"] =1
    mlrobd["traincalib"] = False
    mlrobd["testcalib"] = True
    mlrobd["name"] = "mlrobd"

    ml2robd = {}
    ml2robd["l1"] = 1
    ml2robd["l2"] = ltwo(1, .3, m, alp)  
    ml2robd["l3"] = .3   
    ml2robd["mtl_weight"] = 1
    ml2robd["c"] =1
    ml2robd["traincalib"] = False
    ml2robd["testcalib"] = True
    ml2robd["name"] = "ml2robd"


    ecl20 = {}
    ecl20["l1"] = 1
    ecl20["l3"] = 0.5
    ecl20["l2"] = ltwo(1, .5, m, alp)
    ecl20["mtl_weight"] = 0.6
    ecl20["c"] =1
    ecl20["traincalib"] = True
    ecl20["testcalib"] = True
    ecl20["name"] = "ecl20"

    ecl20b = {}
    ecl20b["l1"] = 1
    ecl20b["l3"] = 0.5
    ecl20b["l2"] = ltwo(1, .5, m, alp)
    ecl20b["mtl_weight"] = 0.2
    ecl20b["c"] =1
    ecl20b["traincalib"] = True
    ecl20b["testcalib"] = True
    ecl20b["name"] = "ecl20b"

    epoch_number = 80 #80 journal change
    lr = 4e-4
    
    #def replication(input_dict, csv_file, epoch_number=25, lr=4e-4):
    for input_dict in [ecl20]: # [pureml, robd, mlrobd, ecl20, ecl20b]:    
        random.seed(10)
        #np.random.seed(10)
        #torch.manual_seed(10)
        l_1 = input_dict["l1"]
        l_2 = input_dict["l2"]
        l_3 = input_dict["l3"]
        n_iter = 0
        n_iter_val = 0
        epoch_num = epoch_number
        batch_size = 50 # journal change 50 #change from 50 to 5 depending on debudding
        #print("Parameters")
        #print("     w     l_1     l_2     l_3     mtl     ")
        #print("  {:.3f}   {:.3f}   {:.3f}   {:.3f}   {:.3f}".format(w, l_1, l_2, l_3, mtl_weight))

        hidden_size = 10
        num_classes = 1

        input_size = 2 * num_classes
        seq_length = 25 #25 #25  journal change
        num_layers = 3

        
        mtl_weight = input_dict["mtl_weight"]

        # df_header = pd.read_csv(csv_file, nrows=1) ## general information (e.g. time zone, elevation)
        df= pd.read_csv(csv_file, header = 2)

        data_raws = load_power_shortage(df)

        
        n_trian_step= 24*60 #24*60   
        n_val_step= 24*30 #  24*30   
        n_test_step= 24*90 # 24*60   3 months instead of two
        offset = 0 #Used for debugging to get zeros in input value
        period = 3 # to use n_train_step has to equal n_test_step

        ## Splitting training and testing dataset
        data_raw = data_raws.reshape([-1,1])
        train_raw=data_raw[offset:n_trian_step+offset, :]
        val_raw=data_raw[n_trian_step+offset:n_trian_step+n_val_step+offset, :]
        test_raw = {}
        for i in range(1,4):    
            test_raw[i] = data_raw[n_trian_step + n_val_step+offset + (i-1)*n_test_step:n_trian_step+n_val_step + i*n_test_step+offset,:]
        #print(f'train raw is {train_raw.size}') 
        test_raw[4] = data_raw[n_trian_step + n_val_step+offset + 0*n_test_step:n_trian_step+n_val_step + 3*n_test_step+offset,:]    
        train_seq = sliding_windows(train_raw, seq_length)
        val_seq = sliding_windows(val_raw, seq_length)

        temp_seq = {}
        bsize = {}
        op_temp_cost_array = {}

        for i in range(1,5):
            temp_seq[i] = sliding_windows(test_raw[i], seq_length)
            op_temp_cost_array[i] = optimal_cost_calc(test_raw[i], seq_length)
        #temp_seq = sliding_windows(temp_raw, seq_length)
            bsize[i] = temp_seq[i].shape[0]
        #print(f'bsize is {bsize}')

        op_val_cost_array = optimal_cost_calc(val_raw, seq_length)
        #op_temp_cost_array = optimal_cost_calc(test_raw, seq_length)

        #print(f'train_seq is {train_seq.size}')
        traject_dataset_train = TrajectCR_Dataset(train_seq, w, mute=mute)
        train_dataloader = DataLoader(traject_dataset_train, batch_size=batch_size, shuffle=False, num_workers=4)

        traject_dataset_val = TrajectCR_Dataset(val_seq, w, mute=mute)
        val_dataloader = DataLoader(traject_dataset_val, batch_size=val_seq.shape[0], shuffle=False, num_workers=4)

        traject_dataset_temp = {}
        temp_dataloader = {}
        
        for i in range(1,5):
            traject_dataset_temp[i] = TrajectCR_Dataset(temp_seq[i], w, mute=mute)
            temp_dataloader[i] = DataLoader(traject_dataset_temp[i], batch_size=bsize[i], shuffle=False, num_workers=4)

        val_seq = val_raw.reshape([1,-1,1])
        val_seq_tensor = torch.from_numpy(val_seq).float()
        if use_cuda: val_seq_tensor = val_seq_tensor.cuda()

        
        #random.seed(10)
        #np.random.seed(10)
        #torch.manual_seed(10)

        lstm = LSTM_unroll(num_classes, input_size, hidden_size, num_layers, 
                            seq_length, w, l_1, l_2, l_3)

        #random.seed(10)
        #np.random.seed(10)
        #torch.manual_seed(10)
        
        lstm_dynamic = LSTM_unroll_dynamic(num_classes, input_size, hidden_size, num_layers, 
                        seq_length, w, l_1, l_2, l_3)

        optimizer = optim.Adam(lstm.parameters(), lr=lr)
        if use_cuda: lstm = lstm.cuda()

        optimizer_dy = optim.Adam(lstm_dynamic.parameters(), lr=lr)
        if use_cuda: lstm_dynamic = lstm_dynamic.cuda()
        #writer_path = base_log_dir + "/mtl_{:.2f}/".format(mtl_weight)
        #writer = SummaryWriter(writer_path)
        #del train_cr
        #from train_unroll_batch import train_cr

        #train_cr_vals(lstm, optimizer, writer, train_dataloader, val_seq_tensor, 
        #            epoch_num, pureml["c"], min_cr, mtl_weight=pureml["mtl_weight"], mute=mute, l_1=pureml["l1"],l_2=pureml["l2"], l_3=pureml["l3"], calib=pureml["calib"])
        
        #train_cr_vals(lstm, optimizer, train_dataloader, val_dataloader, val_seq_tensor,
         #       epoch_num, input_dict["c"], min_cr,temp_seq,input_dict,op_temp_cost_array, df_train, df_test,
         #       df_val, mtl_weight=input_dict["mtl_weight"], mute=mute, l_1=input_dict["l1"],
          #      l_2=input_dict["l2"], l_3=input_dict["l3"], calib=input_dict["traincalib"])
        
        
        train_cr_vals_dynamic(lstm_dynamic, optimizer_dy, train_dataloader, val_dataloader,temp_dataloader, temp_seq,
                              op_val_cost_array, epoch_num, input_dict["c"], min_cr, input_dict, op_temp_cost_array,
                              df_train_dy, df_test_dy, df_val_dy, df_testp, mtl_weight=input_dict["mtl_weight"], mute=mute, 
                              l_1=input_dict["l1"],l_2=input_dict["l2"], l_3=input_dict["l3"], 
                              calib=input_dict["traincalib"])
    '''
        for i in range(1,5):
            #temp_seq = sliding_windows(test_raw, seq_length)
            #temp_seq.shape
            temp_action_seq = torch.zeros(temp_seq[i].shape)
            #temp_action_seq.shape
            lstm.eval()
            tmval = lstm(torch.from_numpy(temp_seq[i]).float(), mode="val", calib=input_dict["testcalib"])
            #print(tmval.shape)
            #print(torch.max(tmval))
            hit_cost = torch.norm(tmval - torch.from_numpy(temp_seq[i]).float(), dim=1)
            hit_cost = (1/2)*(hit_cost)**2
            #print(torch.max(hit_cost))

            #switch_diff = action[:,1:,:] - action[:,:-1,:]
            switch_cost = torch.norm(tmval[:,1:,:] - tmval[:,:-1,:], dim=1)
            switch_cost = (10/2)*(switch_cost)**2
            #print(switch_cost.shape)
            val_cost_norm = hit_cost + switch_cost
            norm_cost = torch.div(val_cost_norm, torch.from_numpy(op_temp_cost_array).unsqueeze(1))
            #print(torch.max(norm_cost))
            average_cost_value = torch.mean(norm_cost).item()
            cr_value = torch.max(norm_cost).item()
            return [average_cost_value, cr_value]
    '''

    os.makedirs('result', exist_ok=True) 
    #df_train.to_csv('result/df_train')
    #df_val.to_csv('result/df_val')
    #df_test.to_csv('result/df_test')

    df_train_dy.to_csv('result/df_2train_dy')
    df_val_dy.to_csv('result/df_2val_dy')
    df_test_dy.to_csv('result/df_2test_dy')
    df_testp.to_csv('result/df_testpone_four')
        
        
    print('Finished Training')