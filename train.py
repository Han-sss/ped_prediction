import torch
import torch.nn as nn

import os
import time
import pickle
import argparse
import numpy as np
from layer import QRNNLayer
from model import QRNNModel

import data.dataloader as data_loader
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

"""
Train the model
"""

device = torch.device('cuda:3')

def create_model(config):  # not to use config.num_symbols, but actual dict size
    print('Creating new model parameters..')
    model = QRNNModel(config.dim_in, config.dim_hid, config.dim_out,
                      QRNNLayer, config.num_layers, config.kernel_size,
                      config.hidden_size, config.batch_size,
                      config.frames, config.dec_size, config.out_size, device)

    # Initialize a model state
    model_state = vars(config)
    model_state['epoch'], model_state['train_steps'] = 0, 0
    model_state['state_dict'] = None
    model_path = os.path.join(config.model_dir, config.model_name)

    # If training stops half way, restart training from last checkpoint of previous training.
    if os.path.exists(model_path):
        print('Reloading model parameters..')
        checkpoint = torch.load(model_path)

        model_state['epoch'] = checkpoint['epoch']
        model_state['train_steps'] = checkpoint['train_steps']
        model.load_state_dict(checkpoint['state_dict'])

    print('Using gpu..')
    model.train().to(device)
    print(next(model.parameters()).is_cuda)

    return model, model_state

def diff_trans(output_diff, index):

    output = torch.zeros_like(output_diff).to(device)

    for i in range(len(output_diff)):
        ind = index[i][-1]
        with open('/home/tfukuda/qrnn/data/labels/'+str(int(ind[0])).zfill(4)+'_ped'+str(int(ind[2]))+'.pkl', 'rb') as f:
            bbox = pickle.load(f)
            
        p_bbox = torch.tensor(bbox[int(ind[1])]).to(device)

        for j in range(len(output_diff[i])):
            output[i][j] = p_bbox + output_diff[i][j]
            p_bbox = output[i][j]
    return output



def train(config):
    print('Loading data..')
    # Creating data for training and test splits
    dataset = data_loader.Seq_Data(config.frames, config.dim_in, config.dim_out)

    # Get random datas by using SubsetRandomSampler
    batch_size = config.batch_size
    test_split = 0.2
    shuffle_dataset = True
    random_seed= 42

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_datas = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True, num_workers=2)
    test_datas = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, drop_last=True, num_workers=2)

    print('train data length: ', len(train_datas))
    print('test data length: ', len(test_datas))

    model, model_state = create_model(config)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    loss = 0.0

    start_time = time.time()

    # Training loop
    print('Training..')
    for epoch in range(config.epochs):
        model_state['epoch'] += 1
        for data, label, index in train_datas:

            data = data.requires_grad_().to(device)
            label = label.to(device)
            index = index.to(device)

            # Execute a single training step
            optimizer.zero_grad()
            output_diff = model(data)
            output = diff_trans(output_diff, index)
            step_loss = criterion(output, label)
            step_loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), config.max_grad_norm)
            optimizer.step()

            loss += float(step_loss) / config.display_freq

            model_state['train_steps'] += 1

            # Display the average loss of training
            if model_state['train_steps'] % config.display_freq == 0:
                avg_loss = float(loss)
                time_elapsed = time.time() - start_time
                step_time = time_elapsed / config.display_freq

                print('Epoch ', model_state['epoch'], 'Step ', model_state['train_steps'], \
                      'Loss {0:.2f}'.format(avg_loss), 'Step-time {0:.2f}'.format(step_time))
                loss = 0.0
                start_time = time.time()

            # Test step start
            if model_state['train_steps'] % config.test_freq == 0:
                model.eval()
                print('Test step')
                test_steps = 0
                test_loss = 0.0

                for test_data, test_label, test_index in test_datas:

                    test_data = test_data.requires_grad_().to(device)
                    test_label = test_label.to(device)
                    test_index = test_index.to(device)

                    test_output_diff = model(test_data)
                    test_output = diff_trans(test_output_diff, test_index)
                    step_loss = criterion(test_output, test_label)
                    test_steps += 1 
                    test_loss += float(step_loss)

                model.train()
                #Display loss of test steps
                print('Test Loss: {0:.2f}'.format(test_loss / test_steps))

            # Save the model checkpoint
            if model_state['train_steps'] % config.save_freq == 0:
                print('Saving the model..')

                model_state['state_dict'] = model.state_dict()
                model_path = os.path.join(config.model_dir, config.model_name)
                torch.save(model_state, model_path)

        # Increase the epoch index of the model
        print('Epoch {0:} DONE'.format(model_state['epoch']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Network parameters
    parser.add_argument('--dim_in', type=float, default=17)
    parser.add_argument('--dim_hid', type=float, default=8)
    parser.add_argument('--dim_out', type=float, default=4)
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dec_size', type=int, default=2)
    parser.add_argument('--out_size', type=int, default=4)

    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--display_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=200)
    parser.add_argument('--test_freq', type=int, default=200)
    parser.add_argument('--model_dir', type=str, default='model/')
    parser.add_argument('--model_name', type=str, default='model_diff_hid-8_dec-2_ep-16.pkl')

    config = parser.parse_args()
    print(config)
    train(config)
    print('DONE')