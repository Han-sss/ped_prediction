import torch
import torch.nn as nn

import os
import argparse
import pickle
from layer import QRNNLayer
from model import QRNNModel

import data.dataloader as data_loader
from torch.utils.data import DataLoader

import cv2

"""
Evaluate model and output pictures with ground truth/predicted bounding boxes
"""

device = torch.device('cuda:3')

def load_model(config):
    if os.path.exists(config.model_path):
        print('Reloading model parameters..')
        checkpoint = torch.load(config.model_path)
        model = QRNNModel(checkpoint['dim_in'], checkpoint['dim_hid'], checkpoint['dim_out'],
                          QRNNLayer, checkpoint['num_layers'], checkpoint['kernel_size'],
                          checkpoint['hidden_size'], checkpoint['batch_size'],
                          checkpoint['frames'], checkpoint['dec_size'], checkpoint['out_size'], device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError('No such file:[{}]'.format(config.model_path))
    for key in config.__dict__:
        checkpoint[key] = config.__dict__[key]
    print('Using gpu..')
    model = model.eval().to(device)
    return model, checkpoint


def diff_trans(output_diff, index):

    output = torch.zeros_like(output_diff).to(device)

    for i in range(len(output_diff)):
        ind = index[i][-1]
        with open('/home/tfukuda/qrnn/data/labels/eval/'+str(int(ind[0])).zfill(4)+'_ped'+str(int(ind[2]))+'.pkl', 'rb') as f:
            bbox = pickle.load(f)
            
        p_bbox = torch.tensor(bbox[int(ind[1])]).to(device)

        for j in range(len(output_diff[i])):
            output[i][j] = p_bbox + output_diff[i][j]
            p_bbox = output[i][j]
    return output



def evaluate(config):
    model, config = load_model(config)
    # Load data to eval
    dataset = data_loader.Seq_Data_eval(config['frames'], config['dim_in'], config['dim_out'])

    eval_datas = DataLoader(dataset, batch_size=config['batch_size'], drop_last=True, num_workers=2)

    criterion = nn.L1Loss()

    loss = 0.0
    step = 0

    print('Evaluation starts..')
    for data, label, index in eval_datas:

        data = data.requires_grad_().to(device)
        label = label.to(device)
        output_diff = model(data)
        output = diff_trans(output_diff, index)
        step_loss = criterion(output, label)

        loss += float(step_loss)
        step += 1
        
        for pred, lab, idx in zip(output, label, index):
            for p, l, i in zip(pred, lab, idx):

                img = cv2.imread('/home/tfukuda/qrnn/data/images/video_'+str(int(i[0])).zfill(4)+'/'+str(int(i[1])+30).zfill(5)+'.png')
                p = torch.round(p)
                cv2.rectangle(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (255,0,0), 3)
                cv2.rectangle(img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0,255,0), 3)
                dir_path = '/home/tfukuda/qrnn/data/eval_images/diff2/'+str(int(i[0])).zfill(4)+'/ped'+str(int(i[2]))+'/'
                os.makedirs(dir_path, exist_ok=True)
                cv2.imwrite(dir_path+str(int(i[1])).zfill(5)+'.png', img)

    print(loss/step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Decoding parameters
    parser.add_argument('-model_path', type=str, default='model/model_diff_hid-8_dec-2_ep-16.pkl')
    parser.add_argument('-batch_size', type=int, default=16)
    
    config = parser.parse_args()

    print(config)
    evaluate(config)
    print('DONE')

    