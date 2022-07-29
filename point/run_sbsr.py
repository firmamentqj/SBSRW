import torch
import torch.nn as nn
import sys
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import scipy
import scipy.spatial

from tqdm import tqdm
import pickle as pkl

import torchvision.transforms as transforms
import torchvision

import argparse
import numpy as np
import time
import os

from util import *
from ops import *

from model.pointnet import *
from model.shape_head import *
from model.sketch_head import *

from torch.utils.tensorboard import SummaryWriter
from dataloader_shrec22 import *

PLOT_SAMPLE_NBS = [242, 7, 549, 112, 34]


parser = argparse.ArgumentParser(description='semantic SBSR')

parser.add_argument('--run_mode', '-rmode',  default="train", choices=["train", "test"],
                    help='The mode of running the code: train, test classification, test retrieval, test rotation robustness, or test occlusion robustness. You have to train before testing')
parser.add_argument('--gpu', type=int,
                    default=0, help='GPU number ')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--batch_size', '-b', default=8, type=int,
                    help='mini-batch size (default: 20)')
parser.add_argument('-r', '--resume', dest='resume',
                    action='store_true', help='continue training from the `setup[weights_file] checkpoint ')
parser.add_argument('--config_file', '-cfg',  default="config_trans.yaml", help='the conifg yaml file for more options.')
parser.add_argument('--task', '-t',  default="cad", choices=["cad", "wild"],
                    help='Train on which tasks')
parser.add_argument('--sketch_head', '-sk',  default="config_trans.yaml",
                    help='sketch head pretrain')


args = parser.parse_args()
args = vars(args)
config = read_yaml(args["config_file"])
setup = {**args, **config}
initialize_setup(setup)

print('Loading data')

torch.cuda.set_device(int(setup["gpu"]))

dset_train = SHREC22("train", setup["task"],setup["data_pth"])
dset_val_sketch = SHREC22("test_sketch", setup["task"],setup["data_pth"])
dset_val_shape = SHREC22("test_shape", setup["task"],setup["data_pth"])

if setup["task"]=="cad":
    print("CLASS CAD")
    classes = 44
else:
    print("CLASS WILD")
    classes = 10

train_loader = DataLoader(dset_train, batch_size=setup["batch_size"],
                          shuffle=True, num_workers=6, collate_fn=collate_fn, drop_last=True)

val_loader_sketch = DataLoader(dset_val_sketch, batch_size=int(setup["batch_size"]),
                        shuffle=False, num_workers=6, collate_fn=collate_fn)

val_loader_shape = DataLoader(dset_val_shape, batch_size=int(setup["batch_size"]),
                        shuffle=False, num_workers=6, collate_fn=collate_fn)

print("classes nb:", classes, "number of train models: ", len(
    dset_train), "number of test sketch: ", len(dset_val_sketch), "number of test shape: ", len(dset_val_shape))


depth2featdim = {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}
assert setup["depth"] in list(
    depth2featdim.keys()), "the requested resnt depth not available"

sketch_head = EncoderCNN(classes)
shape_head = FeatureExtractor(classes,shape_extractor="DGCNN", screatch_feature_extractor=True)

sketch_head.cuda()
shape_head.cuda()

cudnn.benchmark = True

print('Running on ' + str(torch.cuda.current_device()))

print(setup)

lr = setup["learning_rate"]
n_epochs = setup["epochs"]
criterion = nn.CrossEntropyLoss()
tripletloss = nn.TripletMarginLoss(margin=0.2)

params = list(sketch_head.parameters()) + list(shape_head.parameters())
optimizer = torch.optim.AdamW(
    params, lr=lr, weight_decay=setup["weight_decay"])

def train(data_loader, sketch_head,optimizer,  setup):
    train_size = len(data_loader)
    total = 0.0
    correct_3D_pos= 0.0
    correct_3D_neg = 0.0
    correct_2D = 0.0
    total_loss = 0.0
    n = 0

    for i, (points_pos, targets_3D_pos,points_neg, targets_3D_neg,sketches) in enumerate(data_loader):
        optimizer.zero_grad()

        points_pos=points_pos.cuda()
        targets_3D_pos=targets_3D_pos.long().cuda()
        points_neg=points_neg.cuda()
        targets_3D_neg=targets_3D_neg.long().cuda()
        sketches=sketches.cuda()

        glo_feat_3D_pos, _ = shape_head(points_pos)
        glo_feat_3D_neg, _ = shape_head(points_neg)
        glo_feat_2D, _ = sketch_head(sketches)

        loss_3D_pos = criterion(glo_feat_3D_pos, targets_3D_pos)
        loss_3D_neg = criterion(glo_feat_3D_neg, targets_3D_neg)
        loss_2D = criterion(glo_feat_2D, targets_3D_pos)

        triplet = tripletloss(glo_feat_2D, glo_feat_3D_pos, glo_feat_3D_neg)

        loss=(loss_3D_pos+loss_3D_neg)+(loss_2D)*10.0+2.0*triplet

        _, predicted_3D_pos = torch.max(glo_feat_3D_pos.data, 1)
        _, predicted_3D_neg = torch.max(glo_feat_3D_neg.data, 1)
        _, predicted_2D = torch.max(glo_feat_2D.data, 1)

        total += targets_3D_pos.size(0)

        loss.backward()

        optimizer.step()

        if (i + 1) % setup["print_freq"] == 0:
            print("\tIter [%d/%d] Loss: %.4f" %
                  (i + 1, train_size, loss.item()))

        correct_3D_pos += (predicted_3D_pos.cpu() == targets_3D_pos.cpu()).sum()
        correct_3D_neg += (predicted_3D_neg.cpu() == targets_3D_neg.cpu()).sum()
        correct_2D += (predicted_2D.cpu() == targets_3D_pos.cpu()).sum()
        total_loss += loss.item()
        n += 1
        # break
    avg_loss = total_loss / n
    avg_train_acc_3D_pos = 100 * correct_3D_pos / total
    avg_train_acc_3D_neg = 100 * correct_3D_neg / total
    avg_train_acc_2D = 100 * correct_2D / total

    return avg_train_acc_3D_pos,avg_train_acc_3D_neg,avg_train_acc_2D, avg_loss

def evluate(val_loader_sketch,val_loader_shape, sketch_head):

    proj_glo_feat_3D_list=[]
    proj_glo_feat_2D_list=[]

    for i, (sketches,s_label) in enumerate(val_loader_sketch):
        with torch.no_grad():

            sketches = sketches.cuda()

            glo_feat_2D, _ = sketch_head(sketches)

            proj_glo_feat_2D_list.append(glo_feat_2D)

    for i, (shape,shape_label) in enumerate(val_loader_shape):
        with torch.no_grad():
            shape = shape.float().cuda()

            glo_feat_3D_pos, _ = shape_head(shape)

            proj_glo_feat_3D_list.append(glo_feat_3D_pos)


    # return avg_test_acc_3D_pos,avg_test_acc_3D_neg,avg_test_acc_2D, avg_loss
    return proj_glo_feat_3D_list,proj_glo_feat_2D_list

if setup["resume"] or "test" in setup["run_mode"]:
    load_checkpoint(setup, shape_head,sketch_head,optimizer, setup["weights_file"])
    print("LOADING WEGIHT FILE")

if setup["run_mode"] == "train":
    for epoch in range(setup["start_epoch"], n_epochs):
        setup["c_epoch"] = epoch
        print('\n-----------------------------------')
        print('Epoch: [%d/%d]' % (epoch + 1, n_epochs))
        start = time.time()
        sketch_head.train()
        shape_head.train()
        avg_train_acc_3D_pos,avg_train_acc_3D_neg,avg_train_acc_2D, avg_train_loss = train(
            train_loader, sketch_head,optimizer, setup)
        print('Time taken: %.2f sec.' % (time.time() - start))

        avg_train_acc=0.3*(avg_train_acc_3D_pos+avg_train_acc_3D_neg+avg_train_acc_2D)
        print('\nEvaluation:')
        print('\ntrain acc: %.2f - train Loss: %.4f' %
              (avg_train_acc, avg_train_loss))
        print('\ttrain acc 3D pos: %.2f - train acc 3D neg: %.2f - train acc 2D: %.2f' %
              (avg_train_acc_3D_pos, avg_train_acc_3D_neg,avg_train_acc_2D))

        print('\nCurrent best train acc: %.2f' % setup["best_acc"])
        saveables = {'epoch': epoch + 1,
                     'sketch_head': sketch_head.state_dict(),
                     "shape_head": shape_head.state_dict(),
                     'acc': avg_train_acc,
                     'best_acc': setup["best_acc"],
                     'optimizer': optimizer.state_dict(),
                     }
        if setup["save_all"]:
            save_checkpoint(saveables, setup, None,
                            setup["weights_file"])

        if avg_train_acc >= setup["best_acc"]:
            print('\tSaving checkpoint - Acc: %.2f' % avg_train_acc)
            saveables["best_acc"] = avg_train_acc
            setup["best_loss"] = avg_train_loss
            # setup["best_acc"] = avg_test_acc.item()
            setup["best_acc"] = avg_train_acc
            save_checkpoint(saveables, setup, None,
                            setup["weights_file"])
        if (epoch + 1) % setup["lr_decay_freq"] == 0:
            lr *= setup["lr_decay"]
            optimizer = torch.optim.AdamW(params, lr=lr)
            print('Learning rate:', lr)

if setup["run_mode"] == "test":
    print('\n-----------------------------------')
    start = time.time()

    sketch_head.eval()
    shape_head.eval()

    shape_feat, sketch_feat = evluate(
        val_loader_sketch, val_loader_shape, sketch_head)
    shape_feat = torch.cat(shape_feat, axis=0)
    sketch_feat = torch.cat(sketch_feat, axis=0)
    shape_feat = shape_feat.cpu().detach().numpy()
    sketch_feat = sketch_feat.cpu().detach().numpy()

    np.savetxt("./results/"+setup["exp_set"]+"/"+setup["exp_id"]+"/shape_feat.npz", shape_feat)
    np.savetxt("./results/" + setup["exp_set"] + "/" + setup["exp_id"] + "/sketch_feat.npz",sketch_feat)
    print('Time taken: %.2f sec.' % (time.time() - start))
    print('\nDone:')

