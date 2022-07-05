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

from models.pointnet import *
from models.mvtn import *
from models.multi_view import *
from models.renderer import *
from models.sketch_head import *


from torch.utils.tensorboard import SummaryWriter
from dataloader_shrec22 import *


PLOT_SAMPLE_NBS = [242, 7, 549, 112, 34]


parser = argparse.ArgumentParser(description='MVTN-PyTorch')

parser.add_argument('--data_dir',   help='path to 3D dataset')
parser.add_argument('--run_mode', '-rmode',  default="train", choices=["test","train", "test_cls", "test_retr", "test_rot", "test_occ"],
                    help='The mode of running the code: train, test classification, test retrieval, test rotation robustness, or test occlusion robustness. You have to train before testing')
parser.add_argument('--mvnetwork', '-m',  default="mvcnn", choices=["mvcnn", "rotnet", "viewgcn"],
                    help='the type of multi-view network used:')
parser.add_argument('--nb_views', type=int,
                    help='number of views in the multi-view setup')
parser.add_argument('--views_config', '-s',  default="spherical", choices=["circular", "random", "learned_circular", "learned_direct", "spherical", "learned_spherical", "learned_random", "learned_transfer", "custom"],
                    help='the selection type of views ')
parser.add_argument('--gpu', type=int,
                    default=0, help='GPU number ')
parser.add_argument('--dset_variant', '-dsetp', help='The variant used of the `ScanObjectNN` dataset  ',
                    default="obj_only", choices=["obj_only", "with_bg", "hardest"])
parser.add_argument('--pc_rendering', dest='pc_rendering',
                    action='store_true', help='use point cloud renderer instead of mesh renderer  ')
parser.add_argument('--object_color', '-clr',  default="white", choices=["white", "random", "black", "red", "green", "blue", "custom"],
                    help='the selection type of views ')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--batch_size', '-b', default=8, type=int,
                    help='mini-batch size (default: 20)')
parser.add_argument('-r', '--resume', dest='resume',
                    action='store_true', help='continue training from the `setup[weights_file] checkpoint ')
parser.add_argument("--viewgcn_phase", default="all", choices=["all", "first", "second"],
                    help='what stage of training of the ViewGCN ( it has two stages)')
parser.add_argument('--config_file', '-cfg',  default="config.yaml", help='the conifg yaml file for more options.')
parser.add_argument('--task', '-t',  default="cad", choices=["cad", "wild"],
                    help='Train on which tasks')

args = parser.parse_args()
args = vars(args)
config = read_yaml(args["config_file"])
setup = {**args, **config}
if setup["mvnetwork"] in ["rotnet", "mvcnn"]:
    initialize_setup(setup)
else:
    initialize_setup_gcn(setup)

print('Loading data')

if setup["task"]=="cad":
    print("CLASS CAD")
    classes = 44
else:
    print("CLASS WILD")
    classes = 10
torch.cuda.set_device(int(setup["gpu"]))

dset_train = SHREC22("train", setup["task"])
dset_val_sketch = SHREC22("test_sketch", setup["task"])
dset_val_shape = SHREC22("test_shape", setup["task"])

train_loader = DataLoader(dset_train, batch_size=setup["batch_size"],
                          shuffle=True, num_workers=6, collate_fn=collate_fn, drop_last=True)

val_loader_sketch = DataLoader(dset_val_sketch, batch_size=int(setup["batch_size"]),
                        shuffle=False, num_workers=6, collate_fn=collate_fn)
val_loader_shape = DataLoader(dset_val_shape, batch_size=int(setup["batch_size"]),
                        shuffle=False, num_workers=6, collate_fn=collate_fn)

print("classes nb:", classes, "number of train models: ", len(
    train_loader), "number of test models: ", len(val_loader_shape), classes)

if setup["mvnetwork"] == "mvcnn":
    depth2featdim = {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}
    assert setup["depth"] in list(
        depth2featdim.keys()), "the requested resnt depth not available"
    mvnetwork = torchvision.models.__dict__[
        "resnet{}".format(setup["depth"])](setup["pretrained"])
    mvnetwork.fc = nn.Sequential()
    mvnetwork = MVAgregate(mvnetwork, agr_type="max",
                           feat_dim=depth2featdim[setup["depth"]], num_classes=classes)
    print('Using ' + setup["mvnetwork"] + str(setup["depth"]))

assert setup["depth"] in list(
    depth2featdim.keys()), "the requested resnt depth not available"

sketch_head = EncoderCNN(classes)

mvnetwork.cuda()
sketch_head.cuda()
cudnn.benchmark = True

print('Running on ' + str(torch.cuda.current_device()))


lr = setup["learning_rate"]
n_epochs = setup["epochs"]


mvtn = MVTN(setup["nb_views"], views_config=setup["views_config"],
            canonical_elevation=setup["canonical_elevation"], canonical_distance=setup["canonical_distance"],
            shape_features_size=setup["features_size"], transform_distance=setup["transform_distance"], input_view_noise=setup["input_view_noise"], shape_extractor=setup["shape_extractor"], screatch_feature_extractor=setup["screatch_feature_extractor"]).cuda()
mvrenderer = MVRenderer(nb_views=setup["nb_views"], image_size=setup["image_size"], pc_rendering=setup["pc_rendering"], object_color=setup["object_color"], background_color=setup["background_color"],
                        faces_per_pixel=setup["faces_per_pixel"], points_radius=setup["points_radius"],  points_per_pixel=setup["points_per_pixel"], light_direction=setup["light_direction"], cull_backfaces=setup["cull_backfaces"])
print(setup)
criterion = nn.CrossEntropyLoss()

tripletloss = nn.TripletMarginLoss(margin=0.2)

params = list(mvnetwork.parameters()) + list(sketch_head.parameters())
optimizer = torch.optim.AdamW(
    params, lr=lr, weight_decay=setup["weight_decay"])
if setup["is_learning_views"]:
    mvtn_optimizer = torch.optim.AdamW(mvtn.parameters(
    ), lr=setup["mvtn_learning_rate"], weight_decay=setup["mvtn_weight_decay"])
else:
    mvtn_optimizer = None


models_bag = {"mvnetwork": mvnetwork,"sketch_head":sketch_head, "optimizer": optimizer,
              "mvtn": mvtn, "mvtn_optimizer": mvtn_optimizer, "mvrenderer": mvrenderer}


def train(data_loader, sketch_head,models_bag, setup ):
    train_size = len(data_loader)
    total = 0.0
    correct = 0.0
    total_loss = 0.0
    n = 0
    correct_3D_pos= 0.0
    correct_3D_neg = 0.0
    correct_2D = 0.0

    for i, (points_pos, targets_3D_pos,points_neg, targets_3D_neg,sketches) in enumerate(data_loader):
        c_batch_size = targets_3D_pos.shape[0]

        models_bag["optimizer"].zero_grad()
        if setup["is_learning_views"]:
            models_bag["mvtn_optimizer"].zero_grad()
        # ===============pos===============
        azim, elev, dist = models_bag["mvtn"](
            points_pos, c_batch_size=c_batch_size)
        rendered_images, _,_,_ = models_bag["mvrenderer"](
            None, points_pos,  azim=azim, elev=elev, dist=dist)
        rendered_images = regualarize_rendered_views(
            rendered_images, setup["view_reg"], setup["augment_training"], setup["crop_ratio"])
        targets_3D_pos=targets_3D_pos.long()
        targets = targets_3D_pos.cuda()
        targets = Variable(targets)
        outputs = models_bag["mvnetwork"](rendered_images)[0]
        loss_pos = criterion(outputs, targets)
        _, predicted_pos = torch.max(outputs.data, 1)

        # ===============neg===============
        azim_neg, elev_neg, dist_neg = models_bag["mvtn"](
            points_neg, c_batch_size=c_batch_size)
        rendered_images_neg, _, _, _ = models_bag["mvrenderer"](
            None, points_neg, azim=azim_neg, elev=elev_neg, dist=dist_neg)
        rendered_images_neg = regualarize_rendered_views(
            rendered_images_neg, setup["view_reg"], setup["augment_training"], setup["crop_ratio"])
        targets_3D_neg = targets_3D_neg.long()
        targets_neg = targets_3D_neg.cuda()
        targets_neg = Variable(targets_neg)
        outputs_neg = models_bag["mvnetwork"](rendered_images_neg)[0]
        loss_neg = criterion(outputs_neg, targets_neg)
        _, predicted_neg = torch.max(outputs_neg.data, 1)

        # ===============2D================
        sketches = sketches.cuda()
        log_2D, glo_feat_2D = sketch_head(sketches)
        loss_2D = criterion(log_2D, targets)
        _, predicted_2D = torch.max(log_2D.data, 1)

        # ===============loss===============
        triplet = tripletloss(log_2D, outputs, outputs_neg)

        loss=(loss_pos+loss_neg)+loss_2D+triplet*3

        correct_3D_pos += (predicted_pos.cpu() == targets_3D_pos.cpu()).sum()
        correct_3D_neg += (predicted_neg.cpu() == targets_3D_neg.cpu()).sum()
        correct_2D += (predicted_2D.cpu() == targets_3D_pos.cpu()).sum()

        total += targets.size(0)

        loss.backward()

        models_bag["optimizer"].step()
        if (i + 1) % setup["print_freq"] == 0:
            print("\tIter [%d/%d] Loss: %.4f" %
                  (i + 1, train_size, loss.item()))
        total_loss += loss.item()
        n += 1
    avg_loss = total_loss / n
    avg_train_acc_3D_pos = 100 * correct_3D_pos / total
    avg_train_acc_3D_neg = 100 * correct_3D_neg / total
    avg_train_acc_2D = 100 * correct_2D / total

    return avg_train_acc_3D_pos,avg_train_acc_3D_neg,avg_train_acc_2D, avg_loss


def evluate_at_end(val_loader_sketch,val_loader_shape,models_bag,sketch_head,setup):

    proj_glo_feat_3D_list = []
    proj_glo_feat_2D_list = []

    for i, (sketches,sketches_label) in enumerate(val_loader_sketch):
        with torch.no_grad():

            sketches = sketches.cuda()

            glo_feat_2D, _ = sketch_head(sketches)

            proj_glo_feat_2D_list.append(glo_feat_2D)
    print("DONE SKETCH")
    for i,  (shape,shape_label)in enumerate(val_loader_shape):
        with torch.no_grad():
            # shape = shape.float().cuda()
            c_batch_size = shape.shape[0]

            azim, elev, dist = models_bag["mvtn"](
                shape, c_batch_size=c_batch_size)
            rendered_images, _, _, _ = models_bag["mvrenderer"](
                None, shape, azim=azim, elev=elev, dist=dist)

            outputs = models_bag["mvnetwork"](rendered_images)[0]
            outputs=torch.unsqueeze(outputs, 0)
            # print(outputs.shape)
            # sys.exit(0)
            proj_glo_feat_3D_list.append(outputs)

    return proj_glo_feat_3D_list,proj_glo_feat_2D_list

if setup["resume"] or "test" in setup["run_mode"]:
    load_checkpoint(setup, models_bag,sketch_head,setup["weights_file"])

if setup["mvnetwork"] == "mvcnn":
    if setup["run_mode"] == "train":
        for epoch in range(setup["start_epoch"], n_epochs):
            setup["c_epoch"] = epoch
            print('\n-----------------------------------')
            print('Epoch: [%d/%d]' % (epoch + 1, n_epochs))
            start = time.time()
            sketch_head.train()
            models_bag["mvnetwork"].train()
            models_bag["mvtn"].train()
            models_bag["mvrenderer"].train()

            avg_train_acc_3D_pos, avg_train_acc_3D_neg, avg_train_acc_2D, avg_train_loss = train(
                train_loader, sketch_head, models_bag, setup)
            print('Time taken: %.2f sec.' % (time.time() - start))

            avg_train_acc = 0.3 * (avg_train_acc_3D_pos + avg_train_acc_3D_neg + avg_train_acc_2D)
            print('\nEvaluation:')
            print('\ntrain acc: %.2f - train Loss: %.4f' %
                  (avg_train_acc, avg_train_loss))
            print('\ttrain acc 3D pos: %.2f - train acc 3D neg: %.2f - train acc 2D: %.2f' %
                  (avg_train_acc_3D_pos, avg_train_acc_3D_neg, avg_train_acc_2D))

            print('\nCurrent best train acc: %.2f' % setup["best_acc"])

            saveables = {'epoch': epoch + 1,
                         'state_dict': models_bag["mvnetwork"].state_dict(),
                         'sketch_head': sketch_head.state_dict(),
                         "mvtn": models_bag["mvtn"].state_dict(),
                         'best_acc': setup["best_acc"],
                         'optimizer': models_bag["optimizer"].state_dict(),
                         'mvtn_optimizer': None if not setup["is_learning_views"] else models_bag["mvtn_optimizer"].state_dict(),
                         }

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

    shape_feat, sketch_feat = evluate_at_end(
        val_loader_sketch, val_loader_shape, models_bag,sketch_head, setup)
    shape_feat = torch.cat(shape_feat, axis=0)
    sketch_feat = torch.cat(sketch_feat, axis=0)
    shape_feat = shape_feat.cpu().detach().numpy()
    sketch_feat = sketch_feat.cpu().detach().numpy()

    np.savetxt("/scratch/sy2366/view/MVTN/results/"+setup["exp_set"]+"/"+setup["exp_id"]+"/shape_feat_mv_"+setup["task"]+".npz", shape_feat)
    np.savetxt("/scratch/sy2366/view/MVTN/results/" + setup["exp_set"] + "/" + setup["exp_id"] + "/sketch_feat_mv_"+setup["task"]+".npz",sketch_feat)
    print('Time taken: %.2f sec.' % (time.time() - start))
    print('\nDone:')