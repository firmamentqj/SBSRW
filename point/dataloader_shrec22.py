import argparse
import torch
import pickle
import h5py
import torch.utils.data as data
import torchvision.transforms as transforms
import os, sys
from random import randint
from PIL import Image
import random
import torchvision.utils as tutils
import torchvision.transforms.functional as F
import collections
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures
from torch.utils.data import DataLoader

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SHREC22(data.Dataset):
    def __init__(self, mode,task,data_pth):
        self.dset_norm = 2
        self.mode = mode
        self.sampled_points = 2048

        if task=="cad":
            train_sketch_path = os.path.join(data_pth,'shrec22_cad_train_sketch.h5')
            test_sketch_path = os.path.join(data_pth,'shrec22_cad_test_sketch.h5')
            model_path = os.path.join(data_pth,'shrec22_model_cad_train.h5')
            model_test_path=os.path.join(data_pth,'shrec22_model_cad_test.h5')
        else:
            train_sketch_path = os.path.join(data_pth,'shrec22_wild_train_sketch.h5')
            test_sketch_path = os.path.join(data_pth,'shrec22_wild_test_sketch.h5')
            model_path = os.path.join(data_pth,'shrec22_model_wild.h5')
            model_test_path=os.path.join(data_pth,'shrec22_model_wild.h5')
        if mode== 'train':
            self.sketches, self.sketch_labels,self.sid = load_all_sketch_h5(train_sketch_path)
            self.vertes, self.model_labels, self.model_labels_str, self.model_id= load_h5(model_path)
        elif mode=="test_sketch":
            self.sketches, self.sketch_labels,self.sid=load_all_sketch_h5(test_sketch_path)

        else:
            self.vertes, self.model_labels, self.model_labels_str, self.model_id=load_h5(model_test_path)

        self.topil = transforms.ToPILImage()
        self.transform = get_transform()

    def __getitem__(self, idx):

        if self.mode == 'train':
            sketch_img= self.sketches[idx]
            sketch_label = self.sketch_labels[idx]

            n_flip = random.random()
            sketch_img = torch.tensor(sketch_img)
            sketch_img = self.topil(sketch_img)
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
            sketch_img = self.transform(sketch_img)

            # positive model from the same class
            positive_model_id = np.random.choice(np.where(self.model_labels == sketch_label)[0], 1)[0]
            points_pos = self.vertes[positive_model_id][np.random.choice(
                    self.vertes[positive_model_id].shape[0], self.sampled_points), :]

            # negative model from the other class
            negative_model_id=np.random.choice(np.where(self.model_labels != sketch_label)[0], 1)[0]
            points_neg = self.vertes[negative_model_id][np.random.choice(
                self.vertes[negative_model_id].shape[0], self.sampled_points), :]
            neg_lab=self.model_labels[negative_model_id]

            return points_pos,sketch_label,points_neg, neg_lab, sketch_img

        elif self.mode == "test_sketch":
            sketch_img = torch.tensor(self.sketches[idx])
            sketch_label = self.sketch_labels[idx]
            sketch_img = self.topil(sketch_img)
            sketch_img = self.transform(sketch_img)

            return sketch_img,sketch_label
        else:
            shape = self.vertes[idx][np.random.choice(
                self.vertes[idx].shape[0], self.sampled_points), :]
            shape_label = self.model_labels[idx]
            return shape,shape_label

    def __len__(self):
        if self.mode == 'train':
            return len(self.sketches)
        elif self.mode == "test_sketch":
            return len(self.sketches)
        else:
            return len(self.vertes)


def read_off(file):
    if 'OFF' != file.readline().strip():
        raise ('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    faces = np.array(faces)
    verts = np.array(verts)
    return verts, faces

def load_sketch_debug_h5(pth):
    sketches = []
    labels = []
    sid = []
    ii=0
    with h5py.File(pth, 'r') as file:
        for s_id, s_v in file.items():
            ii+=1
            sketches.append(np.array(s_v["sketch"]))
            labels.append(np.array(s_v["label"]))
            sid.append(s_id)
            if ii>=10:
                break

    return sketches, np.array(labels),np.array(sid).astype(int)

def load_test_sketch_h5(pth):
    sketches = []
    sid = []
    with h5py.File(pth, 'r') as file:
        for s_id, s_v in file.items():
            sketches.append(np.array(s_v["sketch"]))
            sid.append(s_id)
    # print(sketches[0])
    # print(sketches[0].dtype())
    #
    # print("???")
    # sys.exit(0)
    return sketches,np.array(sid).astype(int)

def load_all_sketch_h5(pth):
    sketches = []
    labels = []
    sid = []
    with h5py.File(pth, 'r') as file:
        for s_id, s_v in file.items():
            sketches.append(np.array(s_v["sketch"]))
            labels.append(np.array(s_v["label"]))
            sid.append(s_id)


    return sketches, labels,np.array(sid).astype(int)

def load_sketch_h5(pth, ids):
    out = []
    with h5py.File(pth, 'r') as file:
        for id in ids:
            f = file.get(str(id))
            out.append(np.array(f["sketch"]))
        return out

def load_h5(pth):
    points = []
    labels = []
    labels_str = []
    sid = []
    ii = 0
    with h5py.File(pth, 'r') as file:
        for s_id, s_v in file.items():
            ii += 1
            points.append(np.array(s_v["verts"]))
            labels.append(np.array(s_v["label"]))
            labels_str.append(np.array(s_v["label_str"]))
            sid.append(s_id)
            # if ii>=200:
            #     break
    return points, labels, labels_str,sid

def load_shape_test_h5(pth):
    points = []

    ii = 0
    with h5py.File(pth, 'r') as file:
        for s_id, s_v in file.items():
            ii += 1
            points.append(np.array(s_v["verts"]))

            # if ii>=200:
            #     break
    return points

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_transform():
    # transform_list = [transforms.Resize(299), transforms.ToTensor(),
    #                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    transform_list = [transforms.Resize((224, 224)), transforms.ToTensor(),
                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

    return transforms.Compose(transform_list)

def torch_center_and_normalize(points, p="inf"):
    """
    a helper pytorch function that normalize and center 3D points clouds
    """
    N = points.shape[0]
    center = points.mean(0)
    if p != "fro" and p != "no":
        scale = torch.max(torch.norm(points - center, p=float(p), dim=1))
    elif p == "fro":
        scale = torch.norm(points - center, p=p)
    elif p == "no":
        scale = 1.0
    points = points - center.expand(N, 3)
    points = points * (1.0 / float(scale))
    return points

def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'pytorch3d.structures.meshes':
        return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, (int)):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):

        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

