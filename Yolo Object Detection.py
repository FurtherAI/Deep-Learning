import torch as th
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import random_split
from torchvision.ops import box_iou
from torchvision.datasets import VOCDetection, CIFAR10, SVHN
from torch.utils.data import DataLoader

import numpy as np
import cv2
from time import perf_counter
from copy import deepcopy

# @article{yolov3,
#   title={YOLOv3: An Incremental Improvement},
#   author={Redmon, Joseph and Farhadi, Ali},
#   journal = {arXiv},
#   year={2018}
# }

MEAN = np.array([.485, .456, .406])
STD = np.array([.229, .224, .225])

def conv_bn(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, *args, **kwargs),
        nn.BatchNorm2d(out_channels)
    )


class VOCDataModule(pl.LightningDataModule):
    def __init__(self, dir, batch_size=32):
        super().__init__()

        self.dir = dir
        self.batch_size = batch_size

        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.3, .3, .3),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

        self.dims = (3, 224, 224)

    def collate_fn(self, batch):
        images = [item[0] for item in batch]
        labels = [self.prepare_labels(item[1]) for item in batch]
        return [th.stack(images, dim=0), labels]

    def prepare_labels(self, labels):
        name_to_idx = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5,
                       'sheep': 6, 'aeroplane': 7,'bicycle': 8, 'boat': 9, 'bus': 10,
                       'car': 11, 'motorbike': 12, 'train': 13, 'bottle': 14, 'chair': 15,
                       'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19}

        objects = labels['annotation']['object']
        size = labels['annotation']['size']
        size = (int(size['width']), int(size['height']))

        labels = [name_to_idx[object['name']] for object in objects]
        bbox = [(self.process_bbox(object['bndbox'], size)) for object in objects]

        return labels, bbox

    def process_bbox(self, bbox, size):
        center = (((int(bbox['xmin']) + int(bbox['xmax'])) / 2) * (self.dims[1] / size[0]), ((int(bbox['ymin']) + int(bbox['ymax'])) / 2) * (self.dims[2] / size[1]))
        dims = ((int(bbox['xmax']) - int(bbox['xmin'])) * (self.dims[1] / size[0]), (int(bbox['ymax']) - int(bbox['ymin'])) * (self.dims[2] / size[1]))
        return (*center, *dims)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.voc_train = VOCDetection(self.dir, image_set='train', download=False, transform=self.train_transforms)
            self.voc_split = VOCDetection(self.dir, image_set='val', download=False, transform=self.val_transforms)
            self.voc_val = random_split(self.voc_split, (4, len(self.voc_split) - 4))
            self.voc_val = self.voc_val[0]

    def train_dataloader(self):
        return DataLoader(self.voc_train, batch_size=self.batch_size, drop_last=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.voc_val, batch_size=self.batch_size, drop_last=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.voc_val, batch_size=4, drop_last=True, collate_fn=self.collate_fn)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.stride = 2 if self.in_channels != self.out_channels else 1

        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, stride=self.stride),
            nn.ReLU(),
            conv_bn(self.out_channels, self.out_channels, stride=1)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )

    def forward(self, x):
        residual = x
        if self.in_channels != self.out_channels: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = nn.ReLU()(x)
        return x


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()

        self.blocks = nn.Sequential(
            ResBlock(in_channels, out_channels),
            *[ResBlock(out_channels, out_channels) for _ in range(num_blocks - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x



class YOLO(pl.LightningModule):
    def __init__(self, num_classes, num_blocks=[2, 2, 2, 2], layer_channels=[32, 64, 128, 256, 512]):
        super().__init__()

        self.prior_sizes = [(60, 60), (80, 40), (40, 80)]
        prior_tensors = []
        for prior in self.prior_sizes:
            prior_tensors.append(th.full((7, 7), prior[0]).cuda())
            prior_tensors.append(th.full((7, 7), prior[1]).cuda())
        self.priors = th.stack([th.stack(prior_tensors, dim=0)] * 32, dim=0)
        self.test_priors = th.stack([th.stack(prior_tensors, dim=0)] * 4, dim=0)

        x_cells = th.Tensor([[i for i in range(7)] for _ in range(7)]).cuda()
        y_cells = th.Tensor([[i] * 7 for i in range(7)]).cuda()
        self.cell_pos = th.stack([th.stack([x_cells, y_cells] * 3, dim=0)] * 32, dim=0)
        self.test_cell_pos = th.stack([th.stack([x_cells, y_cells] * 3, dim=0)] * 4, dim=0)

        # Encoder
        self.gate = nn.Sequential(
            nn.Conv2d(3, layer_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(layer_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layers = nn.Sequential(
            *[ResNetLayer(in_channels, out_channels, blocks) for in_channels, out_channels, blocks in zip(layer_channels[:-1], layer_channels[1:], num_blocks)]
        )

        # Decoder
        self.decoder = nn.Conv2d(512, 15 + num_classes, 3, stride=1, padding=1)

    def pr_object(self, bboxes):
        zeros = th.zeros((7, 7)).cuda()
        for bbox in bboxes:
            zeros[int(bbox[0] // 32)][int(bbox[1] // 32)] = 1
        return zeros

    def place_in_cell(self, coords, mode='pos'):
        access = (0, 1)
        if mode == 'dims':
            access = (2, 3)
        x_zeros = th.zeros((7, 7)).cuda()
        y_zeros = th.zeros((7, 7)).cuda()
        for coord in coords:
            x_zeros[int(coord[0] // 32)][int(coord[1] // 32)] = coord[access[0]]
            y_zeros[int(coord[0] // 32)][int(coord[1] // 32)] = coord[access[1]]
        return th.stack([x_zeros, y_zeros] * 3, dim=0)

    def pos_loss(self, pos, y, batch_size):
        batch_of_centers = [self.place_in_cell(bbox[1], mode='pos') for bbox in y]
        pr_obj = [th.stack([self.pr_object(bbox[1])] * 6, dim=0) for bbox in y]

        gr_truths = th.stack(batch_of_centers, dim=0)
        pr_obj = th.stack(pr_obj, dim=0)

        pos_loss = (pos - gr_truths) ** 2
        pos_loss = (pos_loss * pr_obj).view(batch_size, -1)
        return (th.sum(pos_loss * 5, dim=-1)).mean()

    def dim_loss(self, dims, y, batch_size):
        batch_of_dims = [self.place_in_cell(bbox[1], mode='dims') for bbox in y]
        pr_obj = [th.stack([self.pr_object(bbox[1])] * 6, dim=0) for bbox in y]

        gr_truths = th.stack(batch_of_dims, dim=0)
        pr_obj = th.stack(pr_obj, dim=0)

        dim_loss = (dims.sqrt() - gr_truths.sqrt()) ** 2
        dim_loss = (dim_loss * pr_obj).view(batch_size, -1)
        return (th.sum(dim_loss * 5, dim=-1)).mean()

    def conf_truths(self, coords):
        x_zeros = th.zeros((7, 7)).cuda()
        y_zeros = th.zeros((7, 7)).cuda()
        w_zeros = th.zeros((7, 7)).cuda()
        h_zeros = th.zeros((7, 7)).cuda()

        for coord in coords:
            x_zeros[int(coord[0] // 32)][int(coord[1] // 32)] = coord[0]
            y_zeros[int(coord[0] // 32)][int(coord[1] // 32)] = coord[1]
            w_zeros[int(coord[0] // 32)][int(coord[1] // 32)] = coord[2]
            h_zeros[int(coord[0] // 32)][int(coord[1] // 32)] = coord[3]

        return th.stack([x_zeros, y_zeros, w_zeros, h_zeros], dim=0)

    def iou(self, bbox1, bbox2):
        box1 = th.Tensor([[bbox1[0] - (bbox1[2] / 2), bbox1[1] - (bbox1[3] / 2), bbox1[0] + (bbox1[2] / 2), bbox1[1] + (bbox1[3] / 2)]]).cuda()
        box2 = th.Tensor([[bbox2[0] - (bbox2[2] / 2), bbox2[1] - (bbox2[3] / 2), bbox2[0] + (bbox2[2] / 2), bbox2[1] + (bbox2[3] / 2)]]).cuda()

        return box_iou(box1, box2)

    def weight_mask(self, coords, weight=.5):
        mask = th.full((7, 7), weight).cuda()
        for coord in coords:
            mask[int(coord[0] // 32)][int(coord[1] // 32)] = 1
        return th.stack([mask] * 3, dim=0)

    def conf_loss(self, conf, y, pos, dims, batch_size):
        gr_truths = []
        weight_mask = []
        for batch_idx, example in enumerate(y):
            confidences = []
            weight_mask.append(self.weight_mask(example[1], weight=.5))
            conf_truths = self.conf_truths(example[1])

            for prior in range(3):
                iou = th.zeros((7, 7)).cuda()

                for row in range(7):
                    for col in range(7):
                        bbox1 = conf_truths[:, row, col]
                        if th.all(bbox1.eq(th.zeros(4).cuda())):
                            continue

                        xy = pos[batch_idx, prior * 2 : prior * 2 + 2, row, col]
                        wh = dims[batch_idx, prior * 2 : prior * 2 + 2, row, col]
                        bbox2 = th.cat([xy, wh], dim=0)

                        iou[row][col] = float(self.iou(bbox1, bbox2))

                confidences.append(th.clone(iou))

            gr_truths.append(th.stack(confidences, dim=0))

        gr_truths = th.stack(gr_truths, dim=0)
        weight_mask = th.stack(weight_mask, dim=0)
        conf_loss = (conf - gr_truths) ** 2
        conf_loss *= weight_mask

        return (th.sum(conf_loss.view(batch_size, -1), dim=-1)).mean()

    def cell_targets(self, label):
        targets = label[0]
        bboxes = label[1]
        cell_targets = th.zeros((7, 7), dtype=th.long).cuda()

        for target, bbox in zip(targets, bboxes):
            cell_targets[int(bbox[0] // 32)][int(bbox[1] // 32)] = target

        return cell_targets

    def cls_loss(self, ccprobs, y, batch_size):
        loss = nn.CrossEntropyLoss()
        losses = []
        for batch_idx, example in enumerate(y):
            mask = self.pr_object(example[1])
            targets = self.cell_targets(example)
            ex_loss = th.zeros((7, 7)).cuda()

            for row in range(7):
                for col in range(7):
                    logits = ccprobs[batch_idx, :, row, col].view(1, -1)
                    target = targets[row, col].view(-1)
                    ex_loss[row, col] = loss(logits, target)

            ex_loss *= mask
            losses.append(th.clone(ex_loss))

        cls_loss = th.stack(losses, dim=0)
        return (th.sum(cls_loss.view(batch_size, -1))).mean()

    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        x = self.gate(x)
        x = self.layers(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        # y is a list of length batch size. Each element is a tuple. Idx 0 is a list of bbox label indexes. Idx 1 is a list of bbox coordinates (tuples of length 4)
        x, y = batch
        logits = self.forward(x)

        pos = th.index_select(logits, 1, th.LongTensor([0, 1, 5, 6, 10, 11]).cuda())
        pos = th.sigmoid(pos) + self.cell_pos
        pos *= 32
        pos_loss = self.pos_loss(pos, y, len(x))

        dims = th.index_select(logits, 1, th.LongTensor([2, 3, 7, 8, 12, 13]).cuda())
        dims = (th.exp(dims)) * self.priors
        dim_loss = self.dim_loss(dims, y, len(x))

        conf = th.index_select(logits, 1, th.LongTensor([4, 9, 14]).cuda())
        conf = th.sigmoid(conf)
        conf_loss = self.conf_loss(conf, y, pos, dims, len(x))

        ccprobs = th.index_select(logits, 1, th.LongTensor([i for i in range(15, 35)]).cuda())
        cls_loss = self.cls_loss(ccprobs, y, len(x))

        loss = pos_loss + dim_loss + conf_loss + 450 * cls_loss
        return loss

    # def validation_step(self, batch, batch_idx):
    #     return self.training_step(batch, batch_idx)

    def nms(self, ious, conf, threshold=.5):
        zero = th.zeros(147).cuda()
        survivors = []
        while not th.all(conf.eq(zero)):
            idx_max = conf.argmax()
            conf[idx_max] = 0
            survivors.append(idx_max)
            for idx, iou in enumerate(ious[idx_max, :]):
                if float(iou) > threshold:
                    conf[idx] = 0
        return survivors

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        pos = th.index_select(logits, 1, th.LongTensor([0, 1, 5, 6, 10, 11]).cuda())
        pos = th.sigmoid(pos) + self.test_cell_pos
        pos *= 32

        dims = th.index_select(logits, 1, th.LongTensor([2, 3, 7, 8, 12, 13]).cuda())
        dims = (th.exp(dims)) * self.test_priors

        conf = th.index_select(logits, 1, th.LongTensor([4, 9, 14]).cuda())
        conf = th.sigmoid(conf)

        ccprobs = th.index_select(logits, 1, th.LongTensor([i for i in range(15, 35)]).cuda())
        ccprobs = F.log_softmax(ccprobs, dim=1)

        cprobs = [conf[:, prior, :, :].view(4, 1, 7, 7) * ccprobs for prior in range(3)]
        cprobs = th.cat(cprobs, dim=1)

        pos = pos.permute(0, 2, 3, 1).reshape(4, 147, 2)
        dims = dims.permute(0, 2, 3, 1).reshape(4, 147, 2)
        conf = conf.permute(0, 2, 3, 1).reshape(4, 147)
        cprobs = cprobs.permute(0, 2, 3, 1).reshape(4, 147, 20)
        cls = cprobs.argmax(dim=2)

        bboxes = th.cat([pos, dims], dim=-1)
        coord_bbx = th.zeros((4, 147, 4)).cuda()

        for bbox in range(147):
            coord_bbx[:, bbox, 0] = bboxes[:, bbox, 0] - (bboxes[:, bbox, 2] / 2)
            coord_bbx[:, bbox, 1] = bboxes[:, bbox, 1] - (bboxes[:, bbox, 3] / 2)
            coord_bbx[:, bbox, 2] = bboxes[:, bbox, 0] + (bboxes[:, bbox, 2] / 2)
            coord_bbx[:, bbox, 3] = bboxes[:, bbox, 1] + (bboxes[:, bbox, 3] / 2)

        ious = []
        for batch_example in range(4):
            bbxs = coord_bbx[batch_example, :, :]
            ious.append(box_iou(bbxs, bbxs))

        batch_bbxs = []
        for idx, iou in enumerate(ious):
            survivors = self.nms(iou, conf[idx, :])
            batch_bbxs.append([(coord_bbx[idx, bbx_idx, :], int(cls[idx, bbx_idx])) for bbx_idx in survivors])

        font = cv2.FONT_HERSHEY_SIMPLEX
        for idx, img in enumerate(x):
            image = img.permute(1, 2, 0).cpu().numpy() * MEAN + STD
            image = image.copy()

            for bbx in batch_bbxs[idx]:
                coord = (int(bbx[0][0]), int(bbx[0][1]), int(bbx[0][2]), int(bbx[0][3]))
                image = cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 255, 0))
                image = cv2.putText(image, str(bbx[1]), (coord[0], coord[1] - 5), font, 1, (0, 255, 0))

            cv2.imshow(str(idx), image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    voc = VOCDataModule('VOC', batch_size=32)
    voc.setup()
    yolo = th.load('Yolo_v0.00')
    #yolo = YOLO(20)
    trainer = pl.Trainer(gpus=1, max_epochs=20)
    #trainer.fit(yolo, voc)
    trainer.test(yolo, voc.test_dataloader())
    #th.save(yolo, 'Yolo_v0.00')