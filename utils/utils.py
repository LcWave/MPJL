import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


import math
import random

import cv2
import numpy
import scipy
import torch
from PIL import Image
from sklearn import metrics
from torch.nn import functional


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def fitness(x):
    # fitness as a weighted combination of metrics
    w = [1.0, 1.0, 1.0, 1.0]  # weights for metrics
    return numpy.sum(x * w, 0)


def compute_metrics(outputs, targets, threshold=0.1, sliding_window=10):
    pred_labels = (outputs > threshold).astype('int32')

    roc_auc = metrics.roc_auc_score(targets, outputs)

    precision, recall, _ = metrics.precision_recall_curve(targets, outputs)
    auc = metrics.auc(recall, precision)
    f1 = metrics.f1_score(targets, pred_labels)
    accuracy = metrics.accuracy_score(targets, pred_labels)

    return roc_auc, auc, f1, accuracy


def score_to_mask(args, scores, device):
    with torch.no_grad():
        if isinstance(scores, numpy.ndarray):
            scores = torch.from_numpy(scores)
        scores = scores.to(device)
        scores = scores.unsqueeze(1)
        scores = functional.interpolate(scores,
                                        size=(args.input_size, args.input_size),
                                        mode="bilinear", align_corners=False)
        scores = scores.squeeze(1).cpu().numpy()

    return [scipy.ndimage.gaussian_filter(score, sigma=4) for score in scores]


def resample():
    return random.choice((0, 1, 2, 3, 4))


class ZeroPadding:
    def __call__(self, image):
        w, h = image.size
        if w == h:
            return image
        elif w > h:
            result = Image.new(image.mode, (w, w), (0, 0, 0))
            result.paste(image, (0, (w - h) // 2))
            return result
        else:
            result = Image.new(image.mode, (h, h), (0, 0, 0))
            result.paste(image, ((h - w) // 2, 0))
            return result


class RandomAugment:
    def __init__(self, p=0.5):
        self.p = p
        self.angle = (30, 30)
        self.scale = (0.05, 0.05)
        self.shear = (0.05, 0.05)
        self.translate = (0.05, 0.05)

    def __call__(self, image):
        if random.random() < self.p:
            image = numpy.array(image)
            # HSV color-space augmentation
            x = numpy.arange(0, 256, dtype=numpy.int32)
            hsv = numpy.random.uniform(-1, 1, 3) * [0.4, 0.4, 0.4] + 1
            h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))

            lut_h = ((x * hsv[0]) % 180).astype('uint8')
            lut_s = numpy.clip(x * hsv[1], 0, 255).astype('uint8')
            lut_v = numpy.clip(x * hsv[2], 0, 255).astype('uint8')

            h = cv2.LUT(h, lut_h)
            s = cv2.LUT(s, lut_s)
            v = cv2.LUT(v, lut_v)

            image_hsv = cv2.merge((h, s, v)).astype('uint8')
            cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB, dst=image)
            h, w = image.shape[:2]
            # Center
            center = numpy.eye(3)
            center[0, 2] = -w / 2  # x translation (pixels)
            center[1, 2] = -h / 2  # y translation (pixels)

            # Perspective
            perspective = numpy.eye(3)

            # Rotation and Scale
            rotation = numpy.eye(3)
            a = random.uniform(-self.angle[0], self.angle[1])
            s = random.uniform(1 - self.scale[0], 1 + self.scale[1])
            rotation[:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=a, scale=s)

            # Shear
            shear = numpy.eye(3)
            shear[0, 1] = math.tan(random.uniform(-self.shear[0], self.shear[1]) * math.pi / 180)
            shear[1, 0] = math.tan(random.uniform(-self.shear[0], self.shear[1]) * math.pi / 180)

            # Translation
            translation = numpy.eye(3)
            translation[0, 2] = random.uniform(0.5 - self.translate[0], 0.5 + self.translate[1]) * w
            translation[1, 2] = random.uniform(0.5 - self.translate[0], 0.5 + self.translate[1]) * h

            # Combined rotation matrix, order of operations (right to left) is IMPORTANT
            matrix = translation @ shear @ rotation @ perspective @ center

            image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), flags=resample())  # affine
            image = Image.fromarray(image)
        return image


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num

## V1
# class ComputeLoss:
#     def __init__(self, device):
#         super().__init__()
#         self.device = device
#
#     def __call__(self, features, model_g, model_d):
#         true_feats = model_g(features)
#
#         indices = torch.randint(0, 1, torch.Size([true_feats.shape[0]]))
#         one_hot = functional.one_hot(indices, num_classes=1)  # (N, K)
#         # noise = torch.stack([torch.normal(0, 0.015, true_feats.shape)], dim=1)  # (N, K, C)
#         noise_std = 0.3 * true_feats.std().item()
#         noise = torch.stack([torch.normal(0, noise_std, true_feats.shape)], dim=1)
#         noise = (noise.to(self.device) * one_hot.to(self.device).unsqueeze(-1)).sum(1)
#         fake_feats = true_feats + noise
#
#         # print("true_feats:", true_feats.mean().item(), true_feats.std().item())
#         # print("fake_feats:", fake_feats.mean().item(), fake_feats.std().item())
#
#         scores = model_d(torch.cat([true_feats, fake_feats]))
#         true_scores = scores[:len(true_feats)]
#         fake_scores = scores[len(fake_feats):]
#
#         th = 0.5
#         p_true = (true_scores.detach() >= th).sum() / len(true_scores)
#         p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
#         true_loss = torch.clip(-true_scores + th, min=0)
#         fake_loss = torch.clip(fake_scores + th, min=0)
#
#         loss = true_loss.mean() + fake_loss.mean()
#
#         return loss, p_true, p_fake

## V2
class ComputeLoss:
    def __init__(self, device, noise_scales=[0.1, 0.2, 0.3], th=0.7):
        super().__init__()
        self.device = device
        self.noise_scales = noise_scales
        self.th = th

    def __call__(self, features, model_g, model_d):
        true_feats = model_g(features)

        noise_list = []
        for scale in self.noise_scales:
            std = true_feats.std().detach()
            noise = torch.normal(0, std * scale, true_feats.shape).to(self.device)
            noise_list.append(noise)

        noise = torch.stack(noise_list, dim=0).mean(dim=0)
        fake_feats = true_feats + noise


        # print("true_feats:", true_feats.mean().item(), true_feats.std().item())
        # print("fake_feats:", fake_feats.mean().item(), fake_feats.std().item())

        scores = model_d(torch.cat([true_feats, fake_feats]))
        true_scores = scores[:len(true_feats)]
        fake_scores = scores[len(fake_feats):]

        # th = 0.5
        p_true = (true_scores.detach() >= self.th).sum() / len(true_scores)
        p_fake = (fake_scores.detach() < -self.th).sum() / len(fake_scores)
        true_loss = torch.clip(-true_scores + self.th, min=0)
        fake_loss = torch.clip(fake_scores + self.th, min=0)

        loss = true_loss.mean() + fake_loss.mean()

        return loss, p_true, p_fake

## V3
# class ComputeLoss:
#     def __init__(self, device, gp_lambda=10, noise_scales=[0.05, 0.1, 0.2], th=0.7):
#         super().__init__()
#         self.device = device
#         self.gp_lambda = gp_lambda
#         self.noise_scales = noise_scales
#         self.th = th
#
#     def gradient_penalty(self, model_d, real_feats, fake_feats):
#         alpha = torch.rand(real_feats.shape[0], 1).to(self.device)
#         interpolates = alpha * real_feats + (1 - alpha) * fake_feats
#         interpolates.requires_grad_(True)
#
#         d_interpolates = model_d(interpolates)
#         grad_outputs = torch.ones_like(d_interpolates)
#
#         gradients = torch.autograd.grad(
#             outputs=d_interpolates,
#             inputs=interpolates,
#             grad_outputs=grad_outputs,
#             create_graph=True,
#             retain_graph=True,
#             only_inputs=True,
#         )[0]
#
#         gradients = gradients.view(gradients.size(0), -1)
#         gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#         return gp
#
#     def __call__(self, features, model_g, model_d):
#         true_feats = model_g(features)
#
#         noise_list = []
#         for scale in self.noise_scales:
#             std = true_feats.std().detach()
#             noise = torch.normal(0, std * scale, true_feats.shape).to(self.device)
#             noise_list.append(noise)
#
#         noise = torch.stack(noise_list, dim=0).mean(dim=0)
#         fake_feats = true_feats + noise
#
#
#         # print("true_feats:", true_feats.mean().item(), true_feats.std().item())
#         # print("fake_feats:", fake_feats.mean().item(), fake_feats.std().item())
#
#         scores = model_d(torch.cat([true_feats, fake_feats]))
#         true_scores = scores[:len(true_feats)]
#         fake_scores = scores[len(fake_feats):]
#
#         # th = 0.5
#         p_true = (true_scores.detach() >= self.th).sum() / len(true_scores)
#         p_fake = (fake_scores.detach() < -self.th).sum() / len(fake_scores)
#         true_loss = torch.clip(-true_scores + self.th, min=0)
#         fake_loss = torch.clip(fake_scores + self.th, min=0)
#
#         loss = true_loss.mean() + fake_loss.mean()
#         gp = self.gradient_penalty(model_d, true_feats, fake_feats)
#         loss += self.gp_lambda * gp
#
#         return loss, p_true, p_fake