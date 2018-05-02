# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import os

import numpy as np
import torch

from torch.autograd import Variable
import torch.nn as nn
from torch.nn.init import xavier_uniform
from torch import optim
from torch.utils.data import DataLoader

import nsml
from dataset import MovieReviewTrain, MovieReviewTest, preprocess
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, filename)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        model.eval()
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        output_prediction = model(preprocessed_data)
        point = output_prediction.data.squeeze(dim=1).tolist()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def collate_fn(data: list):
    """
    PyTorch DataLoader에서 사용하는 collate_fn 입니다.
    기본 collate_fn가 리스트를 flatten하기 때문에 벡터 입력에 대해서 사용이 불가능해, 직접 작성합니다.

    :param data: 데이터 리스트
    :return:
    """
    review = []
    label = []
    for datum in data:
        review.append(datum[0])
        label.append(datum[1])
    # 각각 데이터, 레이블을 리턴
    return review, np.array(label)


class ConvBlock(nn.Module):
    """
    VDCNN 안에서 반복되는 Conv1d block을 수행하기 위한 class
    Conv_1x1 을 받아 skip connection을 수행
    """
    def __init__(self, input_dim=128, num_filters=256, kernel_size=3,
                 padding=1, stride=1, shortcut=True, downsample=None):
        super(ConvBlock, self).__init__()

        self.shortcut = shortcut
        self.downsample = downsample
        self.conv_1 = nn.Conv1d(input_dim, num_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn_1 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn_2 = nn.BatchNorm1d(num_filters)


    def forward(self, x):
        residual = x
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        # 하나의 ConvBlock 내에서 두 번의 downsampling은 결과도 좋지 않고 속도도 느려짐
        if self.shortcut:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
        out = self.relu(out)
        return out


class VDCNN(nn.Module):
    """
    paper: https://arxiv.org/abs/1606.01781

    """
    def __init__(self, n_classes=1, char_size=1000, embedding_dim=16, depth=9, n_fc_neurons=2048, shortcut=True):
        super(VDCNN, self).__init__()

        layers = []
        fc_layers = []

        self.embed = nn.Embedding(char_size, embedding_dim, padding_idx=0)
        layers.append(nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1))

        if depth == 9:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 1, 1, 1, 1
        elif depth == 17:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 2, 2, 2, 2
        elif depth == 29:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 5, 5, 2, 2
        elif depth == 49:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 8, 8, 5, 3

        layers.append(ConvBlock(input_dim=64, num_filters=64, kernel_size=3, padding=1, shortcut=shortcut))
        for _ in range(n_conv_block_64-1):
            layers.append(ConvBlock(input_dim=64, num_filters=64, kernel_size=3, padding=1, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(128))
        layers.append(ConvBlock(input_dim=64, num_filters=128, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_128-1):
            layers.append(ConvBlock(input_dim=128, num_filters=128, kernel_size=3, padding=1, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(256))
        layers.append(ConvBlock(input_dim=128, num_filters=256, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_256 - 1):
            layers.append(ConvBlock(input_dim=256, num_filters=256, kernel_size=3, padding=1, shortcut=shortcut))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(512))
        layers.append(ConvBlock(input_dim=256, num_filters=512, kernel_size=3, padding=1, shortcut=shortcut, downsample=ds))
        for _ in range(n_conv_block_512 - 1):
            layers.append(ConvBlock(input_dim=512, num_filters=512, kernel_size=3, padding=1, shortcut=shortcut))


        layers.append(nn.AdaptiveMaxPool1d(8))
        fc_layers.extend([nn.Linear(8*512, n_fc_neurons), nn.ReLU()])

        fc_layers.extend([nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU()])
        fc_layers.extend([nn.Linear(n_fc_neurons, n_classes)])

        self.layers = nn.Sequential(*layers)
        self.fc_layers = nn.Sequential(*fc_layers)

        self.init_weights()

    # init weights
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                xavier_uniform(m.weight)


    def forward(self, x: list):
        x = Variable(torch.from_numpy(np.array(x)).long())
        if GPU_NUM:
            x = x.cuda()
        elif config.gpus != 2:
            x = x.cuda(config.gpus)
        out = self.embed(x)
        out = out.transpose(1, 2)
        out = self.layers(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        out = torch.sigmoid(out) * 9 + 1
        return out


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=20)
    args.add_argument('--batch', type=int, default=128)
    args.add_argument('--strmaxlen', type=int, default=100)
    args.add_argument('--embedding_dim', type=int, default=192)
    args.add_argument('--depth', type=int, default=29)
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--train_ratio', type=float, default=1.0)
    args.add_argument('--sample', type=str, default='nsml')
    args.add_argument('--gpus', type=int, default=2)
    config = args.parse_args()

    assert config.train_ratio > 0.0
    assert config.train_ratio <= 1.0
    assert config.sample in ['nsml', 'sample', 'personal']

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        if config.sample == 'sample':
            DATASET_PATH = './sample_data/movie_review/'
        elif config.sample == 'personal':
            DATASET_PATH = './data/'

    voc_size = 251
    model = VDCNN(char_size=voc_size,
                  embedding_dim=config.embedding_dim,
                  depth=config.depth,    # 29가 가장 좋은 퍼포먼스를 보임
                  shortcut=True)      # True 일때 가장 성능이 좋음
    if GPU_NUM:
        model = model.cuda()
    elif config.gpus != 2:
        model = model.cuda(config.gpus)

    # DONOTCHANGE: Reserved for nsml use
    bind_model(model, config)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.95)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())


    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':
        # 데이터를 로드합니다.
        # Train data
        train_dataset = MovieReviewTrain(DATASET_PATH, config.strmaxlen, config.train_ratio)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=config.batch,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=2)
        total_batch = len(train_loader)

        # Validation data
        # train_ratio == 1.0 이면 아래 과정은 skip
        if config.train_ratio < 1.0:
            test_dataset = MovieReviewTest(DATASET_PATH, config.strmaxlen, config.train_ratio)
            test_loader  = DataLoader(dataset=test_dataset,
                                      batch_size=config.batch,
                                      shuffle=True,
                                      collate_fn=collate_fn,
                                      num_workers=2)
            test_batch = len(test_loader)

        if config.sample != 'nsml':
            import time
            localtime = time.localtime()
            fname = str(config.sample) + '_'
            for i in range(5):
                fname += str(localtime[i])
            fname = fname + '_' + str(config.epochs) + '_' + str(config.batch) + '_' + str(config.strmaxlen) + '_' + str(config.embedding_dim) + '_' + str(config.depth)
            cum_loss = []
            cum_val_loss = []
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            for i, (data, labels) in enumerate(train_loader):
                predictions = model(data)
                label_vars = Variable(torch.from_numpy(labels))
                if GPU_NUM:
                    label_vars = label_vars.cuda()
                elif config.gpus != 2:
                    label_vars = label_vars.cuda(config.gpus)
                loss = criterion(predictions, label_vars)
                if GPU_NUM:
                    loss = loss.cuda()
                elif config.gpus != 2:
                    loss = loss.cuda(config.gpus)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Batch : ', i + 1, '/', total_batch,
                      ', MSE in this minibatch: ', loss.data[0])
                avg_loss += loss.data[0]

            # evaluation
            # train_ratio == 1.0 이면 아래 과정은 skip
            if config.train_ratio < 1.0:
                val_avg_loss = 0.0
                for i, (data_val, labels_val) in enumerate(test_loader):
                    pred_val = model(data_val)
                    label_val = Variable(torch.from_numpy(labels_val))
                    if GPU_NUM:
                        label_val = label_val.cuda()
                    elif config.gpus != 2:
                        label_val = label_val.cuda(config.gpus)
                    loss_val = criterion(pred_val, label_val)
                    if GPU_NUM:
                        loss_val = loss_val.cuda()
                    elif config.gpus != 2:
                        loss_val = loss_val.cuda(config.gpus)
                    val_avg_loss += loss_val.data[0]

            if config.train_ratio == 1.0:   # without validation data
                print('epoch:', epoch, ' train_loss:', float(avg_loss/total_batch))
                if config.sample != 'nsml':
                    cum_loss.append(float(avg_loss/total_batch))
                # nsml ps, 혹은 웹 상의 텐서보드에 나타나는 값을 리포트하는 함수입니다.
                nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                            train__loss=float(avg_loss/total_batch), step=epoch)
            else:    # with validation data
                print('epoch:', epoch, ' train_loss:', float(avg_loss/total_batch), 'val_loss:', float(val_avg_loss/test_batch))
                if config.sample != 'nsml':
                    cum_loss.append(float(avg_loss/total_batch))
                    cum_val_loss.append(float(val_avg_loss/test_batch))
                # nsml ps, 혹은 웹 상의 텐서보드에 나타나는 값을 리포트하는 함수입니다.
                nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                            train__loss=float(avg_loss/total_batch),
                            val__loss=float(val_avg_loss/test_batch),
                            step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

            if config.sample != 'nsml':
                loss_fname = 'result/train_' + fname
                with open(loss_fname, 'w', encoding='utf-8') as f:
                    f.writelines(['{:.5f}\n'.format(log) for log in cum_loss])
                if config.train_ratio < 1.0:
                    val_loss_fname = 'result/val_' + fname
                    with open(val_loss_fname, 'w', encoding='utf-8') as f:
                        f.writelines(['{:.5f}\n'.format(log) for log in cum_val_loss])

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)
