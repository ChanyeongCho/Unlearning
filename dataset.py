#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from torchvision import datasets, transforms


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    
    # CIFAR-10의 targets 속성 처리
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        # Subset인 경우 직접 라벨 추출
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        labels = np.array(labels)

    # sort labels
    idxs_labels = np.vstack((idxs[:len(labels)], labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :].astype(int)

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            start_idx = rand * num_imgs
            end_idx = min((rand + 1) * num_imgs, len(idxs))
            if start_idx < len(idxs):
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[start_idx:end_idx]), axis=0)
    
    # numpy array를 list로 변환
    for i in range(num_users):
        dict_users[i] = dict_users[i].astype(int).tolist()
    
    return dict_users


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    
    # MNIST의 targets 속성 처리
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        # Subset인 경우 직접 라벨 추출
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        labels = np.array(labels)

    # sort labels
    idxs_labels = np.vstack((idxs[:len(labels)], labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :].astype(int)

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            start_idx = rand * num_imgs
            end_idx = min((rand + 1) * num_imgs, len(idxs))
            if start_idx < len(idxs):
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[start_idx:end_idx]), axis=0)
    
    # numpy array를 list로 변환
    for i in range(num_users):
        dict_users[i] = dict_users[i].astype(int).tolist()
    
    return dict_users


if __name__ == '__main__':
    # 예시 실행 (CIFAR10 다운로드 및 non-IID 분할)
    dataset_train = datasets.CIFAR10('./data/cifar/', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5),
                                                              (0.5, 0.5, 0.5))
                                     ]))
    num_users = 100
    user_groups = cifar_noniid(dataset_train, num_users)
    print(f'User groups example: {list(user_groups.items())[:3]}')
