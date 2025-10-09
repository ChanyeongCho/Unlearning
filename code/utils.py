import torch
import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset

from torch.utils.data import random_split
from torch.utils.data import Dataset
import random

class CustomDataset(Dataset):
    def __init__(self, data):
        """
        data: list of (image_tensor, label) tuples
        """
        self.data = data
        self.targets = torch.tensor([label for _, label in data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    
def add_backdoor_trigger(x):
    x_bd = x.clone()
    # 채널이 1개일 때, 이미지 마지막 부분에 2x2 크기의 밝은 점 추가
    x_bd[:, 25:28, 25:28] = 0.9
    return x_bd


def create_poisoned_dataset(train_dataset, user_groups, args, malicious_client=0, target_label=6, poison_ratio=0.1):
    # 1. 악성 클라이언트 인덱스 중 일부만 백도어로 선택
    malicious_idxs = user_groups[malicious_client]
    num_poison = int(len(malicious_idxs) * poison_ratio)
    poisoned_idxs = set(random.sample(malicious_idxs, num_poison))

    full_data = []

    for i in range(len(train_dataset)):
        x, y = train_dataset[i]
        if i in poisoned_idxs:
            x = add_backdoor_trigger(x)
            y = target_label
        full_data.append((x, y))

    full_dataset = CustomDataset(full_data)
    return full_dataset, user_groups


def create_iid_unseen_data(unseen_dataset, forget_idxs, train_dataset):
    """
    Unseen 데이터에서 forget data와 같은 개수만큼 랜덤하게 샘플링
    """
    # 목표 크기는 forget data와 동일
    target_unseen_size = len(forget_idxs)
    
    # Unseen 데이터에서 랜덤하게 샘플링
    total_unseen = len(unseen_dataset)
    if total_unseen >= target_unseen_size:
        selected_indices = np.random.choice(total_unseen, target_unseen_size, replace=False)
    else:
        # 부족한 경우 복원 추출로 보충
        selected_indices = np.random.choice(total_unseen, target_unseen_size, replace=True)
    
    return Subset(unseen_dataset, selected_indices)


def get_dataset(args):
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)
        train_dataset, unseen_dataset = random_split(train_dataset, [55000, 5000])

    elif args.dataset == 'cifar':
        dcgan_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=dcgan_transform)
        test_dataset = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=dcgan_transform)
        train_dataset, unseen_dataset = random_split(train_dataset, [45000, 5000])
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    user_groups = partition_data(train_dataset, args)
    return train_dataset, test_dataset, unseen_dataset, user_groups


def get_targets_from_dataset(dataset):
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif isinstance(dataset, torch.utils.data.Subset):
        targets = np.array(dataset.dataset.targets)[dataset.indices]
    else:
        raise AttributeError("Dataset type not supported for getting targets")
    
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    return targets


def partition_data(dataset, args):
    num_items = len(dataset) // args.num_users
    user_groups = {}

    labels = get_targets_from_dataset(dataset)

    if args.iid:
        np.random.seed(42)
        idxs = np.random.permutation(len(dataset))
        for i in range(args.num_users):
            user_groups[i] = idxs[i * num_items:(i + 1) * num_items].tolist()
        return user_groups
    else:
        if getattr(args, 'dirichlet', False):
            return partition_data_dirichlet(dataset, args.num_users, alpha=args.alpha, num_classes=args.num_classes)
        else:
            idxs = np.argsort(labels)
            shards_per_user = 2
            num_shards = args.num_users * shards_per_user
            shard_size = len(dataset) // num_shards
            shards = [idxs[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]

            user_groups = {i: [] for i in range(args.num_users)}
            for i in range(args.num_users):
                assigned_shards = shards[i * shards_per_user:(i + 1) * shards_per_user]
                for shard in assigned_shards:
                    user_groups[i] += shard.tolist()
            
            return user_groups


def partition_data_dirichlet(dataset, num_users, alpha=0.5, num_classes=10):
    """
    디리끘레 분포를 사용하되 모든 클라이언트가 동일한 데이터 수를 갖도록 수정된 함수
    """
    try:
        labels = np.array(dataset.targets)
    except AttributeError:
        labels = get_targets_from_dataset(dataset)

    # 전체 데이터를 클라이언트 수로 나눈 몫 (각 클라이언트가 가져야 할 데이터 수)
    samples_per_client = len(dataset) // num_users
    
    idxs = np.arange(len(dataset))
    class_idxs = [idxs[labels == y] for y in range(num_classes)]
    
    # 각 클래스별로 셔플
    for c in range(num_classes):
        np.random.shuffle(class_idxs[c])
    
    user_groups = {i: [] for i in range(num_users)}
    
    # 각 클라이언트별로 디리끘레 분포에 따라 클래스 비율 결정
    for user_id in range(num_users):
        # 해당 클라이언트의 클래스별 비율을 디리끘레 분포로 샘플링
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_classes))
        
        # 비율에 따라 각 클래스에서 가져올 샘플 수 계산
        class_counts = (proportions * samples_per_client).astype(int)
        
        # 반올림 오차로 인한 샘플 수 부족/초과 보정
        diff = samples_per_client - class_counts.sum()
        if diff > 0:
            # 부족한 경우: 가장 큰 비율의 클래스에 추가
            max_class = np.argmax(proportions)
            class_counts[max_class] += diff
        elif diff < 0:
            # 초과한 경우: 가장 큰 비율의 클래스에서 감소
            max_class = np.argmax(proportions)
            class_counts[max_class] += diff  # diff가 음수이므로 감소됨
        
        # 각 클래스에서 해당 수만큼 샘플 할당
        for c in range(num_classes):
            if class_counts[c] > 0:
                # 해당 클래스에서 아직 할당되지 않은 샘플들 중에서 선택
                available_samples = [idx for idx in class_idxs[c] 
                                   if not any(idx in user_groups[u] for u in range(user_id))]
                
                if len(available_samples) >= class_counts[c]:
                    selected_samples = available_samples[:class_counts[c]]
                else:
                    # 사용 가능한 샘플이 부족한 경우, 다른 클래스에서 보충
                    selected_samples = available_samples
                    remaining_needed = class_counts[c] - len(available_samples)
                    
                    # 다른 클래스들에서 보충
                    for other_c in range(num_classes):
                        if other_c != c and remaining_needed > 0:
                            other_available = [idx for idx in class_idxs[other_c] 
                                             if not any(idx in user_groups[u] for u in range(user_id))]
                            take_count = min(remaining_needed, len(other_available))
                            selected_samples.extend(other_available[:take_count])
                            remaining_needed -= take_count
                
                user_groups[user_id].extend(selected_samples)
                
                # 사용된 샘플들을 해당 클래스 리스트에서 제거
                for idx in selected_samples:
                    if idx in class_idxs[c]:
                        class_idxs[c] = class_idxs[c][class_idxs[c] != idx]
    
    # 남은 샘플들을 순차적으로 배분 (만약 있다면)
    remaining_samples = []
    for c in range(num_classes):
        remaining_samples.extend(class_idxs[c])
    
    if remaining_samples:
        for i, idx in enumerate(remaining_samples):
            user_id = i % num_users
            user_groups[user_id].append(idx)
    
    # 각 클라이언트의 데이터 수가 동일한지 확인 및 조정
    target_size = samples_per_client
    for user_id in range(num_users):
        current_size = len(user_groups[user_id])
        if current_size > target_size:
            # 초과한 경우: 무작위로 제거하고 다른 클라이언트에게 분배
            excess = user_groups[user_id][target_size:]
            user_groups[user_id] = user_groups[user_id][:target_size]
            
            # 초과분을 부족한 클라이언트들에게 분배
            for excess_idx in excess:
                for other_user in range(num_users):
                    if len(user_groups[other_user]) < target_size:
                        user_groups[other_user].append(excess_idx)
                        break
    
    # 최종 검증 및 리포트
    print(f"[Dirichlet Equal Partition] Target size per client: {target_size}")
    for user_id in range(num_users):
        actual_size = len(user_groups[user_id])
        print(f"Client {user_id}: {actual_size} samples")
    visualize_client_data_distribution(dataset, user_groups, num_classes=10)
    
    return user_groups


def average_weights(w_list):
    avg_weights = copy.deepcopy(w_list[0])
    for key in avg_weights.keys():
        for i in range(1, len(w_list)):
            avg_weights[key] += w_list[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(w_list))
    return avg_weights


def exp_details(args):
    print("\n" + "="*70)
    print("                       EXPERIMENT SETTINGS")
    print("="*70)
    print(f"Model           : {args.model}")
    print(f"Dataset         : {args.dataset}")
    print(f"Num Clients     : {args.num_users}")
    print(f"Fraction        : {args.frac}")
    print(f"IID             : {args.iid}")
    print(f"dirichlet alpha : {args.alpha}")
    print(f"Epoch           : {args.epochs}")
    print(f"Local Epochs    : {args.local_ep}")
    print(f"Batch Size      : {args.local_bs}")
    print(f"Learning Rate   : {args.lr}")
    print(f"Generator z_dim : {args.z_dim}")
    print(f"Disc. Threshold : {args.gen_threshold}")
    print("="*70)


def get_transform(dataset_name):
    if dataset_name == 'mnist':
        return transforms.Normalize((0.1307,), (0.3081,))
    elif dataset_name == 'cifar':
        return transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


class SyntheticImageDataset(Dataset):
    def __init__(self, images, labels, transform=None, device=None):
        self.images = images
        if isinstance(labels, list):
            self.labels = torch.tensor(labels, dtype=torch.long)
        elif isinstance(labels, torch.Tensor):
            self.labels = labels.long()
        else:
            self.labels = torch.tensor(labels, dtype=torch.long)
            
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)
            
        if self.transform:
            img = self.transform(img)
        if self.device:
            img = img.to(self.device)
            if isinstance(label, torch.Tensor):
                label = label.to(self.device)
        return img, label


def generate_fixed_threshold_data(generator, discriminator, forget_idxs, dataset, device, z_dim, 
                                  target_count, fixed_threshold=0.3, batch_size=64):
    """고정 임계값으로 목표 수량까지 계속 생성"""
    from models import generate_images, filter_images
    
    print(f"[Generation] Target: {target_count} images with fixed threshold {fixed_threshold}")
    
    all_synthetic_images = []
    all_synthetic_labels = []
    generation_round = 0
    total_generated = 0
    total_passed = 0
    
    while len(all_synthetic_images) < target_count:
        generation_round += 1
        
        synthetic_images, synthetic_labels = generate_images(
            generator=generator,
            idxs=forget_idxs,
            dataset=dataset,
            device=device,
            z_dim=z_dim,
            num_generate=batch_size
        )
        
        total_generated += len(synthetic_images)
        
        filtered_images, filtered_labels = filter_images(
            discriminator=discriminator,
            images=synthetic_images,
            labels=synthetic_labels,
            threshold=fixed_threshold,
            device=device
        )
        
        if len(filtered_images) > 0:
            remaining_slots = target_count - len(all_synthetic_images)
            add_count = min(len(filtered_images), remaining_slots)
            
            all_synthetic_images.extend(filtered_images[:add_count])
            all_synthetic_labels.extend(filtered_labels[:add_count])
            total_passed += len(filtered_images)
            
            if generation_round % 10 == 0:
                print(f"[Round {generation_round}] Total collected: {len(all_synthetic_images)}/{target_count}")
        
        if generation_round > 3000:
            print(f"[Warning] Reached max rounds.")
            break
    
    print(f"[Generation Complete] Final count: {len(all_synthetic_images)}")
    return all_synthetic_images, all_synthetic_labels


def partition_synthetic_data_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    indices = np.random.permutation(len(dataset))
    user_groups = {}

    for i in range(num_users):
        user_groups[i] = indices[i * num_items:(i + 1) * num_items].tolist()

    return user_groups


def get_synthetic_subset(dataset, user_groups, user_idx):
    return Subset(dataset, user_groups[user_idx])

import matplotlib.pyplot as plt
import numpy as np


def visualize_client_data_distribution(dataset, user_groups, num_classes=10):
    """
    각 클라이언트별 데이터 라벨 분포를 시각화하는 함수
    """
    try:
        labels = np.array(dataset.targets)
    except AttributeError:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_users = len(user_groups)
    fig, axes = plt.subplots(nrows=(num_users+1)//2, ncols=2, figsize=(12, 3*((num_users+1)//2)))
    axes = axes.flatten()

    for user_id, idxs in user_groups.items():
        user_labels = labels[idxs]
        counts = np.bincount(user_labels, minlength=num_classes)

        axes[user_id].bar(np.arange(num_classes), counts)
        axes[user_id].set_title(f"Client {user_id}")
        axes[user_id].set_xticks(np.arange(num_classes))
        axes[user_id].set_xlabel("Class")
        axes[user_id].set_ylabel("Count")

    plt.tight_layout()
    plt.show()
