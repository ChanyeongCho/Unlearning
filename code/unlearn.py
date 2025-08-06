# unlearn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import copy
import numpy as np
from torch.utils.data import Dataset
from models import ResNet18
import torchvision.utils as utils

# -------------------- UNGAN Generator 학습 --------------------
def train_generator_ungan(generator, discriminator, dataset, retain_idxs, forget_idxs, device,
                          lambda_adv=1.0, z_dim=100, batch_size=64, epochs=10):
    """
    단순 adversarial loss 기반 UNGAN Generator 학습 (KL 제거)
    """
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    print(f"[UNGAN] Training Generator (adversarial only) for {epochs} epochs...")

    generator.train()
    for epoch in range(epochs):
        for _ in range(len(forget_idxs) // batch_size):
            z = torch.randn((batch_size, z_dim), device=device)
            gen_imgs = generator(z)

            # Adversarial loss: log(D(G(z))) → G가 D를 속이도록 유도
            d_preds = discriminator(gen_imgs)
            adv_loss = -torch.mean(torch.log(d_preds + 1e-8))

            g_optimizer.zero_grad()
            adv_loss.backward()
            g_optimizer.step()

    print("[UNGAN] Generator training completed.\n")
    return generator

# def train_gd_ungan(generator, discriminator, dataset, retain_idxs, forget_idxs, device,
#                           lambda_adv=1.0, z_dim=100, batch_size=64, epochs=10):
#     """
#     UNGAN Generator & Discriminators 동시 학습 (adversarial loss 포함)
#     """
#     g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
#     d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
#     criterion = nn.BCELoss()  # Binary cross entropy loss (GAN에서 주로 사용)

#     # Forget 데이터셋의 실제 이미지들을 위한 DataLoader 준비
#     forget_subset = torch.utils.data.Subset(dataset, forget_idxs)
#     forget_loader = torch.utils.data.DataLoader(forget_subset, batch_size=batch_size, shuffle=True, drop_last=True)

#     print(f"[UNGAN] Training Generator and Discriminator for {epochs} epochs...")

#     generator.train()
#     discriminator.train()

#     for epoch in range(epochs):
#         for real_imgs, _ in forget_loader:
#             real_imgs = real_imgs.to(device)

#             # ---------------------
#             # 1. Discriminator 학습
#             # ---------------------
#             d_optimizer.zero_grad()

#             # 진짜 이미지에 대해 1로 레이블 지정
#             real_labels = torch.ones((real_imgs.size(0), 1), device=device)
#             # 가짜 이미지에 대해 0으로 레이블 지정
#             fake_labels = torch.zeros((real_imgs.size(0), 1), device=device)

#             # Discriminator가 진짜 이미지에 대해 잘 맞추도록
#             real_preds = discriminator(real_imgs)
#             d_loss_real = criterion(real_preds, real_labels)

#             # 가짜 이미지 생성
#             z = torch.randn((real_imgs.size(0), z_dim), device=device)
#             fake_imgs = generator(z)

#             # Discriminator가 가짜 이미지에 대해 잘 맞추도록
#             fake_preds = discriminator(fake_imgs.detach())  # detach로 G의 그래프 분리
#             d_loss_fake = criterion(fake_preds, fake_labels)

#             # Discriminator 전체 손실 및 역전파
#             d_loss = d_loss_real + d_loss_fake
#             d_loss.backward()
#             d_optimizer.step()

#             # ---------------------
#             # 2. Generator 학습
#             # ---------------------
#             g_optimizer.zero_grad()

#             # Generator는 Discriminator를 속여야 하므로 진짜(1) 레이블로 학습
#             fake_preds = discriminator(fake_imgs)
#             g_loss = criterion(fake_preds, real_labels)

#             g_loss.backward()
#             g_optimizer.step()

#         print(f"[UNGAN] Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

#     print("[UNGAN] Generator and Discriminator training completed.\n")
#     return generator, discriminator


# ========== 1. 기존 train_gd_ungan 함수를 이것으로 교체 ==========
def train_gd_ungan(generator, discriminator, dataset, retain_idxs, forget_idxs, device,
                   lambda_adv=1.0, z_dim=100, batch_size=64, epochs=10):
    """
    제공하신 코드와 동일한 방식의 DCGAN 학습
    """
    # 제공하신 파라미터 사용
    n_epochs = 5
    batch_size = 128  
    lr1 = 0.0001
    lr2 = 0.0002
    b1 = 0.5
    b2 = 0.999
    latent_dim = 100
    sample_interval = 400

    # Loss function
    adversarial_loss = nn.BCELoss()

    # Optimizers (동일한 학습률)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr1, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr2, betas=(b1, b2))

    # 고정 노이즈
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    # 라벨 값 (제공하신 코드와 동일)
    real_label = 1.
    fake_label = 0.

    # ========== 🔧 핵심: 전체 데이터셋 사용 ==========
    # forget_idxs만 사용하는 대신 전체 데이터셋 사용 (옵션)
    use_full_dataset = True  # 실험해볼 수 있는 옵션
    
    if use_full_dataset:
        # 전체 데이터셋 사용 (제공하신 코드처럼)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"[DCGAN] Using FULL dataset: {len(dataset)} samples")
    else:
        # forget 데이터만 사용 (기존 방식)
        forget_subset = torch.utils.data.Subset(dataset, forget_idxs)
        dataloader = torch.utils.data.DataLoader(forget_subset, batch_size=batch_size, shuffle=True, drop_last=True)
        print(f"[DCGAN] Using forget subset: {len(forget_subset)} samples")

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print(f"[DCGAN] Training with {len(dataloader)} batches per epoch")

    generator.train()
    discriminator.train()

    for epoch in range(n_epochs):
        for i, data in enumerate(dataloader, 0):
            # ========== 제공하신 코드와 동일한 구조 ==========
            # 1. Discriminator 학습
            # 1-1. Real data        
            real_img = data[0].to(device)
            b_size = real_img.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            discriminator.zero_grad()
            output = discriminator(real_img).view(-1)     
            real_loss = adversarial_loss(output, label)
            real_loss.backward()
            D_x = output.mean().item()

            # 1-2. Fake data   
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(fake_label)

            output = discriminator(fake.detach()).view(-1)
            fake_loss = adversarial_loss(output, label)
            fake_loss.backward()

            D_G_z1 = output.mean().item()        
            disc_loss = real_loss + fake_loss

            optimizer_D.step()

            # 2. Generator 학습
            generator.zero_grad()
            label.fill_(real_label)  
            output = discriminator(fake).view(-1)
            gen_loss = adversarial_loss(output, label)
            gen_loss.backward()
            D_G_z2 = output.mean().item()
            optimizer_G.step()

            if i % 50 == 0:
                print('[{}/{}][{}/{}]'.format(epoch+1, n_epochs, i, len(dataloader)))            
                print('Discriminator Loss:{:.4f}\t Generator Loss:{:.4f}\t D(x):{:.4f}\t D(G(z)):{:.4f}/{:.4f}'.format(
                    disc_loss.item(), gen_loss.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(gen_loss.item())
            D_losses.append(disc_loss.item())

            if (iters % sample_interval == 0) or ((epoch == n_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake_sample = generator(fixed_noise).detach().cpu()
                img_list.append(utils.make_grid(fake_sample, padding=2, normalize=True))
                print(f"[Sample Generated] Iter {iters}")

            iters += 1

    print(f"[DCGAN] Training completed! Total iterations: {iters}")
    return generator, discriminator


def train_gd_ungan_with_unseen(generator, discriminator, dataset, retain_idxs, forget_idxs, device,
                   lambda_adv, z_dim, batch_size, epochs, unseen_dataset=None):
    """
    UNGAN Generator & Discriminator 학습
    → Discriminator는 Unseen+Forget 데이터를 모두 Real로 학습
    → Generator는 adversarial loss + (optional) unseen similarity loss 사용
    """
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    criterion = nn.BCELoss()  # GAN용 이진 분류 손실

    # Forget + Unseen 데이터 로더 구성
    forget_subset = torch.utils.data.Subset(dataset, forget_idxs)

    if unseen_dataset is not None:
        real_dataset = torch.utils.data.ConcatDataset([forget_subset, unseen_dataset])
    else:
        real_dataset = forget_subset

    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"[UNGAN] Training Generator and Discriminator for {epochs} epochs...")

    generator.train()
    discriminator.train()
    if unseen_dataset is not None:
        unseen_loader = torch.utils.data.DataLoader(unseen_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        unseen_iter = iter(unseen_loader)
    for epoch in range(epochs):
        for real_imgs, _ in real_loader:
            real_imgs = real_imgs.to(device)

            # ---------------------
            # 1. Discriminator 학습
            # ---------------------
            d_optimizer.zero_grad()

            real_labels = torch.ones((real_imgs.size(0), 1), device=device)
            fake_labels = torch.zeros((real_imgs.size(0), 1), device=device)

            real_preds = discriminator(real_imgs)
            d_loss_real = criterion(real_preds, real_labels)

            z = torch.randn((real_imgs.size(0), z_dim), device=device)
            fake_imgs = generator(z)

            fake_preds = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(fake_preds, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # ---------------------
            # 2. Generator 학습
            # ---------------------
            g_optimizer.zero_grad()
            fake_preds = discriminator(fake_imgs)
            g_loss_adv = criterion(fake_preds, real_labels)

            sim_loss = 0.0
            if unseen_dataset is not None:
                try:
                    unseen_imgs, _ = next(unseen_iter)
                except StopIteration:
                    unseen_iter = iter(unseen_loader)
                    unseen_imgs, _ = next(unseen_iter)

                unseen_imgs = unseen_imgs.to(device)
                min_len = min(fake_imgs.size(0), unseen_imgs.size(0))
                sim_loss = F.mse_loss(fake_imgs[:min_len], unseen_imgs[:min_len])

            g_loss = g_loss_adv + lambda_adv * sim_loss
            g_loss.backward()
            g_optimizer.step()

        print(f"[UNGAN] Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | Sim Loss: {sim_loss:.4f}")

    print("[UNGAN] Generator and Discriminator training completed.\n")
    return generator, discriminator






# -------------------- Synthetic Dataset 정의 --------------------
class SyntheticImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        # 라벨을 텐서로 변환 (DataLoader 호환성)
        if isinstance(labels, list):
            self.labels = torch.tensor(labels, dtype=torch.long)
        elif isinstance(labels, torch.Tensor):
            self.labels = labels.long()
        else:
            self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 이미지와 라벨 모두 텐서로 반환
        image = self.images[idx]
        label = self.labels[idx]
        
        # 이미지가 텐서가 아니면 변환
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
            
        return image, label


# -------------------- IID 분배 --------------------
def partition_synthetic_data_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    indices = np.random.permutation(len(dataset))
    user_groups = {}

    for i in range(num_users):
        user_groups[i] = indices[i * num_items:(i + 1) * num_items].tolist()

    return user_groups


# -------------------- Non-IID 분배 --------------------
def partition_synthetic_data_dirichlet(dataset, num_users, alpha=0.5, num_classes=10):
    """
    Synthetic 데이터셋을 Dirichlet 분포 기반으로 Non-IID하게 분할

    Args:
        dataset: SyntheticImageDataset 
        num_users: 사용자 수 (언러닝 클라이언트 제외, 즉 9명)
        alpha: Dirichlet 분포의 집중도 (작을수록 편향 큼)
        num_classes: 총 클래스 수

    Returns:
        user_groups: Dict[user_id] = list of sample indices
    """
    if isinstance(dataset.labels, torch.Tensor):
        labels = dataset.labels.cpu().numpy()
    else:
        labels = np.array(dataset.labels)

    user_groups = {i: [] for i in range(num_users)}
    idxs = np.arange(len(dataset))

    # 클래스별 인덱스 그룹화
    class_indices = [idxs[labels == y] for y in range(num_classes)]

    for c in range(num_classes):
        np.random.shuffle(class_indices[c])
        class_size = len(class_indices[c])

        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = np.array([
            p * (len(user_groups[i]) < len(dataset) / num_users)
            for i, p in enumerate(proportions)
        ])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * class_size).astype(int)[:-1]

        split = np.split(class_indices[c], proportions)
        for i, idx in enumerate(split):
            user_groups[i] += idx.tolist()

    return user_groups




# -------------------- Subset 추출 --------------------
def get_synthetic_subset(dataset, user_groups, user_idx):
    return Subset(dataset, user_groups[user_idx])
