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
    32×32 CIFAR-10용으로 최적화된 DCGAN 훈련 함수
    """
    import torch
    import torch.nn as nn
    import torchvision.utils as utils
    
    # ========== 하이퍼파라미터 ==========
    niter = epochs
    batch_size = 128  # 더 안정적인 훈련을 위해 유지
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    nz = z_dim
    
    # ========== 데이터로더 설정 ==========
    # retain 데이터 사용 (언러닝 목적)
    retain_subset = torch.utils.data.Subset(dataset, retain_idxs)
    dataloader = torch.utils.data.DataLoader(retain_subset, batch_size=batch_size, 
                                            shuffle=True, drop_last=True)
    print(f"[DCGAN] Using retain subset: {len(retain_subset)} samples")

    # ========== 손실 함수 및 옵티마이저 ==========
    criterion = nn.BCELoss()
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    
    # ========== 고정 노이즈 및 라벨 ==========
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)  # 시각화용
    real_label = 1
    fake_label = 0
    
    # ========== 훈련 기록용 ==========
    img_list = []
    g_loss = []
    d_loss = []
    
    print(f"[DCGAN] Starting training for {niter} epochs with {len(dataloader)} batches per epoch")
    
    generator.train()
    discriminator.train()
    
    # ========== 훈련 루프 ==========
    for epoch in range(niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            batch_size_current = real_cpu.size(0)
            label = torch.full((batch_size_current,), real_label, dtype=torch.float, device=device)

            output = discriminator(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size_current, nz, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = discriminator(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # ========== 로깅 ==========
            if i % 50 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' 
                      % (epoch, niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            g_loss.append(errG.item())
            d_loss.append(errD.item())
            
            # ========== 샘플 이미지 생성 ==========
            if i % 100 == 0:
                print('[DCGAN] Saving sample images...')
                with torch.no_grad():
                    fake_sample = generator(fixed_noise).detach().cpu()
                img_list.append(utils.make_grid(fake_sample, padding=2, normalize=True))

    print(f"[DCGAN] Training completed! Final G_loss: {g_loss[-1]:.4f}, D_loss: {d_loss[-1]:.4f}")
    
    # 훈련 모드에서 평가 모드로 전환
    generator.eval()
    discriminator.eval()
    
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
class SyntheticImageDataset(torch.utils.data.Dataset):
    """합성 이미지 데이터셋 (수정된 버전)"""
    
    def __init__(self, images, labels):
        self.images = images
        #  라벨을 텐서로 변환하여 일관성 보장
        if isinstance(labels, torch.Tensor):
            self.labels = labels
        else:
            self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        #  둘 다 텐서로 반환하여 일관성 보장
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
            
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
