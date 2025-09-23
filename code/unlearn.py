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
                              lambda_adv=1.0, z_dim=100, batch_size=64, epochs=10, unseen_dataset=None, mixing_ratio=0.5):
    """
     수정된 분포 혼합: Forget 분포 특성 + Unseen 시각적 특성
    → "Unseen처럼 보이되 Forget 분포를 따름"
    """
    import torch
    import torch.nn as nn
    import torchvision.utils as utils
    
    # ========== 하이퍼파라미터 ==========
    niter = epochs
    batch_size = 128
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    nz = z_dim
    
    # ========== 분리된 데이터로더 설정 ==========
    forget_subset = torch.utils.data.Subset(dataset, forget_idxs)
    forget_loader = torch.utils.data.DataLoader(forget_subset, batch_size=batch_size//2, 
                                               shuffle=True, drop_last=True)
    
    if unseen_dataset is not None:
        #  평가용과 동일하게 수량 맞춤 (forget_size와 동일한 수로 제한)
        forget_size = len(forget_idxs)
        if len(unseen_dataset) > forget_size:
            # 무작위로 forget_size만큼 선택
            unseen_indices = torch.randperm(len(unseen_dataset))[:forget_size]
            unseen_for_training = torch.utils.data.Subset(unseen_dataset, unseen_indices)
        else:
            unseen_for_training = unseen_dataset
            
        unseen_loader = torch.utils.data.DataLoader(unseen_for_training, batch_size=batch_size//2, 
                                                   shuffle=True, drop_last=True)
        print(f"[DCGAN] Using Forget: {len(forget_subset)} + Unseen: {len(unseen_for_training)} samples with REVERSED distribution mixing")
    else:
        print(f"[DCGAN] Using forget subset only: {len(forget_subset)} samples")
        
        return train_gd_ungan(generator, discriminator, dataset, retain_idxs, forget_idxs, device, lambda_adv, z_dim, batch_size, epochs)

    # ========== 손실 함수 및 옵티마이저 ==========
    criterion = nn.BCELoss()
    mse_criterion = nn.MSELoss()  # 분포 혼합용
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    
    # ========== 고정 노이즈 및 라벨 ==========
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0
    
    # ========== 훈련 기록용 ==========
    img_list = []
    g_loss = []
    d_loss = []
    
    print(f"[DCGAN] Starting REVERSED distribution mixing training for {niter} epochs")
    print(f"[Target] Generate images that LOOK LIKE unseen but FOLLOW forget distribution")
    
    generator.train()
    discriminator.train()
    
    # ==========  수정된 분포 혼합 훈련 루프 ==========
    for epoch in range(niter):
        forget_iter = iter(forget_loader)
        unseen_iter = iter(unseen_loader)
        
        # 더 짧은 데이터로더에 맞춤
        num_batches = min(len(forget_loader), len(unseen_loader))
        
        for i in range(num_batches):
            try:
                forget_batch = next(forget_iter)
                unseen_batch = next(unseen_iter)
            except StopIteration:
                break
                
            forget_imgs = forget_batch[0].to(device)
            unseen_imgs = unseen_batch[0].to(device)
            
            # 배치 크기 맞춤
            min_batch = min(forget_imgs.size(0), unseen_imgs.size(0))
            forget_imgs = forget_imgs[:min_batch]
            unseen_imgs = unseen_imgs[:min_batch]
            
            ############################
            # (1) Update D network: 혼합된 real 데이터로 훈련
            ###########################
            discriminator.zero_grad()
            
            # Real 데이터: Forget + Unseen 혼합
            real_imgs = torch.cat([forget_imgs, unseen_imgs], dim=0)
            batch_size_current = real_imgs.size(0)
            
            label = torch.full((batch_size_current,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_imgs)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Fake 데이터 생성 및 판별
            noise = torch.randn(batch_size_current, nz, 1, 1, device=device)
            fake_imgs = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake_imgs.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) 수정된 Generator 업데이트: REVERSED 분포 혼합
            ###########################
            generator.zero_grad()
            
            # Adversarial loss
            label.fill_(real_label)
            output = discriminator(fake_imgs)
            errG_adv = criterion(output, label)
            
            # 핵심 수정: 역방향 분포-스타일 혼합
            fake_forget_batch = fake_imgs[:min_batch]  # 앞쪽 절반 → Forget 분포 학습용
            fake_unseen_batch = fake_imgs[min_batch:]  # 뒤쪽 절반 → Unseen 스타일 학습용
            
            # **NEW**: Forget의 분포적 특성 학습 (통계적 구조)
            # 채널별 평균과 분산으로 분포 특성 캡처
            forget_dist_mean = forget_imgs.mean(dim=[0, 2, 3])  # [C] - 채널별 전체 평균
            forget_dist_std = forget_imgs.std(dim=[0, 2, 3])    # [C] - 채널별 전체 분산
            
            fake_forget_dist_mean = fake_forget_batch.mean(dim=[0, 2, 3])
            fake_forget_dist_std = fake_forget_batch.std(dim=[0, 2, 3])
            
            # 분포 매칭 손실 (Forget의 통계적 특성을 따라하기)
            distribution_loss = (
                mse_criterion(fake_forget_dist_mean, forget_dist_mean) +
                mse_criterion(fake_forget_dist_std, forget_dist_std)
            ) / 2
            
            #  **NEW**: Unseen의 시각적 스타일 학습 (외관적 특징)
            # 픽셀 레벨 패턴과 텍스처 특성으로 스타일 캡처
            unseen_style_pattern = unseen_imgs.mean(dim=[2, 3])  # [B, C] - 이미지별 채널 평균
            fake_unseen_style_pattern = fake_unseen_batch.mean(dim=[2, 3])
            
            # 추가: 공간적 변화량 (텍스처 복잡도)
            unseen_spatial_var = torch.var(unseen_imgs.view(min_batch, -1), dim=1)  # [B] - 이미지별 픽셀 분산
            fake_unseen_spatial_var = torch.var(fake_unseen_batch.view(min_batch, -1), dim=1)
            
            # 스타일 매칭 손실 (Unseen의 시각적 특성을 따라하기)
            style_loss = (
                mse_criterion(fake_unseen_style_pattern, unseen_style_pattern) +
                mse_criterion(fake_unseen_spatial_var, unseen_spatial_var)
            ) / 2
            
            #  새로운 혼합 비율: 분포 70% + 스타일 30%
            # "Forget 분포를 어느 정도 따르고, Unseen처럼 보이게"
            distribution_weight = mixing_ratio  # 0.2 (분포가 더 중요)
            style_weight = 1 - mixing_ratio              # 0.8 (스타일은 보조)
            
            errG_mixing = distribution_weight * distribution_loss + style_weight * style_loss
            
            # 전체 Generator 손실
            errG = errG_adv + lambda_adv * errG_mixing
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # ========== 로깅 ==========
            if i % 25 == 0:  # 더 자주 로깅
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f (Adv: %.4f, Dist: %.4f, Style: %.4f) D(x): %.4f D(G(z)): %.4f / %.4f' 
                      % (epoch, niter, i, num_batches, errD.item(), errG.item(), errG_adv.item(), 
                         distribution_loss.item(), style_loss.item(), D_x, D_G_z1, D_G_z2))
            
            g_loss.append(errG.item())
            d_loss.append(errD.item())
            
            # ========== 샘플 이미지 생성 ==========
            if i % 50 == 0:
                print('[DCGAN] Saving REVERSED distribution-mixed sample images...')
                with torch.no_grad():
                    fake_sample = generator(fixed_noise).detach().cpu()
                img_list.append(utils.make_grid(fake_sample, padding=2, normalize=True))

    print(f"[DCGAN] REVERSED Distribution mixing training completed! Final G_loss: {g_loss[-1]:.4f}, D_loss: {d_loss[-1]:.4f}")
    print(f"[Result] Generated images should LOOK LIKE unseen but FOLLOW forget distribution patterns")
    
    generator.eval()
    discriminator.eval()
    
    return generator, discriminator



def train_gd_ungan_unseen_only(generator, discriminator, dataset, retain_idxs, forget_idxs, device,
                              lambda_adv=1.0, z_dim=100, batch_size=64, epochs=10, unseen_dataset=None):
    """
    Unseen 데이터만 사용한 DCGAN 훈련: 순수 Unseen 분포 학습
    → "Unseen 데이터의 특성만으로 합성 데이터 생성"
    """
    import torch
    import torch.nn as nn
    import torchvision.utils as utils
    
    # ========== 하이퍼파라미터 ==========
    niter = epochs
    batch_size = 128
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    nz = z_dim
    
    # ========== Unseen 데이터로더 설정 ==========
    if unseen_dataset is not None:
        # forget_size와 동일한 수량으로 제한
        forget_size = len(forget_idxs)
        if len(unseen_dataset) > forget_size:
            # 무작위로 forget_size만큼 선택
            unseen_indices = torch.randperm(len(unseen_dataset))[:forget_size]
            unseen_for_training = torch.utils.data.Subset(unseen_dataset, unseen_indices)
        else:
            unseen_for_training = unseen_dataset
            
        unseen_loader = torch.utils.data.DataLoader(unseen_for_training, batch_size=batch_size, 
                                                   shuffle=True, drop_last=True)
        print(f"[DCGAN] Using UNSEEN ONLY: {len(unseen_for_training)} samples for pure unseen distribution learning")
    else:
        print(f"[DCGAN] No unseen dataset provided, falling back to retain data")
        return train_gd_ungan(generator, discriminator, dataset, retain_idxs, forget_idxs, device, lambda_adv, z_dim, batch_size, epochs)

    # ========== 손실 함수 및 옵티마이저 ==========
    criterion = nn.BCELoss()
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    
    # ========== 고정 노이즈 및 라벨 ==========
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0
    
    # ========== 훈련 기록용 ==========
    img_list = []
    g_loss = []
    d_loss = []
    
    print(f"[DCGAN] Starting UNSEEN ONLY training for {niter} epochs")
    print(f"[Target] Generate images following pure unseen data distribution")
    
    generator.train()
    discriminator.train()
    
    # ========== 순수 Unseen 데이터 훈련 루프 ==========
    for epoch in range(niter):
        for i, data in enumerate(unseen_loader, 0):
            unseen_imgs = data[0].to(device)
            batch_size_current = unseen_imgs.size(0)
            
            ############################
            # (1) Update D network: Unseen 데이터로만 훈련
            ###########################
            discriminator.zero_grad()
            
            # Real 데이터: Unseen만 사용
            label = torch.full((batch_size_current,), real_label, dtype=torch.float, device=device)
            output = discriminator(unseen_imgs)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Fake 데이터 생성 및 판별
            noise = torch.randn(batch_size_current, nz, 1, 1, device=device)
            fake_imgs = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake_imgs.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: 순수 Adversarial 학습
            ###########################
            generator.zero_grad()
            
            # Generator는 Discriminator를 속여야 하므로 진짜(1) 레이블로 학습
            label.fill_(real_label)
            output = discriminator(fake_imgs)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # ========== 로깅 ==========
            if i % 50 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' 
                      % (epoch, niter, i, len(unseen_loader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            g_loss.append(errG.item())
            d_loss.append(errD.item())
            
            # ========== 샘플 이미지 생성 ==========
            if i % 100 == 0:
                print('[DCGAN] Saving UNSEEN ONLY sample images...')
                with torch.no_grad():
                    fake_sample = generator(fixed_noise).detach().cpu()
                img_list.append(utils.make_grid(fake_sample, padding=2, normalize=True))

    print(f"[DCGAN] UNSEEN ONLY training completed! Final G_loss: {g_loss[-1]:.4f}, D_loss: {d_loss[-1]:.4f}")
    print(f"[Result] Generated images follow pure unseen data distribution")
    
    generator.eval()
    discriminator.eval()
    
    return generator, discriminator


def train_gd_ungan_forget_only(generator, discriminator, dataset, retain_idxs, forget_idxs, device,
                              lambda_adv=1.0, z_dim=100, batch_size=64, epochs=10, unseen_dataset=None):
    """
    Forget 데이터만 사용한 DCGAN 훈련: 순수 Forget 분포 학습
    → "Forget 데이터의 특성만으로 합성 데이터 생성"
    """
    import torch
    import torch.nn as nn
    import torchvision.utils as utils
    
    # ========== 하이퍼파라미터 ==========
    niter = epochs
    batch_size = 128
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    nz = z_dim
    
    # ========== Forget 데이터로더 설정 ==========
    forget_subset = torch.utils.data.Subset(dataset, forget_idxs)
    forget_loader = torch.utils.data.DataLoader(forget_subset, batch_size=batch_size, 
                                               shuffle=True, drop_last=True)
    print(f"[DCGAN] Using FORGET ONLY: {len(forget_subset)} samples for pure forget distribution learning")

    # ========== 손실 함수 및 옵티마이저 ==========
    criterion = nn.BCELoss()
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    
    # ========== 고정 노이즈 및 라벨 ==========
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0
    
    # ========== 훈련 기록용 ==========
    img_list = []
    g_loss = []
    d_loss = []
    
    print(f"[DCGAN] Starting FORGET ONLY training for {niter} epochs")
    print(f"[Target] Generate images following pure forget data distribution")
    
    generator.train()
    discriminator.train()
    
    # ========== 순수 Forget 데이터 훈련 루프 ==========
    for epoch in range(niter):
        for i, data in enumerate(forget_loader, 0):
            forget_imgs = data[0].to(device)
            batch_size_current = forget_imgs.size(0)
            
            ############################
            # (1) Update D network: Forget 데이터로만 훈련
            ###########################
            discriminator.zero_grad()
            
            # Real 데이터: Forget만 사용
            label = torch.full((batch_size_current,), real_label, dtype=torch.float, device=device)
            output = discriminator(forget_imgs)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Fake 데이터 생성 및 판별
            noise = torch.randn(batch_size_current, nz, 1, 1, device=device)
            fake_imgs = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake_imgs.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: 순수 Adversarial 학습
            ###########################
            generator.zero_grad()
            
            # Generator는 Discriminator를 속여야 하므로 진짜(1) 레이블로 학습
            label.fill_(real_label)
            output = discriminator(fake_imgs)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # ========== 로깅 ==========
            if i % 50 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' 
                      % (epoch, niter, i, len(forget_loader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            g_loss.append(errG.item())
            d_loss.append(errD.item())
            
            # ========== 샘플 이미지 생성 ==========
            if i % 100 == 0:
                print('[DCGAN] Saving FORGET ONLY sample images...')
                with torch.no_grad():
                    fake_sample = generator(fixed_noise).detach().cpu()
                img_list.append(utils.make_grid(fake_sample, padding=2, normalize=True))

    print(f"[DCGAN] FORGET ONLY training completed! Final G_loss: {g_loss[-1]:.4f}, D_loss: {d_loss[-1]:.4f}")
    print(f"[Result] Generated images follow pure forget data distribution")
    
    generator.eval()
    discriminator.eval()
    
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
