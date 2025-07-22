import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import json


def get_confidences(model, dataset, idxs, device):
    """
    모델이 주어진 데이터셋의 특정 인덱스들에 대해 예측하는 확신도를 추출
    """
    model.eval()
    loader = DataLoader(Subset(dataset, idxs), batch_size=128, shuffle=False)
    confidences = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            max_conf, _ = torch.max(probs, dim=1)
            confidences.extend(max_conf.cpu().numpy())

    return confidences


def evaluate_mia(model, dataset, test_dataset, forget_idxs, retain_idxs, shadow_idxs, device, save_path=None):
    """
    Membership Inference Attack을 통해 unlearning 효과 평가
    
    Args:
        model: 평가할 모델
        dataset: 원본 훈련 데이터셋
        test_dataset: 테스트 데이터셋
        forget_idxs: 제거하려는 데이터 인덱스들
        retain_idxs: 유지하는 데이터 인덱스들 (test_dataset에서)
        shadow_idxs: Shadow model 훈련용 인덱스들 (확실히 member인 데이터)
        device: 디바이스
        save_path: 결과 저장 경로
    
    Returns:
        MIA 결과 딕셔너리 (AUC 포함)
    """
    
    # 1. Shadow 모델 훈련용 confidence 수집
    print("[MIA] Collecting shadow confidences...")
    conf_retain = get_confidences(model, dataset, shadow_idxs, device)  # Member (확실히 훈련됨)
    conf_forget = get_confidences(model, dataset, forget_idxs, device)  # Target (제거하려는 데이터)

    print(f"[MIA] Shadow retain confidence mean: {np.mean(conf_retain):.4f}")
    print(f"[MIA] Shadow forget confidence mean: {np.mean(conf_forget):.4f}")

    # Shadow 데이터로 공격 모델 훈련
    X_shadow = np.array(conf_retain + conf_forget).reshape(-1, 1)
    y_shadow = np.array([1] * len(conf_retain) + [0] * len(conf_forget))  # 1: member, 0: non-member

    # 2. 공격 모델 학습 (Logistic Regression)
    print("[MIA] Training attack model...")
    clf = LogisticRegression(solver='liblinear', random_state=42)
    clf.fit(X_shadow, y_shadow)

    # 3. 평가 대상 confidence 수집
    print("[MIA] Collecting evaluation confidences...")
    # Test dataset의 retain 데이터 (확실히 non-member)
    eval_conf_retain = get_confidences(model, test_dataset, retain_idxs, device)  # Non-member
    # 원본 dataset의 forget 데이터 (unlearning 타겟)
    eval_conf_forget = get_confidences(model, dataset, forget_idxs, device)      # Target

    print(f"[MIA] Eval retain confidence mean: {np.mean(eval_conf_retain):.4f}")
    print(f"[MIA] Eval forget confidence mean: {np.mean(eval_conf_forget):.4f}")

    # 평가 데이터 준비
    X_eval = np.array(eval_conf_retain + eval_conf_forget).reshape(-1, 1)
    y_eval = np.array([0] * len(eval_conf_retain) + [1] * len(eval_conf_forget))  # 0: non-member, 1: member

    # 4. AUC 계산
    print("[MIA] Computing AUC...")
    pred_probs = clf.predict_proba(X_eval)[:, 1]  # member일 확률
    auc = roc_auc_score(y_eval, pred_probs)

    # 결과 정리
    result = {
        'auc': float(auc),
        'n_forget': len(forget_idxs),
        'n_retain': len(retain_idxs),
        'n_shadow': len(shadow_idxs),
        'shadow_retain_conf_mean': float(np.mean(conf_retain)),
        'shadow_forget_conf_mean': float(np.mean(conf_forget)),
        'eval_retain_conf_mean': float(np.mean(eval_conf_retain)),
        'eval_forget_conf_mean': float(np.mean(eval_conf_forget)),
        'interpretation': {
            'auc_meaning': 'Lower AUC indicates better unlearning (harder to distinguish forget data)',
            'perfect_unlearning_auc': 0.5,
            'current_status': 'excellent' if auc < 0.6 else 'good' if auc < 0.7 else 'needs_improvement'
        }
    }

    # 결과 저장
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"[MIA] Results saved to {save_path}")

    return result
