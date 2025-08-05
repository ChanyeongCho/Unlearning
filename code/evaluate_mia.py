import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import json


def get_confidences(model, dataset, idxs, device):
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


def evaluate_classification_accuracy(model, dataset, idxs, device, dataset_name=""):
    """
    특정 데이터셋 인덱스에 대한 분류 정확도 평가
    """
    model.eval()
    model.to(device)
    
    if len(idxs) == 0:
        print(f"[Warning] {dataset_name}: No data to evaluate")
        return 0.0
    
    loader = DataLoader(Subset(dataset, idxs), batch_size=128, shuffle=False)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            predicted = outputs.argmax(dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"[Classification] {dataset_name} Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    return accuracy


def evaluate_synthetic_classification_accuracy(model, synthetic_dataset, device, dataset_name="Synthetic"):
    """
    생성된 합성 데이터셋에 대한 분류 정확도 평가
    """
    model.eval()
    model.to(device)
    
    if len(synthetic_dataset) == 0:
        print(f"[Warning] {dataset_name}: No synthetic data to evaluate")
        return 0.0
    
    loader = DataLoader(synthetic_dataset, batch_size=128, shuffle=False)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            predicted = outputs.argmax(dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"[Classification] {dataset_name} Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    return accuracy


def comprehensive_evaluation(model, train_dataset, test_dataset, forget_idxs, retain_idxs, 
                           synthetic_dataset, device, save_path=None):
    """
    종합적인 분류 성능 평가
    """
    print("\n========== Classification Performance Evaluation ==========")
    
    # 1. 전체 원본 테스트 데이터에 대한 정확도
    all_test_idxs = list(range(len(test_dataset)))
    test_acc = evaluate_classification_accuracy(model, test_dataset, all_test_idxs, device, "Original Test Set")
    
    # 2. Retain set에 대한 정확도
    retain_acc = evaluate_classification_accuracy(model, train_dataset, retain_idxs, device, "Retain Set")
    
    # 3. Forget set에 대한 정확도
    forget_acc = evaluate_classification_accuracy(model, train_dataset, forget_idxs, device, "Forget Set")
    
    # 4. 생성된 합성 데이터에 대한 정확도
    synthetic_acc = 0.0
    if synthetic_dataset is not None:
        synthetic_acc = evaluate_synthetic_classification_accuracy(model, synthetic_dataset, device, "Generated Synthetic Data")
    else:
        print("[Classification] Generated Synthetic Data: No synthetic data available")
    
    print("===========================================================\n")
    
    # 결과 저장
    results = {
        'original_test_accuracy': float(test_acc),
        'retain_set_accuracy': float(retain_acc),
        'forget_set_accuracy': float(forget_acc),
        'synthetic_data_accuracy': float(synthetic_acc),
        'n_retain': len(retain_idxs),
        'n_forget': len(forget_idxs),
        'n_synthetic': len(synthetic_dataset) if synthetic_dataset else 0
    }
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    return results


def evaluate_mia(model, dataset, test_dataset, forget_idxs, retain_idxs, shadow_idxs, device, save_path=None):
    # 1. shadow 모델 훈련용 confidence 수집
    conf_retain = get_confidences(model, dataset, shadow_idxs, device)
    conf_forget = get_confidences(model, dataset, forget_idxs, device)

    print("retain confidence mean:", np.mean(conf_retain))
    print("forget confidence mean:", np.mean(conf_forget))

    X_shadow = np.array(conf_retain + conf_forget).reshape(-1, 1)
    y_shadow = np.array([0] * len(conf_retain) + [1] * len(conf_forget))

    # 2. 공격 모델 학습
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X_shadow, y_shadow)

    # 3. 평가 대상 confidence 수집
    eval_conf_retain = get_confidences(model, test_dataset, retain_idxs, device)
    eval_conf_forget = get_confidences(model, dataset, forget_idxs, device)
    print("evalu retain confidence mean:", np.mean(eval_conf_retain))
    print("evalu forget confidence mean:", np.mean(eval_conf_forget))

    X_eval = np.array(eval_conf_retain + eval_conf_forget).reshape(-1, 1)
    y_eval = np.array([0] * len(eval_conf_retain) + [1] * len(eval_conf_forget))

    # 4. AUC 계산
    pred_probs = clf.predict_proba(X_eval)[:, 1]
    auc = roc_auc_score(y_eval, pred_probs)

    result = {
        'auc': float(auc),
        'n_forget': len(forget_idxs),
        'n_retain': len(retain_idxs),
        'n_shadow': len(shadow_idxs)
    }

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=4)

    return result
