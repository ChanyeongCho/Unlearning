import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import json
import time


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


def evaluate_model_comparison(original_model, retrain_model, finetune_model, unlearn_model,
                            train_dataset, test_dataset, forget_idxs, retain_idxs, 
                            synthetic_dataset, device, save_path=None):
    """
    Original, Retrain, Finetune 모델 종합 비교 평가
    """
    print("\n========== Comprehensive Model Comparison ==========")
    
    models = {
        'Original': original_model,
        'Retrain': retrain_model,
        'Finetune': finetune_model,
        'Unlearn': unlearn_model
    }
    
    all_results = {}
    
    for model_name, model in models.items():
        if model is None:
            print(f"[Warning] {model_name} model is None, skipping...")
            continue
            
        print(f"\n--- Evaluating {model_name} Model ---")
        
        start_time = time.time()
        
        # 1. 테스트 정확도
        all_test_idxs = list(range(len(test_dataset)))
        test_acc = evaluate_classification_accuracy(model, test_dataset, all_test_idxs, device, f"{model_name} Test")
        
        # 2. Retain 정확도  
        retain_acc = evaluate_classification_accuracy(model, train_dataset, retain_idxs, device, f"{model_name} Retain")
        
        # 3. Forget 정확도
        forget_acc = evaluate_classification_accuracy(model, train_dataset, forget_idxs, device, f"{model_name} Forget")
        
        # 4. 합성 데이터 정확도
        synthetic_acc = 0.0
        if synthetic_dataset is not None:
            synthetic_acc = evaluate_synthetic_classification_accuracy(model, synthetic_dataset, device, f"{model_name} Synthetic")
        
        # 5. MIA 평가
        mia_auc = 0.0
        if len(forget_idxs) > 0 and len(retain_idxs) > 0:
            try:
                # 간단한 MIA: Forget과 Retain의 confidence 차이로 평가
                forget_conf = get_confidences(model, train_dataset, forget_idxs, device)
                retain_conf = get_confidences(model, train_dataset, retain_idxs[:len(forget_idxs)], device)  # 동일 수량으로 맞춤
                
                if len(forget_conf) > 0 and len(retain_conf) > 0:
                    X = np.array(forget_conf + retain_conf).reshape(-1, 1)
                    y = np.array([1] * len(forget_conf) + [0] * len(retain_conf))  # Forget=1, Retain=0
                    
                    clf = LogisticRegression(solver='liblinear')
                    clf.fit(X, y)
                    pred_probs = clf.predict_proba(X)[:, 1]
                    mia_auc = roc_auc_score(y, pred_probs)
                    
            except Exception as e:
                print(f"[Warning] MIA evaluation failed for {model_name}: {str(e)}")
        
        evaluation_time = time.time() - start_time
        
        # 결과 저장
        model_results = {
            'test_accuracy': float(test_acc),
            'retain_accuracy': float(retain_acc), 
            'forget_accuracy': float(forget_acc),
            'synthetic_accuracy': float(synthetic_acc),
            'mia_auc': float(mia_auc),
            'evaluation_time': float(evaluation_time),
            'n_retain': len(retain_idxs),
            'n_forget': len(forget_idxs),
            'n_synthetic': len(synthetic_dataset) if synthetic_dataset else 0
        }
        
        all_results[model_name] = model_results
        
        print(f"[Summary] {model_name} - Test: {test_acc:.3f}, Retain: {retain_acc:.3f}, Forget: {forget_acc:.3f}, MIA: {mia_auc:.3f}, Time: {evaluation_time:.2f}s")
    
    print("\n==== Final Comparison Summary ====")
    for model_name, results in all_results.items():
        print(f"{model_name:10} | Test: {results['test_accuracy']:.3f} | Retain: {results['retain_accuracy']:.3f} | Forget: {results['forget_accuracy']:.3f} | MIA: {results['mia_auc']:.3f}")
    
    print("=====================================\n")
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=4)
    
    return all_results


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


def calculate_asr(model, backdoor_dataset, device, target_label=6):
    """
    Attack Success Rate 계산 (백도어 공격 성공률)
    """
    model.eval()
    model.to(device)
    
    if len(backdoor_dataset) == 0:
        return 0.0
    
    loader = DataLoader(backdoor_dataset, batch_size=128, shuffle=False)
    correct_target = 0
    total = 0
    
    with torch.no_grad():
        for x, _ in loader:  # 실제 레이블 무시, 타겟 레이블로 예측되는지 확인
            x = x.to(device)
            outputs = model(x)
            predicted = outputs.argmax(dim=1)
            total += x.size(0)
            correct_target += (predicted == target_label).sum().item()
    
    asr = correct_target / total if total > 0 else 0.0
    print(f"[ASR] Attack Success Rate: {asr*100:.2f}% ({correct_target}/{total})")
    return asr


def evaluate_iid_vs_noniid(models_dict, datasets_dict, device, save_path=None):
    """
    IID vs Non-IID 설정에서의 모델 성능 비교
    """
    print("\n========== IID vs Non-IID Comparison ==========")
    
    results = {}
    
    for setting in ['iid', 'noniid']:
        if setting not in datasets_dict:
            continue
            
        print(f"\n--- {setting.upper()} Setting ---")
        setting_results = {}
        
        # 각 모델별 평가
        for model_name, model in models_dict.items():
            if model is None:
                continue
                
            dataset_info = datasets_dict[setting]
            
            # 기본 정확도 평가
            test_acc = evaluate_classification_accuracy(
                model, dataset_info['test_dataset'], 
                list(range(len(dataset_info['test_dataset']))), 
                device, f"{setting.upper()} {model_name} Test"
            )
            
            retain_acc = evaluate_classification_accuracy(
                model, dataset_info['train_dataset'], 
                dataset_info['retain_idxs'], 
                device, f"{setting.upper()} {model_name} Retain"
            )
            
            forget_acc = evaluate_classification_accuracy(
                model, dataset_info['train_dataset'], 
                dataset_info['forget_idxs'], 
                device, f"{setting.upper()} {model_name} Forget"
            )
            
            setting_results[model_name] = {
                'test_accuracy': float(test_acc),
                'retain_accuracy': float(retain_acc),
                'forget_accuracy': float(forget_acc)
            }
        
        results[setting] = setting_results
        
        # 설정별 요약
        print(f"\n{setting.upper()} Summary:")
        for model_name, metrics in setting_results.items():
            print(f"  {model_name}: Test={metrics['test_accuracy']:.3f}, Retain={metrics['retain_accuracy']:.3f}, Forget={metrics['forget_accuracy']:.3f}")
    
    print("===============================================\n")
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    return results
