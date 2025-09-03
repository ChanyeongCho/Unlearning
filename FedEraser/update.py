import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import copy

class LocalUpdate:
    def __init__(self, args, dataset, idxs=None):
        self.args = args
        self.device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

        # Dataset 설정 (real or synthetic 모두 가능)
        if idxs is not None:
            self.train_loader = DataLoader(Subset(dataset, idxs),
                                           batch_size=self.args.local_bs,
                                           shuffle=True)
        else:
            self.train_loader = DataLoader(dataset,
                                           batch_size=self.args.local_bs,
                                           shuffle=True)

        self.criterion = nn.CrossEntropyLoss()

    def update_weights(self, model, global_round):
        model.to(self.device)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum,weight_decay=1e-4)

        epoch_loss = []

        for epoch in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def FedErase_update_weights(self, model, global_round):
        model.to(self.device)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum,weight_decay=1e-4)
        original_weights = copy.deepcopy(model.state_dict())
        epoch_loss = []

        for epoch in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        
        # final weights after local update
        updated_weights = model.state_dict()

        # FedEraser: compute delta = updated_weights - original_weights
        delta_weights = {}
        for key in updated_weights.keys():
            delta_weights[key] = updated_weights[key] - original_weights[key]

        # 반환: 모델 파라미터와 로컬 손실, 델타 파라미터
        return updated_weights, sum(epoch_loss) / len(epoch_loss), delta_weights

    def synthetic_update_weights(self, model, global_round, synthetic_dataset):
        """
        합성 데이터로 언러닝 클라이언트 업데이트
        """
        model.to(self.device)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-4)
        original_weights = copy.deepcopy(model.state_dict())
        
        # 합성 데이터셋으로 DataLoader 생성
        synthetic_loader = DataLoader(synthetic_dataset, batch_size=self.args.local_bs, shuffle=True)
        
        epoch_loss = []
        
        print(f"[Synthetic Update] Training with {len(synthetic_dataset)} synthetic samples")
        
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(synthetic_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        
        updated_weights = model.state_dict()
        
        # 델타 계산 (FedEraser 스타일)
        delta_weights = {}
        for key in updated_weights.keys():
            delta_weights[key] = updated_weights[key] - original_weights[key]
        
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"[Synthetic Update] Completed with average loss: {avg_loss:.4f}")
        
        return updated_weights, avg_loss, delta_weights

    def short_update_weights(self, model, global_round, short_epochs=None):
        """
        FedEraser 방식: 나머지 클라이언트들의 짧은 업데이트
        """
        if short_epochs is None:
            short_epochs = max(1, self.args.local_ep // 3)  # 기본 에포크의 1/3
        
        model.to(self.device)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-4)
        original_weights = copy.deepcopy(model.state_dict())
        
        epoch_loss = []
        
        print(f"[Short Update] Training for {short_epochs} epochs (reduced from {self.args.local_ep})")
        
        for epoch in range(short_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        
        updated_weights = model.state_dict()
        
        # 델타 계산
        delta_weights = {}
        for key in updated_weights.keys():
            delta_weights[key] = updated_weights[key] - original_weights[key]
        
        return updated_weights, sum(epoch_loss) / len(epoch_loss), delta_weights

    def inference(self, model):
        model.to(self.device)
        model.eval()

        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        test_loss /= len(self.train_loader.dataset)
        accuracy = correct / len(self.train_loader.dataset)

        return accuracy, test_loss


def test_inference(args, model, test_dataset):
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset,
                             batch_size=128,
                             shuffle=False)

    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    return accuracy, test_loss


def combine_weights_federa(global_weights, unlearn_delta, other_deltas, unlearn_weight=0.5):
    """
    FedEraser 스타일 가중치 조합
    global_weights: 현재 글로벌 모델 가중치
    unlearn_delta: 언러닝 클라이언트의 델타
    other_deltas: 나머지 클라이언트들의 델타 리스트
    unlearn_weight: 언러닝 클라이언트의 가중치
    """
    combined_weights = copy.deepcopy(global_weights)
    
    # 나머지 클라이언트들의 평균 델타 계산
    if len(other_deltas) > 0:
        avg_other_delta = {}
        for key in other_deltas[0].keys():
            avg_other_delta[key] = torch.zeros_like(other_deltas[0][key])
            for delta in other_deltas:
                avg_other_delta[key] += delta[key]
            avg_other_delta[key] /= len(other_deltas)
        
        # 가중치 조합: 언러닝 델타와 일반 델타를 결합
        for key in combined_weights.keys():
            combined_weights[key] = global_weights[key] + \
                                 unlearn_weight * unlearn_delta[key] + \
                                 (1 - unlearn_weight) * avg_other_delta[key]
    else:
        # 언러닝 클라이언트만 있는 경우
        for key in combined_weights.keys():
            combined_weights[key] = global_weights[key] + unlearn_delta[key]
    
    return combined_weights
