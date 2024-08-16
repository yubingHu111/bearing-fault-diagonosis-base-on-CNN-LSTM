from joblib import load
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from sklearn.metrics import classification_report
import seaborn as sns

train_acc_loss = load(r"D:\document_s\vs_code\复现实验\CNN LSTM\train_test_metrics")
train_loss = train_acc_loss["train_loss"]
train_acc = train_acc_loss["train_acc"]
test_loss = train_acc_loss["test_loss"]
test_acc = train_acc_loss["test_acc"]
iter_num = train_acc_loss["iter_num"]
lr_rate = train_acc_loss["l_rate"]
best_accuracy = train_acc_loss["best_accuracy"]


def Display(train_loss, train_accuracy, test_loss, test_accuracy, iter_num, lr_rate, best_accuracy):
    num = len(train_loss)
    X = np.arange(num)
    plt.figure(figsize=(10, 5))
    plt.xlabel(f"best_accuracy:{best_accuracy:.3f}")
    plt.title(f"iter:{iter_num} lr:{lr_rate}")
    plt.plot(X, train_loss, color="red", label="train_loss")
    plt.plot(X, test_loss, color="green", label="test_loss")
    plt.plot(X, train_accuracy, color="pink", label="train_accuracy")
    plt.plot(X, test_accuracy, color="blue", label="test_accuracy")

    plt.legend(loc='upper right')  # 图例显示
    plt.show()


Display(train_loss, train_acc, test_loss, test_acc, iter_num, lr_rate, best_accuracy)

test_x_tensor = load(r"D:\document_s\vs_code\复现实验\CNN LSTM\test_x_tensor")
test_y_tensor = load(r"D:\document_s\vs_code\复现实验\CNN LSTM\test_y_tensor")
device = torch.device("cuda")
model = torch.load(r"D:\document_s\vs_code\复现实验\CNN LSTM\best_model_CNN_LSTM.pt")
model = model.to(device)

test_loader = Data.DataLoader(dataset=(Data.TensorDataset(test_x_tensor, test_y_tensor)),
                              batch_size=32, shuffle=True, drop_last=True)

with torch.no_grad():
    real_class = []
    predict_class = []
    for data, lab in test_loader:
        data, lab = data.to(device), lab.to(device)
        output = model(data)
        probabilities = F.softmax(output, dim=1)
        predict = torch.argmax(probabilities, dim=1)
        real_class.extend(lab.tolist())
        predict_class.extend(predict.tolist())
confusion_mat = confusion_matrix(real_class, predict_class)

# 计算F1-score
report = classification_report(real_class, predict_class, digits=4)
print(report)

# 绘制混淆矩阵
label_mapping = {
    0: "C1", 1: "C2", 2: "C3", 3: "C4", 4: "C5", 5: "C6",
    6: "C7", 7: "C8", 8: "C9", 9: "C10", 10: "C11", 11: "C12"
}
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mat, xticklabels=label_mapping.values(),
            yticklabels=label_mapping.values(), annot=True, fmt="d", cmap="summer")
plt.show()