import torch
import torch.nn as nn
import torch.utils.data as Data
from joblib import dump, load
from CNN_LSTM import CNN_LSTM
import torch.nn.functional as F

torch.manual_seed(1)
device = torch.device("cuda")


def DataLoad(batch_size):
    # 加载tensor数据集，并制作成dataloader格式
    train_x_tensor = load(r"D:\document_s\vs_code\复现实验\CNN LSTM\train_x_tensor")
    train_y_tensor = load(r"D:\document_s\vs_code\复现实验\CNN LSTM\train_y_tensor")
    test_x_tensor = load(r"D:\document_s\vs_code\复现实验\CNN LSTM\test_x_tensor")
    test_y_tensor = load(r"D:\document_s\vs_code\复现实验\CNN LSTM\test_y_tensor")

    # dro_last -- 是否丢弃最后一个小批量数据；shuffle--是否打乱数据集
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_x_tensor, train_y_tensor),
                                   batch_size=batch_size, shuffle=True, drop_last=True)

    test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_x_tensor, test_y_tensor),
                                  batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader


def train_model(train_load, test_load, batch_size, epochs, model, optimizer, loss_function):
    # 参数初始化
    model = model.to(device)

    train_size = len(train_load) * batch_size
    test_size = len(test_loader) * batch_size
    best_model = model
    best_accuracy = 0
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    for epoch in range(epochs):
        model.train()
        loss_train = 0
        correct_train = 0
        for train, label in train_load:
            train, label = train.to(device), label.to(device)
            optimizer.zero_grad()
            y_pre = model(train)

            loss = loss_function(y_pre, label)
            probabilities = F.softmax(y_pre, dim=1)  # 计算概率
            predict_label = torch.argmax(probabilities, dim=1)  # 预测标签
            correct_train += (predict_label == label).sum().item()
            loss_train += loss.item()

            loss.backward()  # 反向传播
            optimizer.step()  # 梯度更新
        tr_loss = loss_train / train_size
        tr_correct = correct_train / train_size
        train_loss.append(tr_loss)
        train_acc.append(tr_correct)
        print(f"epoch:{epoch + 1:2}  train_loss:{tr_loss:10.8f}  train_correct:{tr_correct:4.4f}")

        with torch.no_grad():
            loss_test = 0
            correct_test = 0
            for test, lab in test_load:
                test, lab = test.to(device), lab.to(device)
                output = model(test)
                loss = loss_function(output, lab)
                loss_test += loss.item()
                probabilities = F.softmax(output, dim=1)
                predict = torch.argmax(probabilities, dim=1)
                correct_test += (predict == lab).sum().item()

            te_loss = loss_test / test_size
            te_correct = correct_test / test_size
            test_loss.append(te_loss), test_acc.append(te_correct)
            print(f"epoch:{epoch + 1:2}  test_loss:{te_loss:10.8f}  test_acc:{te_correct:4.4f}")

            if te_correct > best_accuracy:
                best_accuracy = te_correct
                best_model = model
    print(f"best_accuracy:{best_accuracy:4.4f}")

    torch.save(best_model, "best_model_CNN_LSTM.pt")
    dump({
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "iter_num": epochs,
        "l_rate": learning_rate,
        "best_accuracy": best_accuracy
    },
        "train_test_metrics")


if __name__ == "__main__":
    batch_size = 32
    train_loader, test_loader = DataLoad(batch_size)

    input_dim = 1
    conv_param = [32, 64, 128]
    lstm_param = [128, 64]
    class_num = 12
    learning_rate = 0.001
    epochs = 50

    # 实例化模型
    # 实例化优化器
    # 实例化损失函数
    model = CNN_LSTM(input_dim=input_dim, conv_param=conv_param, lstm_hid=lstm_param,
                     batch_size=batch_size, class_num=class_num)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    # 训练模型
    train_model(train_loader, test_loader, batch_size, epochs, model, optimizer, loss_function)

