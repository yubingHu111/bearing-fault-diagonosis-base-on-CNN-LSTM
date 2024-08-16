import torch.nn as nn


# 模型构建
class CNN_LSTM(nn.Module):
    def __init__(self, batch_size, input_dim, conv_param, lstm_hid, class_num ):
        """

        :param batch_size: 小批量样本数（int）64
        :param input_dim: 输入维度（int）一维
        :param conv_param: 卷积层每层隐藏单元个数（list）conv_param = [32,64,128]
        :param lstm_hid: LSTM每层隐藏单元个数（list）lstm_hid = [128,64]
        :param class_num: 分类数（int）12
        """
        super().__init__()
        # 参数初始化
        self.batch_size = batch_size
        self.class_num = class_num
        # 初始化CNN模块
        conv_layers = []
        for out_channel in conv_param:
            conv_layers.append(nn.Conv1d(input_dim, out_channel, kernel_size=3, stride=1, padding=int((3 - 1) / 2)))
            # （64，32，1024），（64，64，512），（64，128，256）
            conv_layers.append(nn.ReLU())
            input_dim = out_channel
            conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            # （64，32，512）（64，64，256），output.shape（64，128，128）
        self.convolution = nn.Sequential(*conv_layers)

        # 初始化LSTM模块
        self.lstm_layers = nn.ModuleList()
        # batch_first=True--指定输入和输出的张量格式为 (batch_size, seq_len, input_size)
        self.lstm_layers.append(nn.LSTM(conv_param[-1], lstm_hid[0], batch_first=True))
        for i in range(1, len(lstm_hid)):
            self.lstm_layers.append(nn.LSTM(lstm_hid[i-1], lstm_hid[i], batch_first=True))

        # 定义输出层
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.full_connected = nn.Linear(lstm_hid[-1], class_num)    # 定义全连接层

    def forward(self, batch_seq):
        input_seq = batch_seq.view(self.batch_size, 1, -1)  # (64,1,1024)(样本数，特征数，序列长度)
        con_output = self.convolution(input_seq)            # (64,128,128)(样本数，特征数，序列长度)
        # 将输入维度符合LSTM的格式（batch_size, seq_length, input_size)
        lstm_out = con_output.transpose(1, 2)   # （64，128，128）（样本数，序列长度，input_size）
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(lstm_out)                    # （64,128,128）(64,128,64)
        # 通过平均池化展平操作
        lstm_out = lstm_out.transpose(1, 2)       # (64,64,128)（样本数，input_size，序列长度）
        avg_out = self.avgpool(lstm_out)                    # (64,64,1)(样本数，input_size, 序列长度)
        avg_out = avg_out.reshape(self.batch_size, -1)      # （64，64）
        output = self.full_connected(avg_out)               # （64，12）

        return output

input_dim = 1
conv_param = [32, 64, 128]
lstm_param = [128, 64]
class_num = 12
learning_rate = 0.001
epochs = 50
batch_size = 64
    # 实例化模型
    # 实例化优化器
    # 实例化损失函数
model = CNN_LSTM(input_dim=input_dim, conv_param=conv_param, lstm_hid=lstm_param,
                     batch_size=batch_size, class_num=class_num)
print(model)





