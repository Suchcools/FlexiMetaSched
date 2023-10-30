import torch.nn.modules as nn
import torch
import torch.nn.functional as F

def maml_init_(module):
    torch.nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    torch.nn.init.constant_(module.bias.data, 0.0)
    return module


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, max_pool_factor=1.0):
        super().__init__()
        stride = (int(2 * max_pool_factor))
        self.max_pool = nn.MaxPool1d(kernel_size=stride, stride=stride, ceil_mode=False)
        self.normalize = nn.BatchNorm1d(out_channels, affine=True)
        torch.nn.init.uniform_(self.normalize.weight)
        self.relu = nn.ReLU()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True)
        maml_init_(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class ConvBase(nn.Sequential):
    def __init__(self, hidden=64, channels=1, layers=4, max_pool_factor=1.0):
        core = [ConvBlock(channels, hidden, 3, max_pool_factor)]
        for _ in range(layers - 1):
            core.append(ConvBlock(hidden, hidden, 3, max_pool_factor))
        super(ConvBase, self).__init__(*core)


class CNN4Backbone(ConvBase):
    def forward(self, x):
        x = super(CNN4Backbone, self).forward(x)
        x = x.reshape(x.size(0), -1)
        return x


class Net4CNN(torch.nn.Module):
    def __init__(self, output_size, hidden_size, layers, channels, embedding_size):
        super().__init__()
        self.features = CNN4Backbone(hidden_size, channels, layers, max_pool_factor=4 // layers)
        self.classifier = torch.nn.Linear(embedding_size, output_size, bias=True)
        maml_init_(self.classifier)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class MLP(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, dropout=0.5):
        super(MLP, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.bn1 = torch.nn.BatchNorm1d(n_hidden)

        self.hidden_2 = torch.nn.Linear(n_hidden, n_hidden//2)
        self.bn2 = torch.nn.BatchNorm1d(n_hidden//2)

        self.hidden_3 = torch.nn.Linear(n_hidden//2, n_hidden//4)  # hidden layer
        self.bn3 = torch.nn.BatchNorm1d(n_hidden//4)

        self.hidden_4 = torch.nn.Linear(n_hidden // 4, n_hidden // 8)  # hidden layer
        self.bn4 = torch.nn.BatchNorm1d(n_hidden // 8)

        self.out = torch.nn.Linear(n_hidden//8, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden_1(x))  # activation function for hidden layer
        x = self.dropout(self.bn1(x))
        x = F.relu(self.hidden_2(x))  # activation function for hidden layer
        x = self.dropout(self.bn2(x))
        x = F.relu(self.hidden_3(x))  # activation function for hidden layer
        x = self.dropout(self.bn3(x))
        x = F.relu(self.hidden_4(x))  # activation function for hidden layer
        x = self.dropout(self.bn4(x))
        x = self.out(x)
        return x