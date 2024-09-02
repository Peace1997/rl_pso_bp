import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from torch.utils.tensorboard import SummaryWriter

EPOCH = 1000  # 训练次数


def data_processing():
    # 读取每一行的数据
    with open("../data_new.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    csvfile.close()
    # 将每个str数据转换为float
    for i in range(len(rows)):
        for j in range(7):
            rows[i][j] = float(rows[i][j])

    # 读取每一列的数据
    columns = []
    for i in range(7):
        column = [row[i] for row in rows]
        columns.append(column)

    # 获取每一列数据的最大值和最小值
    columns_max = []
    columns_min = []
    for i in range(7):
        one_max = max(columns[i])
        one_min = min(columns[i])
        columns_max.append(one_max)
        columns_min.append(one_min)

    # 输入输出数据归一化处理
    for i in range(len(rows)):
        for j in range(7):
            rows[i][j] = 0.8 * (float(rows[i][j]) - columns_min[j]) / (columns_max[j] - columns_min[j]) + 0.1

    # 前33个数据作为训练数据，后5个数据作为测试数据
    train_data_x = []
    train_data_y = []
    test_data_x = []
    test_data_y = []
    for i in range(189):  # 189
        train_data_x.append(torch.tensor(rows[i][:6], dtype=torch.float))
        train_data_y.append(torch.tensor(rows[i][-1], dtype=torch.float))
    for i in range(189, 237):  # 189,237
        test_data_x.append(torch.tensor(rows[i][:6], dtype=torch.float))
        test_data_y.append(torch.tensor(rows[i][-1], dtype=torch.float))

    # 训练数据和测试数据
    train_data_x = torch.tensor([item.cpu().detach().numpy() for item in train_data_x])
    train_data_y = torch.tensor(train_data_y, dtype=torch.float)
    test_data_x = torch.tensor([item.cpu().detach().numpy() for item in test_data_x])
    test_data_y = torch.tensor(test_data_y, dtype=torch.float)
    return train_data_x, train_data_y, test_data_x, test_data_y


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        """
        构建神经网络
        :param n_feature: 输入层神经元个数
        :param n_hidden: 隐藏层神经元个数
        :param n_output: 输出层神经元个数
        """
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.constant_(self.hidden.bias, 0.0)
        torch.nn.init.xavier_uniform_(self.predict.weight)
        torch.nn.init.constant_(self.predict.bias, 0.0)
        self.layer_norm = torch.nn.LayerNorm(n_hidden)

    def forward(self, x):
        x = self.hidden(x)
        x = F.elu(x)
        x = self.predict(x)
        return x


# 初始化神经网络常用参数


class Particle:
    def __init__(self, dim, x_range, v_range):
        """
        单个粒子结构
        :param dim: 粒子维度
        :param x_range: 位置范围
        :param v_range: 速度范围
        """
        self.x = np.random.uniform(x_range[0], x_range[1], dim)
        self.v = np.random.uniform(v_range[0], v_range[1], (dim,))
        self.Pbest = np.inf
        self.Pbest_pos = np.zeros((dim,))
        self.Pbest_net = Net(6, 13, 1)
        self.Pbest_net_optimizer = torch.optim.Adam(self.Pbest_net.parameters(), lr=0.001)


class PSO_BP:
    def __init__(self, num_particle, dim, x_range, v_range, w_max, w_min, max_iter, min_fitness):
        """
        一群粒子结构
        :param num_particle: 粒子数
        :param dim: 粒子维度
        :param x_range: 位置范围
        :param v_range: 速度范围
        :param w_range: 惯性权重
        :param c1: 学习因子1
        :param c2: 学习因子2
        """
        self.p = np.array([Particle(dim, x_range, v_range) for i in range(num_particle)])
        self.Gbest_loss = 1
        self.Pre_Gbest_loss = 1
        self.Gbest_pos = np.zeros(dim)
        self.x_range = x_range
        self.v_range = v_range
        self.w_max = w_max
        self.w_min = w_min
        self.num_particle = num_particle
        self.dim = dim

        self.c1 = 2  # 学习因子1
        self.c2 = 2  # 学习因子2
        self.w = 0.5
        self.max_iter = max_iter
        self.min_fitness = min_fitness
        self.iter = 0

        self.iter_no_improve_step = 0

        self.particle_net = Net(6, 13, 1)
        self.Gbest_net = Net(6, 13, 1)
        self.Gbest_optimizer = torch.optim.Adam(self.Gbest_net.parameters(), lr=0.001)

    def optimizer(self, X, Y, c1, c2):
        """
        迭代优化

        :param X: 输入数据
        :param Y: 标签值
        :param iter: 迭代次数
        :return:
        """
        writer = SummaryWriter("./logs/5")
        self.iter = 0
        self.c1, self.c2 = c1, c2
        while not (self.Gbest_loss < self.min_fitness or self.iter > self.max_iter):
            self.iter += 1

            for particle in self.p:
                fitness = self.forward_and_compute_particle_loss(X, Y, particle.x)

                if fitness - particle.Pbest < 1.0e-64:
                    particle.Pbest = fitness
                    particle.Pbest_pos = particle.x.copy()
                    self.update_and_backward_Pbest(
                        particle.Pbest_net, particle.Pbest_net_optimizer, X, Y, particle.Pbest_pos
                    )
                else:
                    particle.Pbest = self.backward_Pbest(
                        particle.Pbest_net, particle.Pbest_net_optimizer, X, Y
                    )

                if fitness - self.Gbest_loss < 1.0e-64:
                    self.Gbest_loss = fitness
                    self.Gbest_pos = particle.x.copy()
                    self.update_and_backward_Gbest(X, Y, self.Gbest_pos)

                if particle.Pbest - self.Gbest_loss < 1.0e-64:
                    self.Gbest_loss = particle.Pbest
                    self.Gbest_net.hidden.bias = particle.Pbest_net.hidden.bias
                    self.Gbest_net.hidden.weight = particle.Pbest_net.hidden.weight
                    self.Gbest_net.predict.bias = particle.Pbest_net.predict.bias
                    self.Gbest_net.predict.weight = particle.Pbest_net.predict.weight
                    Gbest_prediction = self.Gbest_net(X)
                    Gbest_loss = F.mse_loss(Gbest_prediction, Y)
                    self.Gbest_optimizer.zero_grad()
                    Gbest_loss.backward()
                    self.Gbest_optimizer.step()

                else:
                    self.Gbest_loss = self.backward_Gbest(X, Y)
                    # print(self.Gbest_loss)
            writer.add_scalar("fitness ", self.Gbest_loss, self.iter)
            for particle in self.p:
                # w = self.w_max - (self.w_max-self.w_min)*(self.iter/self.max_iter)
                # w = 1

                particle.v = (
                    self.w * particle.v
                    + (self.c1 * np.random.uniform(1.0, 1.0, (self.dim,)) * (particle.Pbest_pos - particle.x))
                    + (self.c2 * np.random.uniform(1.1, 1.0, (self.dim,)) * (self.Gbest_pos - particle.x))
                )
                # particle.v  =np.clip(particle.v,-0.2,0.2)
                particle.x = particle.x + particle.v

        print("train Gbest loss:", self.Gbest_loss, "\n pos_bp iter: ", self.iter)

    def one_optimizer(self, X, Y, w, c1, c2):
        """
        迭代优化

        :param X: 输入数据
        :param Y: 标签值
        :param iter: 迭代次数
        :return:
        """
        self.w, self.c1, self.c2 = w, c1, c2
        mean_pbets, mean_gbest, mean_x, mean_diverse = [0.0] * 105, [0.0] * 105, [0.0] * 105, 0
        for particle in self.p:
            fitness = self.forward_and_compute_particle_loss(X, Y, particle.x)

            if fitness - particle.Pbest < 1.0e-64:
                particle.Pbest = fitness
                particle.Pbest_pos = particle.x.copy()
                self.update_and_backward_Pbest(
                    particle.Pbest_net, particle.Pbest_net_optimizer, X, Y, particle.Pbest_pos
                )
            else:
                particle.Pbest = self.backward_Pbest(particle.Pbest_net, particle.Pbest_net_optimizer, X, Y)

            if fitness - self.Gbest_loss < 1.0e-64:
                self.Gbest_loss = fitness
                self.Gbest_pos = particle.x.copy()
                self.update_and_backward_Gbest(X, Y, self.Gbest_pos)

            if particle.Pbest - self.Gbest_loss < 1.0e-64:
                self.Gbest_loss = particle.Pbest
                self.Gbest_net.hidden.bias = particle.Pbest_net.hidden.bias
                self.Gbest_net.hidden.weight = particle.Pbest_net.hidden.weight
                self.Gbest_net.predict.bias = particle.Pbest_net.predict.bias
                self.Gbest_net.predict.weight = particle.Pbest_net.predict.weight
                Gbest_prediction = self.Gbest_net(X)
                Gbest_loss = F.mse_loss(Gbest_prediction, Y)
                self.Gbest_optimizer.zero_grad()
                Gbest_loss.backward()
                self.Gbest_optimizer.step()

            else:
                self.Gbest_loss = self.backward_Gbest(X, Y)
                # print(self.Gbest_loss)
        for particle in self.p:
            # self.w = self.w_max - (self.w_max-self.w_min)*(self.iter/self.max_iter)
            # w = 1
            mean_pbets += particle.Pbest_pos - particle.x
            mean_gbest += self.Gbest_pos - particle.x
            mean_x += particle.x
            particle.v = (
                self.w * particle.v
                + (self.c1 * np.random.uniform(1.0, 1.0, (self.dim,)) * (particle.Pbest_pos - particle.x))
                + (self.c2 * np.random.uniform(1.0, 1.0, (self.dim,)) * (self.Gbest_pos - particle.x))
            )
            particle.v = np.clip(particle.v, -0.5, 0.5)
            particle.x = particle.x + particle.v
            particle.x = np.clip(particle.v, -2, 2)
        mean_x /= self.num_particle
        for i in self.p:
            mean_diverse += (i.x - mean_x) ** 2
        mean_diverse = mean_diverse**0.5
        mean_diverse = mean_diverse.mean()
        mean_pbets, mean_gbest = [x / self.num_particle for x in mean_pbets], [
            x / self.num_particle for x in mean_gbest
        ]
        return self.Gbest_loss, mean_pbets, mean_gbest, mean_diverse

    def final_one_optimizer(self, X, Y, w, c1, c2):
        """
        迭代优化

        :param X: 输入数据
        :param Y: 标签值
        :param iter: 迭代次数
        :return:
        """
        self.w, self.c1, self.c2 = w, c1, c2
        mean_pbest_loss = 0.0
        mean_pbets, mean_gbest, mean_x, mean_diverse = [0.0] * 105, [0.0] * 105, [0.0] * 105, 0
        for particle in self.p:
            fitness = self.forward_and_compute_particle_loss(X, Y, particle.x)

            if fitness - particle.Pbest < 1.0e-64:
                particle.Pbest = fitness
                particle.Pbest_pos = particle.x.copy()
                self.update_and_backward_Pbest(
                    particle.Pbest_net, particle.Pbest_net_optimizer, X, Y, particle.Pbest_pos
                )
            else:
                particle.Pbest = self.backward_Pbest(particle.Pbest_net, particle.Pbest_net_optimizer, X, Y)
            mean_pbest_loss += particle.Pbest
            if fitness - self.Gbest_loss < 1.0e-64:
                self.Gbest_loss = fitness
                self.Gbest_pos = particle.x.copy()
                self.update_and_backward_Gbest(X, Y, self.Gbest_pos)

            if particle.Pbest - self.Gbest_loss < 1.0e-64:
                self.Gbest_loss = particle.Pbest
                self.Gbest_net.hidden.bias = particle.Pbest_net.hidden.bias
                self.Gbest_net.hidden.weight = particle.Pbest_net.hidden.weight
                self.Gbest_net.predict.bias = particle.Pbest_net.predict.bias
                self.Gbest_net.predict.weight = particle.Pbest_net.predict.weight
                Gbest_prediction = self.Gbest_net(X)
                Gbest_loss = F.mse_loss(Gbest_prediction, Y)
                self.Gbest_optimizer.zero_grad()
                Gbest_loss.backward()
                self.Gbest_optimizer.step()

            else:
                self.Gbest_loss = self.backward_Gbest(X, Y)
                # print(self.Gbest_loss)
        mean_pbest_loss /= self.num_particle
        for particle in self.p:
            # self.w = self.w_max - (self.w_max-self.w_min)*(self.iter/self.max_iter)
            # w = 1
            mean_pbets += particle.Pbest_pos - particle.x
            mean_gbest += self.Gbest_pos - particle.x
            mean_x += particle.x
            particle.v = (
                self.w * particle.v
                + (self.c1 * np.random.uniform(1.0, 1.0, (self.dim,)) * (particle.Pbest_pos - particle.x))
                + (self.c2 * np.random.uniform(1.0, 1.0, (self.dim,)) * (self.Gbest_pos - particle.x))
            )
            particle.v = np.clip(particle.v, -0.5, 0.5)
            particle.x = particle.x + particle.v
            particle.x = np.clip(particle.v, -2, 2)
        mean_x /= self.num_particle
        for i in self.p:
            mean_diverse += (i.x - mean_x) ** 2
        mean_diverse = mean_diverse**0.5
        mean_diverse = mean_diverse.mean()
        # mean_pbets ,mean_gbest = [x / self.num_particle  for x in mean_pbets], [x / self.num_particle  for x in mean_gbest]

        # obs list
        obs = np.array([0.0] * 5)
        obs[0] = (mean_pbest_loss - self.Gbest_loss).detach().numpy() * 10
        # obs[1] = self.Gbest_loss
        if self.iter == 0:
            obs[1] = 0
            self.Pre_Gbest_loss = self.Gbest_loss

        else:
            obs[1] = (
                self.Pre_Gbest_loss - self.Gbest_loss
            ).detach().numpy() * 10000  # np.exp(-(self.Pre_Gbest_loss -self.Gbest_loss).detach().numpy())
            a = float(f"{self.Pre_Gbest_loss-self.Gbest_loss:.4f}")
            if a > 0:
                self.iter_no_improve_step = self.iter
            self.Pre_Gbest_loss = self.Gbest_loss
        obs[2] = (self.iter - self.iter_no_improve_step) / self.max_iter
        obs[3] = mean_diverse / 10
        obs[4] = self.iter / (self.max_iter)
        return obs

    def get_Gbest_pos(self):
        return self.Gbest_pos

    def forward_and_compute_particle_loss(self, X, Y, particle_x):
        """
        神经网络前向传播并计算误差(适应度值)
        :param X: 待训练数据
        :param Y: 标签数据
        :param particle: 单个粒子
        :return: 适应度值
        """
        particle_x = torch.tensor(particle_x, dtype=torch.float)
        self.particle_net.hidden.weight.data = torch.nn.Parameter(particle_x[:78].reshape(13, 6))
        self.particle_net.hidden.bias = torch.nn.Parameter(particle_x[78:91])
        self.particle_net.predict.weight.data = torch.nn.Parameter(particle_x[91:104].reshape(1, 13))
        self.particle_net.predict.bias = torch.nn.Parameter(particle_x[104:105])
        prediction = self.particle_net(X)
        loss = F.mse_loss(prediction.reshape(-1), Y)
        return loss

    # 更新权重并反向传播更新Pbest神经网络
    def update_and_backward_Pbest(self, Pbest_net, Pbest_net_optimizer, X, Y, Pbest_pos):
        Pbest_pos = torch.tensor(Pbest_pos, dtype=torch.float)

        Pbest_net.hidden.weight.data = torch.nn.Parameter(Pbest_pos[:78].reshape(13, 6))
        Pbest_net.hidden.bias = torch.nn.Parameter(Pbest_pos[78:91])
        Pbest_net.predict.weight = torch.nn.Parameter(Pbest_pos[91:104].reshape(1, 13))
        Pbest_net.predict.bias = torch.nn.Parameter(Pbest_pos[104:105])

        Pbest_prediction = Pbest_net(X)
        Pbest_loss = F.mse_loss(Pbest_prediction.reshape(-1), Y)
        Pbest_net_optimizer.zero_grad()
        Pbest_loss.backward()
        Pbest_net_optimizer.step()

    # 反向传播更新神经网络
    def backward_Pbest(self, Pbest_net, Pbest_net_optimizer, X, Y):
        Pbest_prediction = Pbest_net(X)
        Pbest_loss = F.mse_loss(Pbest_prediction.reshape(-1), Y)
        Pbest_net_optimizer.zero_grad()
        Pbest_loss.backward()
        Pbest_net_optimizer.step()

        return Pbest_loss

    # 更新权重并反向传播更新Gbest神经网络
    def update_and_backward_Gbest(self, X, Y, Gbest_pos):
        Gbest_pos = torch.tensor(Gbest_pos, dtype=torch.float)

        self.Gbest_net.hidden.weight = torch.nn.Parameter(Gbest_pos[:78].reshape(13, 6))
        self.Gbest_net.hidden.bias = torch.nn.Parameter(Gbest_pos[78:91])
        self.Gbest_net.predict.weight = torch.nn.Parameter(Gbest_pos[91:104].reshape(1, 13))
        self.Gbest_net.predict.bias = torch.nn.Parameter(Gbest_pos[104:105])

        Gbest_prediction = self.Gbest_net(X)
        Gbest_loss = F.mse_loss(Gbest_prediction.reshape(-1), Y)
        self.Gbest_optimizer.zero_grad()
        Gbest_loss.backward()
        self.Gbest_optimizer.step()

    def backward_Gbest(self, X, Y):
        Gbest_prediction = self.Gbest_net(X)
        Gbest_loss = F.mse_loss(Gbest_prediction.reshape(-1), Y)
        self.Gbest_optimizer.zero_grad()
        Gbest_loss.backward()
        self.Gbest_optimizer.step()

        return Gbest_loss

    def compute_obs(self, c1, c2):
        train_data_x, train_data_y, test_data_x, test_data_y = data_processing()
        iter = self.optimizer(test_data_x, test_data_y, c1, c2)
        train_output = self.Gbest_net(test_data_x)
        print("test_data_y (label):", test_data_y)
        print("train_output:", train_output)
        train_loss = F.mse_loss(train_output.reshape(-1), test_data_y)

        print("\ntrain loss:", train_loss)

        obs = (
            [train_loss.detach().numpy().tolist()]
            + self.Gbest_net.hidden.weight.detach().numpy().ravel().tolist()
            + self.Gbest_net.hidden.bias.detach().numpy().ravel().tolist()
            + self.Gbest_net.predict.weight.detach().numpy().ravel().tolist()
            + self.Gbest_net.predict.bias.detach().numpy().ravel().tolist()
        )
        np.array(obs)
        return obs

    def compute_one_obs(self, w, c1, c2):
        train_data_x, train_data_y, test_data_x, test_data_y = data_processing()
        for i in range(4):
            train_loss, mean_pbest, mean_gbest, mean_diverse = self.one_optimizer(
                train_data_x, train_data_y, w, c1, c2
            )
        # obs = [
        #           train_loss.detach().numpy().tolist()] + self.Gbest_net.hidden.weight.detach().numpy().ravel().tolist() + \
        #       self.Gbest_net.hidden.bias.detach().numpy().ravel().tolist() + self.Gbest_net.predict.weight.detach().numpy().ravel().tolist() + \
        #       self.Gbest_net.predict.bias.detach().numpy().ravel().tolist()
        obs = [train_loss.detach().numpy().tolist()] + mean_gbest + mean_pbest
        np.array(obs)
        return obs, mean_diverse

    def final_compute_one_obs(self, w, c1, c2):
        train_data_x, train_data_y, test_data_x, test_data_y = data_processing()
        for i in range(5):
            obs = self.final_one_optimizer(train_data_x, train_data_y, w, c1, c2)
        return obs

        # print("test min loss:",min)


if __name__ == "__main__":
    num_particle = 20  # 种群数量
    dim = (13 * 6) + 13 + (13 * 1) + 1  # 粒子维数
    x_range = [-1, 1]  # 位置范围
    v_range = [-0.5, 0.5]  # 速度范围
    w_max = 0.9  # 惯性权重最大值
    w_min = 0.4  # 惯性权重最小值
    max_iter = 300  # 最大迭代次数
    min_fitness = 0.001  # 目标适应值
    w, c1, c2 = 0.5, 0.5, 1.9  # 2.0,2.0#0.5,1.9#2.0,2.0#0.5,1.9
    pso_bp = PSO_BP(num_particle, dim, x_range, v_range, w_max, w_min, max_iter, min_fitness)
    print(pso_bp.final_compute_one_obs(w, c1, c2))
