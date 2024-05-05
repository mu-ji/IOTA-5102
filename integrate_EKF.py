import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim, hid_dims, out_dim):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential()
        dims = [input_dim] + hid_dims + [out_dim]
        for i in range(len(dims)-1):
            self.mlp.add_module('lay_{}'.format(i),nn.Linear(in_features=dims[i], out_features=dims[i+1]))
            if i+2 < len(dims):
                self.mlp.add_module('act_{}'.format(i), nn.ReLU())
    def reset_parameters(self):
        for i, l in enumerate(self.mlp):
            if type(l) == nn.Linear:
                nn.init.xavier_normal_(l.weight)

    def forward(self, x):
        return self.mlp(x)  
    
class DeepONet_ZL(nn.Module):

    def __init__(self, m, n_yloc, out_dim = 2, p = 32, hidden_dim = 128):
        
        super(DeepONet_ZL, self).__init__()
        
        # self.branch_net = MLP(m,[hidden_dim],p)
        self.branch_net = MLP(m,[hidden_dim],p)
        
        self.trunk_net = nn.Sequential(
                                        nn.Linear(1, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, p),
                                        nn.ReLU(), # add activation function at the last layer                                  
                                        )
        self.p = p
        self.out_dim = out_dim
        self.b = nn.Parameter(torch.randn(1), requires_grad=True)
        
    def forward(self, ux, y_loc):
               
        branch_out = self.branch_net(ux) 
        # print(branch_out.shape)
        # branch_out = torch.reshape(branch_out, (-1,self.p,self.out_dim))
        
        trunk_out = self.trunk_net(y_loc)
        # print(trunk_out.shape)
        
        G_uy = torch.einsum("bp,mp->bm", branch_out, trunk_out) + self.b # G(u)(y), n is number of evaluation points
        
        return G_uy
    
integrate_model = torch.load('integrate_model_1.pth')

m = 150 # m = # of sensor locations
x = torch.linspace(-2.0,2.0, steps = m).unsqueeze(-1).permute((1,0))
y_loc = x
y_loc_test = y_loc.permute((1,0))

# test3
ux_test = torch.sin(x) * torch.sin(x)
G_uy_test = x/2 - 1/4 * torch.sin(2*x)

G_uy_test_pred = integrate_model(ux_test,y_loc_test)

plt.figure()
plt.subplot(2,1,1)
plt.plot(x[0,:], ux_test[0,:])
plt.title(r"$Input Function: u(x) = sin(x)^2$")
plt.subplot(2,1,2)
plt.plot(y_loc[0,:], G_uy_test[0,:].detach().numpy(), color = "silver", lw = 3, label = "ground truth")
plt.plot(y_loc[0,:], G_uy_test_pred[0,:].detach().numpy(), "--", color = "blue",label = "prediction")
plt.legend()
plt.title(r"$Output Function: G(u)(x) = \frac{x}{2} - \frac{1}{4}sin(2x)$") 
plt.show()

measurement = G_uy_test[0,:].detach().numpy() + np.random.normal(0, 0.1, len(G_uy_test[0,:].detach().numpy()))

plt.figure()
plt.plot(y_loc[0,:], G_uy_test[0,:].detach().numpy(), color = "silver", lw = 3, label = "ground truth")
plt.scatter(y_loc[0,:], measurement, c='r', label='measurement')
plt.legend()
plt.show()

def particle_filter(num_particles, initial_particles, process_noise, measurement_noise, measurements):
    num_steps = len(measurements)
    num_particles = len(initial_particles)
    particles = np.zeros((num_particles, num_steps))
    weights = np.ones(num_particles) / num_particles
    estimates = []
    
    for t in range(num_steps):
        # 预测步骤
        if t == 0:
            particles[:, t] = initial_particles
        else:
            particles[:, t] = particles[:, t - 1] + np.random.normal(0, process_noise, size=num_particles)
        
        # 计算权重
        weights *= np.exp(-0.5 * ((particles[:, t] - measurements[t])**2) / measurement_noise**2)
        weights /= np.sum(weights)
        
        # 重采样
        indices = np.random.choice(np.arange(num_particles), size=num_particles, replace=True, p=weights)
        particles[:, t] = particles[indices, t]
        
        # 计算估计值
        estimate = np.mean(particles[:, t])
        estimates.append(estimate)
    
    return estimates

# 示例数据
measurements = list(measurement)
initial_particles = np.random.normal(0, 1, size=100000)
process_noise = 0.1
measurement_noise = 0.2

# 执行粒子滤波器
estimates = particle_filter(len(initial_particles), initial_particles, process_noise, measurement_noise, measurements)

plt.figure()
plt.plot(y_loc[0,:], G_uy_test[0,:].detach().numpy(), color = "silver", lw = 3, label = "ground truth")
#plt.plot(y_loc[0,:], estimates, color = "y", lw = 3, label = "partical filter")
plt.plot(y_loc[0,:], G_uy_test_pred[0,:].detach().numpy(), "--", color = "blue",label = "prediction")
plt.scatter(y_loc[0,:], measurement, c='r', label='measurement')
plt.legend()
plt.show()

transition_model = MLP(2,[32,64,32,16],1)

inputs = G_uy_test_pred[0,:].detach().numpy()[:-1]
controls = ux_test[0,:].detach().numpy()[:-1]

inputs = np.vstack((inputs,controls))

outputs = G_uy_test_pred[0,:].detach().numpy()[1:]

# 转换数据为张量
inputs = torch.tensor(inputs, dtype=torch.float32).permute((1,0))
targets = torch.tensor(outputs, dtype=torch.float32).unsqueeze(dim=1)

inputs.require_grad = True

criterion = nn.MSELoss()
optimizer = optim.SGD(transition_model.parameters(), lr=0.01)

# 训练迭代
num_epochs = 5000
for epoch in range(num_epochs):
    # 前向传播
    outputs = transition_model(inputs)
    loss = criterion(outputs, targets)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印训练信息
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


test_input = torch.tensor(inputs[5], dtype=torch.float32).unsqueeze(dim=0)  # 输入一个新样本
test_input.requires_grad = True
predicted_output = transition_model(test_input)

grads = torch.autograd.grad(predicted_output, test_input, torch.ones_like(predicted_output), retain_graph=True)[0]

H = 1
R = 0.1 # 测量噪声
P = 1
Q = 0.01
w = 1
v = 1
x = measurement[0]
x_list = [x]
x_pre_list = []
for i in range(len(y_loc[0,:])):
    #x_(k+1) = NN(x_k)
    #Z_k = H * k_x + R
    #x_(k+1) = x_k + partial * (x_(k+1) - x_k)
    
    if type(x) != torch.tensor:

        test_input = torch.tensor([x,np.array(ux_test[0,:].detach().numpy()[i])], dtype=torch.float32).unsqueeze(dim=0)  # 输入一个新样本
    
    test_input.requires_grad = True

    x = transition_model(test_input)
    x_pre = x[0]
    x_pre_list.append(x_pre.detach().numpy())
    A = torch.autograd.grad(x, test_input, torch.ones_like(x), retain_graph=True)[0]
    A = A[0][0]
    print(A)
    x = x.detach().float()
    P_pred = A*P*A + w*Q*w      # 估计误差协方差预测 
            
    y = measurement[i] - H * x

    K = (P_pred*H)/(H*P_pred*H + v*R*v)

    x = x + K*y  # 状态更新
    x = x[0]
    P = (1 - K*H)*P_pred             # 估计误差协方差更新
    x_list.append(x)

print(x_pre_list[10])
plt.figure()
plt.plot(y_loc[0,:], G_uy_test[0,:].detach().numpy(), color = "silver", lw = 3, label = "ground truth")
#plt.plot(y_loc[0,:], estimates, color = "y", lw = 3, label = "partical filter")
plt.plot(y_loc[0,:], x_list[1:], color = "g", lw = 3, label = "EKF")
plt.plot(y_loc[0,:], x_pre_list, color = "pink", lw = 3, label = "EKF_pre")
plt.plot(y_loc[0,:], G_uy_test_pred[0,:].detach().numpy(), "--", color = "blue",label = "prediction")
plt.scatter(y_loc[0,:], measurement, c='r', label='measurement')
plt.legend()
plt.show()

Q = 0.1
x = measurement[0]
NolearningEKF_list = []
for i in range(len(y_loc[0,:])):

    x = x #+ np.random.uniform(low=-0.1, high=0.1)

    A = 1
    P_pred = A*P*A + w*Q*w      # 估计误差协方差预测 
            
    y = measurement[i] - H * x

    K = (P_pred*H)/(H*P_pred*H + v*R*v)

    x = x + K*y  # 状态更新
    P = (1 - K*H)*P_pred             # 估计误差协方差更新
    NolearningEKF_list.append(x)

print(x_pre_list[10])
plt.figure()
plt.plot(y_loc[0,:], G_uy_test[0,:].detach().numpy(), color = "silver", lw = 3, label = "ground truth")
#plt.plot(y_loc[0,:], estimates, color = "y", lw = 3, label = "partical filter")
plt.plot(y_loc[0,:], x_list[1:], color = "g", lw = 3, label = "EKF")
plt.plot(y_loc[0,:], NolearningEKF_list, color = "pink", lw = 3, label = "Nolearning_EKF")
plt.plot(y_loc[0,:], G_uy_test_pred[0,:].detach().numpy(), "--", color = "blue",label = "prediction")
plt.scatter(y_loc[0,:], measurement, c='r', label='measurement')
plt.legend()
plt.show()


x = y_loc[0,:]
y = measurement
degree = 3  # 多项式的阶数
coefficients = np.polyfit(x, y, degree)

p = np.poly1d(coefficients)  # 创建多项式函数
x_fit = y_loc[0,:]
y_fit = p(x_fit)  # 计算拟合曲线的因变量值

EKF_error = np.mean(np.abs((G_uy_test[0,:] - torch.tensor(x_list[1:], dtype=torch.float32).unsqueeze(dim=0)).detach().numpy()))
NolearningEKF_error = np.mean(np.abs((G_uy_test[0,:] - torch.tensor(NolearningEKF_list, dtype=torch.float32).unsqueeze(dim=0)).detach().numpy()))
LS_error = np.mean(np.abs((G_uy_test[0,:] - torch.tensor(y_fit, dtype=torch.float32).unsqueeze(dim=0)).detach().numpy()))
Partical_error = np.mean(np.abs((G_uy_test[0,:] - torch.tensor(estimates, dtype=torch.float32).unsqueeze(dim=0)).detach().numpy()))
print(EKF_error)
print(NolearningEKF_error)
print(LS_error)
print(Partical_error)

plt.figure()
plt.plot(y_loc[0,:], G_uy_test[0,:].detach().numpy(), color = "silver", lw = 3, label = "ground truth")
plt.plot(y_loc[0,:], estimates, color = "y", lw = 3, label = 'PF error = {}'.format(Partical_error))
plt.plot(y_loc[0,:], x_list[1:], color = "g", lw = 3, label = "EKF error = {}".format(EKF_error))
plt.plot(y_loc[0,:], NolearningEKF_list, color = "pink", lw = 3, label = "Nolearning_EKF error = {}".format(NolearningEKF_error))
plt.plot(y_loc[0,:], G_uy_test_pred[0,:].detach().numpy(), "--", color = "blue",label = "prediction")
plt.scatter(y_loc[0,:], measurement, c='r', label='measurement')
plt.plot(x_fit, y_fit, color='gray', label='Least square error = {}'.format(LS_error))  # 绘制拟合曲线
plt.grid()
plt.legend()
plt.show()

