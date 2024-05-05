import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
    
# data generation
m = 150 # m = # of sensor locations
x = torch.linspace(-2.0,2.0, steps = m).unsqueeze(-1).permute((1,0))
y_loc = x
# training data
n_batch = 50

ux = torch.zeros((n_batch, m))
G_uy_train = torch.zeros((n_batch, m))

r1 = 20
r2 = -10

for i in range(n_batch):
    
    w = (r1 - r2) * torch.rand(1) + r2
    b = torch.rand(1)
    
    ux_i = torch.cos(w*x)
    G_uy_train_i = 1/w * torch.sin(w*y_loc) #+ np.random.normal(0, 0.1, len(y_loc))
    
    ux[i,:] = ux_i
    G_uy_train[i,:] = G_uy_train_i

y_loc_train = y_loc.permute((1,0))
# testing data
w_test = 5
ux_test = torch.cos(w_test*x)
G_uy_test = 1/w_test*torch.sin(w_test*y_loc)
y_loc_test = y_loc.permute((1,0))

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


out_dim = G_uy_train.shape[0] 
model = DeepONet_ZL(m, m, out_dim)
model_mlp = MLP(m,[128, 128],m)

lr = 1e-3
num_epochs = 30000
global_step = 0
epoch_loss = 0

params = model.parameters()
optimizer = optim.Adam(params, lr= lr)

for epoch in range(num_epochs):
    
    global_step += 1            
    optimizer.zero_grad()
    
    G_uy_pred = model(ux,y_loc_train)
    
    loss = torch.mean(torch.abs(G_uy_pred - G_uy_train))   
    loss.backward()
    optimizer.step() 
    
    if not epoch % 1000:
        
        G_uy_test_pred = model(ux_test,y_loc_test)
        loss_test = torch.mean(torch.abs(G_uy_test_pred - G_uy_test)) 
        
        
        print('epoch: {}, loss_train: {:.4f}'.format(epoch, loss)) 
        print('epoch: {}, loss_test: {:.4f}'.format(epoch, loss_test)) 
        

params_mlp = model_mlp.parameters()
optimizer_mlp = optim.Adam(params_mlp, lr= lr)
        
for epoch in range(num_epochs):
    
    global_step += 1            
    optimizer_mlp.zero_grad()
    
    G_uy_pred_mlp = model_mlp(ux)
    
    loss_mlp = torch.mean(torch.abs(G_uy_pred_mlp - G_uy_train))   
    loss_mlp.backward()
    optimizer_mlp.step() 
    
    if not epoch % 1000:
        
        G_uy_test_pred_mlp = model_mlp(ux_test)
        loss_test_mlp = torch.mean(torch.abs(G_uy_test_pred_mlp - G_uy_test)) 
        
        
        print('epoch: {}, loss_train: {:.4f}'.format(epoch, loss_mlp)) 
        print('epoch: {}, loss_test: {:.4f}'.format(epoch, loss_test_mlp))         

torch.save(model,'integrate_model_1.pth')
G_uy_pred = model(ux,y_loc_train)
G_uy_pred_mlp = model_mlp(ux)
plt.figure()
plt.subplot(2,1,1)
plt.plot(x[0,:], ux[1,:])
plt.title("Input Function")
plt.subplot(2,1,2)
plt.plot(x[0,:], G_uy_train[1,:].detach().numpy(),color = "silver", lw = 3, label = "ground truth")
plt.plot(x[0,:], G_uy_pred[1,:].detach().numpy(), "--", color = "blue", label = "prediction (DeepONet)")
plt.plot(x[0,:], G_uy_pred_mlp[1,:].detach().numpy(), ":", color = "black", label = "prediction (MLP)")
plt.legend()
plt.title("Output Function")

# test1_a
w_test = 5
ux_test = torch.cos(w_test*x)
#ux_test.requires_grad = True
G_uy_test = 1/w_test*torch.sin(w_test*y_loc)
y_loc_test = y_loc.permute((1,0))

G_uy_test_pred = model(ux_test,y_loc_test)
#grads = torch.autograd.grad(G_uy_test_pred, ux_test, torch.ones_like(G_uy_test_pred), retain_graph=True)[0]
#print(grads)
G_uy_test_pred_mlp = model_mlp(ux_test)
plt.figure()
plt.subplot(2,1,1)
plt.plot(x[0,:], ux_test[0,:])
plt.title(r"$Input Function: cos(5x)$")
plt.subplot(2,1,2)
plt.plot(x[0,:],G_uy_test[0,:].detach().numpy(), color = "silver", lw = 3, label = "ground truth")
plt.plot(x[0,:],G_uy_test_pred[0,:].detach().numpy(), "--", color = "blue",label = "prediction (DeepONet)")
plt.plot(x[0,:],G_uy_pred_mlp[0,:].detach().numpy(), ":", color = "black", label = "prediction (MLP)")
plt.legend()
plt.title(r"$Output Function: \frac{1}{5}sin(5x)$")
plt.show()

# test2
ux_test = x**2 + 5
G_uy_test = 1/3 * x**3 + 5*x

G_uy_test_pred = model(ux_test,y_loc_test)
G_uy_test_pred_mlp = model_mlp(ux_test)

plt.figure()
plt.subplot(2,1,1)
plt.plot(x[0,:], ux_test[0,:])
plt.title(r"$Input Function: u(x) = x^2 + 5$")
plt.subplot(2,1,2)
plt.plot(y_loc[0,:], G_uy_test[0,:].detach().numpy(), color = "silver", lw = 3, label = "ground truth")
plt.plot(y_loc[0,:], G_uy_test_pred[0,:].detach().numpy(), "--", color = "blue",label = "prediction")
plt.plot(y_loc[0,:], G_uy_pred_mlp[0,:].detach().numpy(), ":", color = "black", label = "prediction (MLP)")
plt.legend()
plt.title(r"$Output Function: G(u)(x) = \frac{1}{3}x^3 + 5x$")
plt.show()        

# test3
ux_test = torch.sin(x) * torch.sin(x) 
G_uy_test = x/2 - 1/4 * torch.sin(2*x)

G_uy_test_pred = model(ux_test,y_loc_test)
G_uy_test_pred_mlp = model_mlp(ux_test)

plt.figure()
plt.subplot(2,1,1)
plt.plot(x[0,:], ux_test[0,:])
plt.title(r"$Input Function: u(x) = sin(x)^2$")
plt.subplot(2,1,2)
plt.plot(y_loc[0,:], G_uy_test[0,:].detach().numpy(), color = "silver", lw = 3, label = "ground truth")
plt.plot(y_loc[0,:], G_uy_test_pred[0,:].detach().numpy(), "--", color = "blue",label = "prediction")
plt.plot(y_loc[0,:], G_uy_pred_mlp[0,:].detach().numpy(), ":", color = "black", label = "prediction (MLP)")
plt.legend()
plt.title(r"$Output Function: G(u)(x) = \frac{x}{2} - \frac{1}{4}sin(2x)$") 
plt.show()




   