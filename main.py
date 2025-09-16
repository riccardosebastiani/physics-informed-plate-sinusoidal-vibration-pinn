from torch.utils.data import DataLoader
import Data_Set as dataSet
import torch
import math
import numpy as np
import WAlg as loss
import modules
import training
import matplotlib.pyplot as plt
from dataio import get_mgrid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # TODO
print('device:', device)

num_epochs = 200
n_step = 50
batch_size = 32
lr = 0.001
batch_size_domain = 300
batch_size_boundary = 50

steps_til_summary = 10
opt_model = 'sine'
mode = 'pinn'
clip_grad = 1.0
use_lbfgs = False
relo = True
total_length = 1
max_epochs_without_improvement = 100

ntot = 1320

W = 10
H = 10
T = 0.2
E = 210000
nue = 0.35
p0 = 0.15
den = 405

paf = 1e9
C = torch.tensor([
    [10.8, 0, 0],
    [0, 0.8424, 0],
    [0, 0, 0.697]
])

ni = torch.tensor([0.372, 0.04, 0])
fac = T**3/(12*(1-ni[0]*ni[1]))
gac = (1/12) * T**3
D = [[0 for _ in range(3)] for _ in range(3)]
D[0][0] = fac * C[0][0] * paf
D[1][1] = fac * C[1][1] * paf
D[1][0] = fac * C[1][1] * ni[0] * paf
D[0][1] = fac * C[1][1] * ni[0] * paf
D[2][2] = gac * C[2][2] * paf
D = torch.tensor(D)
print('D:', D)
n = 4
m = 4

a = n * np.pi / W
b = m * np.pi / H

percentage_of_known_points = 15 #%
T0 = 1
omega = 2*np.pi

nkp = percentage_of_known_points * batch_size_domain // 100
known_points_x = torch.rand((nkp, 1)) * W
known_points_y = torch.rand((nkp, 1)) * H
known_points_t = torch.rand((nkp, 1))

mul_fac = 10e9

denom = D[0,0] * a**4 + 2*(D[0,1] + 2*D[2,2]) * a**2 * b**2 + D[1,1] * b**4 - 4 * np.pi**2 * den * T
def load(x,y,t):
    return p0 * torch.sin(x * a) * torch.sin(y * b) * torch.sin(t * omega) * denom
    
def u_val(x, y, t):
    p_amp = p0
    return p_amp * torch.sin(x * a) * torch.sin(y * b) * torch.sin(t * omega)


denom = D[0,0] * a**4 + 2*(D[0,1] + 2*D[2,2]) * a**2 * b**2 + D[1,1] * b**4 - 4 * np.pi**2 * den * T
p_amp = p0 + 0.02

def u_in(x,y):
    return 1

plate = dataSet.KirchhoffDataset(u_val=u_val, T=T, nue=nue, E=E, W=W, H=H, D=D, p_amp=p_amp, load=load, T0=T0, u_in=u_in, total_length=total_length, den=den,
                                 omega=omega, batch_size_domain=batch_size_domain, batch_size_boundary=
                                 batch_size_boundary, known_points_x=known_points_x, known_points_y=known_points_y, known_points_t=known_points_t,
                                 nkp=nkp, device=device)
data_loader = DataLoader(plate, shuffle=True, batch_size=batch_size, pin_memory=False, num_workers=0)
model = modules.PINNet(out_features=1, type=opt_model, mode=mode)
model.to(device)

history_loss = {'L_f': [], 'L_b0': [], 'L_b2': [], 'L_u': [], 'L_t': []}
if not relo:
    loss_fn = loss.KirchhoffLoss(plate)
    kirchhoff_metric = loss.KirchhoffMetric(plate)
    history_lambda = None
    metric_lam = None
else:
    loss_fn = loss.ReLoBRaLoKirchhoffLoss(plate, temperature=0.00001, rho=0.9, alpha=0.9)
    kirchhoff_metric = loss.KirchhoffMetric(plate)
    history_lambda = {'L_f_lambda': [], 'L_b0_lambda': [], 'L_b2_lambda': [], 'L_t_lambda': []}
    metric_lam = loss.ReLoBRaLoLambdaMetric(loss_fn)

training.train(model=model, train_dataloader=data_loader, epochs=num_epochs, n_step=n_step, lr=lr,
               steps_til_summary=steps_til_summary, loss_fn=loss_fn, history_loss=history_loss,
               history_lambda=history_lambda,
               metric=kirchhoff_metric, metric_lam=metric_lam, clip_grad=clip_grad,
               use_lbfgs=False, max_epochs_without_improvement=max_epochs_without_improvement)
model.eval()
snaps = [0.08, 0.1, 0.5]

plate.visualise(model, snaps[0])
plate.visualise(model, snaps[1])
plate.visualise(model, snaps[2])

fig = plt.figure(figsize=(6, 4.5), dpi=100)
plt.plot(torch.log(torch.tensor(history_loss['L_f'])), label='$L_f$ governing equation')
plt.plot(torch.log(torch.tensor(history_loss['L_b0'])), label='$L_{b0}$ Dirichlet boundaries')
plt.plot(torch.log(torch.tensor(history_loss['L_b2'])), label='$L_{b2}$ Moment boundaries')
plt.plot(torch.log(torch.tensor(history_loss['L_t'])), label='$L_t$ Known points')
plt.plot(torch.log(torch.tensor(history_loss['L_u'])), label='$L_u$ analytical solution')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Log-loss')
plt.title('Loss evolution Kirchhoff PDE')
plt.savefig('kirchhoff_loss_unscaled')
plt.show()

if metric_lam is not None:
    fig2 = plt.figure(figsize=(6, 4.5), dpi=100)
    plt.plot(history_lambda['L_f_lambda'], label='$\lambda_f$ governing equation')
    plt.plot(history_lambda['L_b0_lambda'], label='$\lambda_{b0}$ Dirichlet boundaries')
    plt.plot(history_lambda['L_b2_lambda'], label='$\lambda_{b2}$ Moment boundaries')
    plt.plot(history_lambda['L_t_lambda'], label='$\lambda_{t}$ Known points')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('scalings lambda')  # $\lambda$')
    plt.title('ReLoBRaLo weights on Kirchhoff PDE')
    plt.savefig('kirchhoff_lambdas_relobralo')
    plt.show()
