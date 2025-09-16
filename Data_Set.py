import torch
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import dataio
import math

EPS = 1e-6


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def compute_derivatives(x, y, t, u):

    dudx = gradient(u, x)
    dudy = gradient(u, y)
    dudt = gradient(u, t)

    dudxx = gradient(dudx, x)
    dudyy = gradient(dudy, y)
    dudtt = gradient(dudt, t)

    dudxxx = gradient(dudxx, x)
    dudxxy = gradient(dudxx, y)
    dudyyy = gradient(dudy, y)

    dudxxxx = gradient(dudxxx, x)
    dudxxyy = gradient(dudxxy, y)
    dudyyyy = gradient(dudyyy, y)

    return dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy, dudtt, dudt


def compute_moments(Dx, Dy, Dxy, nue, dudxx, dudyy):
    mx = - Dx * dudxx - Dxy * dudyy
    my = - Dxy * dudxx - Dy * dudyy

    return mx, my


def min_max_normalization(data, min_val, max_val):
    return 2 * (data - min_val) / (max_val - min_val) - 1


def inv(data):
    return (data + 1) * 10 / 2


class KirchhoffDataset(Dataset):

    def __init__(self, u_val, T, nue, E, H, W, D, load, T0, p_amp, u_in, total_length, den: float, omega: float, batch_size_domain,
                 batch_size_boundary, known_points_x, known_points_y, known_points_t, nkp, device):

        self.u_val = u_val
        self.T = T
        self.nue = nue
        self.E = E
        self.H = H
        self.W = W
        self.num_terms = 7
        self.total_length = total_length
        self.den = den
        self.omega = omega
        self.batch_size_domain = batch_size_domain
        self.batch_size_boundary = batch_size_boundary
        self.known_points_x = known_points_x
        self.known_points_y = known_points_y
        self.known_points_t = known_points_t
        self.nkp = nkp
        self.D = D
        self.p = load
        self.T0 = T0
        self.device = device
        self.count = 0
        self.full_count = 20
        self.u_in = u_in
        self.p_amp = p_amp
        self.max = torch.max(self.u_val(self.known_points_x, self.known_points_y, self.known_points_t))
        self.min = torch.min(self.u_val(self.known_points_x, self.known_points_y, self.known_points_t))


    def __getitem__(self, item):
        x, y, t = self.training_batch()
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        xyt = torch.cat([x, y, t], dim=-1)
        return {'coords': xyt}

    def __len__(self):
        return self.total_length

    def training_batch(self) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:

        x_t = self.known_points_x
        x_in = torch.rand((self.batch_size_domain, 1)) * self.W
        x_b1 = torch.zeros((self.batch_size_boundary, 1))
        x_b2 = torch.zeros((self.batch_size_boundary, 1)) + self.W
        x_b3 = torch.rand((self.batch_size_boundary, 1)) * self.W
        x_b4 = torch.rand((self.batch_size_boundary, 1)) * self.W

        x = torch.cat([x_t, x_in, x_b1, x_b2, x_b3, x_b4], dim=0)  # .to(self.device)

        y_t = self.known_points_y
        y_in = torch.rand((self.batch_size_domain, 1)) * self.H
        y_b1 = torch.rand((self.batch_size_boundary, 1)) * self.H
        y_b2 = torch.rand((self.batch_size_boundary, 1)) * self.H
        y_b3 = torch.zeros((self.batch_size_boundary, 1))
        y_b4 = torch.zeros((self.batch_size_boundary, 1)) + self.H
        y = torch.cat([y_t, y_in, y_b1, y_b2, y_b3, y_b4], dim=0)  # .to(self.device)

        t_t = self.known_points_t
        time = torch.zeros(self.batch_size_domain + 2*self.batch_size_boundary, 1).uniform_(0, 2 * (self.count / self.full_count))
        t = torch.tensor(time, dtype=torch.float32).view(-1,1)
        t_in = torch.zeros((2 * self.batch_size_boundary, 1))

        t = torch.cat([t_t, t, t_in], dim=0)

        #x = min_max_normalization(x, 0, self.W)
        #y = min_max_normalization(y, 0, self.H)

        x = x.to(self.device)  # CUDA
        y = y.to(self.device)
        t = t.to(self.device)

        if self.count == self.full_count:
            self.count = 1
        else:
            self.count += 1
        return x, y, t

    def validation_batch(self, snap, grid_width=32, grid_height=32):
        x, y = np.mgrid[0:self.W:complex(0, grid_width), 0:self.H:complex(0, grid_height)]
        x = torch.tensor(x.reshape(grid_width * grid_height, 1), dtype=torch.float32)  # .to(self.device)
        y = torch.tensor(y.reshape(grid_width * grid_height, 1), dtype=torch.float32)  # .to(self.device)
        t = torch.ones((grid_width * grid_height, 1), dtype=torch.float32)
        t.fill_(snap)

        x = x[None, ...]
        y = y[None, ...]
        t = t[None, ...]
        x = x.to(self.device)  # CUDA
        y = y.to(self.device)
        t = t.to(self.device)

        # Assuming self.u_val is a PyTorch function
        u = self.u_val(x, y, t)
        return x, y, t, u

    def compute_loss(self, x, y, t, preds, eval=False):
        # governing equation loss
        preds = np.squeeze(preds, axis=0)
        x = np.squeeze(x, axis=0)
        y = np.squeeze(y, axis=0)
        t = np.squeeze(t, axis=0)
        x_t = x[:self.nkp]
        y_t = y[:self.nkp]
        t_t = t[:self.nkp]
        u_t = np.squeeze(preds[:self.nkp, 0:1])
        u = np.squeeze(preds[:, 0:1])
        dudxx = np.squeeze(preds[:, 1:2])
        dudyy = np.squeeze(preds[:, 2:3])
        dudxxxx = np.squeeze(preds[:, 3:4])
        dudyyyy = np.squeeze(preds[:, 4:5])
        dudxxyy = np.squeeze(preds[:, 5:6])
        dudtt = np.squeeze(preds[:, 6:7])
        dudt = np.squeeze(preds[:, 7:8])
        err_t = self.u_val(x_t, y_t, t_t) - u_t
        f = self.D[0, 0] * dudxxxx + 2 * (self.D[0, 1] + 2 * self.D[2, 2]) * dudxxyy + self.D[1, 1] * dudyyyy - self.den * self.T * dudtt - self.p(x, y, t)
        #++, --, -+, +-
        L_f = f ** 2
        L_t = 0*err_t ** 2
        # if a point is on either of the boundaries, its value is 1 and 0 otherwise
        x_lower = torch.where(x <= 0 + EPS, torch.tensor(1.0, device=self.device),
                              torch.tensor(0.0, device=self.device))  # CUDA
        x_upper = torch.where(x >= self.W - EPS, torch.tensor(1.0, device=self.device),
                              torch.tensor(0.0, device=self.device))
        y_lower = torch.where(y <= 0 + EPS, torch.tensor(1.0, device=self.device),
                              torch.tensor(0.0, device=self.device))
        y_upper = torch.where(y >= self.H - EPS, torch.tensor(1.0, device=self.device),
                              torch.tensor(0.0, device=self.device))

        L_b0 = torch.mul((x_lower + x_upper + y_lower + y_upper), u) ** 2
        # compute 2nd order boundary condition loss
        mx, my = compute_moments(self.D[0, 0], self.D[1, 1], self.D[0, 1], self.nue, dudxx, dudyy)
        L_b2 = torch.mul((x_lower + x_upper), mx) ** 2 + torch.mul((y_lower + y_upper), my) ** 2


        center_low_x = torch.where(x >= self.W//2 - EPS, torch.tensor(1.0, device=self.device),
                              torch.tensor(0.0, device=self.device))
        center_low_y = torch.where(y >= self.H // 2 - EPS, torch.tensor(1.0, device=self.device),
                                   torch.tensor(0.0, device=self.device))
        center_up_x= torch.where(x <= self.W // 2 + EPS, torch.tensor(1.0, device=self.device),
                                   torch.tensor(0.0, device=self.device))
        center_up_y = torch.where(y <= self.H // 2 + EPS, torch.tensor(1.0, device=self.device),
                                   torch.tensor(0.0, device=self.device))
        center = center_low_x + center_low_y + center_up_x + center_up_y

        L_c = 0*(torch.mul(center, u) - torch.mul(center, self.u_val(x,y,t)))**2
        t_in = torch.where(t <= EPS, t,
                              torch.tensor(0.0, device=self.device))
        t_in_uno = torch.where(t <= EPS, torch.tensor(1.0, device=self.device),
                              torch.tensor(0.0, device=self.device))

        L_t0 = (torch.mul(t_in_uno, u) - torch.mul(t_in_uno, self.u_val(x,y,t)))**2
        L_t1 = torch.mul(t_in_uno, dudt) ** 2
        if eval:
            L_u = (self.u_val(x, y, t) - u) #** 2

            return L_f, L_b0, L_b2, L_u, L_t, L_t0, L_t1, L_c
        return L_f, L_b0, L_b2, L_t, L_t0, L_t1, L_c

    def __validation_results(self, pinn, snap, image_width=32, image_height=32):
        x, y, t, u_real = self.validation_batch(snap, image_width, image_height)
        x = torch.tensor(x.clone().detach().reshape(image_width * image_height, 1), dtype=torch.float32)
        y = torch.tensor(y.clone().detach().reshape(image_width * image_height, 1), dtype=torch.float32)
        t = torch.tensor(t.clone().detach().reshape(image_width * image_height, 1), dtype=torch.float32)

        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        t = t.unsqueeze(1)
        x = x.to(self.device)  # CUDA
        y = y.to(self.device)
        t = t.to(self.device)

        c = {'coords': torch.cat([x, y, t], dim=-1).float()}
        pred = pinn(c, eval=True)['model_out']
        u_pred, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy, dudtt, dudt = (
            pred[:, :, 0:1], pred[:, :, 1:2], pred[:, :, 2:3], pred[:, :, 3:4], pred[:, :, 4:5], pred[:, :, 5:6], pred[:, :, 6:7], pred[:, :, 7:]
        )
        mx, my = compute_moments(self.D[0,0], self.D[1,1], self.D[0,1], self.nue, dudxx, dudyy)
        f = self.D[0, 0] * dudxxxx + 2 * (self.D[0, 1] + 2 * self.D[2, 2]) * dudxxyy + self.D[1, 1] * dudyyyy + self.den * dudtt
        p = self.p(x, y, t)
        return u_real, u_pred, mx, my, f, p, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy, dudtt, dudt

    def visualise(self, pinn=None, snap = 0, image_width=32, image_height=32):
        if pinn is None:
            x, y, t, u_real = self.validation_batch(image_width, image_height)
            l = self.p(x,y,t).cpu.detach().numpy().reshape(image_height, image_width)
            fig, axs = plt.subplots(1, 1, figsize=(8, 3.2), dpi=100)
            self.__show_image(
                l, #u_real.cpu.reshape(image_width, image_height),
                title='Deformation',
                z_label='[m]'
            )
            plt.tight_layout()
            plt.show()

        else:
            u_real, u_pred, mx, my, f, p, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy, dudtt, dudt = self.__validation_results(pinn, snap, image_width, image_height)

            NMSE = (np.linalg.norm(u_real.cpu().detach().numpy() - u_pred.cpu().detach().numpy()) ** 2) / (np.linalg.norm(u_real.cpu().detach().numpy()) ** 2)
            print('NMSE:', NMSE)

            u_real = u_real.cpu().detach().numpy().reshape(image_width, image_height)  # CUDA
            u_pred = u_pred.cpu().detach().numpy().reshape(image_width, image_height)  # CUDA
            self.__plot_3d(u_real, 'Real Displacement (m)')

            # Plot 3D for u_pred
            self.__plot_3d(u_pred, 'Predicted Displacement (m)')

            f = f.cpu().detach().numpy().reshape(image_width, image_height)  # CUDA
            p = p.cpu().detach().numpy().reshape(image_width, image_height)

            #self.__plot_3d(p, 'Load')

            fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
            self.__show_image(u_pred, axs[0], 'Predicted Displacement (m)')
            self.__show_image(u_real, axs[1], 'Real Displacement (m)')

            fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
            self.__show_image(f, axs[0], 'f')
            self.__show_image(p, axs[1], 'p')

            dudtt = dudtt.cpu().detach().numpy().reshape(image_width, image_height)
            dudxxxx = dudxxxx.cpu().detach().numpy().reshape(image_width, image_height)

            fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
            self.__show_image(dudxxxx, axs[0], 'dudxxxx')
            self.__show_image(dudtt, axs[1], 'dudtt')

            dudyyyy = dudyyyy.cpu().detach().numpy().reshape(image_width, image_height)
            dudxxyy = dudxxyy.cpu().detach().numpy().reshape(image_width, image_height)

            fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
            self.__show_image(dudxxyy, axs[0], 'dudxxyy')
            self.__show_image(dudyyyy, axs[1], 'dudyyyy')

            dudt = dudt.cpu().detach().numpy().reshape(image_width, image_height)

            fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
            self.__show_image(dudt, axs[0], 'dudt')

            for ax in axs.flat:
                ax.label_outer()

            plt.tight_layout()
            plt.show()

    def __show_image(self, img, axis=None, title='', x_label='x [m]', y_label='y [m]', z_label=''):
        if axis is None:
            _, axis = plt.subplots(1, 1, figsize=(4,2), dpi=100)
        if title == 'Predicted Displacement (m)' or 'Real Displacement (m)':
            im = axis.imshow(np.rot90(img, k=3), cmap='plasma', origin='lower', aspect='auto')#, vmax=self.max, vmin=self.min)
        if title == 'Squared Error Displacement':
            im = axis.imshow(np.rot90(img, k=3), cmap='plasma', origin='lower', aspect='auto')#, vmax=self.max/100, vmin=self.min/100)
        else:
            im = axis.imshow(np.rot90(img, k=3), cmap='plasma', origin='lower', aspect='auto')
        cb = plt.colorbar(im, label=z_label, ax=axis)
        #axis.set_aspect(0.5)
        axis.set_xticks([0, img.shape[0] - 1])
        axis.set_xticklabels([0, self.W])
        axis.set_yticks([0, img.shape[1] - 1])
        axis.set_yticklabels([0, self.H])
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(title)
        return im

    def __plot_3d(self, data, title=''):

        X, Y = np.mgrid[0:self.W:complex(0, 32), 0:self.H:complex(0, 32)]
        Z = data

        z_size = self.p_amp

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        #ax.set_box_aspect([2, 1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.invert_xaxis()
        #ax.set_zlim(-z_size, z_size)
        plt.show()



    def vis(self, pinn):
        sl = 256

        with torch.no_grad():
            frames = [0.0, 0.5, 0.75, 1]
            coords = [dataio.get_mgrid((1, sl, sl), dim=3)[None, ...].cuda() for f in frames]
            for idx, f in enumerate(frames):
                coords[idx][..., 0] = f
            coords = torch.cat(coords, dim=0)

            Nslice = 10
            output = torch.zeros(coords.shape[0], coords.shape[1], 1)
            split = int(coords.shape[1] / Nslice)
            for i in range(Nslice):
                pred = pinn({'coords': coords[:, i * split:(i + 1) * split, :]}, eval=True)['model_out']
                output[:, i * split:(i + 1) * split, :] = pred.cpu()

        pred = output.view(len(frames), 1, sl, sl)

        fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
        self.__show_image(pred[0,0,:,:], axs[0], '0 frame')
        self.__show_image(pred[1,0,:,:], axs[1], '1 frame')

        fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
        self.__show_image(pred[2, 0, :, :], axs[0], '2 frame')
        self.__show_image(pred[3, 0, :, :], axs[1], '3 frame')
