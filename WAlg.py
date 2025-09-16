from Data_Set import KirchhoffDataset
import torch
import torch.nn as nn


class CustomVariable:
    def __init__(self, initial_value, trainable=True, dtype=torch.float32):
        self.data = torch.nn.Parameter(torch.tensor(initial_value, dtype=dtype), requires_grad=trainable)

    def assign(self, new_value):
        self.data.data = torch.tensor(new_value, dtype=self.data.dtype)


class KirchhoffLoss(torch.nn.Module):
    def __init__(self, plate: KirchhoffDataset):
        super(KirchhoffLoss, self).__init__()
        self.plate = plate

    def call(self, preds, xy):
        xy = xy['model_coords']
        x, y, t = xy[:, :, 0], xy[:, :, 1], xy[:, :, 2]
        preds = preds['model_out']
        L_f, L_b0, L_b2, L_t, L_t0, L_t1,L_c = self.plate.compute_loss(x, y, t, preds)
        return {'L_f': L_f, 'L_b0': L_b0, 'L_b2': L_b2, 'L_t': L_t, 'L_t0': L_t0, 'L_t1': L_t1, 'L_c': L_c}


class ReLoBRaLoKirchhoffLoss(KirchhoffLoss):

    def __init__(self, plate: KirchhoffDataset, alpha: float = 0.999, temperature: float = 1., rho: float = 0.9999):
        super().__init__(plate)
        self.plate = plate
        self.alpha = torch.tensor(alpha)
        self.temperature = temperature
        self.rho = rho
        self.call_count = CustomVariable(0., trainable=False, dtype=torch.float32)

        self.lambdas = [CustomVariable(1., trainable=False) for _ in range(plate.num_terms)]
        self.last_losses = [CustomVariable(1., trainable=False) for _ in range(plate.num_terms)]
        self.init_losses = [CustomVariable(1., trainable=False) for _ in range(plate.num_terms)]

    def call(self, preds, xy):
        xy = xy['coords']
        x, y, t = xy[:, :, 0], xy[:, :, 1], xy[:, :, 2]
        preds = preds['model_out']
        EPS = 1e-7

        losses = [torch.mean(loss) for loss in self.plate.compute_loss(x, y, t, preds)]

        cond1 = torch.tensor(self.call_count.data.item() == 0, dtype=torch.bool)
        cond2 = torch.tensor(self.call_count.data.item() == 1, dtype=torch.bool)

        alpha = torch.where(cond1, torch.tensor(1.0),
                            torch.where(cond2, torch.tensor(0.0),
                                        self.alpha))
        cond3 = torch.rand(1).item() < self.rho
        rho = torch.where(cond1, torch.tensor(1.0),
                          torch.where(cond2, torch.tensor(1.0),
                                      torch.tensor(cond3, dtype=torch.float32)))

        # Calcola nuove lambdas w.r.t. le losses nella precedente iterazione
        lambdas_hat = [losses[i].item() / (self.last_losses[i].data.item() * self.temperature + EPS)
                       for i in range(len(losses))]

        lambdas_hat = torch.tensor(lambdas_hat)
        lambdas_hat = (torch.nn.functional.softmax(lambdas_hat - torch.max(lambdas_hat), dim=-1)
                       * torch.tensor(len(losses), dtype=torch.float32))

        init_lambdas_hat = [losses[i].item() / (self.init_losses[i].data.item() * self.temperature + EPS)
                            for i in range(len(losses))]

        init_lambdas_hat = torch.tensor(init_lambdas_hat)
        init_lambdas_hat = (torch.nn.functional.softmax(init_lambdas_hat - torch.max(init_lambdas_hat), dim=-1)
                            * torch.tensor(len(losses), dtype=torch.float32))

        new_lambdas = [
            (rho * alpha * self.lambdas[i].data + (1 - rho) * alpha * init_lambdas_hat[i] + (1 - alpha)
             * lambdas_hat[i]) for i in range(len(losses))]
        self.lambdas = [var.detach().requires_grad_(False) for var in new_lambdas]

        l = [lam * loss for lam, loss in zip(self.lambdas, losses)]
        self.last_losses = [loss.clone().detach() for loss in losses]

        first_iteration = torch.tensor(self.call_count.data.item() < 1, dtype=torch.float32)

        for i, (var, loss) in enumerate(zip(self.init_losses, losses)):
            self.init_losses[i].data = (loss.data * first_iteration + var.data * (1 - first_iteration)).detach()
        self.call_count.data += 1
        return {'L_f': l[0], 'L_b0': l[1], 'L_b2': l[2], 'L_t': l[3], 'L_t0': l[4], 'L_t1': l[5], 'L_c': l[6]}


class KirchhoffMetric(nn.Module):
    def __init__(self, plate, name='kirchhoff_metric'):
        super(KirchhoffMetric, self).__init__()
        self.plate = plate
        self.L_f_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.L_b0_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.L_b2_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.L_t_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.L_t0_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.L_t1_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.L_c_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.L_u_mean = nn.Parameter(torch.zeros(1), requires_grad=False)

    def update_state(self, xy, y_pred, losses=None, sample_weight=None):
        xy = xy['coords']
        y_pred = y_pred['model_out']
        x, y, t = xy[:, :, 0], xy[:, :, 1], xy[:, :, 2]

        L_f, L_b0, L_b2, L_u, L_t , L_t0, L_t1, L_c = self.plate.compute_loss(x, y, t, y_pred, eval=True)
        self.L_f_mean.data = torch.mean(L_f)
        self.L_b0_mean.data = torch.mean(L_b0)
        self.L_b2_mean.data = torch.mean(L_b2)
        self.L_u_mean.data = torch.mean(L_u)
        self.L_t_mean.data = torch.mean(L_t)
        self.L_t1_mean.data = torch.mean(L_t1)
        self.L_t0_mean.data = torch.mean(L_t0)
        self.L_c_mean.data = torch.mean(L_c)

    def reset_state(self):
        self.L_f_mean.data = torch.zeros(1)
        self.L_b0_mean.data = torch.zeros(1)
        self.L_b2_mean.data = torch.zeros(1)
        self.L_u_mean.data = torch.zeros(1)
        self.L_t_mean.data = torch.zeros(1)
        self.L_t0_mean.data = torch.zeros(1)
        self.L_t1_mean.data = torch.zeros(1)
        self.L_c_mean.data = torch.zeros(1)

    def result(self):
        return {'L_f': self.L_f_mean.item(),
                'L_b0': self.L_b0_mean.item(),
                'L_b2': self.L_b2_mean.item(),
                'L_u': self.L_u_mean.item(),
                'L_t': self.L_t_mean.item(),
                'L_t0': self.L_t0_mean.item(),
                'L_t1': self.L_t1_mean.item(),
                'L_c': self.L_c_mean.item()}


class ReLoBRaLoLambdaMetric(nn.Module):
    def __init__(self, loss, name='relobralo_lambda_metric'):
        super(ReLoBRaLoLambdaMetric, self).__init__()
        self.loss = loss
        self.L_f_lambda_mean = CustomVariable(0.0, trainable=False)
        self.L_b0_lambda_mean = CustomVariable(0.0, trainable=False)
        self.L_b2_lambda_mean = CustomVariable(0.0, trainable=False)
        self.L_t_lambda_mean = CustomVariable(0.0, trainable=False)
        self.L_t0_lambda_mean = CustomVariable(0.0, trainable=False)
        self.L_t1_lambda_mean = CustomVariable(0.0, trainable=False)
        self.L_c_lambda_mean = CustomVariable(0.0, trainable=False)

    def update_state(self, xy, y_pred, sample_weight=None):
        L_f_lambda, L_b0_lambda, L_b2_lambda, L_t_lambda, L_t0_lambda, L_t1_lambda, L_c_lambda = self.loss.lambdas
        self.L_f_lambda_mean.assign(L_f_lambda.data.data.item())
        self.L_b0_lambda_mean.assign(L_b0_lambda.data.data.item())
        self.L_b2_lambda_mean.assign(L_b2_lambda.data.item())
        self.L_t_lambda_mean.assign(L_t_lambda.data.item())
        self.L_t0_lambda_mean.assign(L_t0_lambda.data.item())
        self.L_t1_lambda_mean.assign(L_t1_lambda.data.item())
        self.L_c_lambda_mean.assign(L_c_lambda.data.item())

    def reset_state(self):
        self.L_f_lambda_mean.assign(0.0)
        self.L_b0_lambda_mean.assign(0.0)
        self.L_b2_lambda_mean.assign(0.0)
        self.L_t_lambda_mean.assign(0.0)
        self.L_t0_lambda_mean.assign(0.0)
        self.L_t1_lambda_mean.assign(0.0)
        self.L_c_lambda_mean.assign(0.0)


    def result(self):
        return {'L_f': self.L_f_lambda_mean.data.data,
                'L_b0': self.L_b0_lambda_mean.data.data,
                'L_b2': self.L_b2_lambda_mean.data.data,
                'L_t': self.L_t_lambda_mean.data.data,
                'L_t0': self.L_t0_lambda_mean.data.data,
                'L_t1': self.L_t1_lambda_mean.data.data,
                'L_c': self.L_c_lambda_mean.data.data}
