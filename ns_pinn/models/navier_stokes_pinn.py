from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
import lightning as L

from models.components.mlp import MLP


class NavierStokes2DPINN(L.LightningModule):
    def __init__(self,
                 layers: list[int],
                 activation: str,
                 dropout: float,
                 learning_rate: float,
                 lr_decay: float,
                 weight_decay: float,
                 data_loss_coef: float = 1,
                 physics_loss_coef: float = 1,
                 rho: float = 1e3,
                 mu: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        match activation:
            case 'hardswish':
                activation_module = nn.Hardswish
            case 'silu':
                activation_module = nn.SiLU
            case 'tanh':
                activation_module = nn.Tanh
            case _:
                # logger.critical(
                #     f'Unsuported activation: {cfg.model.nn.activation}')
                exit()
        self.net = MLP(layers, activation_module, dropout)
        self.optim = torch.optim.Adam(
            self.net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.data_mse = MeanSquaredError()
        self.physics_mse = MeanSquaredError()
        self.val_metrics = MetricCollection({
            'MAE': MeanAbsoluteError(),
            'MAPE': MeanAbsolutePercentageError(),
            'MSE': MeanSquaredError()
        }, prefix='validation/')
        self.data_loss_coef = data_loss_coef
        self.physics_loss_coef = physics_loss_coef
        self.rho = rho
        self.mu = mu
        self.lr_decay = lr_decay

    def configure_optimizers(self):
        lr_scheduler = {
            'scheduler': LambdaLR(self.optim, lr_lambda=lambda epoch: self.lr_decay ** epoch),
            'name': 'train/learning_rate'
        }
        return [self.optim], [lr_scheduler]

    def data_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(prediction, target)

    def physics_loss(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, u: torch.Tensor, v: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_t = grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        p_x = grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        e1 = (self.rho*(u_t + u*u_x + v*u_y) + p_x) - (self.mu*(u_xx + u_yy))
        e2 = (self.rho*(v_t + u*v_x + v*v_y) + p_y) - (self.mu*(v_xx + v_yy))
        e3 = u_x + v_y
        residuals = torch.cat((e1, e2, e3))

        return self.physics_mse(residuals, torch.zeros_like(residuals))

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y, t, u, v = batch
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        data = torch.cat((x, y, t), dim=1)
        targets = torch.cat((u, v), dim=1)
        predictions = self.net(data)
        d_loss = self.data_mse(predictions[:, 0:2].contiguous(), targets)
        u_hat, v_hat, p_hat = predictions.split(1, dim=1)
        p_loss = self.physics_loss(x, y, t, u_hat, v_hat, p_hat)
        final_loss = (d_loss*self.data_loss_coef) + \
            (p_loss*self.physics_loss_coef)
        self.log('train/data_loss', self.data_mse)
        self.log('train/physics_loss', self.physics_mse)
        self.log('train/final_loss', final_loss)
        return final_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y, t, u, v = batch
        data = torch.cat((x, y, t), dim=1)
        targets = torch.cat((u, v), dim=1)
        predictions = self.net(data)
        self.val_metrics.update(predictions[:, 0:2].contiguous(), targets)

    def on_validation_epoch_end(self) -> None:
        output = self.val_metrics.compute()
        self.log_dict(output)
        self.val_metrics.reset()

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y, t, u, v = batch
        data = torch.cat((x, y, t), dim=1)
        return self.net(data)
