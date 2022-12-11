from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torchmetrics import MetricCollection
import pytorch_lightning as pl


class NavierStokes2DPINN(pl.LightningModule):
    def __init__(self,
                 net: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 val_metrics: MetricCollection,
                 lr_scheduler: Union[torch.optim.lr_scheduler._LRScheduler, None] = None,
                 data_loss_coef: float = 1,
                 physics_loss_coef: float = 1,
                 rho: float = 1e3,
                 mu: float = 1e-3) -> None:
        super().__init__()
        self.net = net
        self.optim = optimizer
        self.scheduler = lr_scheduler
        self.val_metrics = val_metrics
        self.data_loss_coef = data_loss_coef
        self.physics_loss_coef = physics_loss_coef
        self.rho = rho
        self.mu = mu

    def configure_optimizers(self):
        if not self.scheduler:
            return self.optim
        return {'optimizer': self.optim, 'lr_scheduler': self.scheduler}

    def data_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(prediction, target)

    def physics_loss(self, prediction: torch.Tensor, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, rho: float, mu: float) -> torch.Tensor:
        u, v, p = prediction.split(1, dim=1)

        u_x = grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        u_t = grad(u, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]

        v_x = grad(v, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        v_y = grad(v, y, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        v_t = grad(v, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]

        p_x = grad(p, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        p_y = grad(p, y, grad_outputs=torch.ones_like(x), create_graph=True)[0]

        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(x),
                    create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(x),
                    create_graph=True)[0]

        v_xx = grad(v_x, x, grad_outputs=torch.ones_like(x),
                    create_graph=True)[0]
        v_yy = grad(v_y, y, grad_outputs=torch.ones_like(x),
                    create_graph=True)[0]

        e1 = (rho*(u_t + u*u_x + v*u_y) + p_x) - (mu*(u_xx + u_yy))
        e2 = (rho*(v_t + u*v_x + v*v_y) + p_y) - (mu*(v_xx + v_yy))
        e3 = u_x + v_y
        residuals = torch.cat((e1, e2, e3))

        return F.mse_loss(residuals, torch.zeros_like(residuals))

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        sample, labels = batch
        sample.requires_grad = True
        sample = sample.type(torch.float32)
        labels = labels.type(torch.float32)
        # Split needed for calculation of the physics loss without breaking torch graph
        x, y, t = sample.split(1, dim=1)
        sample = torch.cat((x, y, t), dim=1)
        predictions = self.net(sample)
        d_loss = self.data_loss(predictions[:, 0:2], labels[:, 0:2])
        p_loss = self.physics_loss(predictions, x, y, t, self.rho, self.mu)
        final_loss = (d_loss*self.data_loss_coef) + \
            (p_loss*self.physics_loss_coef)
        self.log('training/data_loss', d_loss)
        self.log('training/physics_loss', p_loss)
        self.log('training/final_loss', final_loss)
        return final_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        sample, labels = batch
        sample = sample.type(torch.float32)
        labels = labels.type(torch.float32)
        predictions = self.net(sample)
        self.val_metrics.update(predictions[:, 0:2], labels)

    def validation_epoch_end(self, outputs) -> None:
        output = self.val_metrics.compute()
        self.log_dict(output)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.net.forward(batch)
