import torch


class MSESymmetricJointLoss(torch.nn.MSELoss):
    def forward(self, input, target):
        mse_loss = super(MSESymmetricJointLoss, self).forward(input, target)
        symmetric_loss = torch.mean(torch.abs(torch.mean(input, dim=1)))
        return mse_loss + symmetric_loss


class MSESymmetricMuscleLoss(torch.nn.MSELoss):
    def forward(self, input, target):
        mse_loss = super(MSESymmetricMuscleLoss, self).forward(input, target)
        dorsal = torch.relu(input[:, :, 0:48])  # clip action
        ventral = torch.relu(input[:, :, 48:96])  # clip action
        symmetric_loss = torch.mean(torch.abs(torch.mean(dorsal - ventral, dim=1)))
        return mse_loss + symmetric_loss
