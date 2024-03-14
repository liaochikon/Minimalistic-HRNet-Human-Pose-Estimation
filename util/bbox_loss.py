import torch.nn as nn

class IOULoss(nn.Module):
    def __init__(self, reduction = 'mean'):
        super(IOULoss, self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)

    def forward(self, output, target):
        batch_size = output.size(0)
        cat_num = output.size(1)
        heatmaps_pred = output.reshape((batch_size, cat_num, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, cat_num, -1)).split(1, 1)
        loss = 0

        for idx in range(cat_num):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss
    