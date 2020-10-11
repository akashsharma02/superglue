import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, make_matching_plot
import matplotlib.cm as cm


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):
            for key in data:
                if key != "filename" and key != "image0" and key != "image1":
                    if type(data[key]) == torch.Tensor:
                        data[key] = data[key].to(self.device)
                    else:
                        data[key] = torch.stack(data[key]).to(self.device)

            # data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output['scores'], data['target_matches'])
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, data['target_matches']))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

                image0, image1 = data['image0'].cpu().numpy().squeeze(), data['image1'].cpu().numpy().squeeze()
                kpts0, kpts1 = data['keypoints0'].cpu().numpy().squeeze(), data['keypoints1'].cpu().numpy().squeeze()
                matches, conf = output['matches0'].cpu().detach().numpy(), output['matching_scores0'].cpu().detach().numpy()

                valid = matches > -1
                mkpts0, mkpts1 = kpts0[valid], kpts1[matches[valid]]
                mconf = conf[valid]
                color = cm.jet(mconf)
                stem = data['filename']
                text = []

                plot = make_matching_plot(
                    image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                    text, show_keypoints=True)
                self.writer.add_image(f'{batch_idx}_matches', np.transpose(plot, (2, 0, 1)))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                for key in data:
                    if key != "filename" and key != "image0" and key != "image1":
                        if type(data[key]) == torch.Tensor:
                            data[key] = data[key].to(self.device)
                        else:
                            data[key] = torch.stack(data[key]).to(self.device)

                output = self.model(data)
                loss = self.criterion(output['scores'], data['target_matches'])

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                image0, image1 = data['image0'].cpu().numpy().squeeze(), data['image1'].cpu().numpy().squeeze()
                kpts0, kpts1 = data['keypoints0'].cpu().numpy().squeeze(), data['keypoints1'].cpu().numpy().squeeze()
                matches, conf = output['matches0'].cpu().detach().numpy(), output['matching_scores0'].cpu().detach().numpy()

                valid = matches > -1
                mkpts0, mkpts1 = kpts0[valid], kpts1[matches[valid]]
                mconf = conf[valid]
                color = cm.jet(mconf)
                stem = data['filename']
                text = []

                plot = make_matching_plot(
                    image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                    text, show_keypoints=True)
                self.writer.add_image(f'{batch_idx}_matches', np.transpose(plot, (2, 0, 1)))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)




