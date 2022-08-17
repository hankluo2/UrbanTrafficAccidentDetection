import logging
from pathlib import Path
import os
import time
from collections import OrderedDict
import numpy as np
from sklearn.metrics import auc, roc_curve

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
"""
Written by Eitan Kosman
"""


def update_pedict(buffer_data: np.ndarray, pred_clip_score: np.ndarray, center: int,
                  frames_per_clip: int) -> np.ndarray:
    half_len = pred_clip_score.shape[0] // 2 * frames_per_clip
    if center - half_len < 0:  # here 32 = pred_clip_score.shape[0] / 2 * frames_per_clip
        start = 0
    elif center + half_len >= buffer_data.shape[0]:
        start = buffer_data.shape[0] - 2 * half_len - 1
    else:
        start = center - half_len

    cur = start
    for pred in pred_clip_score:
        # print(cur, pred)
        # buffer_data[cur:cur + frames_per_clip] = np.array([pred]).repeat(frames_per_clip)
        buffer_data[cur:cur + frames_per_clip] = pred

        # for idx in range(cur, cur + frames_per_clip):
        #     buffer_data[idx] = pred
        # print(buffer_data)
        cur += frames_per_clip

    return buffer_data


def get_torch_device():
    """
    Retrieves the device to run torch models, with preferability to GPU (denoted as cuda by torch)
    Returns: Device to run the models
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path, model_prototype, is_pred):
    """
    Loads a Pytorch model
    Args:
        model_path: path to the model to load

    Returns: a model loaded from the specified path

    """
    logging.info(f"Load the model from: {model_path}")
    # model_dict = torch.load(model_path, map_location='cpu')
    model_dict = torch.load(model_path)
    try:
        model_prototype.load_state_dict(model_dict)
        logging.info(model_prototype)
        return model_prototype
    except:
        # TODO: this is a patch to fix a bug.
        if is_pred:
            new_model_dict = OrderedDict([(k.replace("model.", ""), v) for (k, v) in model_dict.items()])
            model_prototype.load_state_dict(new_model_dict)
            # model_prototype.load_state_dict(model_dict)
        else:
            new_model_dict = OrderedDict([("model." + k, v) for (k, v) in model_dict.items()])
            model_prototype.load_state_dict(new_model_dict)

        logging.info(model_prototype)
        return model_prototype


def get_loader_shape(loader):
    assert len(loader) != 0
    return loader[0].shape


class TorchModel(nn.Module):
    """
    Wrapper class for a torch model to make it comfortable to train and load models
    """
    def __init__(self, model):
        super(TorchModel, self).__init__()
        self.device = get_torch_device()
        self.iteration = 0
        self.model = model
        self.is_data_parallel = False
        self.callbacks = []

    def register_callback(self, callback_fn):
        """
        Register a callback to be called after each evaluation run
        Args:
            callback_fn: a callable that accepts 2 inputs (output, target)
                            - output is the model's output
                            - target is the values of the target variable
        """
        self.callbacks.append(callback_fn)

    def data_parallel(self):
        """
        Transfers the model to data parallel mode
        """
        self.is_data_parallel = True
        if not isinstance(self.model, torch.nn.DataParallel):
            # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
            self.model = torch.nn.DataParallel(self.model, device_ids=[0])

        return self

    @classmethod
    def load_model(cls, model_path, model_prototype, is_pred=False):
        """
        Loads a pickled model
        Args:
            model_path: path to the pickled model

        Returns: TorchModel class instance wrapping the provided model
        """
        return cls(load_model(model_path, model_prototype, is_pred))

    def notify_callbacks(self, notification, *args, **kwargs):
        for callback in self.callbacks:
            try:
                method = getattr(callback, notification)
                method(*args, **kwargs)
            except (AttributeError, TypeError) as e:
                logging.error(
                    f"callback {callback.__class__.__name__} doesn't fully implement the required interface {e}"
                )

    def fit(
        self,
        train_iter,
        criterion,
        optimizer,
        eval_iter=None,
        epochs=10,
        network_model_path_base=None,
        save_every=None,
        evaluate_every=None,
        args=None,
    ):
        """

        Args:
            train_iter: iterator for training
            criterion: loss function
            optimizer: optimizer for the algorithm
            eval_iter: iterator for evaluation
            epochs: amount of epochs
            network_model_path_base: where to save the models
            save_every: saving model checkpoints every specified amount of epochs
            evaluate_every: perform evaluation every specified amount of epochs. If the evaluation is expensive,
                            you probably want ot choose a high value for this
        """
        criterion = criterion.to(self.device)
        self.notify_callbacks('on_training_start', epochs)

        last_auc = 0
        for epoch in range(epochs):
            train_loss = self.do_epoch(criterion=criterion,
                                       optimizer=optimizer,
                                       data_iter=train_iter,
                                       epoch=epoch)

            if save_every and network_model_path_base and epoch % save_every == 0:
                logging.info(f"Save the model after epoch {epoch}")
                self.save(os.path.join(network_model_path_base, f'epoch_{epoch}.pt'))

            val_loss = None
            if eval_iter and evaluate_every and epoch % evaluate_every == 0:
                logging.info(f"Evaluating after epoch {epoch}")

                # val_loss = self.evaluate(
                #     criterion=criterion,
                #     data_iter=eval_iter,
                # )
                # ==========================================================
                # put evaluation here, directly on datasets
                y_trues = np.array([])
                y_preds = np.array([])
                last_peak_file = None
                frame_gt_dir = Path(args.frames_gt_dir)

                # placeholder of caches
                peaks_cache = None
                frame_gt_cache = None
                video_pred_cache = None
                Path(args.anomaly_score_dir).mkdir(parents=True, exist_ok=True)

                with torch.no_grad():
                    for features, label, peaks_name, peaks_index in eval_iter:
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        features = features.to(device)
                        outputs = self.model(features).squeeze(-1)

                        peaks_path = Path(args.peaks_dir) / peaks_name
                        frame_gt_path = frame_gt_dir / peaks_name

                        if last_peak_file != peaks_name:
                            # save last
                            if video_pred_cache is not None:
                                np.save(
                                    Path(args.anomaly_score_dir) / last_peak_file, video_pred_cache)

                            # load next
                            peaks_cache = np.load(
                                peaks_path, allow_pickle=True)  # load new file if not loaded
                            frame_gt_cache = np.load(frame_gt_path, allow_pickle=True)
                            video_pred_cache = np.array([0.] * frame_gt_cache.shape[0])
                            last_peak_file = peaks_name  # update

                        center = peaks_cache[peaks_index]

                        clip_num = outputs.shape[0]
                        # clip_gt = label.cpu().numpy().repeat(clip_num)
                        clip_gt = label.repeat(clip_num)
                        # scores = outputs[0].cpu().numpy()
                        scores = outputs.cpu()
                        video_pred_cache = update_pedict(video_pred_cache, scores, center,
                                                         args.clip_frame_num)

                        y_trues = np.concatenate((y_trues, clip_gt))
                        y_preds = np.concatenate((y_preds, scores))

                    fpr, tpr, _ = roc_curve(y_trues, y_preds)
                    rec_auc = auc(fpr, tpr)
                    logging.info(f"rec_auc={rec_auc}")
                    if rec_auc > last_auc:
                        last_auc = rec_auc
                        logging.info(f"Save the model after epoch {epoch}, auc={last_auc}")
                        if args.train.ext:
                            save_name = f'{args.dataset}_epoch_{epoch}_{last_auc:.4f}_ext.pt'
                        else:
                            save_name = f'{args.dataset}_epoch_{epoch}_{last_auc:.4f}.pt'
                        self.save(os.path.join(network_model_path_base, save_name))
                        np.save(Path(args.anomaly_score_dir) / last_peak_file, video_pred_cache)

                # ==========================================================

            self.notify_callbacks('on_training_iteration_end', train_loss, val_loss)

        self.notify_callbacks('on_training_end', self.model)
        # Save the last model anyway...
        if network_model_path_base:
            self.save(os.path.join(network_model_path_base, f'epoch_{epoch + 1}.pt'))

    def evaluate(self, criterion, data_iter):
        """
        Evaluates the model
        Args:
            criterion: Loss function for calculating the evaluation

            data_iter: torch data iterator
        """
        self.eval()
        self.notify_callbacks('on_evaluation_start', len(data_iter))
        total_loss = 0

        with torch.no_grad():
            for iteration, batch in enumerate(data_iter):
                # batch = self.data_to_device(batch, self.device)
                inputs, targets = batch
                inputs = self.data_to_device(inputs, self.device)
                targets = self.data_to_device(targets, self.device)

                # outputs = self.model(batch)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                self.notify_callbacks('on_evaluation_step', iteration,
                                      outputs.detach().cpu(),
                                      targets.detach().cpu(), loss.item())

                total_loss += loss.item()

        loss = total_loss / len(data_iter)
        self.notify_callbacks('on_evaluation_end')
        return loss

    def do_epoch(self, criterion, optimizer, data_iter, epoch):
        total_loss = 0
        total_time = 0
        self.train()
        self.notify_callbacks('on_epoch_start', epoch, len(data_iter))
        for iteration, (batch, targets) in enumerate(data_iter):
            self.iteration += 1
            start_time = time.time()
            batch = self.data_to_device(batch, self.device)
            targets = self.data_to_device(targets, self.device)

            outputs = self.model(batch)

            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            end_time = time.time()

            total_time += end_time - start_time

            self.notify_callbacks(
                'on_epoch_step',
                self.iteration,
                iteration,
                loss.item(),
            )
            self.iteration += 1

        loss = total_loss / len(data_iter)

        self.notify_callbacks('on_epoch_end', loss)
        return loss

    def data_to_device(self, data, device):
        """
        Transfers a tensor data to a device
        Args:
            data: torch tensor
            device: target device
        """
        if type(data) == list:
            data = [d.to(device) for d in data]
        elif type(data) == tuple:
            data = tuple([d.to(device) for d in data])
        else:
            data = data.to(device)

        return data

    def save(self, model_path):
        """
        Saves the model to the given path. If currently using data parallel, the method
        will save the original model and not the data parallel instance of it
        Args:
            model_path: target path to save the model to
        """
        if self.is_data_parallel:
            torch.save(self.model.module.state_dict(), model_path)
        else:
            torch.save(self.model.state_dict(), model_path)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
