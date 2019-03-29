import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from dataloader import ReutersDataset
from vocabulary import Vocabulary

log = logging.getLogger(__name__)

class Trainer(object):

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.device = torch.device(args.device)
        # self.dataset = args.dataset
        self.model_type = args.model
        assert self.model_type in {"sl"}
        self.model_id = args.model_id
        self.learning_rate = args.learning_rate
        
        # load data
        self.dataset_sizes = {}
        self.datasets = {}
        self.dataloaders = {}
        self._load_data(args)

        log.info("Device: {}".format(self.device))

        # load model
        self.model = None
        self._create_model(args)
        
    def _get_save_path(self):
        return os.path.join("results", self.model_id)


    def _load_data(self, args):
        self.vocabulary = Vocabulary(True, 5, True, "./reuters/stopwords")
        for split in {"training", "test"}:
            self.datasets[split] = ReutersDataset("reuters", split, self.vocabulary)
            self.dataset_sizes[split] = len(self.datasets[split])
            self.dataloaders[split] = DataLoader(self.datasets[split],
                                                 batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def _create_model(self, args):
        if self.model_type == "sl":
            pass
        else:
            raise ValueError("unknown model")

    def run_epoch(self, epoch, phase, device, optimizer):
        log.info("Phase: {}".format(phase))
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_n = 0.0

        n_batches = (self.dataset_sizes[phase] // self.batch_size) + 1
        # Iterate over data.
        for batch_idx, (_id, encoded_labels, id_doc, text, prep_text, label) in enumerate(self.dataloaders[phase], 1):

            inputs = inputs.to(device)

            if phase == "train":
                # zero the parameter gradients
                optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                all_loss = self.model.compute_loss(inputs)
                loss = all_loss["loss"]

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_n += inputs.size(0)
            if batch_idx % 50 == 0:
                log.info("\t[{}/{}] Batch {}/{}: Loss: {:.4f}".format(phase,
                                                                      epoch,
                                                                      batch_idx,
                                                                      n_batches,
                                                                      running_loss / running_n))

        epoch_loss = running_loss / self.dataset_sizes[phase]

        log.info('{} Loss: {:.4f}'.format(
            phase, epoch_loss))

        return epoch_loss

    def train(self, num_epochs):

        root_path = self._get_save_path()

        model_path = os.path.join(root_path, "best_model.pkl")

        device = torch.device(self.device)

        self.model = self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        since = time.time()

        for epoch in range(1, num_epochs + 1):
            log.info('Epoch {}/{}'.format(epoch, num_epochs))

            train_loss = self.run_epoch(epoch,
                                        "train", device, optimizer)

            if math.isnan(train_loss):
                raise ValueError("NaN loss encountered")

            val_loss = self.run_epoch(epoch, "val", device, None)

            if math.isnan(val_loss):
                raise ValueError("NaN loss encountered")

            save_path = os.path.join(
                self._get_save_path(), self.MODEL_WTS_DIR, "model_{}.wts".format(epoch))

            torch.save(self.model.state_dict(), save_path)
            
        time_elapsed = time.time() - since
        log.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    def test(self):
        device = torch.device(self.device)
        self.model = self.model.to(device)
        self.run_epoch(0, "test", device, None)