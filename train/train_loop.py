import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from datetime import datetime
import os
import blobfile as bf
import logging
logging.getLogger().setLevel(logging.INFO)

class TrainLoop:
    def __init__(self, args, model, device, data_train=None, data_val=None, data_test=None):
        self.args = args
        self.data_train = data_train
        self.data_val = data_val
        self.model = model
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.batch_size = args.batch_size
        self.num_epochs = 40000
        self.logs_dir = args.logs_dir
        self.save_dir = args.save_dir
        self.sync_cuda = torch.cuda.is_available()
        self.resume_checkpoint = args.resume_checkpoint
        self.step = 0
        self.criterion = nn.CrossEntropyLoss()
        if self.resume_checkpoint:
            self._load_and_sync_parameters()

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.opt = optim.AdamW([{'params': self.model.parameters(), 'lr':self.lr, 'weight_decay':self.weight_decay}])
        # check！load optimizer state
        if self.resume_checkpoint:
            self._load_optimizer_state()


    def run_loop(self):
        # check！create new tb folder，avoiding cover
        prefix = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
        tb_dir = os.path.join(self.logs_dir, "tb_"+prefix)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir, flush_secs=10)
        self.model.train()

        for epoch in range(self.num_epochs):
                running_loss = 0.0
                for batch in tqdm(self.data_train):
                    data, target = batch
                    self.optimizer.zero_grad()
                    
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item() * data.size(0)
                
                epoch_loss = running_loss / len(self.data_train.dataset)
                print('Epoch: {}, Phase: {}, Loss: {}'.format(epoch, self.step+self.resume_step, epoch_loss))

                if self.step % self.log_interval == 0:                  
                    writer.add_scalar('Loss/{}'.format(self.step+self.resume_step), epoch_loss, epoch)
                    # check! record learning rate，check if lr is too close to zero
                    current_lr = self.opt.param_groups[0]['lr']
                    writer.add_scalars("learning_rate", {'lr': current_lr}, self.step)
                
                # check！check valid loss, use for early stop
                if self.step % 5000 == 0:
                    self.model.eval()
                    val_loss = self.evaluate()

                    writer.add_scalars("losses/losses", {'eval':val_loss}, self.step,)

                    if val_loss < self.min_val_loss:
                        # check! record valid result, use for anlysising training state
                        self.evaluate_save()
                        print('New min valid loss, cur is {}, new is {}, saving the model...'.format(self.min_val_loss, val_loss))
                        self.save(name='best_val_')
                        self.min_val_loss = val_loss
                    self.model.train()

                self.step += 1
        writer.close()
        print('Finished Training')


    def evaluate(self):
        total_val_loss = 0
        for batch in tqdm(self.data_val):
            data, target = batch
            output = self.model(data)
            loss = self.criterion(output, target)
            running_loss += loss.item() * data.size(0)
        val_loss = running_loss / len(self.data_val.dataset)
        return val_loss



    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
    
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=self.device
                )
            )

    
    def _load_optimizer_state(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
    
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)

            opt_checkpoint = bf.join(bf.dirname(resume_checkpoint), f"opt{self.resume_step:09}_ok.pt")
            if bf.exists(opt_checkpoint):
                logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
                state_dict = dist_util.load_state_dict(
                    opt_checkpoint, map_location=self.device
                )
                self.opt.load_state_dict(state_dict)


    def evaluate_save(self):
        """
            save valid result
        """
        pass


def parse_resume_step_from_filename(filename):

    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    split1 = split1.replace('_ok', '')
    try:
        return int(split1)
    except ValueError:
        return 0



def find_resume_checkpoint():
    return None

