"""
Define the solid recover model
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math
import numpy as np 
from sr_net import sr_single_omic, multi_model
import custom_scheduler
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class NetworkConfig:
    feature_num: int
    hidden_dims: List[int]
    dropout: float
    layernorm_eps: float
    activation: str
    input_dropout: float
    vae_weight: float
    ce_weights: Optional[Dict[str, float]]
    class_dict: Optional[Dict[str, int]]

@dataclass
class OptimizerConfig:
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    anneal_steps: int
    min_lr: float

@dataclass
class TrainingConfig:
    device: str
    training_steps: int
    eval_steps: int
    save_steps: int
    log_dir: str
    save_dir: str
    run_name: str

class single_sr:

    def __init__(self, config: Dict):
        network_config = NetworkConfig(**config['network'])
        optimizer_config = OptimizerConfig(**config['optimizer'])
        training_config = TrainingConfig(**config['training'])

        self.model_type = config['omic']['model_type']

        self.feature_num = network_config.feature_num
        self.hidden_dims = network_config.hidden_dims
        self.dropout = network_config.dropout
        self.layernorm_eps = network_config.layernorm_eps
        self.activation = network_config.activation
        self.input_dropout = network_config.input_dropout
        self.vae_weight = network_config.vae_weight
        self.ce_weights = network_config.ce_weights
        self.class_dict = network_config.class_dict

        # Define sr_single_omic model based on config parameters
        self.model = sr_single_omic(
            feature_num=self.feature_num,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            layernorm_eps=self.layernorm_eps,
            activation=self.activation,
            input_dropout=self.input_dropout,
            vae_weight=self.vae_weight,
            ce_weights=self.ce_weights,
            class_dict=self.class_dict
        )

        # Get optimizer and scheduler parameters
        self.learning_rate = optimizer_config.learning_rate
        self.weight_decay = optimizer_config.weight_decay
        self.warmup_steps = optimizer_config.warmup_steps
        self.anneal_steps = optimizer_config.anneal_steps
        self.min_lr = optimizer_config.min_lr

        # Get training parameters
        self.device = training_config.device
        self.training_steps = training_config.training_steps
        self.eval_steps = training_config.eval_steps
        self.save_steps = training_config.save_steps

        self.log_dir = training_config.log_dir
        self.save_dir = training_config.save_dir
        self.run_name = training_config.run_name
    
    def set_optimizer(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = custom_scheduler.WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=self.warmup_steps,
            anneal_steps=self.anneal_steps,
            min_lr=self.min_lr)
        return optimizer, scheduler

    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint from the given path.
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer, scheduler = self.set_optimizer()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['step']
        print(f"Checkpoint loaded from {checkpoint_path} at step {start_step}")
        return optimizer, scheduler, start_step

    def eval(self,
             eval_loader,
             writer,
             eval_point):
        """
        Evaluate the model on the validation dataset.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0  # New variable to record the total number of samples
        sub_losses = {}
        with torch.no_grad():
            for batch in eval_loader:
                """
                Data preparation
                """
                feature = batch['feature'].to(self.device)

                if self.class_dict is None:
                    label_dic = None
                else:
                    label_dic = {}
                    for key in self.class_dict:
                        label_dic[key] = batch[key].to(self.device)

                """
                Loss forward 
                """
                outputs = self.model(feature, label_dic)

                loss = outputs['loss']
                total_loss += loss.item() * feature.size(0)  # Accumulate loss by multiplying with the number of samples in the current batch
                total_samples += feature.size(0)  # Accumulate the number of samples in the current batch

                for key, value in outputs.items():
                    if 'loss' in key and key != 'loss':
                        if key not in sub_losses:
                            sub_losses[key] = 0.0
                        sub_losses[key] += value.item() * feature.size(0)  # Accumulate sub-loss by multiplying with the number of samples in the current batch

        avg_loss = total_loss / total_samples  # Calculate average loss using the total number of samples
        writer.add_scalar('loss/val', avg_loss, eval_point)
        print(f"Validation loss at step {eval_point}: {avg_loss}")

        for key, value in sub_losses.items():
            avg_sub_loss = value / total_samples  # Calculate average sub-loss using the total number of samples
            writer.add_scalar(f'{key}/val', avg_sub_loss, eval_point)
            print(f"Validation {key} at step {eval_point}: {avg_sub_loss}")

    def train_model(self,
              train_loader,
              val_loader=None,
              checkpoint_path=None):

        """
        Define optimizer, scheduler, writer
        """

        if checkpoint_path is not None:
            optimizer, scheduler, start_step = self.load_checkpoint(checkpoint_path)
        else:
            optimizer, scheduler = self.set_optimizer()
            start_step = 0

        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')

        self.save_dir = os.path.join(self.save_dir, '_'.join([self.run_name,current_time]))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.log_dir = os.path.join(self.log_dir, '_'.join([self.run_name, current_time]))
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        writer = SummaryWriter(log_dir = self.log_dir)

        """
        Training loop 
        """
        steps = start_step
        L = len(train_loader)
        epoch_num = self.training_steps // L + 1

        self.model.to(self.device)
        self.model.train()

        for _ in range(epoch_num):
            for _, batch in enumerate(train_loader):

                """
                Data preparation
                """
                feature = batch['feature'].to(self.device)

                if self.class_dict is None:
                    label_dic = None
                else:
                    label_dic = {}
                    for key in self.class_dict:
                        label_dic[key] = batch[key].to(self.device)

                """
                Loss backward 
                """
                outputs = self.model(feature, label_dic)

                loss = outputs['loss']
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                """
                Update writer
                """

                steps += 1
                for key, values in outputs.items():
                    if 'loss' in key:
                        writer.add_scalar(f'{key}/train', values, steps)
                writer.add_scalar('lr', scheduler.get_last_lr()[0], steps)

                """
                Save checkpoint when reach save step
                """
                if steps % self.save_steps == 0:
                    checkpoint_path = os.path.join(self.save_dir, f'checkpoint_{steps}.pth')
                    torch.save({
                        'step': steps,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': loss.item(),
                    }, checkpoint_path)
                    print(f"Checkpoint saved at {checkpoint_path}")

                """
                Perform evaluation when reach eval_steps
                """
                if steps % self.eval_steps == 0:
                    self.eval(eval_loader=val_loader,
                               writer=writer,
                               eval_point=steps)
                    self.model.train()
                if steps >= self.training_steps:
                    break

        ## Save the final model
        if steps % self.save_steps != 0:
            checkpoint_path = os.path.join(self.save_dir, f'checkpoint_{steps}.pth')
            torch.save({
                'step': steps,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': loss.item(),
            }, checkpoint_path)
            print(f"Final checkpoint saved at {checkpoint_path}")

        return None