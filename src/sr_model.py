from sr_net import *
from lr_scheduler import *
from sr_dataset import *
from sr_loss import * 
from typing import List, Dict, Union, Any
from anndata import AnnData 
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import sparse
from torch.utils.data import Dataset, DataLoader 
import torch 
import os 
from datetime import datetime
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter


class single_sr:

    '''
    Define the single omic sr model
    '''
    def __init__(self,
                 feature_num: int, 
                 hidden_params: Union[Dict[str,int], List[int]],
                 embed_dim: int,
                 use_rmsnorm = True,
                 use_residual = False, 
                 dropout_p = 0.05,
                 vae_model = True,):
        
        if vae_model:
            self.model = sr_vae(feature_num, hidden_params, embed_dim, use_rmsnorm, use_residual, dropout_p)
            self.model_type = 'sr_vae'
        else:
            self.model = sr_ae(feature_num, hidden_params, embed_dim, use_rmsnorm, use_residual, dropout_p)
            self.model_type = 'sr_ae'
    @staticmethod
    def _adata_format(adata: AnnData) -> torch.Tensor:
            '''
            Check the .X format of adata and convert it to torch.Tensor (float32).

            Supports:
            - torch.Tensor
            - scipy.sparse matrix (csr, csc, etc.)
            - numpy.matrix

            Returns:
                torch.Tensor: of dtype float32, shape (n_obs, n_vars)
            '''
            X = adata.X

            # Case 1: already a torch.Tensor
            if isinstance(X, torch.Tensor):
                return X.float()  # ensure float32

            # Case 2: scipy sparse matrix
            if sparse.issparse(X):
                X = X.toarray()  # convert to dense numpy array

            # Case 3: numpy.ndarray (most common)
            if isinstance(X, np.ndarray):
                # Ensure contiguous array for efficiency
                X = np.ascontiguousarray(X)
                tensor = torch.from_numpy(X).float()
                return tensor

    def set_dataset(self, 
                    adata: AnnData,
                    train_idx: np.ndarray = None,
                    test_idx: np.ndarray = None,
                    test_size: float = None,
                    random_state: int = 42):
        '''
        Args:
            adata: AnnData object,
            train_idx: np.array, array([int]) corresponding to train samples indices
            test_idx: np.array, array([int]) corresponding to test samples indices
            test_size: float, default = 0.1, test dataset fraction
            random_state: int, default = 42, random seed for train test split
        '''

        if train_idx is None and test_idx is None:
            if test_size is None:
                test_size = min(0.1, 50000/adata.shape[0])
            train_idx, test_idx = train_test_split(np.arange(adata.shape[0]), test_size = test_size, random_state = random_state)
        
        train_data = adata[train_idx,:]
        test_data = adata[test_idx,:]

        self.train_dataset = single_data(self._adata_format(train_data))
        self.test_dataset = single_data(self._adata_format(test_data))
        return None 
    
    def set_dataset(self, train_dataset:Dataset, test_dataset: Dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
    def set_dataloader(self, batch_size: int = 128):
        self.train_loader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = True)
        self.test_loader = DataLoader(self.test_dataset, batch_size = batch_size, shuffle = False)
        return None

    def set_optimizer(self, 
                      lr: float ,
                      warmup_steps: int, 
                      steady_1_steps: int, 
                      cosine_anneal_steps: int, 
                      min_lr: float = 1e-6):
        '''
        Set optimizer, default is AdamW
        '''

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = lr)
        self.scheduler = sr_scheduler(self.optimizer, 
                                     warmup_steps = warmup_steps, 
                                     steady_1_steps = steady_1_steps, 
                                     cosine_anneal_steps = cosine_anneal_steps, 
                                     min_lr = min_lr)
        
    def set_project(self, project_dir):
        '''
        create a project directory, save the model checkpoint, tensorboard log 
        create a tensorboard log writer
        '''

        if not os.path.exists(project_dir):
            os.makedirs(project_dir, exist_ok= True)
        
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            project_dir = f'{project_dir}_{timestamp}'
            os.makedirs(project_dir, exist_ok=True)


        self.project_dir = project_dir
        self.model_dir = os.path.join(self.project_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        self.log_dir = os.path.join(self.project_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir = self.log_dir)

    def set_loss(self,beta: float = 1.0):
        if self.model_type == 'sr_vae':
            self.loss = VAE_loss(kl_weight = beta)
        else:
            self.loss = AE_loss( kl_weight = beta)

    def calculate_loss(self, outputs, x):
        if self.model_type == 'sr_vae':
            x_recon = outputs['x_recon']
            z_mu = outputs['z_mu']
            z_logvar = outputs['z_logvar']
            loss_dic = self.loss(x_recon, x, z_mu, z_logvar)
        
        if self.model_type == 'sr_ae':
            x_recon = outputs['x_recon']
            loss_dic = self.loss(x_recon, x)
        return loss_dic

    def eval_model(self, eval_points, device = 'cuda'):
        self.model.eval()
        total_counts = 0
        eval_dic = {}

        with torch.no_grad():
            for batch in self.test_loader:
                x = batch['feature'].to(device)
                outputs = self.model(x)
                B = x.shape[0]

                loss_dic = self.calculate_loss(outputs, x)
                
                total_counts += B 
                for key, value in loss_dic.items():
                    eval_dic[f'{key}/val'] = eval_dic.get(f'{key}/val', 0) + value.item()*B 

        for key, value in eval_dic.items():
            self.writer.add_scalar(key, value/total_counts, eval_points)

    def init_model(self, checkpoint_path = None):
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def train_model(self, 
                    train_steps,
                    eval_points,
                    save_points,
                    device = 'cuda'):

        '''
        Training loop
        '''

        steps = 0 
        L = len(self.train_loader)
        epoch_num = train_steps // L + 1
        self.model.to(device) 

        self.model.train()

        for _ in range(epoch_num):
            for batch in tqdm(self.train_loader):

                feature = batch['feature'].to(device)

                outputs = self.model(feature)
                loss_dic = self.calculate_loss(outputs, feature)

                loss = loss_dic['loss']
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad() 

                steps += 1 
                for key, value in loss_dic.items():
                    if 'loss' in key:
                        self.writer.add_scalar(f'{key}/train', value, steps)
                self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], steps)

                if steps > train_steps:
                    self.eval_model(eval_points = steps, device = device)
                    ckpt_path = os.path.join(self.model_dir, f'ckpt_{steps}.pth')
                    torch.save({'model_state_dict': self.model.state_dict()}, ckpt_path)
                    break 
                if steps % eval_points == 0:
                    self.eval_model(eval_points = steps, device = device) 
                if steps % save_points == 0:
                    ckpt_path = os.path.join(self.model_dir, f'ckpt_{steps}.pth')
                    torch.save({'model_state_dict': self.model.state_dict()}, ckpt_path)
            if steps > train_steps:
                break 

        print('SR model training completed.')

    def get_embedding(self, adata: AnnData, embedding_keys: List[str] = ['z_embed'],  device: str = 'cuda', batch_size: int = 128):

        '''
        legal embedding_keys: z_encoder, z_mu, z_logvar, z_embed, x_recon, default: z_embed
        '''
        X = self._adata_format(adata)
        infer_dataset = single_data(X)
        infer_loader = DataLoader(infer_dataset, batch_size = batch_size, shuffle = False)

        self.model.eval()
        total_counts = 0
        eval_dic = {}
        embed_dic = {key: [] for key in embedding_keys}


        with torch.no_grad():
            for batch in infer_loader:
                x = batch['feature'].to(device)
                outputs = self.model(x)
                B = x.shape[0]

                loss_dic = self.calculate_loss(outputs, x)
                
                total_counts += B
                for key, value in loss_dic.items():
                    eval_dic[f'{key}/val'] = eval_dic.get(f'{key}/val', 0) + value.item()*B  

                for key in embedding_keys:
                    if key in outputs:
                        tensor = outputs[key].detach().cpu()
                        embed_dic[key].append(tensor.numpy())
                    else:
                        print(f"Warning: requested embedding key '{key}' not found in model outputs.")

        for key, value in eval_dic.items():
            print(f'Infer result {key}: {value/total_counts:.4f}')

        for key in embedding_keys:
            if len(embed_dic[key]) > 0:
                embed = np.concatenate(embed_dic[key], axis = 0)
                adata.obsm[f'sr_{key}'] = embed 
        return adata


        


                
        
    

        