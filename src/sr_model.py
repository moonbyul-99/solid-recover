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
import joblib
from muon import MuData


class Base_sr(nn.Module):
    def __init__(self):
        super().__init__()
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

            # # Case 2: scipy sparse matrix
            # if sparse.issparse(X):
            #     X = X.toarray()  # convert to dense numpy array
            # Case 2: scipy sparse matrix
            if sparse.issparse(X):
                chunk_size = 1000
                if not sparse.isspmatrix_csr(X):
                    X = X.tocsr()  # ensure CSR format for efficient row slicing

                n_rows = X.shape[0]
                chunks = []
                # Use tqdm to show progress
                for start in tqdm(range(0, n_rows, chunk_size), desc="Converting sparse matrix to dense"):
                    end = min(start + chunk_size, n_rows)
                    chunk_dense = X[start:end].toarray()  # convert current block to dense
                    chunks.append(chunk_dense)
                X = np.concatenate(chunks, axis=0)

            # Case 3: numpy.ndarray (most common)
            if isinstance(X, np.ndarray):
                # Ensure contiguous array for efficiency
                X = np.ascontiguousarray(X)
                tensor = torch.from_numpy(X).float()
                return tensor

    def set_dataset(self, train_dataset:Dataset, test_dataset: Dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
    
    def set_dataloader(self, batch_size: int = 128):
        self.batch_size = batch_size

        self.train_loader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = True)
        self.test_loader = DataLoader(self.test_dataset, batch_size = batch_size, shuffle = False)
        return None

    def set_loss(self):
        raise NotImplementedError('Subclass must implement this method')
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

    def _process_and_calculate_loss(self, batch, device):

        '''
        Perform the forward pass and calculate loss
        return outputs_dic and loss_dic
        '''
        raise NotImplementedError('Subclass must implement this method')



    def eval_model(self, eval_points, device = 'cuda'):
        self.model.eval()
    
        total_counts = 0
        eval_dic = {}

        with torch.no_grad():
            for batch in self.test_loader:
                outputs, loss_dic = self._process_and_calculate_loss(batch, device)
                for key in outputs:
                    if 'recon' in key:
                        B = outputs[key].shape[0]
                        break 
                
                total_counts += B 
                for key, value in loss_dic.items():
                    if 'loss' in key:
                        eval_dic[f'{key}/val'] = eval_dic.get(f'{key}/val', 0) + value.item()*B 

        for key, value in eval_dic.items():
            self.writer.add_scalar(key, value/total_counts, eval_points)

    def load_model(self, checkpoint_path = None):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location = device)
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

                '''避免对比学习中最后一个batch中样本数目过少'''
                if isinstance(batch, dict):
                    current_batch_size = next(iter(batch.values())).size(0)
                else:
                    current_batch_size = None 

                if current_batch_size is not None and current_batch_size <= 0.8*self.train_loader.batch_size:
                    continue

                outputs, loss_dic = self._process_and_calculate_loss(batch, device)

                loss = loss_dic['loss']
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad() 

                steps += 1 
                for key, value in loss_dic.items():
                    if 'loss' in key:
                        self.writer.add_scalar(f'{key}/train', value, steps)
                    if 'logit_scale' in key:
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


        

class single_sr(Base_sr):

    '''
    Define the single omic sr model
    '''
    def __init__(self,
                 feature_num: int, 
                 hidden_params: Union[Dict[str,int], List[int]],
                 embed_dim: int,
                 use_rmsnorm = True,
                 use_residual = False, 
                 dropout_p = 0.05,):
        super().__init__()

        self.model = sr_vae(feature_num, hidden_params, embed_dim, use_rmsnorm, use_residual, dropout_p)
        self.embed_dim = self.model.embed_dim
        self.classifiers = None
        self.labelencoders = None

    def create_dataset(self, 
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

    def set_loss(self,beta: float = 1.0):
        self.model.set_loss(beta)
    def _process_and_calculate_loss(self, batch, device):
        feature = batch['feature'].to(device)
        outputs, loss_dic = self.model(feature)
        return outputs, loss_dic
    def get_embedding(self, adata: AnnData, embedding_keys: List[str] = ['z_embed'],  device: str = 'cuda', batch_size: int = 128,):

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
    
    def add_classifiers(self, labelencoder_dic: Dict):
        if isinstance(self.model, sr_vae):
            embed_dim = self.model.mu_proj[0].out_features
        elif isinstance(self.model, sr_ae):
            embed_dim = self.model.embed_proj[0].out_features
        else:
            raise TypeError("Unsupported backbone type. Must be sr_vae or sr_ae.")
        
        for category_key in labelencoder_dic:
            path = labelencoder_dic[category_key]

            if self.labelencoders is None:
                self.labelencoders = {category_key: joblib.load(path)}
            else:
                self.labelencoders[category_key] = joblib.load(path)

            class_num = self.labelencoders[category_key].classes_.shape[0]
            print(f'{category_key} label encoder has {class_num} classes.')

            new_classifier =nn.Linear(embed_dim, class_num)
            if self.classifiers is None:
                self.classifiers = nn.ModuleDict({category_key: new_classifier})
            else:
                self.classifiers[category_key] = new_classifier
        
    
    def train_classifiers(self, 
                           dataset: Dataset,
                           project_dir: str,
                           lr_dic: Union[Dict[str,float],float],
                           test_size: float,
                           train_steps: int,
                           eval_points: int,
                           batch_size: int = 512,
                           device: str = 'cuda',
                           random_seed: int =42):
        
        if isinstance(lr_dic, float):
            lr_dic = {category_key: lr_dic for category_key in self.classifiers}

        '''
        train test split
        '''

        Dataset_dic = dataset.train_test_split(
            test_size=test_size,
            seed=random_seed
        )

        train_dataset, test_dataset = Dataset_dic['train'], Dataset_dic['test']
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

        '''
        pretrained model freeze
        '''
        for param in self.model.parameters():
            param.requires_grad = False

        '''
        set optimizers, only optimizers the classifiers
        '''
        optimizers = {}
        for category_key in self.classifiers:
            adapter = self.classifiers[category_key]
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr_dic[category_key])
            optimizers[category_key] = optimizer
        '''
        set logger and save dir
        '''
        os.makedirs(project_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logger_dir = os.path.join(project_dir, f'classification_logs_{timestamp}')
        model_dir = os.path.join(project_dir, f'classification_models_{timestamp}')
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(logger_dir, exist_ok=True)
        writer = SummaryWriter(log_dir = logger_dir)

        '''
        set model to device
        '''
        self.model.to(device)
        self.classifiers.to(device)

        '''
        train loop 
        '''
        epoch_num = train_steps // len(train_loader) + 1 
        steps = 0 

        for epoch in range(epoch_num):
            for batch in train_loader:
                
                x = batch['feature'].to(device)
                outputs = self.model(x)
                z = outputs['z_embed'] 

                self.classifiers.train()
                for category_key in self.classifiers:
                    adapter = self.classifiers[category_key]
                    optimizer = optimizers[category_key]

                    optimizer.zero_grad()
                    logits = adapter(z)
                    loss = F.cross_entropy(logits, batch[category_key].to(device))
                    loss.backward()
                    optimizer.step() 

                    writer.add_scalar(f'{category_key}_loss/train', loss.item(), steps)
                steps += 1

                
                if steps == train_steps or steps % eval_points == 0:
                    '''
                    evaluation
                    '''
                    self.model.eval()
                    self.classifiers.eval()

                    eval_dic = {}
                    total_counts = 0
                    with torch.no_grad():
                        for batch in test_loader:
                            x = batch['feature'].to(device)
                            outputs = self.model(x)
                            z = outputs['z_embed'] 
                            total_counts += logits.shape[0]

                            for category_key in self.classifiers:
                                adapter = self.classifiers[category_key]
                                logits = adapter(z)
                                loss = F.cross_entropy(logits, batch[category_key].to(device))
                                
                                eval_dic[category_key] = eval_dic.get(category_key, 0) + loss.item() * logits.shape[0]

                    for category_key in self.classifiers:
                        writer.add_scalar(f'{category_key}_loss/test', eval_dic[category_key] / total_counts, steps)

                if steps == train_steps:
                    torch.save(self.classifiers.state_dict(), os.path.join(model_dir,f'classifier_adapter_{steps}.pth'))
                    break 

            if steps == train_steps:
                break
    @torch.no_grad()
    def classify(self, adata: AnnData, device: str = 'cuda', pred_probability: bool = False):
        self.model.to(device)
        self.classifiers.to(device)

        X = self._adata_format(adata)
        infer_dataset = single_data(X)
        infer_loader = DataLoader(infer_dataset, batch_size = 128, shuffle = False)
        infer_dic = {key: [] for key in self.classifiers}

        for batch in infer_loader:
            x = batch['feature'].to(device)
            outputs = self.model(x)
            z = outputs['z_embed']

            for category_key in self.classifiers:
                adapter = self.classifiers[category_key]
                logits = adapter(z)
                if pred_probability:
                    predicted_probs = F.softmax(logits, dim=1)
                    predicted_probs_np = predicted_probs.cpu().numpy()
                    infer_dic[category_key].append(predicted_probs_np)
                else:
                    predicted_indices = torch.argmax(logits, dim=1)
                    predicted_indices_np = predicted_indices.cpu().numpy()
                    predicted_labels = self.labelencoders[category_key].inverse_transform(predicted_indices_np)
                    infer_dic[category_key].extend(predicted_labels)

        if pred_probability:
            for category_key in infer_dic:
                prob_df = pd.DataFrame(np.concatenate(infer_dic[category_key], axis=0), columns = self.labelencoders[category_key].classes_)
                adata.uns[f'pred_prob_{category_key}'] = prob_df
        else:
            for category_key in infer_dic:
                adata.obs[f'pred_{category_key}'] = infer_dic[category_key]
        return adata

    def load_classifiers(self, ckpt_path: str):
        checkpoint = torch.load(ckpt_path)
        self.classifiers.load_state_dict(checkpoint)

class pair_sr_scratch(Base_sr):

    '''
    paired solid recover model, training from scratch
    '''
    def __init__(self, 
                feature_num_1:int, 
                 feature_num_2:int, 
                 hidden_params_1: Union[Dict[str,int], List[int]],
                 hidden_params_2: Union[Dict[str,int], List[int]],
                 embed_dim: int,
                 use_rmsnorm = True,
                 use_residual = False, 
                 dropout_p = 0.05,):
        super().__init__()
        self.model = sr_pair_vae(feature_num_1, feature_num_2, hidden_params_1, hidden_params_2, embed_dim, use_rmsnorm, use_residual, dropout_p)#, clip_temperature)
        self.model_type = 'sr_vae-sr_vae'
    
    def create_dataset(self, 
                    mdata: MuData,
                    key_1: str,
                    key_2: str,
                    train_idx: np.ndarray = None,
                    test_idx: np.ndarray = None,
                    test_size: float = None,
                    random_state: int = 42):
        '''
        Args:
            mdata: MuData object,
            key_1: str, key corresponding to first modality
            key_2: str, key corresponding to second modality
            train_idx: np.array, array([int]) or array([str]) corresponding to train samples indices 
            test_idx: np.array, array([int]) or array([str]) corresponding to test samples indices
            test_size: float, default = 0.1, test dataset fraction
            random_state: int, default = 42, random seed for train test split
        '''

        if train_idx is None and test_idx is None:
            if test_size is None:
                test_size = min(0.1, 50000/adata.shape[0])
            train_idx, test_idx = train_test_split(np.arange(mdata.shape[0]), test_size = test_size, random_state = random_state)
        
        train_data_1 = mdata[key_1][train_idx,:]
        test_data_1 = mdata[key_1][test_idx,:]

        train_data_2 = mdata[key_2][train_idx,:]
        test_data_2 = mdata[key_2][test_idx,:]

        self.train_dataset = pair_data(Base_sr._adata_format(train_data_1), Base_sr._adata_format(train_data_2))
        self.test_dataset = pair_data(Base_sr._adata_format(test_data_1), Base_sr._adata_format(test_data_2))
    
    def set_loss(self, 
                 vae_beta_1: float = 1.0, 
                 vae_beta_2: float = 1.0, 
                 clip_weight: float = 1.0,
                 cross_recon_1: float = 0.2,
                 cross_recon_2: float = 0.2,
                 temperature: float = 0.07,
                 trainable_clip_temperature: bool = False,
                 use_weight = False,
                 top_k_ratio = 0.1,
                 bottom_k_ratio = 0.1,
                 weight_top = 0.1,
                 weight_bottom = 2.0):
        self.model.set_loss(vae_beta_1, vae_beta_2, clip_weight, cross_recon_1, cross_recon_2, temperature, trainable_clip_temperature,
                            use_weight, top_k_ratio, bottom_k_ratio, weight_top, weight_bottom)

        # self.loss = VAE_clip_loss(vae_beta_1, vae_beta_2, clip_weight, cross_recon_1, cross_recon_2, temperature)
        # self.loss.clip_loss.logit_scale.requires_grad = trainable_clip_temperature
    # def calculate_loss(self, x1, x2, sr_pair_out:Dict):
    #     return self.loss(x1, x2, sr_pair_out)
    
    def _process_and_calculate_loss(self, batch, device):

        x1 = batch['omic_1'].to(device)
        x2 = batch['omic_2'].to(device)
        outputs, loss_dic = self.model(x1,x2)
        # outputs = self.model(x1, x2)
        # loss_dic = self.calculate_loss(x1, x2, outputs) 
        # loss_dic['logit_scale'] = self.loss.clip_loss.logit_scale.item()
        return outputs, loss_dic
        
class pair_sr_pretrain(pair_sr_scratch):

    '''
    Training solid recover model from pretrained single_sr model
    '''

    def __init__(self, 
                 feature_num_1:int, 
                 feature_num_2:int, 
                 hidden_params_1: Union[Dict[str,int], List[int]],
                 hidden_params_2: Union[Dict[str,int], List[int]],
                 embed_dim:int,
                 use_rmsnorm = True,
                 use_residual = False, 
                 dropout_p = 0.05,):
        
        super().__init__(
            feature_num_1=feature_num_1,
            feature_num_2=feature_num_2,
            hidden_params_1=hidden_params_1,
            hidden_params_2=hidden_params_2,
            embed_dim=embed_dim,
            use_rmsnorm=use_rmsnorm,
            use_residual=use_residual,
            dropout_p=dropout_p,
        )

        print('init pair_model') 
        self.model = sr_pair_vae(feature_num_1, feature_num_2, hidden_params_1, hidden_params_2, embed_dim, use_rmsnorm, use_residual, dropout_p)
    @staticmethod
    def _load_partial_weights(model, ckpt_path_or_dict, verbose=True):
        '''
        加载部分权重

        说明：
        sr_pair_vae 允许两个模态的 sr_vae 拥有不同的 embed_dim, 在构建sr_pair_vae 时，会在encoder 部分添加额外的
        '''
        if isinstance(ckpt_path_or_dict, str):
            ckpt = torch.load(ckpt_path_or_dict, map_location='cpu')
        else:
            ckpt = ckpt_path_or_dict

        model_dict = model.state_dict()
        filtered_ckpt = {
            k: v for k, v in ckpt.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        if verbose:
            print(f"[Partial Load] Loaded {len(filtered_ckpt)}/{len(model_dict)} layers from checkpoint.")
            if len(filtered_ckpt) == 0:
                print("⚠️  Warning: No matching keys found!")

        model_dict.update(filtered_ckpt)
        model.load_state_dict(model_dict, strict=False)
        return len(filtered_ckpt)
        
    def load_pretrained_model(self, omic_1_ckpt: str, omic_2_ckpt: str):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        print('load omic 1 pretrained model')
        checkpoint = torch.load(omic_1_ckpt, map_location = device)
        self.model.model_1.load_state_dict(checkpoint['model_state_dict'])

        print('load omic 2 pretrained model')
        checkpoint = torch.load(omic_2_ckpt, map_location = device)
        self.model.model_2.load_state_dict(checkpoint['model_state_dict'])

    


# class pair_sr_atlas(Base_sr):

#     '''
#     Training solid recover model from atlas scale single omic data and aligned using paired omics data
#     '''

#     def __init__(self, 
#                  feature_num_1:int, 
#                  feature_num_2:int, 
#                  hidden_params_1: Union[Dict[str,int], List[int]],
#                  hidden_params_2: Union[Dict[str,int], List[int]],
#                  embed_dim: int,
#                  use_rmsnorm = True,
#                  use_residual = False, 
#                  dropout_p = 0.05,):
        

#         super().__init__()
        
#         print('init omic 1 model')
#         '''omic 1 model'''
#         self.model_1 = sr_vae(feature_num = feature_num_1,
#                               hidden_params= hidden_params_1,
#                               embed_dim = embed_dim,
#                               use_rmsnorm= use_rmsnorm, 
#                               use_residual= use_residual, 
#                               dropout_p= dropout_p)
        
#         print('init omic 2 model')
#         '''omic 2 model'''
#         self.model_2 = sr_vae(feature_num = feature_num_2,
#                               hidden_params= hidden_params_2,
#                               embed_dim = embed_dim,
#                               use_rmsnorm= use_rmsnorm, 
#                               use_residual= use_residual, 
#                               dropout_p= dropout_p)
        
#     def set_atlas_dataset(self, train_dataset_1, test_dataset_1, train_dataset_2, test_dataset_2):
#         self.train_dataset_1 = train_dataset_1
#         self.test_dataset_1 = test_dataset_1
#         self.train_dataset_2 = train_dataset_2
#         self.test_dataset_2 = test_dataset_2

#     def set_atlas_dataloader(self, batch_size_1: int = 1024, batch_size_2: int = 1024):
#         self.batch_size_1 = batch_size_1

#         self.train_loader_1 = DataLoader(self.train_dataset_1, batch_size=self.batch_size_1, shuffle=True)
#         self.test_loader_1 = DataLoader(self.test_dataset_1, batch_size=self.batch_size_1, shuffle=False)

#         self.train_loader_2 = DataLoader(self.train_dataset_2, batch_size=batch_size_2, shuffle=True)
#         self.test_loader_2 = DataLoader(self.test_dataset_2, batch_size=batch_size_2, shuffle=False)

#     def set_pretrain_loss(self, beta_1: float = 1.0, beta_2: float = 1.0):
#         self.model_1.set_loss(beta_1)
#         self.model_2.set_loss(beta_2)

#     def _process_and_calculate_pretrain_loss(self, batch, device):
#         feature = batch['feature'].to(device)
#         outputs,loss_dic = self.model(feature)
#         loss_dic = self.calculate_loss(outputs, feature)
#         return outputs, loss_dic

        
        

# class pair_sr:

#     def __init__(self, model_1: single_sr, model_2: single_sr, embed_dim: int):
#         self.model_1 = model_1 
#         self.model_2 = model_2 
#         d1 = self.model_1.embed_dim 
#         d2 = self.model_2.embed_dim 
#         self.embed_dim = embed_dim

#         self.adapter_1 = nn.Linead(d1,embed_dim)
#         self.proj_1 = nn.Linear(embed_dim,d1) 

#         self.adapter_2 = nn.Linear(d2, embed_dim)
#         self.proj_2 = nn.Linear(embed_dim,d2)

#     def forward