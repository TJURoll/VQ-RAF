import re
import os
import random
import pickle
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def get_embedding_model(llm_name, dataset_name):
    module_path = '.'.join(['embedding_models', llm_name])
    module = importlib.import_module(module_path)
    for attr in dir(module):
        if attr.lower() == llm_name.lower():
            return getattr(module, attr)(dataset_name)


class Quantizer(nn.Module):
    def __init__(self, codebook_dim = 256, word_num = 8, layer_num=1, dataset_name = 'amazon', llm_name="llama2"):
        super().__init__()
        self.word_num = word_num
        self.layer_num = layer_num
        self.dataset_name = dataset_name

        self.embedding_model = get_embedding_model(llm_name, dataset_name)

        # filter vocabulary
        tokens_df = self.embedding_model.filter_and_get_vocabulary() # two columns: token_id, token
        tokens_df.to_csv(f"./data/vocabulary/vocabulary_{dataset_name}_{llm_name}.csv", index=False) # save the filtered vocabulary

        # get token embedding & construct linear mapping
        token_id = torch.tensor(tokens_df['token_id'].values)
        self.register_buffer('token_id', token_id)
        self.vocabulary = tokens_df['token'].to_numpy()
        input_embeddings = self.embedding_model.get_token_embedding_matrix()
        codebook_tensor = input_embeddings(self.token_id).detach() # no grad
        self.register_buffer('codebook_tensor_pca', codebook_tensor)
        self.codebook_mapping = nn.Linear(codebook_tensor.shape[-1], codebook_dim)

        # TODO: use pca for dimensional reduction first
        # from torch_pca import PCA

        itm_dict = {'amazon': 'book', 'yelp': 'restaurant', 'steam': 'game'}

        # prompt
        prompt_usr = "The user and his likes can be described as the following words:"
        prompt_itm = f"The {itm_dict[dataset_name]} attracts those who can be described as the following words:"
        prompt_embedding = self.embedding_model.get_text_token_embeddings([prompt_usr, prompt_itm, prompt_itm])
        prompt_comma = self.embedding_model.get_text_token_embeddings(",")
        
        # ids_usr = self.tokenizer(prompt_usr, return_tensors="pt")["input_ids"]
        # ids_itm = self.tokenizer(prompt_itm, return_tensors="pt")["input_ids"]
        # prompt_embedding_usr = self.model.get_input_embeddings()(ids_usr).detach()
        # prompt_embedding_itm = self.model.get_input_embeddings()(ids_itm).detach()
        # prompt_embedding = torch.cat([prompt_embedding_usr, prompt_embedding_itm, prompt_embedding_itm], dim=0)
        # ids_comma = self.tokenizer(",", return_tensors="pt")["input_ids"][:,1:]
        # prompt_comma = self.model.get_input_embeddings()(ids_comma).detach()

        self.register_buffer('prompt_embedding', prompt_embedding)
        self.register_buffer('prompt_comma', prompt_comma)

    def reverse_codebook_mapping(self, x):
        reverse_weight = torch.pinverse(self.codebook_mapping.weight.detach()).t()
        x = x - self.codebook_mapping.bias.detach()
        x = torch.matmul(x, reverse_weight)
        return x
    
    def forward(self, z_e):
        if hasattr(self, 'codebook_mapping'):
            mapped_codebook = self.codebook_mapping(self.codebook_tensor_pca)
        else:
            mapped_codebook = self.codebook_tensor_pca

        # # RQ
        vq_loss = torch.tensor(0.0, device=z_e.device)
        commitment_loss = torch.tensor(0.0, device=z_e.device)
        z_q = torch.zeros_like(z_e, device=z_e.device)
        tensor_indices = torch.empty(z_e.shape[0], 1, device=z_e.device, dtype=torch.long)
        token_to_quantize = torch.empty(z_e.shape[0], 1, z_e.shape[1], device=z_e.device, dtype=torch.float32)

        for i in range(self.layer_num):
            residual = z_e - z_q
            dist_matrix = torch.sum(residual**2, dim=1, keepdim=True) + \
                torch.sum(mapped_codebook**2, dim=1) - 2 * \
                torch.matmul(residual, mapped_codebook.t())
            dist_matrix = dist_matrix.reshape(-1, self.word_num, dist_matrix.shape[-1]) 
            batch_indices = torch.arange(dist_matrix.shape[0], device=dist_matrix.device)

            layer_min_dist_indices = torch.empty(dist_matrix.shape[0], self.word_num, device=dist_matrix.device, dtype=torch.long)
            
            for j in range(self.word_num):
                word_j_min_dist_indices = torch.argmin(dist_matrix[:,j,:], dim=1)
                dist_matrix[batch_indices, : , word_j_min_dist_indices] = float('inf') # set the min distance to inf
                layer_min_dist_indices[:, j] = word_j_min_dist_indices

            layer_min_dist_indices = layer_min_dist_indices.reshape(-1)

            z_q += mapped_codebook[layer_min_dist_indices]

            # ||e-sg[z_e]||^2
            vq_loss += torch.mean((z_q - z_e.detach())**2)
            # ||z_e-sg[e]||^2
            commitment_loss += torch.mean((z_e - z_q.detach())**2)

            if i == 0:
                tensor_indices[:, i] = layer_min_dist_indices
                token_to_quantize[:, i, :] = residual
                # vq_loss = torch.mean((z_q - z_e.detach())**2)
                # commitment_loss = torch.mean((z_e - z_q.detach())**2)

            
        # straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach() 


        return z_q_st, 0.75 * vq_loss + 0.25 * commitment_loss, self.get_token_id(tensor_indices), token_to_quantize
    

    def get_token_id(self, codebook_tensor_indices: torch.Tensor) -> torch.Tensor:
        return self.token_id[codebook_tensor_indices]


    def get_token(self, codebook_tensor_indices: np.ndarray) -> list[str]:
        # return [self.vocabulary[idx] for idx in codebook_tensor_indices]
        return self.vocabulary[codebook_tensor_indices].tolist()
    
    def forward_explain(self, z_e):
        if hasattr(self, 'codebook_mapping'):
            mapped_codebook = self.codebook_mapping(self.codebook_tensor_pca)
        else:
            mapped_codebook = self.codebook_tensor_pca

        z_q = torch.zeros_like(z_e, device=z_e.device)
        tensor_indices = torch.empty(z_e.shape[0], 1, device=z_e.device, dtype=torch.long)
        token_to_quantize = torch.empty(z_e.shape[0], 1, z_e.shape[1], device=z_e.device, dtype=torch.float32)

        # # RQ
        for i in range(self.layer_num):
            residual = z_e - z_q
            dist_matrix = torch.sum(residual**2, dim=1, keepdim=True) + \
                torch.sum(mapped_codebook**2, dim=1) - 2 * \
                torch.matmul(residual, mapped_codebook.t())
            dist_matrix = dist_matrix.reshape(-1, self.word_num, dist_matrix.shape[-1]) 
            batch_indices = torch.arange(dist_matrix.shape[0], device=dist_matrix.device)

            layer_min_dist_indices = torch.empty(dist_matrix.shape[0], self.word_num, device=dist_matrix.device, dtype=torch.long)
            
            for j in range(self.word_num):
                word_j_min_dist_indices = torch.argmin(dist_matrix[:,j,:], dim=1)
                dist_matrix[batch_indices, : , word_j_min_dist_indices] = float('inf') # set the min distance to inf
                layer_min_dist_indices[:, j] = word_j_min_dist_indices

            layer_min_dist_indices = layer_min_dist_indices.reshape(-1)

            z_q += mapped_codebook[layer_min_dist_indices]

            if i == 0:
                tensor_indices[:, i] = layer_min_dist_indices
                token_to_quantize[:, i, :] = residual

        return z_q, self.get_token(tensor_indices.reshape(-1).cpu().numpy()), self.get_token_id(tensor_indices), token_to_quantize
    
    def get_collaborative_representations(self, token_to_quantize, token_id):
        token_to_quantize = token_to_quantize.reshape(-1, token_to_quantize.shape[-1]) # batch_size*word_num*layer_num, word_dim
        words_embedding = self.reverse_codebook_mapping(token_to_quantize) # batch_size*word_num*layer_num, token_dim
        # if words_embedding.requires_grad:
        #     words_embedding.register_hook(lambda grad: print("Gradient:", grad)) # register hook to print gradient
        words_embedding = words_embedding.reshape(-1, self.word_num, 1, words_embedding.shape[-1]) # batch_size, word_num,layer_num, token_dim

        words_embedding_2 = self.embedding_model.get_token_embedding_matrix()(token_id) # batch_size, word_num,layer_num token_dim
        words_embedding = words_embedding + (words_embedding_2 - words_embedding).detach() # batch_size, word_num,layer_num, token_dim

        words_embedding = words_embedding.reshape(-1, self.word_num * 1, words_embedding.shape[-1]) # batch_size, word_num*layer_num, token_dim

        words_embedding_comma = torch.zeros(words_embedding.shape[0], words_embedding.shape[1] * 2 - 1, words_embedding.shape[2], device=words_embedding.device)
        words_embedding_comma[:,::2,:] = words_embedding
        words_embedding_comma[:,1::2,:] = self.prompt_comma.squeeze()

        batch_prompt_embedding = self.prompt_embedding.repeat(words_embedding_comma.shape[0] // 3, 1, 1)
        combined_embedding = torch.cat([batch_prompt_embedding, words_embedding_comma], dim=1)

        combined_embedding = self.embedding_model.add_special_tokens_for_embeddings(combined_embedding)
        collaborative_representations = self.embedding_model.encode_embeddings(token_embeddings=combined_embedding)

        # attention_mask = torch.ones(combined_embedding.shape[0], combined_embedding.shape[1], dtype=torch.long, device=combined_embedding.device)
        # collaborative_representations = self.model.encode(features={"inputs_embeds": combined_embedding, "attention_mask": attention_mask})

        return collaborative_representations
    
        
class LinearLayer_2(nn.Module):
    def __init__(self, input_dim, mlp_num, output_dim):
        super().__init__()
        #    ：input_dim -> hidden_dim
        self.hidden_dim = output_dim  #              
        self.linear1 = nn.ModuleList([nn.Linear(input_dim, self.hidden_dim) for _ in range(mlp_num)])
        #    ：hidden_dim -> output_dim
        self.linear2 = nn.ModuleList([nn.Linear(self.hidden_dim, output_dim) for _ in range(mlp_num)])
        # Reverse  ：  
        self.reverse_linear1 = nn.Linear(mlp_num * output_dim, mlp_num * self.hidden_dim)
        self.reverse_linear2 = nn.Linear(mlp_num * self.hidden_dim, input_dim)
        
    def forward(self, x):
        outputs = []
        for i in range(len(self.linear1)):
            hidden = F.relu(self.linear1[i](x))  #      
            out = self.linear2[i](hidden)       #    
            outputs.append(out)
        x = torch.stack(outputs, dim=1)  # Shape: (batch_size, mlp_num, output_dim)
        return x
    
    def reverse(self, x):
        x = x.reshape(x.shape[0], -1)  #   
        x = F.relu(self.reverse_linear1(x))  #      
        x = self.reverse_linear2(x)         #    
        return x

class LinearLayer(nn.Module):
    def __init__(self, input_dim, mlp_num, output_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(mlp_num, input_dim, output_dim))
        nn.init.orthogonal_(self.weights)
        self.bias = nn.Parameter(torch.randn(mlp_num, output_dim))

        self.reverse_linear = nn.Linear(mlp_num*output_dim, input_dim)
        
    def forward(self, x):
        x = torch.einsum('bi,cio->bco', x, self.weights)
        x = x + self.bias
        return x
    
    def compute_pinv_weights(self):
        weights_pinv = []
        for i in range(self.weights.size(0)):
            weight = self.weights[i].detach() 
            weight_pinv = torch.pinverse(weight)
            weights_pinv.append(weight_pinv)
        weights_pinv = torch.stack(weights_pinv, dim=0)
        return weights_pinv
    
    def reverse_pinv(self, x):
        weights_pinv = self.compute_pinv_weights()
        x = x - self.bias.detach()
        x = torch.einsum('bco,coi->bci', x, weights_pinv)
        x = x.mean(dim=1)
        return x
    
    def reverse(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.reverse_linear(x)
        return x
    

class FACE(nn.Module):
    def __init__(self, input_dim, word_num, word_dim, dataset_name, llm_name):
        super().__init__()
        self.input_dim = input_dim
        self.word_num = word_num
        self.layer_num = 3
        self.word_dim = word_dim
        nhead = 1
        num_layers = 1

        self.linear_encoder = LinearLayer(input_dim=self.input_dim, mlp_num=self.word_num, output_dim=self.word_dim)
        self.transformer_layer = TransformerEncoderLayer(d_model=self.word_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.transformer_decoder = TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.quantizer = Quantizer(codebook_dim=self.word_dim, word_num = self.word_num,layer_num=self.layer_num, dataset_name=dataset_name, llm_name=llm_name)

    def forward(self, x):
        # Encoder
        linear_out = self.linear_encoder(x) # batch_size, word_num, word_dim

        # # Transformer
        z_e = self.transformer_encoder(linear_out) # batch_size, word_num, word_dim
        # z_e = linear_out

        # collaborative_representations = self.quantizer.get_collaborative_representations(z_e)
        collaborative_representations = None
        
        # VQ
        z_e_reshape = z_e.reshape(-1, self.word_dim) # batch_size*word_num, word_dim

        z_q_reshape, vq_loss, token_id_reshape, token_to_quantize_reshape = self.quantizer(z_e_reshape) # do vq

        z_q = z_q_reshape.reshape(x.shape[0], self.word_num, self.word_dim) # batch_size, word_num, word_dim
        token_id = token_id_reshape.reshape(x.shape[0], self.word_num, 1) # batch_size, word_num, self.layer_num
        token_to_quantize = token_to_quantize_reshape.reshape(x.shape[0], self.word_num, 1, token_to_quantize_reshape.shape[-1]) # batch_size, word_num, self.layer_num, token_dim

        collaborative_representations_2 = self.quantizer.get_collaborative_representations(token_to_quantize, token_id)
        # collaborative_representations_2 = None

        # Decoder
        trans_decoder_out = self.transformer_decoder(z_q) # batch_size, word_num, word_dim
        # trans_decoder_out = self.transformer_decoder(z_q.detach()) #    ，  vq   
        # trans_decoder_out = z_q

        decoded = self.linear_encoder.reverse(trans_decoder_out)

        recons_loss = F.mse_loss(decoded, x.detach())
        # recons_loss = F.mse_loss(decoded, x) #    ，     origin x   
        
        return decoded, vq_loss, recons_loss, collaborative_representations_2
    

    def forward_no_align(self, x):
        # Encoder
        linear_out = self.linear_encoder(x) # batch_size, word_num, word_dim

        # # Transformer
        z_e = self.transformer_encoder(linear_out) # batch_size, word_num, word_dim
        # z_e = linear_out

        # collaborative_representations = None
        
        # VQ
        z_e_reshape = z_e.reshape(-1, self.word_dim) # batch_size*word_num, word_dim

        z_q_reshape, vq_loss, _, _ = self.quantizer(z_e_reshape) # do vq

        z_q = z_q_reshape.reshape(x.shape[0], self.word_num, self.word_dim) # batch_size, word_num, word_dim

        collaborative_representations_2 = None

        # Decoder
        trans_decoder_out = self.transformer_decoder(z_q) # batch_size, word_num, word_dim
        # trans_decoder_out = self.transformer_decoder(z_q.detach()) #    ，  vq   
        # trans_decoder_out = z_q

        decoded = self.linear_encoder.reverse(trans_decoder_out)

        recons_loss = F.mse_loss(decoded, x.detach())
        # recons_loss = F.mse_loss(decoded, x) #    ，     origin x   
        
        return decoded, vq_loss, recons_loss, collaborative_representations_2
        
    def forward_explain(self, x):
        linear_out = self.linear_encoder(x)
        z_e = self.transformer_encoder(linear_out) # batch_size, word_num, word_dim
        # z_e = linear_out

        # collaborative_representations = self.quantizer.get_collaborative_representations(z_e)
        # collaborative_representations = None

        z_e_reshape = z_e.reshape(-1, self.word_dim) # batch_size*word_num, word_dim
        z_q_reshape, explain_words, token_id_reshape, token_to_quantize_reshape = self.quantizer.forward_explain(z_e_reshape) # get the words
        explain_words = [explain_words[i:i + self.word_num * 1] for i in range(0, len(explain_words), self.word_num * 1)]
        explain_words = [",".join(words) for words in explain_words]

        z_q = z_q_reshape.reshape(x.shape[0], self.word_num, self.word_dim) # batch_size, word_num, word_dim
        token_id = token_id_reshape.reshape(x.shape[0], self.word_num, 1) # batch_size, word_num, self.layer_num
        token_to_quantize = token_to_quantize_reshape.reshape(x.shape[0], self.word_num, 1, token_to_quantize_reshape.shape[-1]) # batch_size, word_num, self.layer_num, token_dim

        # get the collaborative representations (after)
        collaborative_representations_2 = self.quantizer.get_collaborative_representations(token_to_quantize, token_id)
        # collaborative_representations_2 = None

        print("==========VQ(first entity)==========")
        print("**Before" ,z_e[0].norm(p=2,dim=1).detach().cpu().numpy())
        print(z_e[0][0][:5].detach().cpu().numpy())
        print("**After", z_q[0].norm(p=2,dim=1).detach().cpu().numpy())
        print(z_q[0][0][:5].detach().cpu().numpy())
        print("**Distance", F.pairwise_distance(z_e[0], z_q[0], p=2).detach().cpu().numpy())
        print(F.mse_loss(z_e, z_q))

        trans_decoder_out = self.transformer_decoder(z_q) # batch_size, word_num, word_dim
        # trans_decoder_out = z_q

        decoded = self.linear_encoder.reverse(trans_decoder_out)
        print("==========Reconstruction==========")
        print("**Init" ,x.norm(p=2,dim=1).detach().cpu().numpy())
        print(x[0][:5].detach().cpu().numpy())
        print("**Reconstruction", decoded.norm(p=2,dim=1).detach().cpu().numpy())
        print(decoded[0][:5].detach().cpu().numpy())
        print("**Distance", F.pairwise_distance(x, decoded, p=2).detach().cpu().numpy())
        print(F.mse_loss(x, decoded))

        return explain_words, collaborative_representations_2
    
    def forward_no_align_explain(self, x):
        linear_out = self.linear_encoder(x)
        z_e = self.transformer_encoder(linear_out) # batch_size, word_num, word_dim
        # z_e = linear_out

        collaborative_representations = None

        z_e_reshape = z_e.reshape(-1, self.word_dim) # batch_size*word_num, word_dim
        z_q_reshape, explain_words, _, _ = self.quantizer.forward_explain(z_e_reshape) # get the words
        explain_words = [explain_words[i:i + self.word_num * 1] for i in range(0, len(explain_words), self.word_num * 1)]
        explain_words = [",".join(words) for words in explain_words]

        z_q = z_q_reshape.reshape(x.shape[0], self.word_num, self.word_dim) # batch_size, word_num, word_dim

        # get the collaborative representations (after)
        collaborative_representations_2 = None

        print("==========VQ(first entity)==========")
        print("**Before" ,z_e[0].norm(p=2,dim=1).detach().cpu().numpy())
        print(z_e[0][0][:5].detach().cpu().numpy())
        print("**After", z_q[0].norm(p=2,dim=1).detach().cpu().numpy())
        print(z_q[0][0][:5].detach().cpu().numpy())
        print("**Distance", F.pairwise_distance(z_e[0], z_q[0], p=2).detach().cpu().numpy())
        print(F.mse_loss(z_e, z_q))

        trans_decoder_out = self.transformer_decoder(z_q) # batch_size, word_num, word_dim
        # trans_decoder_out = z_q

        decoded = self.linear_encoder.reverse(trans_decoder_out)
        print("==========Reconstruction==========")
        print("**Init" ,x.norm(p=2,dim=1).detach().cpu().numpy())
        print(x[0][:5].detach().cpu().numpy())
        print("**Reconstruction", decoded.norm(p=2,dim=1).detach().cpu().numpy())
        print(decoded[0][:5].detach().cpu().numpy())
        print("**Distance", F.pairwise_distance(x, decoded, p=2).detach().cpu().numpy())
        print(F.mse_loss(x, decoded))

        return explain_words, collaborative_representations_2

    
if __name__ == "__main__":
    # Example usage

    linear_layer = LinearLayer(input_dim=256, mlp_num=1, output_dim=384)
    optimizer = torch.optim.Adam(linear_layer.parameters(), lr=0.01)
    data = torch.randn(32, 256)
    for i in range(100):
        x = linear_layer(data)
        data_recover = linear_layer.reverse_pinv(x)

        print(data[:5], data_recover[:5], F.mse_loss(data, data_recover))

        loss = torch.sum(x * torch.randn_like(x))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

        
    