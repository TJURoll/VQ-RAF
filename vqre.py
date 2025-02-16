import re
import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# from sklearn.decomposition import PCA
from torch_pca import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def is_english_word(word):
    return bool(re.fullmatch(r"[a-zA-Z]+", word))

class CodeBook(nn.Module):
    def __init__(self, codebook_dim = 512, word_num = 10, pca_dim = 64,
                 llm_name="../LLMs/llama2-embedding"):
        super().__init__()
        self.word_num = word_num
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        model = AutoModel.from_pretrained(llm_name, trust_remote_code = True)

        for param in model.parameters():
            param.requires_grad = False

        self.tokenizer = tokenizer
        self.model = model
        self.llm_batch_size = 1024

        tokens_df = self._get_vocabulary(tokenizer) # two columns: token_id, token
        tokens_df.to_csv('./data/vocabulary/vocabulary.csv', index=False)

        self.token_id: torch.tensor = torch.tensor(tokens_df['token_id'].values)
        self.vocabulary: list[str] = tokens_df['token'].tolist()
        input_embeddings = model.get_input_embeddings()
        codebook_tensor = input_embeddings(self.token_id).detach() # no grad

        pca_dim = codebook_tensor.shape[-1]

        self.register_buffer('codebook_tensor_pca', codebook_tensor)

        # construct mapping if needed
        self.codebook_mlp = nn.Linear(pca_dim, codebook_dim)

        # prompt
        # prompt = "The user/item is described by the following words: "
        # ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        # prompt_embedding = model.get_input_embeddings()(ids)
        # self.register_buffer('prompt_embedding', prompt_embedding)

        prompt_usr = "The user and his likes can be described as the following words:"
        prompt_itm = "The restaurant attracts those who can be described as the following words:"
        ids_usr = tokenizer(prompt_usr, return_tensors="pt")["input_ids"]
        ids_itm = tokenizer(prompt_itm, return_tensors="pt")["input_ids"]
        prompt_embedding_usr = model.get_input_embeddings()(ids_usr).detach()
        prompt_embedding_itm = model.get_input_embeddings()(ids_itm).detach()
        prompt_embedding = torch.cat([prompt_embedding_usr, prompt_embedding_itm, prompt_embedding_itm], dim=0)

        ids_comma = tokenizer(",", return_tensors="pt")["input_ids"][:,1:]
        prompt_comma = model.get_input_embeddings()(ids_comma).detach()

        self.register_buffer('prompt_embedding', prompt_embedding)
        self.register_buffer('prompt_comma', prompt_comma)

    def _coca60000_vocabulary(self, path = './data/vocabulary/word_frequency_list_60000_English.xlsx', voc_word_num = 20000, pos = ['N','J']):
        coca_df = pd.read_excel(path)
        coca_df = coca_df[['PoS','word']]
        coca_df = coca_df.map(lambda x: x.strip() if type(x) == str else x)
        coca_df = coca_df[coca_df['word'].apply(lambda x: len(x) > 0)] # remove the words with length 0
        coca_df['word'] = coca_df['word'].apply(lambda x: x[1:-1] if x[0] == '(' and x[-1] == ')' else x) # remove the brackets of the word
        coca_df.drop_duplicates(subset=['word'], inplace=True, keep='first') # keep the most frequent PoS of each word
        coca_df = coca_df[coca_df['PoS'].isin(pos)] # only keep the words with pos in pos
        coca_df = coca_df.iloc[:voc_word_num] # only keep the first word_num words
        return coca_df['word'].tolist()
    
    def _profile_vocabulary(self, path = "./data/yelp"):
        usrprf_path = os.path.join(path, 'usr_prf.pkl')
        itmprf_path = os.path.join(path, 'itm_prf.pkl')
        with open(usrprf_path, 'rb') as f:
            usrprf = pickle.load(f)
            usrprf = [v['profile'] for v in usrprf.values()]
            
        with open(itmprf_path, 'rb') as f:
            itmprf = pickle.load(f)
            itmprf = [v['profile'] for v in itmprf.values()]

        prfs = usrprf + itmprf
        prfs = "\n".join(prfs)

        prfs = prfs.lower()

        prfs_split = re.split(r'[^a-zA-Z]+', prfs)
        word_freq = Counter(prfs_split)
        common_words = [word for word, freq in word_freq.items() if freq > 100]
        return common_words

    
    def _get_vocabulary(self, tokenizer):
        vocal_dict = tokenizer.get_vocab()
        tokens = vocal_dict.keys()
        tokens_list = list(tokens)

        tokens_df = pd.DataFrame()
        tokens_df['token'] = tokens_list
        # the id of token
        tokens_df['token_id'] = tokens_df['token'].apply(tokenizer.convert_tokens_to_ids)
        # remove the non-suffix token
        tokens_df = tokens_df[tokens_df['token'].apply(lambda x: x[0] == '▁')]
        # to string
        tokens_df['token'] = tokens_df['token'].apply(lambda x: tokenizer.convert_tokens_to_string([x]))
        # keep the English words
        tokens_df = tokens_df[tokens_df['token'].apply(is_english_word)]
        # remove the words with capital letters
        tokens_df = tokens_df[~tokens_df['token'].str.contains('[A-Z]')]
        # load the vocabulary of COCA60000
        vocabulary_coca60000 = self._coca60000_vocabulary()
        vocabulary_profile = self._profile_vocabulary()

        # find the intersection of the vocabulary of COCA60000 and the tokens
        tokens_df = tokens_df[tokens_df['token'].isin(vocabulary_coca60000)]
        tokens_df = tokens_df[tokens_df['token'].isin(vocabulary_profile)]

        # sort the tokens by token_id
        tokens_df.sort_values(by='token_id', inplace=True)
        return tokens_df
    
    def reverse_codebook_mapping(self, x):
        reverse_weight = torch.pinverse(self.codebook_mlp.weight.detach()).t()
        x = x - self.codebook_mlp.bias.detach()
        x = torch.matmul(x, reverse_weight)
        return x
    
    def forward(self, z_e):
        if hasattr(self, 'codebook_mlp'):
            mapped_codebook = self.codebook_mlp(self.codebook_tensor_pca)
        else:
            mapped_codebook = self.codebook_tensor_pca

        dist_matrix = torch.sum(z_e**2, dim=1, keepdim=True) + \
                        torch.sum(mapped_codebook**2, dim=1) - 2 * \
                        torch.matmul(z_e, mapped_codebook.t())
        
        dist_matrix = dist_matrix.reshape(-1, self.word_num, dist_matrix.shape[-1])
        batch_indices = torch.arange(dist_matrix.shape[0], device=dist_matrix.device).unsqueeze(1)
        
        min_dist_indices = None
        for i in range(self.word_num):
            word_i_min_dist_indices = torch.argmin(dist_matrix[:,i,:], dim=1, keepdim=True)
            dist_matrix[batch_indices, :, word_i_min_dist_indices] = float('inf')
            if min_dist_indices is None:
                min_dist_indices = word_i_min_dist_indices
            else:
                min_dist_indices = torch.cat([min_dist_indices, word_i_min_dist_indices], dim=1)
        min_dist_indices = min_dist_indices.reshape(-1)

        z_q = mapped_codebook[min_dist_indices]

        # straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach() 

        # ||e-sg[z_e]||^2
        vq_loss = torch.mean((z_q - z_e.detach())**2)
        # ||z_e-sg[e]||^2
        commitment_loss = torch.mean((z_e - z_q.detach())**2)

        return z_q_st, 0.75 * vq_loss + 0.25 * commitment_loss
    

    def explain(self, codebook_tensor_indices: list[int]):
        return [self.vocabulary[idx] for idx in codebook_tensor_indices]
    
    def forward_explain(self, z_e):
        if hasattr(self, 'codebook_mlp'):
            mapped_codebook = self.codebook_mlp(self.codebook_tensor_pca)
        else:
            mapped_codebook = self.codebook_tensor_pca

        dist_matrix = torch.sum(z_e**2, dim=1, keepdim=True) + \
                        torch.sum(mapped_codebook**2, dim=1) - 2 * \
                        torch.matmul(z_e, mapped_codebook.t())

        min_dist, min_dist_indices = torch.min(dist_matrix, dim=1)
        z_q = mapped_codebook[min_dist_indices]
        return z_q, self.explain(min_dist_indices.tolist())
    
    def get_collaborative_representations(self, z_e):
        z_e = z_e.reshape(-1, z_e.shape[-1])
        words_embedding = self.reverse_codebook_mapping(z_e)
        words_embedding = words_embedding.reshape(-1, self.word_num, words_embedding.shape[-1])

        words_embedding_comma = torch.zeros(words_embedding.shape[0], words_embedding.shape[1] * 2 - 1, words_embedding.shape[2], device=words_embedding.device)
        words_embedding_comma[:,::2,:] = words_embedding
        words_embedding_comma[:,1::2,:] = self.prompt_comma.squeeze()


        batch_prompt_embedding = torch.cat([self.prompt_embedding] * (words_embedding_comma.shape[0] // 3), dim=0)
        combined_embedding = torch.cat([batch_prompt_embedding, words_embedding_comma], dim=1)

        attention_mask = torch.ones(combined_embedding.shape[0], combined_embedding.shape[1], dtype=torch.long, device=combined_embedding.device)
        collaborative_representations = self.model.encode(features={"inputs_embeds": combined_embedding, "attention_mask": attention_mask})

        return collaborative_representations
    
        
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
    

class VQRE(nn.Module):
    def __init__(self, input_dim, word_num, word_dim):
        super().__init__()
        self.input_dim = input_dim
        self.word_num = word_num
        self.word_dim = word_dim
        nhead = 1
        num_layers = 1

        self.linear_encoder = LinearLayer(input_dim=self.input_dim, mlp_num=self.word_num, output_dim=self.word_dim)
        self.transformer_layer = TransformerEncoderLayer(d_model=self.word_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.transformer_decoder = TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.codebook = CodeBook(codebook_dim=self.word_dim, word_num = self.word_num, pca_dim=-1)

    def forward(self, x):
        # Encoder
        linear_out = self.linear_encoder(x) # batch_size, word_num, word_dim

        # # Transformer
        z_e = self.transformer_encoder(linear_out) # batch_size, word_num, word_dim
        # z_e = linear_out

        collaborative_representations = self.codebook.get_collaborative_representations(z_e)
        
        # VQ
        z_e_reshape = z_e.reshape(-1, self.word_dim) # batch_size*word_num, word_dim

        z_q_reshape, vq_loss = self.codebook(z_e_reshape) # do vq

        z_q = z_q_reshape.reshape(x.shape[0], self.word_num, self.word_dim) # batch_size, word_num, word_dim

        # Decoder
        trans_decoder_out = self.transformer_decoder(z_q) # batch_size, word_num, word_dim
        # trans_decoder_out = z_q

        decoded = self.linear_encoder.reverse(trans_decoder_out)

        recons_loss = F.mse_loss(decoded, x.detach())
        
        return decoded, vq_loss, recons_loss, collaborative_representations
    
    def forward_reconstruction(self, x):
        linear_out = self.linear_encoder(x)
        z_e = self.transformer_encoder(linear_out) # batch_size, word_num, word_dim
        z_e_reshape = z_e.reshape(-1, self.word_dim) # batch_size*word_num, word_dim
        z_q_reshape, vq_loss = self.codebook(z_e_reshape) # do vq
        z_q = z_q_reshape.reshape(x.shape[0], self.word_num, self.word_dim) # batch_size, word_num, word_dim
        trans_decoder_out = self.transformer_decoder(z_q) # batch_size, word_num, word_dim
        decoded = self.linear_encoder.reverse(trans_decoder_out)
        return decoded
    

    def forward_explain(self, x):
        linear_out = self.linear_encoder(x)
        z_e = self.transformer_encoder(linear_out) # batch_size, word_num, word_dim

        collaborative_representations = self.codebook.get_collaborative_representations(z_e)

        z_e_reshape = z_e.reshape(-1, self.word_dim) # batch_size*word_num, word_dim
        z_q_reshape, explain_words = self.codebook.forward_explain(z_e_reshape) # get the words
        explain_words = [explain_words[i:i + self.word_num] for i in range(0, len(explain_words), self.word_num)]
        explain_words = [" ".join(words) for words in explain_words]

        z_q = z_q_reshape.reshape(x.shape[0], self.word_num, self.word_dim) # batch_size, word_num, word_dim
        print("==========VQ(first entity)==========")
        print("**Before" ,z_e[0].norm(p=2,dim=1).detach().cpu().numpy())
        print(z_e[0][0][:5].detach().cpu().numpy())
        print("**After", z_q[0].norm(p=2,dim=1).detach().cpu().numpy())
        print(z_q[0][0][:5].detach().cpu().numpy())
        print("**Distance", F.pairwise_distance(z_e[0], z_q[0], p=2).detach().cpu().numpy())
        print(F.mse_loss(z_e, z_q))

        trans_decoder_out = self.transformer_decoder(z_q) # batch_size, word_num, word_dim
        decoded = self.linear_encoder.reverse(trans_decoder_out)
        print("==========Reconstruction==========")
        print("**Init" ,x.norm(p=2,dim=1).detach().cpu().numpy())
        print(x[0][:5].detach().cpu().numpy())
        print("**Reconstruction", decoded.norm(p=2,dim=1).detach().cpu().numpy())
        print(decoded[0][:5].detach().cpu().numpy())
        print("**Distance", F.pairwise_distance(x, decoded, p=2).detach().cpu().numpy())
        print(F.mse_loss(x, decoded))

        return explain_words, collaborative_representations

    
            
        
    