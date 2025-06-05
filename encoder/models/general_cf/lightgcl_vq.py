import torch as t
import numpy as np
from torch import nn
from models.base_model import BaseModel
from config.configurator import configs
from models.aug_utils import SvdDecomposition
from models.loss_utils import cal_bpr_loss, reg_params, cal_align_loss
from face import FACE

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class LightGCL_vq(BaseModel):
    def __init__(self, data_handler):
        super(LightGCL_vq, self).__init__(data_handler)

        train_mat = data_handler._load_one_mat(data_handler.trn_file)
        rowD = np.array(train_mat.sum(1)).squeeze()
        colD = np.array(train_mat.sum(0)).squeeze()
        for i in range(len(train_mat.data)):
            train_mat.data[i] = train_mat.data[i] / pow(rowD[train_mat.row[i]] * colD[train_mat.col[i]], 0.5)
        adj_norm = self._scipy_sparse_mat_to_torch_sparse_tensor(train_mat)
        self.adj = adj_norm.coalesce().cuda()

        self.svd_decompose = SvdDecomposition(svd_q=configs['model']['svd_q'])
        self.ut, self.vt, self.u_mul_s, self.v_mul_s = self.svd_decompose(self.adj)

        self.temp = configs['model']['temp']
        self.dropout = configs['model']['dropout']
        self.layer_num = configs['model']['layer_num']
        self.cl_weight = configs['model']['cl_weight']
        self.reg_weight = configs['model']['reg_weight']

        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

        self.act = nn.LeakyReLU(0.5)
        self.Ws = nn.ModuleList([W_contrastive(self.embedding_size) for i in range(self.layer_num)])
        self.is_training = True

        self.usrprf_repre = t.tensor(configs['usrprf_repre']).float().cuda()
        self.itmprf_repre = t.tensor(configs['itmprf_repre']).float().cuda()

        # vq
        self.word_num = self.hyper_config['word_num']
        self.word_dim = self.hyper_config['word_dim']
        self.vq_weight = self.hyper_config['vq_weight']
        self.recons_weight = self.hyper_config['recons_weight']
        self.align_weight = self.hyper_config['align_weight']
        self.face = FACE(input_dim=self.embedding_size, word_num=self.word_num, word_dim = self.word_dim, dataset_name = configs['data']['name'], llm_name=configs['llm'])

        if "load_model" in configs['optimizer']:
            model_name = configs['optimizer']["load_model"]
            save_dir_path = './encoder/checkpoint/{}'.format(model_name)
            self._load_parameters('{}/{}-{}-{}.pth'.format(save_dir_path, model_name, configs['data']['name'], configs['train']['seed']))
            print("Successfully load model from {}".format('{}/{}-{}-{}.pth'.format(save_dir_path, configs['optimizer']["load_model"], configs['data']['name'], configs['train']['seed'])))
        elif "load_all" in configs['optimizer']:
            model_name = configs['optimizer']["load_all"]
            save_dir_path = './encoder/checkpoint/{}'.format(model_name)
            self.load_state_dict(t.load('{}/{}-{}-{}_all.pth'.format(save_dir_path, model_name, configs['data']['name'], configs['train']['seed'])))
            print("Successfully load model from {}".format('{}/{}-{}-{}_all.pth'.format(save_dir_path, configs['optimizer']["load_all"], configs['data']['name'], configs['train']['seed'])))

        self.E_u_list = [None] * (self.layer_num+1)
        self.E_i_list = [None] * (self.layer_num+1)
        self.E_u_list[0] = self.user_embeds
        self.E_i_list[0] = self.item_embeds
        self.Z_u_list = [None] * (self.layer_num+1)
        self.Z_i_list = [None] * (self.layer_num+1)
        self.G_u_list = [None] * (self.layer_num+1)
        self.G_i_list = [None] * (self.layer_num+1)
        self.G_u_list[0] = self.user_embeds
        self.G_i_list[0] = self.item_embeds
        self.E_u = None
        self.E_i = None

    def _load_parameters(self, path):
        params = t.load(path)
        self.user_embeds = nn.Parameter(params['user_embeds'])
        self.item_embeds = nn.Parameter(params['item_embeds'])
        

    def _scipy_sparse_mat_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = t.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = t.from_numpy(sparse_mx.data)
        shape = t.Size(sparse_mx.shape)
        return t.sparse.FloatTensor(indices, values, shape)

    def _spmm(self,sp, emb):
        sp = sp.coalesce()
        cols = sp.indices()[1]
        rows = sp.indices()[0]
        col_segs = emb[cols] * t.unsqueeze(sp.values(),dim=1)
        result = t.zeros((sp.shape[0],emb.shape[1])).cuda()
        result.index_add_(0, rows, col_segs)
        return result

    def _sparse_dropout(self, mat, dropout):
        indices = mat.indices()
        values = nn.functional.dropout(mat.values(), p=dropout)
        size = mat.size()
        return t.sparse.FloatTensor(indices, values, size)

    def forward(self, test=False):
        if test and self.E_u is not None:
            return self.E_u, self.E_i
        for layer in range(1, self.layer_num+1):
            # GNN propagation
            self.Z_u_list[layer] = self._spmm(self._sparse_dropout(self.adj,self.dropout), self.E_i_list[layer-1])
            self.Z_i_list[layer] = self._spmm(self._sparse_dropout(self.adj,self.dropout).transpose(0,1), self.E_u_list[layer-1])

            # svd_adj propagation
            vt_ei = self.vt @ self.E_i_list[layer-1]
            self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
            ut_eu = self.ut @ self.E_u_list[layer-1]
            self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

            # aggregate
            self.E_u_list[layer] = self.Z_u_list[layer]  # + self.E_u_list[layer-1]
            self.E_i_list[layer] = self.Z_i_list[layer]  # + self.E_i_list[layer-1]

        # aggregate across layers
        self.G_u = sum(self.G_u_list)
        self.G_i = sum(self.G_i_list)
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)

        return self.E_u, self.E_i
    
    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds
    
    def _pick_prfs(self, user_prfs, item_prfs, batch_data):
        ancs, poss, negs = batch_data
        anc_prfs = [user_prfs[anc.item()] for anc in ancs]
        pos_prfs = [item_prfs[pos.item()] for pos in poss]
        neg_prfs = [item_prfs[neg.item()] for neg in negs]
        return anc_prfs, pos_prfs, neg_prfs


    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward()
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]

        # do vq
        entity_embeds = t.cat([anc_embeds, pos_embeds, neg_embeds], dim=0)
        if "load_model" in configs['optimizer']:
            entity_embeds_vq, vq_loss, recons_loss, colla_repre = self.face.forward_no_align(entity_embeds)
        else:
            entity_embeds_vq, vq_loss, recons_loss, colla_repre = self.face(entity_embeds)

        if configs['optimizer']['use_recons']:
            anc_embeds_vq, pos_embeds_vq, neg_embeds_vq = t.split(entity_embeds_vq, [anc_embeds.shape[0], pos_embeds.shape[0], neg_embeds.shape[0]], dim=0)
            bpr_loss = cal_bpr_loss(anc_embeds_vq, pos_embeds_vq, neg_embeds_vq) / anc_embeds.shape[0]
        else:
            bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0] 
        # get the semantic representations
        ancprf_repre, posprf_repre, negprf_repre = self._pick_embeds(self.usrprf_repre, self.itmprf_repre, batch_data)
        semantic_repre = t.cat([ancprf_repre, posprf_repre, negprf_repre], dim=0)
        align_loss = cal_align_loss(colla_repre, semantic_repre) 

        # pos_scores = (anc_embeds * pos_embeds).sum(-1)
        # neg_scores = (anc_embeds * neg_embeds).sum(-1)
        # bpr_loss = -(pos_scores - neg_scores).sigmoid().log().mean()

        G_u_norm = self.G_u
        E_u_norm = self.E_u
        G_i_norm = self.G_i
        E_i_norm = self.E_i
        neg_score = t.log(t.exp(G_u_norm[ancs] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += t.log(t.exp(G_i_norm[poss] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
        pos_score = (t.clamp((G_u_norm[ancs] * E_u_norm[ancs]).sum(1) / self.temp, -5.0, 5.0)).mean() + \
                    (t.clamp((G_i_norm[poss] * E_i_norm[poss]).sum(1) / self.temp, -5.0, 5.0)).mean()
        cl_loss = -pos_score + neg_score

        loss = bpr_loss + self.cl_weight * cl_loss + self.vq_weight * vq_loss + self.recons_weight * recons_loss + self.align_weight * align_loss
        losses = {'bpr_loss': bpr_loss, 'cl_loss': cl_loss, 'vq_loss': vq_loss, 'recons_loss': recons_loss, 'align_loss': align_loss}
        return loss, losses
    
    def get_explanation(self, batch_data, indices = [0, 1, 4, 9, 16, 25, 36, 49], save = False):
        self.is_training = False
        user_embeds, item_embeds = self.forward(test=True)

        selected_data = [entity_data[indices] for entity_data in batch_data]

        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, selected_data)

        entity_embeds = t.cat([anc_embeds, pos_embeds, neg_embeds], dim=1)
        entity_embeds = entity_embeds.reshape(-1, self.embedding_size)

        if "load_model" in configs['optimizer']:
            explain_words, colla_repre, colla_repre_2 = self.face.forward_no_align_explain(entity_embeds)
        else:
            explain_words, colla_repre, colla_repre_2 = self.face.forward_explain(entity_embeds)

        ancprf_repre, posprf_repre, negprf_repre = self._pick_embeds(self.usrprf_repre, self.itmprf_repre, selected_data)
        semantic_repre = t.cat([ancprf_repre, posprf_repre, negprf_repre], dim=1)
        semantic_repre = semantic_repre.reshape(-1, self.usrprf_repre.shape[1])


        if colla_repre is not None:
            sim_matrix = t.matmul(colla_repre, semantic_repre.T)
            sim_matrix_2 = t.matmul(colla_repre_2, semantic_repre.T)
            print("sim_matrix", sim_matrix[:10, :10])
            print("diag avg: ", t.diag(sim_matrix).mean().item())
            print("non-diag avg: ", ((sim_matrix.sum() - t.diag(sim_matrix).sum()) / (sim_matrix.shape[0] * sim_matrix.shape[1] - sim_matrix.shape[0])).item())

            print("sim_matrix_2", sim_matrix_2[:10, :10])
            print("diag avg: ", t.diag(sim_matrix_2).mean().item())
            print("non-diag avg: ", ((sim_matrix_2.sum() - t.diag(sim_matrix_2).sum()) / (sim_matrix_2.shape[0] * sim_matrix_2.shape[1] - sim_matrix_2.shape[0])).item())
        

            explain_words = [explain_words[i:i+3] for i in range(0, len(explain_words), 3)]
            anc_prfs, pos_prfs, neg_prfs = self._pick_prfs(configs['usrprf'], configs['itmprf'], selected_data)
            entity_prfs = list(zip(anc_prfs, pos_prfs, neg_prfs))
            for i in range(len(indices)):
                print("- USER: ", entity_prfs[i][0])
                print(explain_words[i][0], sim_matrix[i*3][i*3].item(), sim_matrix_2[i*3][i*3].item())
                print("- POS: ", entity_prfs[i][1])
                print(explain_words[i][1], sim_matrix[i*3+1][i*3+1].item(), sim_matrix_2[i*3+1][i*3+1].item())
                print("- NEG: ", entity_prfs[i][2])
                print(explain_words[i][2], sim_matrix[i*3+2][i*3+2].item(), sim_matrix_2[i*3+2][i*3+2].item())

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(test=True)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        if configs['optimizer']['use_recons']:
            entity_embeds = t.cat([pck_user_embeds, item_embeds], dim=0)
            entity_embeds_vq, _, _, _ = self.face.forward_no_align(entity_embeds)
            pck_user_embeds_vq, item_embeds_vq = t.split(entity_embeds_vq, [pck_user_embeds.shape[0], item_embeds.shape[0]], dim=0)
            full_preds = pck_user_embeds_vq @ item_embeds_vq.T
        else:
            full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
    
    def full_predict_2(self, batch_data):
        user_embeds, item_embeds = self.forward(test=True)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        entity_embeds = t.cat([pck_user_embeds, item_embeds], dim=0)
        entity_embeds_vq, _, _, _ = self.face.forward_no_align(entity_embeds)
        pck_user_embeds_vq, item_embeds_vq = t.split(entity_embeds_vq, [pck_user_embeds.shape[0], item_embeds.shape[0]], dim=0)
        full_preds = pck_user_embeds_vq @ item_embeds_vq.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
    


class W_contrastive(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_uniform_(t.empty(d,d)))

    def forward(self,x):
        return x @ self.W