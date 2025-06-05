import pickle
from face import FACE
import torch as t
from torch import nn
from torch.nn import functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss, cal_align_loss
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class LightGCN_plus_vq(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN_plus_vq, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.keep_rate = configs['model']['keep_rate']
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        self.usrprf_repre = t.tensor(configs['usrprf_repre']).float().cuda()
        self.itmprf_repre = t.tensor(configs['itmprf_repre']).float().cuda()

        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']

        # semantic-embeddings
        self.usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        self.mlp = nn.Sequential(
            nn.Linear(self.usrprf_embeds.shape[1], (self.usrprf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrprf_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )

        self._init_weight()

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
            
    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def _load_parameters(self, path):
        params = t.load(path)
        self.user_embeds = nn.Parameter(params['user_embeds'])
        self.item_embeds = nn.Parameter(params['item_embeds'])
    
    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)
    
    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:]

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
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)

        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, batch_data)

        # do vq
        entity_embeds = t.cat([anc_embeds, pos_embeds, neg_embeds], dim=0)
        if "load_model" in configs['optimizer']:
            entity_embeds_vq, vq_loss, recons_loss, colla_repre = self.face.forward_no_align(entity_embeds)
        else:
            entity_embeds_vq, vq_loss, recons_loss, colla_repre = self.face(entity_embeds)

        # get the semantic representations
        ancprf_repre, posprf_repre, negprf_repre = self._pick_embeds(self.usrprf_repre, self.itmprf_repre, batch_data)
        semantic_repre = t.cat([ancprf_repre, posprf_repre, negprf_repre], dim=0)
        if configs['optimizer']['use_recons']:
            anc_embeds_vq, pos_embeds_vq, neg_embeds_vq = t.split(entity_embeds_vq, [anc_embeds.shape[0], pos_embeds.shape[0], neg_embeds.shape[0]], dim=0)
            bpr_loss = cal_bpr_loss(anc_embeds_vq, pos_embeds_vq, neg_embeds_vq) / anc_embeds.shape[0]
        else:
            bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0] 

        usrprf_embeds = self.mlp(self.usrprf_embeds)
        itmprf_embeds = self.mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)

        kd_loss = cal_infonce_loss(anc_embeds, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(neg_embeds, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss /= anc_embeds.shape[0]
        align_loss = cal_align_loss(colla_repre, semantic_repre) 

        loss = bpr_loss + self.kd_weight * kd_loss + self.vq_weight * vq_loss + self.recons_weight * recons_loss + self.align_weight * align_loss
        losses = {'bpr_loss': bpr_loss, 'kd_loss': kd_loss, 'align_loss': align_loss, 'vq_loss': vq_loss, 'recons_loss': recons_loss}
        return loss, losses

    def get_explanation(self, batch_data, indices = [0, 1, 4, 9, 16, 25, 36, 49], save = False):
        self.is_training = False
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)

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
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
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
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
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
    
    