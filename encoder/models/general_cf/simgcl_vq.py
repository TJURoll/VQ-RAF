import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.general_cf.lightgcn_vq import LightGCN_vq
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss, cal_align_loss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class SimGCL_vq(LightGCN_vq):
    def __init__(self, data_handler):
        super(SimGCL_vq, self).__init__(data_handler)

        # hyper-parameter
        self.cl_weight = self.hyper_config['cl_weight']
        self.temperature = self.hyper_config['temperature']
        self.eps = self.hyper_config['eps']

    def _perturb_embedding(self, embeds):
        noise = (F.normalize(t.rand(embeds.shape).cuda(), p=2) * t.sign(embeds)) * self.eps
        return embeds + noise
    
    def forward(self, adj=None, perturb=False):
        if adj is None:
            adj = self.adj
        if not perturb:
            return super(SimGCL_vq, self).forward(adj, 1.0)
        embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
        embeds_list = [embeds]
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds = self._perturb_embedding(embeds)
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        return embeds[:self.user_num], embeds[self.user_num:]
    
    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds
        
    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds1, item_embeds1 = self.forward(self.adj, perturb=True)
        user_embeds2, item_embeds2 = self.forward(self.adj, perturb=True)
        user_embeds3, item_embeds3 = self.forward(self.adj, perturb=False)

        anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)

        # do vq
        entity_embeds = t.cat([anc_embeds3, pos_embeds3, neg_embeds3], dim=0)
        if "load_model" in configs['optimizer']:
            entity_embeds_vq, vq_loss, recons_loss, colla_repre = self.face.forward_no_align(entity_embeds)
        else:
            entity_embeds_vq, vq_loss, recons_loss, colla_repre = self.face(entity_embeds)

        # get the semantic representations
        ancprf_repre, posprf_repre, negprf_repre = self._pick_embeds(self.usrprf_repre, self.itmprf_repre, batch_data)
        semantic_repre = t.cat([ancprf_repre, posprf_repre, negprf_repre], dim=0)

        if configs['optimizer']['use_recons']:
            anc_embeds_vq, pos_embeds_vq, neg_embeds_vq = t.split(entity_embeds_vq, [anc_embeds3.shape[0], pos_embeds3.shape[0], neg_embeds3.shape[0]], dim=0)
            bpr_loss = cal_bpr_loss(anc_embeds_vq, pos_embeds_vq, neg_embeds_vq) / anc_embeds3.shape[0]
        else:
            bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0] 
        cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.temperature) + cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.temperature)
        cl_loss /= anc_embeds1.shape[0]
        align_loss = cal_align_loss(colla_repre, semantic_repre) 

        loss = bpr_loss + self.cl_weight * cl_loss + self.vq_weight * vq_loss + self.recons_weight * recons_loss + self.align_weight * align_loss
        losses = {'bpr_loss': bpr_loss, 'cl_loss': cl_loss, 'vq_loss': vq_loss, 'recons_loss': recons_loss, 'align_loss': align_loss}
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
        user_embeds, item_embeds = self.forward(self.adj, False)
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
        user_embeds, item_embeds = self.forward(self.adj, False)
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
    