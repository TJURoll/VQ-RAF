import os
import time
import wandb
import random
import numpy as np
from numpy import random
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.optim as optim
from trainer.metrics import Metric
from models.bulid_model import build_model
from config.configurator import configs


def init_seed():
    if 'reproducible' in configs['train']:
        if configs['train']['reproducible']:
            seed = configs['train']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, data_handler, logger):
        self.data_handler = data_handler
        self.logger = logger
        self.metric = Metric()

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            if "mode" in optim_config and optim_config['mode'] == "finetune":
                self.optimizer = optim.Adam(# model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
                [
                    {"params": model.vqraf.parameters(), "lr": optim_config['lr']},
                    {"params": list(set(model.parameters()) - set(model.vqraf.parameters())),
                     "lr": optim_config['lr'] / 10, "weight_decay": 2 * model.hyper_config['reg_weight']},
                ])
            else:
                self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay= 2 * model.hyper_config['reg_weight'])
                

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        # start this epoch
        model.train()
        for i, tem in tqdm(enumerate(train_dataloader), desc=f'[Epoch {epoch_idx}]', total=len(train_dataloader)):
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))

            self.optimizer.zero_grad()

            loss, loss_dict = model.cal_loss(batch_data)
            ep_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            wandb.log({f'Batch/{key}': value for key, value in loss_dict.items()})

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

            # if i == 0 and configs['model']['name'][-2:] == 'vq':
            #     model.eval()
            #     print("\033[43mStart to explain\033[0m")
            #     with torch.no_grad():
            #         model.get_explanation(batch_data)
            #     model.train()
            #     print("\033[43mFinish explaining\033[0m")
            #     print()

        wandb.log({f'Epoch/{key}': value for key, value in loss_log_dict.items()})


    def train(self, model):
        now_patience = 0
        best_epoch = 0
        best_recall = -1e9
        self.create_optimizer(model)
        train_config = configs['train']
        for epoch_idx in range(train_config['epoch']):
            # train
            self.train_epoch(model, epoch_idx)
            # evaluate
            if epoch_idx % train_config['test_step'] == 0:
                eval_result = self.evaluate(model)

                if eval_result['recall'][-1] > best_recall:
                    now_patience = 0
                    best_epoch = epoch_idx
                    best_recall = eval_result['recall'][-1]
                    best_state_dict = deepcopy(model.state_dict())
                else:
                    now_patience += 1

                # early stop
                if now_patience == configs['train']['patience']:
                    break

        # evaluation again
        model = build_model(self.data_handler).to(configs['device'])
        model.load_state_dict(best_state_dict)
        self.evaluate(model)

        # final test
        model = build_model(self.data_handler).to(configs['device'])
        model.load_state_dict(best_state_dict)
        test_result = self.test(model)

        # save result
        self.save_model(model)
        print("Best Epoch {}. Final test result: {}.".format(best_epoch, test_result))

    def evaluate(self, model):
        model.eval()
        eval_result = self.metric.eval(model, self.data_handler.valid_dataloader)
        self.logger.log_eval(eval_result, configs['test']['k'], data_type='Valid')
        return eval_result


    def test(self, model):
        model.eval()
        eval_result = self.metric.eval(model, self.data_handler.test_dataloader)
        self.logger.log_eval(eval_result, configs['test']['k'], data_type='Valid')
        return eval_result
    
    def save_model(self, model):
        if configs['train']['save_model']:
            model_state_dict = model.state_dict()
            model_name = configs['model']['name']
            save_dir_path = './encoder/checkpoint/{}'.format(model_name)
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)
            torch.save(model_state_dict, '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, configs['data']['name'], configs['train']['seed']))
            print("Save model parameters to {}".format('{}/{}-{}-{}.pth'.format(save_dir_path, model_name, configs['data']['name'], configs['train']['seed'])))

    def load_model(self, model):
        if 'pretrain_path' in configs['train']:
            pretrain_path = configs['train']['pretrain_path']
            model.load_state_dict(torch.load(pretrain_path))
            print(
                "Load model parameters from {}".format(pretrain_path))
            
    # def explain(self, model):
    #     model.eval()
    #     train_dataloader = self.data_handler.train_dataloader
    #     train_dataloader.dataset.sample_negs()

    #     eval_result = self.evaluate(model)
    #     self.logger.log_eval(eval_result, configs['test']['k'], data_type='Validation')

    #     for _, tem in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
    #         batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
    #         model.get_explanation(batch_data)


class AutoCFTrainer(Trainer):
    def __init__(self, data_handler, logger):
        super(AutoCFTrainer, self).__init__(data_handler, logger)
        self.fix_steps = configs['model']['fix_steps']

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        # start this epoch
        model.train()
        for i, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))

            if i % self.fix_steps == 0:
                sampScores, seeds = model.sample_subgraphs()
                encoderAdj, decoderAdj = model.mask_subgraphs(seeds)

            loss, loss_dict = model.cal_loss(batch_data, encoderAdj, decoderAdj)

            if i % self.fix_steps == 0:
                localGlobalLoss = -sampScores.mean()
                loss += localGlobalLoss
                loss_dict['infomax_loss'] = localGlobalLoss

            ep_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        # writer.add_scalar('Loss/train', ep_loss / steps, epoch_idx)
        wandb.log({f'Epoch/{key}': value for key, value in loss_log_dict.items()})



