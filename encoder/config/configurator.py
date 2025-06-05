import os
import yaml
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn

def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser(description='RLMRec')
    parser.add_argument('--model', type=str, default='lightgcn_vq', help='Model name')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Random Seed')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    parser.add_argument('--remark', type=str, default='Default', help='Remark for wandb')
    parser.add_argument('--llm', type=str, default='miniLM', help='LLM name', choices=['llama2', 'miniLM', 'gpt', 'qwen2'])
    parser.add_argument('--cstep', type=str, default=None, choices=['load_model', 'load_all'])
    parser.add_argument('--use_recons', type=bool, default=None)
    args, _ = parser.parse_known_args()

    # cuda
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # model name
    if model is not None:
        model_name = model.lower()
    elif args.model is not None:
        model_name = args.model.lower()
    else:
        model_name = 'default'
        # print("Read the default (blank) configuration.")

    # dataset
    if dataset is not None:
        args.dataset = dataset

    # read yml file
    with open('./encoder/config/modelconf/{}.yml'.format(model_name), encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)
        configs['model']['name'] = configs['model']['name'].lower()
        if 'tune' not in configs:
            configs['tune'] = {'enable': False}
        configs['device'] = args.device
        if args.dataset is not None:
            configs['data']['name'] = args.dataset
        if args.seed is not None:
            configs['train']['seed'] = args.seed

        if args.cstep is not None: # for face only
            if args.cstep == 'load_model':
                del configs['optimizer']['load_all']
            elif args.cstep == 'load_all':
                del configs['optimizer']['load_model']
        if args.use_recons is not None:
            configs['optimizer']['use_recons'] = args.use_recons
        

        # # semantic embeddings for RLMRec
        # usrprf_embeds_path = "./data/{}/usr_emb_np.pkl".format(configs['data']['name'])
        # itmprf_embeds_path = "./data/{}/itm_emb_np.pkl".format(configs['data']['name'])
        # with open(usrprf_embeds_path, 'rb') as f:
        #     configs['usrprf_embeds'] = pickle.load(f)
        # with open(itmprf_embeds_path, 'rb') as f:
        #     configs['itmprf_embeds'] = pickle.load(f)

        # semantic representations for FACE
        usrprf_repre_path = f"./data/{configs['data']['name']}/usr_repre_np_{args.llm}.pkl"
        itmprf_repre_path = f"./data/{configs['data']['name']}/itm_repre_np_{args.llm}.pkl"
        with open(usrprf_repre_path, 'rb') as f:
            configs['usrprf_repre'] = pickle.load(f)
        with open(itmprf_repre_path, 'rb') as f:
            configs['itmprf_repre'] = pickle.load(f)

        # # semantic embeddings for RLMRec
        # usrprf_repre_path = "./data/{}/usr_emb_np.pkl".format(configs['data']['name'])
        # itmprf_repre_path = "./data/{}/itm_emb_np.pkl".format(configs['data']['name'])
        # with open(usrprf_repre_path, 'rb') as f:
        #     configs['usrprf_repre'] = pickle.load(f)
        # with open(itmprf_repre_path, 'rb') as f:
        #     configs['itmprf_repre'] = pickle.load(f)
        
        usrprf_path = "./data/{}/usr_prf.pkl".format(configs['data']['name'])
        itmprf_path = "./data/{}/itm_prf.pkl".format(configs['data']['name'])
        with open(usrprf_path, 'rb') as f:
            usrprf = pickle.load(f)
            configs['usrprf'] = {k: v['profile'] for k, v in usrprf.items()}
        with open(itmprf_path, 'rb') as f:
            itmprf = pickle.load(f)
            configs['itmprf'] = {k: v['profile'] for k, v in itmprf.items()}

    configs['remark'] = args.remark
    configs['llm'] = args.llm
    return configs

configs = parse_configure()
