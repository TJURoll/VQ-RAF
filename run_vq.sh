#LightGCN
python encoder/train_encoder.py --dataset amazon --cuda 0 --model lightgcn --remark lgn --seed 427
python encoder/train_encoder.py --dataset amazon --cuda 0 --model lightgcn_vq --cstep load_model --remark lgn_miniLM_2 --seed 427
python encoder/train_encoder.py --dataset amazon --cuda 0 --model lightgcn_vq --cstep load_all --remark lgn_miniLM_3 --seed 427
#GMF
python encoder/train_encoder.py --dataset amazon --cuda 0 --model gmf --remark gmf --seed 427
python encoder/train_encoder.py --dataset amazon --cuda 0 --model gmf_vq --cstep load_model --remark gmf_miniLM_2 --seed 427
python encoder/train_encoder.py --dataset amazon --cuda 0 --model gmf_vq --cstep load_all --remark gmf_miniLM_3 --seed 427
