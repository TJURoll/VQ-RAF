# FACE: A general Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens

## 1. Special Files

`encoder/models/general_cf/` all models, including base models (end without vq) and VQ-RAF enhanced models (end with vq)

`encoder/face.py` the implementation of "Vector-quantized Disentangled Representation Mapping"



## 2. Run

### 2.1. Environments

`Python >= 3.9`

```bash
pip install -r requirements.txt
```



### 2.2. Preparations

+ **datasets:** Download the dataset from https://drive.google.com/file/d/1PzePFsBcYofG1MV2FisFLBM2lMytbMdW/view?usp=sharing
+ **embedding model:** Download the pretrained embedding model from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
+ **generate summary embeddings:** Use `generation/generate_repre_miniLM.py`for generating summary embeddings for all users & items



### 2.4. Pretrain the Base Model

Let's start with the example LightGCN and LightGCN+FACE on Amazon dataset.

```
python encoder/train_encoder.py --model lightgcn --dataset amazon
```

Make sure the checkpoint is saved.



### 2.5. Train

After pretraining the base model, we run FACE on its basis. It includes two step for mapping and aligning respectively:

```
python encoder/train_encoder.py --model lightgcn_vq --dataset amazon --cstep load_model 
python encoder/train_encoder.py --model lightgcn_vq --dataset amazon --cstep load_all
```

