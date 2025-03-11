# VQ-RAF

## 1. Special Files

`encoder/models/general_cf/` all models, including base models (end without vq) and VQ-RAF enhanced models (end with vq)

`encoder/vqraf.py` the implementation of "Vector-quantized Disentangled Representation Mapping"



## 2. Run

### 2.1. Environments

`Python >= 3.9`

```bash
pip install -r requirements.txt
```



### 2.2. Preparations

+ **datasets:** Download the dataset from https://drive.google.com/file/d/1PzePFsBcYofG1MV2FisFLBM2lMytbMdW/view?usp=sharing
+ **embedding model:** Download the pretrained embedding model (a smaller one) from https://huggingface.co/mesolitica/llama2-embedding-1b-8k

+ **generate summary embedding:** Use `generation/generate_repre.ipynb`for generating summary embeddings for all users & items



### 2.4. Pretrain the Base Model

Let's start with the example LightGCN and LightGCN+VQ-RAF on Amazon dataset.

```
python encoder/train_encoder.py --model lightgcn --dataset amazon
```

Make sure the checkpoint is saved.



### 2.5. Train

```
python encoder/train_encoder.py --model lightgcn_vq --dataset amazon
```

