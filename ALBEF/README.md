## PELA: Learning Parameter-Efficient Models with Low-Rank Approximation.

We build our model based upon the original [ALBEF](https://github.com/salesforce/ALBEF) repo.
We sincerely appreciate the Junnan' generous sharing and great contribution!

#### Most of the settings follow ALBEF except for the decreased batch-size due to resource constraints.

![model structure](/imgs/model.png)

### Requirements:
* pytorch 1.8.0
* transformers 4.8.1
* timm 0.4.9

### Download:
We reuse the json file extracted by ALBEF.
* <a href="https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/data.tar.gz"> Dataset json files for downstream tasks</a>
* <a href="https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/json_pretrain.zip"> Dataset json files for pre-training</a> (the image paths in each json file need to be changed to your own directory)

### Intermediate Pre-training 4M image-text data:
1. Prepare training json files where each json file contains a list. Each item in the list is a dictonary with two key-value pairs: {'image': path_of_image, 'caption': text_of_image}.
2. In ```configs/Pretrain.yaml```, set the paths for the json files.
3. Download the original 4M [checkpoint](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF_4M.pth).
4. Compress the 4M checkpoints using low-rank approximation.
      ```
      python weight_replacement.py
      ```
5. Pre-train the model using 4 NVIDIA RTX A5000 GPUs:
      ```
      OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 --use_env Pretrain.py
      ```

### Downstream tasks
Our only modification to downstream fine-tuning lies in the following three lines of codes:
```python
import models.low_rank as low_rank
module_lr = low_rank.ModuleLowRank(name_omit=args.name_omit, is_approximate=False)
model = module_lr(model)
```
You can easily find them on every downstream scripts.

All the downstream tutorial can be found at ALBEF, we simply copy them here.
#### Fine-tuning VQA
      ```
      OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 --use_env VQA.py
      ```
#### Fine-tuning Visual Entailment
      ```
      OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 --use_env VE.py
      ```
#### Fine-tuning Retrieval@Flickr30K
      ```
      OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py
      ```
#### Fine-tuning Retrieval@MSCOCO
      ```
      OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
      --config ./configs/Retrieval_flickr.yaml \
      ```
#### Fine-tuning Visual Grounding on RefCOCO+
      ```
      OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 --use_env Grounding.py
      ```

<!-- ### Citation
If you find this code to be useful for your research, please consider citing.
<pre>
@inproceedings{ALBEF,
      title={Align before Fuse: Vision and Language Representation Learning with Momentum Distillation},
      author={Junnan Li and Ramprasaath R. Selvaraju and Akhilesh Deepak Gotmare and Shafiq Joty and Caiming Xiong and Steven Hoi},
      year={2021},
      booktitle={NeurIPS},
}</pre> -->
