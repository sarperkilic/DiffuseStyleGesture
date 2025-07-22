# DiffuseStyleGesture: Stylized Audio-Driven Co-Speech Gesture Generation with Diffusion Models

# Audio → Upper-Body Gesture (PoC)

*Small-scale adaptation of **DiffuseStyleGesture** for a 9-joint torso dataset.*


DiffuseStyleGesture project provides 2 codebases (original and DiffuseStyleGesture+ (BEAT-TWH datasets) ). 

Main codebase uses ZeroEGGS(ZEGGS) dataset and LMDB as a storage format. DiffuseStyleGesture+ codebase gives 2 optian as a dataset, BEAT and TWH, features are packed into a single HDF5 file instead of LMDB. 

I started from the original ZEGGS branch. It is data-prep script is small and readable, easy to plug in new joint set—only change one linear-input size, and also for audio conditioning it is used MFCC or WavLM features, while other codebase expects audio + text + speaker ID. 

### How to adapt the code to provided data
My NumPy array with shape `(num_frames, 9, 7)`, so I flatten → (9 joints × 7 values) = 63 floats per frame. Then created a new LMDB creator script named mydata_to_lmdb.py.

The original script walks through a folder of BVH motion files and their matching WAVs, then calls a BVH-to-array converter. My motion is already in .npy arrays with a different shape and joint set, so I changed the parser and mydata_to_lmdb.py that directly scans my *.npy + *.wav, flattens poses, extracts MFCC, writes LMDB + mean/std.

-  Feature dimensions: the script hard-codes the ZEGGS joint order (upper-body + hands + head) and writes a (T, 263) vector per frame; I only need (T, 63).

### Training Code Changes:

-  In the config file, configs/my_upperbody.yml, I set **motion_dim** to 63 (Matches the flattened 9 × 7 pose vector), **audio_feat** to **mfcc** (MFCC is simpler for the PoC), **audio_feature_dim** to 13 (MFCC gives 13 coefficients; WavLM would be 1024), **n_poses** to 110 (n_poses controls temporal context the diffusion model sees and predicts. window_size in LocalAttention is set to 11, 110 divides 11 and keeps a decent context window)

- main/mydiffusion_zeggs/mdm_dataset.py changed Dataloader key names

     pose = clip["poses"]      # instead of "motion_raw"
     
- main/mydiffusion_zeggs/mdm.py

    self.njoints = motion_dim 

    self.input_feats = motion_dim

*See `configs/my_upperbody.yml` for all hyper-params.*


### Setup Instructions

To create a conda environment and install the required libraries: 

```
conda create -n DiffuseStyleGesture python=3.7
conda activate DiffuseStyleGesture
pip install -r requirements.txt 
```

Then, split .npy files and .wav audio files and move them to ./datasets/npy and .datasets/wav directories respectively. Run lmdb generator script as below:

```
cd main/mydiffusion_zeggs
python mydata_to_lmdb.py \
       --npy_dir   ../../datasets/npy \
       --wav_dir   ../../datasets/wav \
       --out_lmdb ../../datasets/mydata
```
npy, wav files folder path and output lmdb folder path must be given, now they are given as defualt value. This script generated the LMDB file. 

Outputs:

train_lmdb & valid_lmdb

mean.npz & std.npz

Then, model training can be started as below command:

```
cd ./main/mydiffusion_zeggs/
python3 end2end.py --config=./configs/my_upper_body.yml --no_cuda 0 --gpu 0
```
The trained model will be saved to **sarper_lmdb/output_trained_model** (as defined in the config file).

My trained model can be download from: (https://drive.google.com/drive/folders/1kugCHeq3oHTd6yZkX1OKCRhqPFtqUpGW?usp=sharing)

For sampling, sample.py script 
```
python3 sample.py \
  --config ./configs/my_upper_body.yml \
  --model_path /home/codeway/srper/DiffuseStyleGesture/sarper_lmdb/output_trained_model/model000050000.pt \
  --audio_path /home/codeway/srper/DiffuseStyleGesture/datasets/wav/001_Neutral_0_mirror_x_1_0.wav \
  --save_dir /home/codeway/srper/DiffuseStyleGesture/sarper_lmdb/inference_result \
  --max_len 0
```
Produces *.npy gesture file. 

To visualize this npy file, **npy2video.py** script can be run. 

https://github.com/user-attachments/assets/c35da404-d3e7-40f0-8586-4bd8b9430902


### Analysis of Result:

Positive points:
- Stable training curve, loss drops constantly, network accepts 63-dim skeleton & MFCC without.
- Generated gestures (see attached video), upper-torso sway, shoulder shifts, periodic elbow swings roughly synchronized to speech energy peaks. 
  
Negative Points

- Motion amplitude low

Conclusion: PoC succeeds in proving code-path correctness and basic audio-to-motion coupling, but quality is low. 

### Should we run full training (~3 days on 1× V100)?

YES, a 3-day V100 run is justified; expected to lift motion fidelity from “demo” to “usable”. 

For the PoC  MFCC chosen rather than WavLM.

This keeps the pipeline lightweight, faster data prep, shorter training time, smaller GPU footprint, simpler dependency stack.

But, MFCCs carry only coarse prosody; WavLM’s contextual embeddings
encode phonetic and speaker cues that typically yield livelier, better-timed gestures.

Also, more epochs and more data bring unique noise/conditioning pairs, this improves the result.



##

### [![arXiv](https://img.shields.io/badge/arXiv-2305.04919-red.svg)](https://arxiv.org/abs/2305.04919) | [Demo](https://youtu.be/Nzom6gkQ2tM) | [Presentation Video](https://youtu.be/IbpxX1xUo64) | [Conference archive](https://www.ijcai.org/proceedings/2023/0650.pdf)

<div align=center>
<img src="Framework.png" width="750px">
</div>

## Further Work

📢 [QPGesture](https://github.com/YoungSeng/QPGesture) - Based on motion matching, the upper body gesture.

📢 [UnifiedGesture](https://github.com/YoungSeng/UnifiedGesture) - Training on multiple gesture datasets, refine the gestures.

## News

📢 **9/Oct/23** - We obtained the [**REPRODUCIBILITY AWARD**](https://genea-workshop.github.io/2023/challenge/#reproducibility-award) by [GENEA](https://genea-workshop.github.io/2023/) Committee, so we strongly recommend trying [DiffuseStyleGesture+](BEAT-TWH-main) in advance compared to code of DiffuseStyleGesture is partially optimized.

📢 **29/Aug/23** - Release the [paper](https://arxiv.org/abs/2308.13879) of DiffuseStyleGesture+, refer to the official [paper](https://arxiv.org/abs/2308.12646) of [GENEA Challenge 2023](https://genea-workshop.github.io/2023/challenge/) to get more.

📢 **5/Aug/23** - Release code and pre-trained models of [DiffuseStyleGesture+](BEAT-TWH-main) on BEAT and TWH.

📢 **31/Jul/23** - Upload a [tutorial video](visualize_gesture_using_Blender.md) on visualizing gestures. 

📢 **25/Jun/23** - Upload presentation video.

📢 **9/May/23** - First release - arxiv, demo, code, pre-trained models on ZEGGS and [issue](https://github.com/YoungSeng/DiffuseStyleGesture/issues/1#issue-1702250404).


## 1. Getting started

This code was tested on `NVIDIA GeForce RTX 2080 Ti` and requires:

* conda3 or miniconda3

```
conda create -n DiffuseStyleGesture python=3.7
conda activate DiffuseStyleGesture
pip install -r requirements.txt 
```

[//]: # (-i https://pypi.tuna.tsinghua.edu.cn/simple)

## 2. Quick Start

1. Download pre-trained model from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/8ade7c73e05c4549ac6b/) or [Google Cloud](https://drive.google.com/file/d/1RlusxWJFJMyauXdbfbI_XreJwVRnrBv_/view?usp=share_link)
and put it into `./main/mydiffusion_zeggs/`.
2. Download the [WavLM Large](https://github.com/microsoft/unilm/tree/master/wavlm) and put it into `./main/mydiffusion_zeggs/WavLM/`.
3. cd `./main/mydiffusion_zeggs/` and run 
```python
python sample.py --config=./configs/DiffuseStyleGesture.yml --no_cuda 0 --gpu 0 --model_path './model000450000.pt' --audiowavlm_path "./015_Happy_4_x_1_0.wav" --max_len 320
```
You will get the `.bvh` file named `yyyymmdd_hhmmss_smoothing_SG_minibatch_320_[1, 0, 0, 0, 0, 0]_123456.bvh` in the `sample_dir` folder, which can then be visualized using [Blender](https://www.blender.org/) with the following result (To visualize bvh with Blender see this [issue](https://github.com/YoungSeng/DiffuseStyleGesture/issues/8) and this [tutorial video](visualize_gesture_using_Blender.md)):


https://github.com/YoungSeng/DiffuseStyleGesture/assets/37477030/2ef7aa70-69e0-4fd9-a551-6b8a5d075d17


The parameter `no_cuda` and `gpu` need to be the same, i.e. the GPU you want to use; `max_len` is the length you want to generate, this parameter should be `0` if you want to generate the whole length; if you want to **use your own audio**, you should rename your audio file name as `xxx_style_xxx.wav`, e.g. `000_Neutral_xxx.wav` (Happy, Sad, ...). please refer to [this issue](https://github.com/YoungSeng/DiffuseStyleGesture/issues/8#issuecomment-1620027786) to set the style and intensity you want.


## 3. Train your own model

### (1) Get ZEGGS dataset

Same as [ZEGGS](https://github.com/ubisoft/ubisoft-laforge-ZeroEGGS).

An example is as follows.
Download original ZEGGS datasets from [here](https://github.com/ubisoft/ubisoft-laforge-ZeroEGGS) and put it in `./ubisoft-laforge-ZeroEGGS-main/data/` folder.
Then `cd ./ubisoft-laforge-ZeroEGGS-main/ZEGGS` and run `python data_pipeline.py` to process the dataset.
You will get `./ubisoft-laforge-ZeroEGGS-main/data/processed_v1/trimmed/train/` and `./ubisoft-laforge-ZeroEGGS-main/data/processed_v1/trimmed/test/` folders.

If you find it difficult to obtain and process the data, you can download the data after it has been processed by ZEGGS from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/ba5f3b33d94b4cba875b/) or [Baidu Cloud](https://pan.baidu.com/s/1KakkGpRZWfaJzfN5gQvPAw?pwd=vfuc).
And put it in `./ubisoft-laforge-ZeroEGGS-main/data/processed_v1/trimmed/` folder.


### (2) Process ZEGGS dataset

```
cd ./main/mydiffusion_zeggs/
python zeggs_data_to_lmdb.py
```

### (3) Train

```
python end2end.py --config=./configs/DiffuseStyleGesture.yml --no_cuda 0 --gpu 0
```
The model will save in `./main/mydiffusion_zeggs/zeggs_mymodel3_wavlm/` folder.


<!-- Here is our video. Characters from [here](https://www.mixamo.com/#/?page=2&type=Character).  -->
<!-- https://github.com/YoungSeng/DiffuseStyleGesture/assets/37477030/6ae45c42-2275-422b-b0e7-f291e59646eb -->


## Reference
Our work mainly inspired by: [MDM](https://github.com/GuyTevet/motion-diffusion-model), [Text2Gesture](https://github.com/youngwoo-yoon/Co-Speech_Gesture_Generation), [Listen, denoise, action!](https://arxiv.org/abs/2211.09707)

## Citation
If you find this code useful in your research, please cite:

```
@inproceedings{ijcai2023p650,
  title     = {DiffuseStyleGesture: Stylized Audio-Driven Co-Speech Gesture Generation with Diffusion Models},
  author    = {Yang, Sicheng and Wu, Zhiyong and Li, Minglei and Zhang, Zhensong and Hao, Lei and Bao, Weihong and Cheng, Ming and Xiao, Long},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {5860--5868},
  year      = {2023},
  month     = {8},
  doi       = {10.24963/ijcai.2023/650},
  url       = {https://doi.org/10.24963/ijcai.2023/650},
}
```

Please feel free to contact us ([yangsc21@mails.tsinghua.edu.cn](yangsc21@mails.tsinghua.edu.cn)) with any question or concerns.
