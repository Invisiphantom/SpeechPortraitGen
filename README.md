# SpeechPortraitGen

[![LICENSE](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Invisiphantom/SpeechPortraitGen/blob/main/LICENSE)


### :book: 参考项目

:arrow_forward: [Hubert](https://huggingface.co/docs/transformers/model_doc/hubert): HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units <br>
:arrow_forward: [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer): SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers <br>
:arrow_forward: [Deep3DFaceRecon](https://github.com/Microsoft/Deep3DFaceReconstruction): Accurate 3D Face Reconstruction with Weakly-Supervised Learning <br>
:arrow_forward: [Real3DPortrait](https://github.com/yerfor/Real3DPortrait): One-shot Realistic 3D Talking Portrait Synthesis <br>
:arrow_forward: [GFP-GAN](https://github.com/TencentARC/GFPGAN): Towards Real-World Blind Face Restoration with Generative Facial Prior <br>


### :wrench: 环境配置

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [MiniConda](https://docs.anaconda.com/free/miniconda/)

```bash
sudo apt install -y gcc g++ gdb cmake zip portaudio19-dev libglu1-mesa-dev
conda create -n sp python=3.9
conda activate sp
conda install conda-forge::ffmpeg

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pytorch3d::pytorch3d

pip install cython openmim==0.3.9
mim install mmcv==2.1.0

pip install -r docs/requirements.txt -v
bash docs/init_ckpt.sh
python3 app.py
```

