# SpeechPortraitGen

## 环境配置

```bash
sudo apt install gcc g++ gdb cmake zip
conda create -n rp python=3.9
conda activate rp
conda install conda-forge::ffmpeg

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pytorch3d::pytorch3d

pip install cython openmim==0.3.9
mim install mmcv==2.1.0

pip install -r docs/requirements.txt -v
```

