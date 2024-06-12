cd ../deep_3drecon/BFM
gdown -c https://drive.google.com/uc?id=1SPM3IHsyNAaVMwqZZGV6QVaV7I2Hly0v
gdown -c https://drive.google.com/uc?id=1MSldX9UChKEb3AXLVTPzZQcsbGD4VmGF
gdown -c https://drive.google.com/uc?id=180ciTvm16peWrcpl4DOekT9eUQ-lJfMU
gdown -c https://drive.google.com/uc?id=1KX9MyGueFB3M-X0Ss152x_johyTXHTfU
gdown -c https://drive.google.com/uc?id=19-NyZn_I0_mkF-F5GPyFMwQJ_-WecZIL
gdown -c https://drive.google.com/uc?id=11ouQ7Wr2I-JKStp2Fd1afedmWeuifhof
gdown -c https://drive.google.com/uc?id=18ICIvQoKX-7feYWP61RbpppzDuYTptCq
gdown -c https://drive.google.com/uc?id=1VktuY46m0v_n_d4nvOupauJkK4LF6mHE
cd ../..

cd checkpoints
gdown -c https://drive.google.com/uc?id=1gz8A6xestHp__GbZT5qozb43YaybRJhZ
gdown -c https://drive.google.com/uc?id=1gSUIw2AkkKnlLJnNfS2FCqtaVw9tw3QF
unzip 240210_real3dportrait_orig.zip && rm 240210_real3dportrait_orig.zip
unzip pretrained_ckpts.zip && rm pretrained_ckpts.zip

mkdir -p gfpgan && cd gfpgan
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth