# STB-VMM: Swin Transformer Based Video Motion Magnification

**Ricard Lado Roigé, Marco A. Pérez**

*[IQS School of Engineering](https://www.iqs.edu/en 'IQS'), Universitat Ramon Llull*

---

This repository contains the official implementation of the [STB-VMM: Swin Transformer Based Video Motion Magnification](https://www.iqs.edu/en 'paper') paper in PyTorch.

The goal of Video Motion Magnification techniques is to magnify small motions in a video to reveal previously invisible or unseen movement. Its uses extend from bio-medical applications and deep fake detection to structural modal analysis and predictive maintenance. However, discerning small motion from noise is a complex task, especially when attempting to magnify very subtle often sub-pixel movement. As a result, motion magnification techniques generally suffer from noisy and blurry outputs. This work presents a new state-of-the-art model based on the Swin Transformer, which offers better tolerance to noisy inputs as well as higher-quality outputs that exhibit less noise, blurriness and artifacts than prior-art. Improvements in output image quality will enable more precise measurements for any application reliant on magnified video sequences, and may enable further development of video motion magnification techniques in new technical fields.

<p style="text-align: center;"><img src="https://user-images.githubusercontent.com/25719985/176877923-ac6c27cd-5b97-4fed-aedd-739d10ef679b.png" alt="Architecture Overview" width="500"/></p>

---
## Install dependencies
```bash
pip install -r requirements.txt
```

FFMPEG is also required to run *process_video.sh*

---
## Testing
To test STB-VMM just run the script named *process_video.sh* with the appropriate arguments. 

For example:

```bash
bash process_video.sh -amp 20 -i ../demo_video/baby.mp4 -m ckpt/ckpt_e49.pth.tar -o STB-VMM_demo_x20_static -s ../demo_video/ -f 30
```
*Note: To magnify any video a pre-trained checkpoint is required.*

---
## Training
To train the STB-VMM model use *train.py* with the appropriate arguments. The training dataset can be downloaded from [here](https://github.com/12dmodel/deep_motion_mag). 

For example:

```bash
python3 train.py -d ../data/train -j 32 -b 5 --lr 0.00001 --epochs 50 #--resume ckpt/ckpt_e01.pth.tar
```

---
## Citation
```bibtex
@article{lado2022_STB-VMM,
  title={STB-VMM: Swin Transformer Based Video Motion Magnification},
  author={Lado-Roigé, Ricard and P{\'{e}}rez, Marco A.},
  journal={},
  year={2022}
}
```

---
## Acknowledgements

This implementation borrows from the awesome works of:
- [Learning-based Video Motion Magnification](https://github.com/12dmodel/deep_motion_mag 'Tensorflow implementation of Learning-based Video Motion Magnification')
- [Motion Magnification PyTorch](https://github.com/kkjh0723/motion_magnification_pytorch 'Jinhyung')
- [Pytorch Image Models](https://github.com/rwightman/pytorch-image-models 'Ross Wightman')
- [SwinIR](https://github.com/JingyunLiang/SwinIR 'Image Restoration Using Swin Transformer')
