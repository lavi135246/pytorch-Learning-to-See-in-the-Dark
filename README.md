# pytorch-Learning-to-See-in-the-Dark
Learning to See in the Dark using pytorch 0.4.0

### Original tensorflow version
Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, "Learning to See in the Dark", in CVPR, 2018. <br/>
[Tensorflow code](https://github.com/cchen156/Learning-to-See-in-the-Dark) <br/>
[Paper](http://cchen156.web.engr.illinois.edu/paper/18CVPR_SID.pdf)


## Requirements
- 64 GB RAM
- GTX 1080
- PyTorch 0.4.0
- RawPy 0.10
- Scipy


## Download Dataset
Download `download_dataset.py` from the [original code](https://github.com/cchen156/Learning-to-See-in-the-Dark) and put it in the top level directory of this project and execute:
`python download_dataset.py`

## Training
`python train_Sony.py`
- It will save model and generate result images every 100 epoch. 
- The trained models will be saved in `saved_model/` and the result images will be saved in `result_Sony/`.
- The right side of the image is the result of current output and the left side shows the ground truth. 

## Testing
### Download trained model
You can download the trained pytorch model [here](https://drive.google.com/file/d/1qVYtDEObRAox8SDH4Tbqs2s117v7tFWG/view?usp=sharing) and put it in folder `saved_model/`. The trained model is only for `.ARW` photos taken by Sony cameras.
`python test_Sony.py`
- Pytorch somehow needs more GPU resources than Tensorflow. Therefore, it is impossible to take in the whole image.
- Testing will only take 1024 * 1024 pixels from the test images. 
- This testing script is only for checking the performance of the trained model.
- The result will be saved in `test_result_Sony` with gt as ground truth images, scale as scaled images, ori as input images, and out as output images.

### Todo
I have tried to feed the sliced images into the model and put the result back to the original size. But there still remains two problems:
1. The edges of the sliced images are quite obvious in the recovered image.
2. There is no padding SAME in pytorch. Hence, images with incompetible shape might result in errors.
