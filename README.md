# Style-Transfer-with-Target-Feature-Palette-and-Attention-Coloring

## Example Results
### Diverse Stylized Results
<img src="imgs/single_results.jpg" width="800"></img>

Single style transfer with two content images with seven different style images.

### Multi-style Transfer Results
<img src="imgs/multi_results.png" height="300"></img>

Three multi-stylized images are generated using four input images (i.e., one content image and three style images) with different settings.

## Usage
### Prerequisites
- Python 3.7
- Pytorch 0.4.1
- TorchVision 0.2.1
- Pillow
- Visdom
- Cupy

### Getting Started
#### Installation
Clone this repo:
```bash
git clone https://github.com/SuhyeonHa/Style-Transfer-with-Target-Feature-Palette-and-Attention-Coloring
cd Style-Transfer-with-Target-Feature-Palette-and-Attention-Coloring
```
#### Dataset
- [MS-COCO(train2014)](http://images.cocodataset.org/zips/train2014.zip) for content dataset
- [Painter by Numbers](https://www.kaggle.com/c/painter-by-numbers/overview) for style dataset 

#### Model Training
```bash
train.py --train_content_dir D:\Dataset/train2014 --train_style_dir D:\Dataset\painter-by-numbers/train
```
#### Use a Pre-trained Model
- Single style transfer
```bash
test.py --content input/content/sailboat.jpg --style input/style/flower_of_life.jpg --test_mode single_style_transfer
```
- Multi style transfer
```bash
test.py --content input/content/sailboat.jpg --style_dir input/style --test_mode multi_style_transfer
```

## Acknowledgement
Our implementation is based on
- [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN)
- [SANET](https://github.com/GlebBrykin/SANET)
