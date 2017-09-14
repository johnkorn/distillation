## Knowledge distillation experiments

### How to run the code

Dependencies: Keras, Tensorflow, Numpy

* Train teacher model:

```python train.py --file data/matlab/emnist-letters.mat --model cnn```
* Train perceptron normally

```python train.py --file data/matlab/emnist-letters.mat --model mlp```
* Train student network with knowledge distillation:

```python train.py --file data/matlab/emnist-letters.mat --model student --teacher bin/cnn_64_128_1024_30model.h5```

### Results
[EMNIST-letters](https://www.nist.gov/itl/iad/image-group/emnist-dataset) dataset was used for experiments (26 classes of hand-written letters of english alphabet)

As a teacher network a simple cnn with `3378970` parameters (2 conv layers with 64 and 128 filters each, 1024 neurons on fully-connected layer) was trained for 26 epochs and was early stopped on plateau. Its validation accuracy was _94.4%_

As a student network a 1-layer perceptron with 512 hidden units and `415258` total parameters was used (8 times smaller than teacher network). First it was trained alone for 50 epochs, val acc was _91.6%_.

Knowledge distillation approach was used with different combinations of `temperature` and `lambda` parameters. Best performance was achieved with `temp=10, lambda=0.5`. Student network trained that way for 50 epochs got val acc of _92.2%_. 

So, the accuracy increase is less than 1% comparing to classicaly trained perceptron. But still we got some improvement. Actually all reports that people did, show similar results on different tasks: 1-2% quality increase. So we may say that reported results were reproduced on emnist-letters dataset. 

[Knowledge distillation](https://arxiv.org/abs/1503.02531) parameters (temperature and lambda) must be tuned for each specific task. To get better accuracy gain additional similar techniques may be tested, e.g. [deep mutual leraning](https://arxiv.org/abs/1706.00384) or [fitnets](https://arxiv.org/abs/1412.6550). 

