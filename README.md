
# Image Colorization


## Getting Started


### Prerequisites


#### Building training environment

Putting the following in one folder

```
* demo.py
* net.py
* normalize.py
* train.py
* resize.py
* utils.py
* images/...
* resized/...
* model/vgg16-20160129.tfmodel
```

The images/... folder will contain .jpg images for training, it can be any color image that you downloaded.

The resized/... folder will be an empty folder to store resized image for training

The model/... folder will contain the file vgg16-20160129.tfmodel, which can be found on google

### Training

#### Resizing

The original image need to be resized to size 224x224 for training

```bash
python3 -m dataset.resize <args>
```

Use `-h` to see the available options

In our example:

```bash
python3 -m dataset.resize -s images -o resized
```

#### Training

Train the neural network:

```bash
python3 train.py
```

Make sure you have folder resized/... containing the resized images for training, and folder model/... with the vgg16 model, Otherwise you may need to edit the path.


#### Results

The program will generate a folder summary/... , it contains the output image during training. 

## Built With

* [Python 3.6.1](https://www.python.org/) - Programming platform
* [Tensorflow 1.3.0](https://www.tensorflow.org/) - Package for building neural network


