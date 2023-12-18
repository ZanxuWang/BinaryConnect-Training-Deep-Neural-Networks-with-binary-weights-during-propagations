# e4040-2023Fall-project
## BinaryConnect: Training Deep Neural Networks with binary weights during propagations


#Tools and enviroment used in the project 
Tensorflow 2.10 
Keras
CUDA v11.2
cuDNN v8.9.7 for CUDA 11.x
The models are mainly trained and tested locally on a RTX4090 mobile GPU, tested on GCP as well in the end,  training time is recorded using RTX4090 mobile GPU.

#Instruction to run the code
The core algorithm with customized layers for MLP and CNN, customized graident function, binarization function, andd training function is implemented in BinaryConnect.py
Classes and functions in BinaryConnect.py will be used in MNIST_BinaryConnect.ipynb, cifar10_BinaryConnect.ipynb and svhn_BinaryConnect.ipynb.

The three jupyter notebooks contain code for loading & preprossing data, training the model and relavent plots on 3 different datasets as indicated by the file name.

MNIST_BinaryConnect.ipynb: Binarized MLP model was trained on this notebook, weights for the first layer is extracted and plotted by a histogram.
Training time for an epoch is roughly 9 sec, with early stopping applied, to repeat the experiment 6 times takes around 7 hours.

cifar10_BinaryConnect.ipynb: Binarized CNN model was trained on this notebook, data is preprocessed through gloabl contrast normalization and ZCA whitening. The val loss and val error rate per epoch trained was plotted. Training time for an epoch around 90 secs, with early stopping applied, training time takes around 6 hours per experiment.

svhn_BinaryConnect.ipynb: The same Binarized CNN model as used in cifar10 was trained on this notebook, data is preprocessed in the same way. The val loss and val error rate per epoch trained was plotted. Training time for an epoch around 90 secs, with early stopping applied, training time takes around 6 hours per experiment.

Dataset: MNIST and cifar10 can be loaded through keras, SVHN data is loaded through an url and fetched online, relevant code is written in SVHN notebook.

To verify the results on a particular dataset, simply run the corresponding notebook directly.

# Organization of this directory

```
./
├── BinaryConnect.py
├── E4040.2023Fall.ZXYF.report.zw2864.pdf
├── MNIST_BinaryConnect.ipynb
├── README.md
├── __pycache__
│   ├── BinaryConnect.cpython-310.pyc
│   └── BinaryConnect_MNIST.cpython-310.pyc
├── cifar10_BinaryConnect.ipynb
├── figures
│   ├── zw2864_gcp_work_example_screenshot_1.png
│   ├── zw2864_gcp_work_example_screenshot_2.png
│   └── zw2864_gcp_work_example_screenshot_3.png
└── svhn_BninaryConnect.ipynb

2 directories, 11 files
```


```python

```

