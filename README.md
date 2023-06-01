## Federated Learning via Plurality Vote
![](https://img.shields.io/badge/Python-3-blue) ![](https://img.shields.io/badge/Pytorch-1.9.0-blue) ![](https://img.shields.io/badge/status-maintained-green) 

In this [paper](https://arxiv.org/abs/2110.02998), we propose a new scheme named federated learning via plurality vote (FedVote). In each communication round of FedVote, clients transmit binary or ternary weights to the server with low communication overhead. The model parameters are aggregated via weighted voting to enhance the resilience against Byzantine attacks. When deployed for inference, the model with binary or ternary weights is resource-friendly to edge devices. Our results demonstrate that the proposed method can reduce quantization error and converges faster compared to the methods directly quantizing the model updates.

<br />

### Prerequisites

```bash
pip3 install -r requirements.txt
```
The datasets have been preprocessed under `data` directory.

<br />

### Example


Run the example with fashion mnist dataset: 
```bash
python3 train_fmnist.py
```
The script will load the configuration file `config_fmnist.yaml` and data matrices under `data/fmnist/`. 

You can change the dataset, for example, to CIFAR-10, by modifying the `config.yaml` to 
```
test_data_dir:  data/cifar/test.dat
train_data_dir: data/cifar/train.dat
sample_size:
- 32
- 32
channels: 3
classes: 10
```

<br />

