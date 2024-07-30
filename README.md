# Implicit Representations via Operator Learning (ICML 24) 
[[Paper]](https://proceedings.mlr.press/v235/pal24a.html) [[Slide]]() [[Poster]](/asset/poster.pdf) [[Supplementary Material]](https://uwmadison.box.com/s/0hfedqrkdh2glmpph7jiv7wy9zl0ypt1)

#### Sourav Pal, Harshavardhan Adepu, Clinton Wang, Polina Golland, Vikas Singh
![OINR Pipeline](/asset/pipeline.jpeg?raw=true)

## Abstract
The idea of representing a signal as the weights of a neural network, called Implicit Neural Representations (INRs), has led to exciting implications for compression, view synthesis and 3D volumetric data understanding. One problem in this setting pertains to the use of INRs for downstream processing tasks. Despite some conceptual results, this remains challenging because the INR for a given image/signal often exists in isolation. What does the neighborhood around a given INR correspond to? Based on this question, we offer an operator theoretic reformulation of the INR model, which we call Operator INR (or O-INR). At a high level, instead of mapping positional encodings to a signal, O-INR maps one function space to another function space. A practical form of this general casting is obtained by appealing to Integral Transforms. The resultant model does not need multi-layer perceptrons (MLPs), used in most existing INR models -- we show that convolutions are sufficient and offer benefits including numerically stable behavior. We show that O-INR can easily handle most problem settings in the literature, and offers a similar performance profile as baselines. These benefits come with minimal, if any, compromise.

## Installation
It is suggested to create a separate environment to run experiments. The code is written to be compatiable with Python 3.8+ and Pytorch 1.12+
```
conda create --name oinr python=3.9
conda activate oinr
```
Install dependencies using the provided requirement file.
```
pip install -r req.txt
```
Additionally, one needs to install the [Pytorch Wavelets](https://pytorch-wavelets.readthedocs.io/en/latest/index.html) library for running experiments pertaining to the instantiation of O-INR as Calderon-Zygmund operator. We list the install instruction below:
```
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
```
In order to run the experiments, with some default data, please download a sample "data" folder from [here](https://uwmadison.box.com/s/gow8xu9d90nrkpq8atayaximwwrh81dq) and place them adjacent to the requirement file.

## Code
We have organized the code used for the paper according to the experiments. Please navigate to the folders "2dimage", "3dvolume", "cz_operator", and "sequence" to execute code for specific experiments, each have their corresponding README within. The folder "downstream" has sub-folders each with their own instructions on running experiments pertaining to "interpolate", "derivative" and "weight_space". Additionally, we make use of code from the paper [CKConv: Continuous Kernel Convolution For Sequential Data](https://github.com/dwromero/ckconv) as the implementation for continuous convolutions. We have included the required files here in the "ckconv" folder.

## Reference
If you find our paper helpful and/or use this code, please cite our publication at ICML 2024.
```

@InProceedings{pmlr-v235-pal24a,
  title = 	 {Implicit Representations via Operator Learning},
  author =       {Pal, Sourav and Adepu, Harshavardhan and Wang, Clinton and Golland, Polina and Singh, Vikas},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {39022--39041},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/pal24a/pal24a.pdf},
  url = 	 {https://proceedings.mlr.press/v235/pal24a.html},
  abstract = 	 {The idea of representing a signal as the weights of a neural network, called <em>Implicit Neural Representations</em> (INRs), has led to exciting implications for compression, view synthesis and 3D volumetric data understanding. One problem in this setting pertains to the use of INRs for downstream processing tasks. Despite some conceptual results, this remains challenging because the INR for a given image/signal often exists in isolation. What does the neighborhood around a given INR correspond to? Based on this question, we offer an operator theoretic reformulation of the INR model, which we call Operator INR (or O-INR). At a high level, instead of mapping positional encodings to a signal, O-INR maps one function space to another function space. A practical form of this general casting is obtained by appealing to Integral Transforms. The resultant model does not need multi-layer perceptrons (MLPs), used in most existing INR models – we show that convolutions are sufficient and offer benefits including numerically stable behavior. We show that O-INR can easily handle most problem settings in the literature, and offers a similar performance profile as baselines. These benefits come with minimal, if any, compromise. Our code is available at https://github.com/vsingh-group/oinr.}
}

```
