# Biomechanics-informed Generative Model for Myocardial Motion Tracking via Latent Space Exploration

Code accompanying the paper of the same title (Under Review).

## Introduction

Myocardial motion and deformation are rich descriptors that characterise cardiac function. Image registration, as the most commonly used technique for myocardial motion tracking, is an ill-posed inverse problem which often requires prior assumptions on the solution space. In contrast to most existing approaches which impose explicit generic regularisation such as smoothness, in this work we propose a novel method that can implicitly learn an application-specific biomechanics-informed prior and embed it into a neural network-parameterised transformation model. Particularly, the proposed method leverages a variational autoencoder-based generative model to learn a manifold for biomechanically plausible deformations. The motion tracking then can be performed via traversing the learnt manifold to search for the optimal transformations while considering the sequence information. The proposed method is validated on three public cardiac cine MRI datasets with comprehensive evaluations. The results demonstrate that the proposed method can outperform other methods, yielding higher motion tracking accuracy with reasonable volume preservation and better generalisability to varying data distributions. It also enables better estimates of myocardial strains, which indicates the potential of the method in characterising spatiotemporal signatures for understanding cardiovascular diseases.

The toy dataset is borrowed from <https://acdc.creatis.insa-lyon.fr/description/databases.html>.

## Usage

To train the VAE regulariser:

  python VAE_regularisation.py

----

To train the registration network:

  python main.py
