# Generative Myocardial Motion Tracking via Latent Space Exploration with Biomechanics-informed Prior

Demo code accompanying the paper of the same title.

## Introduction

Myocardial motion and deformation are rich descriptors that characterise cardiac function. Image registration, as the most commonly used technique for myocardial motion tracking, is an ill-posed inverse problem which often requires prior assumptions on the solution space. In contrast to most existing approaches which impose explicit generic regularisation such as smoothness, in this work we propose a novel method that can implicitly learn an application-specific biomechanics-informed prior and embed it into a neural network-parameterised transformation model. Particularly, the proposed method leverages a variational autoencoder-based generative model to learn a manifold for biomechanically plausible deformations. The motion tracking then can be performed via traversing the learnt manifold to search for the optimal transformations while considering the sequence information. The proposed method is validated on cardiac cine MRI data.

The toy dataset is borrowed from <https://acdc.creatis.insa-lyon.fr/description/databases.html>.

## Usage

To train the data-driven biomechanical prior:

  python main_prior.py

----

To perform the motion tracking:

  python main_motion_tracking.py

## Citation and Acknowledgement

Chen Qin, Shuo Wang, Chen Chen, Wenjia Bai, Daniel Rueckert. Generative myocardial motion tracking via latent space exploration with biomechanics-informed prior, Medical Image Analysis, 2023. https://doi.org/10.1016/j.media.2022.102682.
