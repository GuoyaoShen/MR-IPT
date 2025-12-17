# MR-IPT

This is the code repo for: 

[Magnetic resonance image processing transformer for general accelerated image restoration](https://www.nature.com/articles/s41598-025-23851-w).


## Overview
Recent advancements in deep learning have enabled the development of generalizable models that 
achieve state-of-the-art performance across various imaging tasks. Vision Transformer (ViT)-based 
architectures, in particular, have demonstrated strong feature extraction capabilities when 
pre-trained on large-scale datasets. In this work, we introduce the Magnetic Resonance Image 
Processing Transformer (MR-IPT), a ViT-based image-domain framework designed to enhance the 
generalizability and robustness of accelerated MRI restoration. Unlike conventional deep learning 
models that require separate training for different acceleration factors, MR-IPT is pre-trained 
on a large-scale dataset encompassing multiple undersampling patterns and acceleration settings, 
enabling a unified framework. By leveraging a shared transformer backbone, MR-IPT effectively 
learns universal feature representations, allowing it to generalize across diverse restoration tasks. 
Extensive experiments demonstrate that MR-IPT outperforms both CNN-based and existing transformer-based 
methods, achieving superior quality across varying acceleration factors and sampling masks. Moreover, 
MR-IPT exhibits strong robustness, maintaining high performance even under unseen acquisition setups, 
highlighting its potential as a scalable and efficient solution for accelerated MRI. Our findings 
suggest that transformer-based general models can significantly advance MRI restoration, offering 
improved adaptability and stability compared to traditional deep learning approaches.


### MR-IPT Framework
<p align="center">
    <img src="/figs/fig1.png" width="500" />
</p>

### Reconstruction Examples
<p align="center">
    <img src="/figs/fig2.png" width="1000" />
    <img src="/figs/fig3.png" width="500" />
</p>