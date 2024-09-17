<h2 align="center">Robust Whole-body PET Image Denoising Using 3D Diffusion Models: Evaluation Across Various Scanners, Tracers, and Dose Levels</h2>
<div align="center">
  <span class="author-block">
    Boxiao Yu<sup>1</sup>,</span>
  <span class="author-block">
   Savas Ozdemir<sup>2</sup>,</span>
  <span class="author-block">
    Yafei Dong<sup>3</sup>,</span>
  <span class="author-block">
    Wei Shao<sup>4</sup>,</span>
  <span class="author-block">
    Tinsu Pan<sup>5</sup>,</span>
  <span class="author-block">
    Kuangyu Shi<sup>6</sup>,</span>
  <span class="author-block">
    Kuang Gong<sup>1</sup>
</div>
<div align="center">
  <span class="author-block"><sup>1</sup>J. Crayton Pruitt Family Department of Biomedical Engineering, 
     University of Florida;</span>
  <span class="author-block"><sup>2</sup>Department of Raiology, University of Florida;</span>
  <span class="author-block"><sup>3</sup>Yale School of Medicine, Yale University;</span>
  <span class="author-block"><sup>4</sup>Department of Medicine, University of Florida;</span>
  <span class="author-block"><sup>5</sup>Department of Imaging Physics, University of Texas MD Anderson Cancer Center;</span>
  <span class="author-block"><sup>6</sup>Department of Nuclear Medicine, University of Bern</span>
</div>

## Purpose

Whole-body PET imaging plays an essential role in cancer diagnosis and treatment but suffers from low image quality. Traditional deep learning-based denoising methods work well for a specific acquisition but are less effective in handling diverse PET protocols. In this study, we proposed and validated a 3D Denoising Diffusion Probabilistic Model (3D DDPM) as a robust and universal solution for whole-body PET image denoising. 

## Method

<p align="center">
  <img align="middle" width="800" src="figs/model_framework.png"/>
</p>
The proposed 3D DDPM gradually injects noise into the images during the forward diffusion phase, allowing the model to learn to reconstruct the clean data during the reverse diffusion process. A 3D convolutional network was trained using high-quality data from the Biograph Vision Quadra PET/CT scanner to generate the score function, enabling the model to capture accurate PET distribution information extracted from the total-body datasets. The trained 3D DDPM was evaluated on datasets from four scanners, four tracer types, and six dose levels representing a broad spectrum of clinical scenarios.

## Result

<p align="center">
  <img align="middle" width="800" src="figs/results.png"/>
</p>
The proposed 3D DDPM consistently outperformed 2D DDPM, 3D UNet, and 3D GAN, demonstrating its superior denoising performance across all tested conditions. Additionally, the model’s uncertainty maps exhibited lower variance, reflecting its higher confidence in its outputs. 

## Code

The code for this project will be uploaded soon. Please stay tuned.

## Contact

For any questions or inquiries, please contact boxiao.yu@ufl.edu.

