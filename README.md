<div align=center>
  
# *TaylorSeer*: From Reusing to Forecasting: Accelerating Diffusion Models with *TaylorSeers*

<p>
<a href='https://arxiv.org/abs/2410.05317'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://toca2024.github.io/ToCa/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
</p>

</div>

## 🔥 News

* `2025/03/10` 🚀🚀 Our latest work "From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers" is released! Codes are available at [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)! TaylorSeer supports lossless compression at a rate of 4.99x on FLUX.1-dev (with a latency factor of 3.53x) and high-quality acceleration at a compression rate of 5.00x on HunyuanVideo (with a latency factor of 4.65x)! We hope *TaylorSeer* can move the paradigm of feature caching methods from reusing to forecasting.For more details, please refer to our latest research paper.
* `2025/02/19` 🚀🚀 ToCa solution for **FLUX** has been officially released after adjustments, now achieving up to **3.14× lossless acceleration** (in FLOPs)!
* `2025/01/22` 💥💥 ToCa is honored to be accepted by ICLR 2025!
* `2024/12/29` 🚀🚀 We release our work [DuCa](https://arxiv.org/abs/2412.18911) about accelerating diffusion transformers for FREE, which achieves nearly lossless acceleration of **2.50×** on [OpenSora](https://github.com/hpcaitech/Open-Sora)! 🎉 **DuCa also overcomes the limitation of ToCa by fully supporting FlashAttention, enabling broader compatibility and efficiency improvements.**
* `2024/12/24` 🤗🤗 We release an open-sourse repo "[Awesome-Token-Reduction-for-Model-Compression](https://github.com/xuyang-liu16/Awesome-Token-Reduction-for-Model-Compression)", which collects recent awesome token reduction papers! Feel free to contribute your suggestions!
* `2024/12/10` 💥💥 Our team's recent work, **SiTo** (https://github.com/EvelynZhang-epiclab/SiTo), has been accepted to **AAAI 2025**. It accelerates diffusion models through adaptive **Token Pruning**.
* `2024/07/15` 🤗🤗 We release an open-sourse repo "[Awesome-Generation-Acceleration](https://github.com/xuyang-liu16/Awesome-Generation-Acceleration)", which collects recent awesome generation accleration papers! Feel free to contribute your suggestions!

## 🛠 Installation

``` cmd
git clone https://github.com/Shenyi-Z/TaylorSeer.git
```


## TaylorSeer-FLUX

TaylorSeer achieved a lossless computational compression of 4.99$\times$ and a Latency Speedup of 3.53$\times$ on FLUX.1-dev, as measured by [ImageReward](https://github.com/THUDM/ImageReward) for comprehensive quality. To run TaylorSeer-FLUX, see [TaylorSeer-FLUX](TaylorSeer-FLUX.md).

## TaylorSeer-HunyuanVideo

TaylorSeer achieved a computational compression of 5.00 $\times$ and a remarkable Latency Speedup of 4.65 $\times$ on HunyuanVideo, as comprehensively measured by the [VBench](https://github.com/Vchitect/VBench) metric. Compared to previous methods, it demonstrated significant improvements in both acceleration efficiency and quality. To run TaylorSeer-HunyuanVideo, see [TaylorSeer-HunyuanVideo](TaylorSeer-HunyuanVideo.md).

## TayorSeer-DiT

TaylorSeer achieved a lossless computational compression of 2.77 $\times$ on the base model DiT, as comprehensively evaluated by metrics such as FID. Its performance across various acceleration ratios significantly surpassed previous methods. For instance, in an extreme scenario with a 4.53 $\times$ compression ratio, TaylorSeer's FID only increased by 0.33 from the non-accelerated baseline of 2.32, reaching 2.65, while ToCa and DuCa exhibited FID scores above 6.0 under the same conditions. To run TaylorSeer-DiT,see [TaylorSeer-DiT](TaylorSeer-DiT.md).


## 👍 Acknowledgements

- Thanks to [DiT](https://github.com/facebookresearch/DiT) for their great work and codebase upon which we build TaylorSeer-DiT.
- Thanks to [FLUX](https://github.com/black-forest-labs/flux) for their great work and codebase upon which we build TaylorSeer-FLUX.
- Thanks to [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) for their great work and codebase upon which we build TaylorSeer-HunyuanVideo.
- Thanks to [ImageReward](https://github.com/THUDM/ImageReward) for Text-to-Image quality evaluation.
- Thanks to [VBench](https://github.com/Vchitect/VBench) for Text-to-Video quality evaluation.


## :e-mail: Contact

If you have any questions, please email [`shenyizou@outlook.com`](mailto:shenyizou@outlook.com).
