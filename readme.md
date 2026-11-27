> Modernization branch: see [core installation, optional components and compatibility](docs/modernization/README.md) and [validation results](docs/modernization/validation.md).

![](https://bigscity-libcity-docs.readthedocs.io/en/latest/_images/logo.png)

------

[![ACM SIGSpatial](https://img.shields.io/badge/ACM%20SIGSPATIAL'21-LibCity-orange)](https://dl.acm.org/doi/10.1145/3474717.3483923) [![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/) [![Pytorch](https://img.shields.io/badge/Pytorch-2.6%2B-blue)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-Apache%202.0-blue)](./LICENSE.txt) [![star](https://img.shields.io/github/stars/LibCity/Bigscity-LibCity?style=social)](https://github.com/LibCity/Bigscity-LibCity/stargazers) [![fork](https://img.shields.io/github/forks/LibCity/Bigscity-LibCity?style=social)](https://github.com/LibCity/Bigscity-LibCity/network/members) 

# LibCity（阡陌）

[HomePage](https://libcity.ai/) | [Docs](https://bigscity-libcity-docs.readthedocs.io/en/latest/index.html) | [Datasets](https://github.com/LibCity/Bigscity-LibCity-Datasets) | [Conference Paper](https://dl.acm.org/doi/10.1145/3474717.3483923) | [Full Paper](https://arxiv.org/abs/2304.14343) | [Paper List](https://github.com/LibCity/Bigscity-LibCity-Paper) | [Experiment Tool](https://github.com/LibCity/Bigscity-LibCity-WebTool) | [EA&B Track Paper](https://arxiv.org/abs/2308.12899) | [中文版](https://github.com/LibCity/Bigscity-LibCity/blob/master/readme_zh.md) 

LibCity is a unified, comprehensive, and extensible library, which provides researchers with a credible experimental tool and a convenient development framework in the traffic prediction field. Our library is implemented based on PyTorch and includes all the necessary steps or components related to traffic prediction into a systematic pipeline, allowing researchers to conduct comprehensive experiments. Our library will contribute to the standardization and reproducibility in the field of traffic prediction.

LibCity currently supports the following tasks:

* Traffic State Prediction
  * Traffic Flow Prediction
  * Traffic Speed Prediction
  * On-Demand Service Prediction
  * Origin-destination Matrix Prediction
  * Traffic Accidents Prediction
* Trajectory Next-Location Prediction
* Estimated Time of Arrival
* Map Matching
* Road Network Representation Learning

## Features

* **Unified**: LibCity builds a systematic pipeline to implement, use and evaluate traffic prediction models in a unified platform. We design basic spatial-temporal data storage, unified model instantiation interfaces, and standardized evaluation procedure.

* **Comprehensive**: The original library provides a broad reference model warehouse; the current runtime supports the explicitly listed migrated models. Meanwhile, LibCity collects 52 commonly used datasets of different sources and implements a series of commonly used evaluation metrics and strategies for performance evaluation. 

* **Extensible**: LibCity enables a modular design of different components, allowing users to flexibly insert customized components into the library. Therefore, new researchers can easily develop new models with the support of LibCity.

## LibCity News

[![new](https://github.com/RUCAIBox/RecBole/raw/master/asset/new.gif)](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif) **04/12/2024**: We release M-LibCity v0.1, the MindSpore version of LibCity. This initial version supports nine traffic prediction tasks, including traffic state forecasting, trajectory next-hop prediction, and estimated time of arrival. Furthermore, M-LibCity also supports multi-cards parallel acceleration on various backends including GPU and NPU. For more detailed information and additional resources, please visit this [link](https://github.com/LibCity/M-LibCity).

[![new](https://github.com/RUCAIBox/RecBole/raw/master/asset/new.gif)](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif) **11/04/2023**: Our creative design *ZongHeng* based on LibCity won the first prize in the 9th China Graduate Contest on Smart-city Technology and Creative Design! [Weixin](https://mp.weixin.qq.com/s/rAiTaVtnG9Kub_BBW502qw)

[![new](https://github.com/RUCAIBox/RecBole/raw/master/asset/new.gif)](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif)**08/24/2023:** We published a paper titled *Unified Data Management and Comprehensive Performance Evaluation for Urban Spatial-Temporal Prediction [Experiment, Analysis & Benchmark]*, including (1) a Unified Storage Format for urban spatial-temporal data, (2) a Technical Development Roadmap for urban spatial-temporal prediction models, (3) Extensive Experiments and Performance Evaluation using 18 models and 20 datasets. [[Paper]](https://arxiv.org/abs/2308.12899).

[![new](https://github.com/RUCAIBox/RecBole/raw/master/asset/new.gif)](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif)**06/20/2023:** We released the Beijing trajectory dataset collected in November 2015, including 1018312 trajectories. We obtained the corresponding road network data from OpenStreetMap and preprocessed the trajectory data to get the Beijing trajectory dataset matched to the road network, and we believed that this dataset could promote the development of urban trajectory mining tasks. Please refer to this [link](https://github.com/aptx1231/START) to obtain it and ensure that this data is **used for research purposes only**. 

[![new](https://github.com/RUCAIBox/RecBole/raw/master/asset/new.gif)](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif) **06/04/2023**: LibCity won the second prize in the 3rd China Science Open Source Software Creativity Competition! [Weixin](https://mp.weixin.qq.com/s?__biz=MzA3NzM4OTc4Mw==&mid=2454775999&idx=1&sn=881a31468c5cd472ed72967b487837cf&chksm=88f68207bf810b1157ac622ae0beba0a1f2ca8ece38fa5c743c4e082c30d9e27d23b92b61530&scene=126&sessionid=1687198811#rd)

[![new](https://github.com/RUCAIBox/RecBole/raw/master/asset/new.gif)](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif) **04/27/2023:** We published a [full paper](https://arxiv.org/abs/2304.14343) titled *LibCity: A Unified Library Towards Efficient and Comprehensive Urban Spatial-Temporal Prediction*, which provides more details of LibCity. [[Paper\]](https://arxiv.org/abs/2304.14343).

[![new](https://github.com/RUCAIBox/RecBole/raw/master/asset/new.gif)](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif) **11/19/2022**: Our self-attention-based traffic flow prediction model **PDFormer** developed on LibCity was accepted by **AAAI2023**, please check this [link](https://github.com/BUAABIGSCity/PDFormer) for more details.

[![new](https://github.com/RUCAIBox/RecBole/raw/master/asset/new.gif)](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif) **08/05/2022**: We develop an **experiment management tool** for the LibCity, which allows users to complete the experiments in a visual interface. The link to the code repository is [here](https://github.com/LibCity/Bigscity-LibCity-WebTool). Some introduction (in Chinese): [Weixin](https://mp.weixin.qq.com/s?__biz=MzA3NzM4OTc4Mw==&mid=2454773897&idx=1&sn=e09cc3fc7dd772a579dd10730f8fadd8&chksm=88f68a31bf810327849442c6af4bf59d5042dfb9871247239a49f070dbeb9f321b41706da157&scene=126&&sessionid=1669002707#rd), [Zhihu](https://zhuanlan.zhihu.com/p/550605104)

**04/27/2022**: We release the first version of LibCity **v0.3**, the latest version, supporting 9 types of spatio-temporal prediction tasks, covering more than 60 prediction models and nearly 40 urban spatio-temporal datasets.

**11/24/2021**: We provide some introductory tutorials (in Chinese) on Zhihu, [link1](https://zhuanlan.zhihu.com/p/401186930), [link2](https://zhuanlan.zhihu.com/p/400814990), [link3](https://zhuanlan.zhihu.com/p/400819354), [link4](https://zhuanlan.zhihu.com/p/400821482), [link5](https://zhuanlan.zhihu.com/p/401190615), [link6](https://zhuanlan.zhihu.com/p/436191860)....

**11/10/2021**: We provide a document that describes in detail the format of the [atomic files](https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/data/atomic_files.html) defined by LibCity. You can download [English Version](https://libcity.ai/A-Unified-Storage-Format-of-Traffic-Data-Atomic-Files-in-LibCity.pdf) and [Chinese Version](https://libcity.ai/LibCity%E4%B8%AD%E4%BA%A4%E9%80%9A%E6%95%B0%E6%8D%AE%E7%BB%9F%E4%B8%80%E5%AD%98%E5%82%A8%E6%A0%BC%E5%BC%8F-%E5%8E%9F%E5%AD%90%E6%96%87%E4%BB%B6.pdf) here for details.

**11/07/2021**: We have a presentation on ACM SIGSPATIAL 2021 Local Track to introduce LibCity. You can download [LibCity Presentation Slide(Chinese)](https://libcity.ai/LibCity-城市时空预测深度学习开源平台.pdf) and [LibCity Chinese Tutorial](https://libcity.ai/LibCity-中文Tutorial.pptx) here.

**11/07/2021**: We have a presentation on ACM SIGSPATIAL 2021 Main Track to introduce LibCity. Here are the [Presentation Video(English)](https://www.bilibili.com/video/BV19q4y1g7Rh/) and the [Presentation Slide(English)](https://libcity.ai/LibCity-Presentation.pdf).

## Overall Framework

![](https://bigscity-libcity-docs.readthedocs.io/en/latest/_images/framework.png)

* **Configuration Module**: Responsible for managing all the parameters involved in the framework.
* **Data Module**: Responsible for loading datasets and data preprocessing operations.
* **Model Module**: Responsible for initializing the reproduced baseline model or custom model.
* **Evaluation Module**: Responsible for evaluating model prediction results through multiple indicators.
* **Execution Module**: Responsible for model training and prediction.

## Installation

This branch runs from source with Python 3.11+, PyTorch 2.6+ and NumPy 2.
See [pyproject.toml](pyproject.toml) for version ranges. Check out this repository's
refactor/libcity-modernization branch. The following example creates atomic data
and writes runs outside the checkout; no downloaded dataset is needed.

## Quick-Start

```powershell
python -m pip install -e .
python tools/make_runtime_fixture.py --directory ../libcity-demo
python run_model.py train --config ../libcity-demo/gru.json --set runtime.exp_id=train
python run_model.py evaluate --config ../libcity-demo/gru.json --checkpoint ../libcity-demo/runs/train/checkpoints/best.pt --set runtime.exp_id=evaluate
python run_model.py train --config ../libcity-demo/gru.json --resume ../libcity-demo/runs/train/checkpoints/last.pt --set training.epochs=3 --set runtime.exp_id=resumed
python run_hyper.py --config ../libcity-demo/gru.json --space hyper_example.json --max-trials 4
```

The CLI uses train/evaluate subcommands and layered JSON. Paths in JSON are relative
to that file. Python callers use `libcity.pipeline.experiment.run_experiment`.
Search trials use validation data; only the selected weights receive a test evaluation.

There are 13 supported runtime entries across five tasks, covering 15 of the original
100 task/model records. The other 85 records remain pending. Retained model source is
reference material, not an additional supported execution API. See the
[migration inventory](docs/modernization/runtime/model-migration.csv),
[five runnable examples](examples/README.md) and [current guide](docs/modernization/runtime/README_zh.md).

Atomic files remain the input format. Target windows are disjoint between splits;
normalization fits the training interval only. Each run owns its config, logs,
metrics, best.pt inference weights, last.pt complete training state and predictions.
Historical checkpoint formats are not automatically loaded.

## TensorBoard Visualization

TensorBoard is optional: install `python -m pip install -e '.[tensorboard]'` and add
`--set runtime.tensorboard=true` to training. Then run
`tensorboard --logdir ../libcity-demo/runs/train/tensorboard`. Evaluation creates no
training writer. For development install `python -m pip install -e '.[dev,transformer]'`
and run `python -m pytest -q`. See the [validation records](docs/modernization/README.md)
for actual CPU and available accelerator coverage.

## Reproduced Model List

For a list of all models reproduced in LibCity, see [Docs](https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/model.html), where you can see the abbreviation of the model and the corresponding papers and citations.

## Tutorial

In order to facilitate users to use LibCity, we provide users with some tutorials:

- We gave lectures on both ACM SIGSPATIAL 2021 Main Track and Local Track. For related lecture videos and Slides, please see our [HomePage](https://libcity.ai/#/tutorial) (in Chinese and English).
- We provide entry-level tutorials (in Chinese and English) in the documentation.
  - [Install and quick start](https://bigscity-libcity-docs.readthedocs.io/en/latest/tutorial/install_quick_start.html)  & [安装和快速上手](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/tutorial/install_quick_start.html)
  - [Run an existing model in LibCity](https://bigscity-libcity-docs.readthedocs.io/en/latest/tutorial/run_model.html) & [运行LibCity中已复现的模型](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/tutorial/run_model.html)
  - [Add a new model to LibCity](https://bigscity-libcity-docs.readthedocs.io/en/latest/tutorial/add_model.html)  & [在LibCity中添加新模型](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/tutorial/add_model.html)
  - [Tuning the model with automatic tool](https://bigscity-libcity-docs.readthedocs.io/en/latest/tutorial/hyper_tune.html) & [使用自动化工具调参](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/tutorial/hyper_tune.html)
  - [Visualize Atomic Files](https://bigscity-libcity-docs.readthedocs.io/en/latest/tutorial/data_visualization.html) & [原子文件可视化](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/tutorial/data_visualization.html)
- In order to facilitate the use of domestic users in China, we provide an introductory tutorial (in Chinese) on Zhihu.
  - [LibCity：一个统一、全面、可扩展的交通预测算法库](https://zhuanlan.zhihu.com/p/401186930)
  - [LibCity入门教程（1）——安装与快速上手](https://zhuanlan.zhihu.com/p/400814990)
  - [LibCity入门教程（2）——运行LibCity中已复现的模型](https://zhuanlan.zhihu.com/p/400819354)
  - [LibCity入门教程（3）——在LibCity中添加新模型](https://zhuanlan.zhihu.com/p/400821482)
  - [LibCity入门教程（4）—— 自动化调参工具](https://zhuanlan.zhihu.com/p/401190615)
  - [北航BIGSCity课题组提出LibCity工具库：城市时空预测深度学习开源平台](https://zhuanlan.zhihu.com/p/436191860)

## Contribution

The LibCity is mainly developed and maintained by Beihang Interest Group on SmartCity ([BIGSCITY](https://www.bigcity.ai/)). The core developers of this library are [@aptx1231](https://github.com/aptx1231) and [@WenMellors](https://github.com/WenMellors). 

Several co-developers have also participated in the reproduction of  the model, the list of contributions of which is presented in the [reproduction contribution list](./contribution_list.md).

If you encounter a bug or have any suggestion, please contact us by [raising an issue](https://github.com/LibCity/Bigscity-LibCity/issues). You can also contact us by sending an email to bigscity@126.com.

## Cite

Our paper is accepted by ACM SIGSPATIAL 2021. If you find LibCity useful for your research or development, please cite our [paper](https://dl.acm.org/doi/10.1145/3474717.3483923).

```
@inproceedings{libcity,
  author = {Wang, Jingyuan and Jiang, Jiawei and Jiang, Wenjun and Li, Chao and Zhao, Wayne Xin},
  title = {LibCity: An Open Library for Traffic Prediction},
  year = {2021},
  isbn = {9781450386647},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3474717.3483923},
  doi = {10.1145/3474717.3483923},
  booktitle = {Proceedings of the 29th International Conference on Advances in Geographic Information Systems},
  pages = {145–148},
  numpages = {4},
  keywords = {Spatial-temporal System, Reproducibility, Traffic Prediction},
  location = {Beijing, China},
  series = {SIGSPATIAL '21}
}
```

For the long paper, please cite it as follows:

```
@article{libcitylong,
  title={LibCity: A Unified Library Towards Efficient and Comprehensive Urban Spatial-Temporal Prediction}, 
  author={Jiang, Jiawei and Han, Chengkai and Jiang, Wenjun and Zhao, Wayne Xin and Wang, Jingyuan},
  journal={arXiv preprint arXiv:2304.14343},
  year={2023}
}
```

## License

LibCity uses [Apache License 2.0](https://github.com/LibCity/Bigscity-LibCity/blob/master/LICENSE.txt).

## Stargazers

[![Stargazers repo roster for @LibCity/Bigscity-LibCity](https://reporoster.com/stars/LibCity/Bigscity-LibCity)](https://github.com/LibCity/Bigscity-LibCity/stargazers)

## Forkers

[![Forkers repo roster for @LibCity/Bigscity-LibCity](https://reporoster.com/forks/LibCity/Bigscity-LibCity)](https://github.com/LibCity/Bigscity-LibCity/network/members)
