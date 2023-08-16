# MVImgNet for 3D Understanding
Point-MAE is a novel scheme of masked autoencoders for point cloud self-supervised learning. It outperforms all other self-supervised learning methods on ScanObjectNN and ModelNet40 in classification tasks in 2022.

Unlike the original [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) which utilizes ShapeNet for pretraining, our approach employs MVPNet, generated from Colmap Dense Reconstruction using [MVImgNet](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet). 

MVPNet is a new large-scale, real-world 3D object point cloud dataset derived from the dense reconstruction on [MVImgNet](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet). It comprises 80,000 point clouds across 150 categories. MVPNet stands out from existing 3D object datasets due to its significantly richer collection of real-world object point clouds. Its abundant categories encompass a wide range of common objects encountered in everyday life.

![A variety of 3D object point clouds in MVPNet](figure/mvp_ds.png)
*A variety of 3D object point clouds in MVPNet*

## Requirements
PyTorch >= 1.7.0 < 1.11.0;
python >= 3.7;
CUDA >= 9.0;
GCC >= 4.9;
torchvision;

```
pip install -r requirements.txt
```

```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## Datasets

We provide the preprocessed [MVPNet](link_to_MVPNet_dataset) dataset. You can download it and extract the contents to the `data` folder in your project directory.

## Instructions
* Pre-training: ``CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/pretrain_MVPNet.yaml --exp_name <output_file_name>``
* Fine-tuning: ``CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_scan_hardest.yaml --finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>``



## Acknowledgements

Our codes are built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE)

## Reference
```
@misc{yu2023mvimgnet,
    title={Mvimgnet: A large-scale dataset of multi-view images},
    author={Yu, Xianggang and Xu, Mutian and Zhang, Yidan and Liu, Haolin and Ye, Chongjie and Wu, Yushuang and Yan, Zizheng and others},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={9150--9161},
    year={2023}
}
```
```
@misc{pang2022masked,
    title={Masked Autoencoders for Point Cloud Self-supervised Learning},
    author={Yatian Pang and Wenxiao Wang and Francis E. H. Tay and Wei Liu and Yonghong Tian and Li Yuan},
    year={2022},
    eprint={2203.06604},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
