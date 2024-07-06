# DragSwin

## Citing StyleSwin

```
@misc{zhang2021styleswin,
      title={StyleSwin: Transformer-based GAN for High-resolution Image Generation}, 
      author={Bowen Zhang and Shuyang Gu and Bo Zhang and Jianmin Bao and Dong Chen and Fang Wen and Yong Wang and Baining Guo},
      year={2021},
      eprint={2112.10762},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Responsible AI Considerations

Our work does not directly modify the exiting images which may alter the identity or expression of the people. We discourage the use of our work in such applications as it is not designed to do so. We have quantitatively verified that the proposed method does not show evident disparity, on gender and ages as the model mostly follows the dataset distribution; however, we encourage additional care if you intend to use the system on certain demographic groups. We also encourage use of fair and representative data when training on customized data. We caution that the high-resolution images produced by our model may potentially be misused for impersonating humans and viable solutions so avoid this include adding tags or watermarks when distributing the generated photos.

## Acknowledgements

This code borrows heavily from [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer). We also thank the contributors of code [Positional Encoding in GANs](https://github.com/open-mmlab/mmgeneration/blob/master/configs/positional_encoding_in_gans/README.md), [DiffAug](https://github.com/mit-han-lab/data-efficient-gans), [StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN) and [GIQA](https://github.com/cientgu/GIQA).

## Maintenance

This is the codebase for our research work. Please open a GitHub issue for any help. If you have any questions regarding the technical details, feel free to contact [zhangbowen@mail.ustc.edu.cn](zhangbowen@mail.ustc.edu.cn) or [zhanbo@microsoft.com](zhanbo@microsoft.com).


## License
The codes and the pretrained model in this repository are under the MIT license as specified by the LICENSE file. We use our labeled dataset to train the scratch detection model.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
