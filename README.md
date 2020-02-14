# interCNN
Public PyTorch code implementation of our work: [Iterative Interaction Training for Segmentation Editing Networks](https://link.springer.com/chapter/10.1007/978-3-030-00919-9_42) which was presented at the Machine Learning in Medical Imaging Workshop in MICCAI 2018.

If you find this code helpful for your research please cite the following paper:
```
@inproceedings{bredell2018iterative,
  title={Iterative Interaction Training for Segmentation Editing Networks},
  author={Bredell, Gustav and Tanner, Christine and Konukoglu, Ender},
  booktitle={International Workshop on Machine Learning in Medical Imaging},
  pages={363--370},
  year={2018},
  organization={Springer}
}
```
Method
---
Here however, we present an unique training strategy for convolutional neural networks (CNNs) trained on top of an automatic method to enable interactive segmentation editing that is not limited to binary segmentation. By utilizing a robot-user during training, we closely mimic realistic use cases to achieve optimal editing performance.


Dataset
---
The data used to train the method was from the [NCI_ISBI_2013 challenge](https://wiki.cancerimagingarchive.net/display/Public/NCI-ISBI+2013+Challenge+-+Automated+Segmentation+of+Prostate+Structures)

Code
---
1. Create a data-split list and save it somewhere
2. Configure the pahts.py script to match where the data is and where results should be saved
3. Train autoCNN
4. Use the trained autoCNN to train an interCNN
5. Evaluate the results with the eval scripts


