# Prior-Guided-SAM
A robust near-eye image segmentation method based on [SAM](https://github.com/facebookresearch/segment-anything) and spatial Adapter, which freezes the pre-trained SAM encoder and inserts a lightweight spatial adaptation module to achieve efficient domain transfer while preserving the strong generalization and segmentation capability of the foundation model. 


## Requirement

Download [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth), and put it at ./checkpoint/sam/

## Dataset

@article{fuhl2021teyed,
  title={TEyeD: Over 20 million real-world eye images with pupil, eyelid, and iris 2d and 3d segmentations, 2d and 3d landmarks, 3d eyeball, gaze vector, and eye movement types},
  author={Wolfgang Fuhl and Gjergji Kasneci and Enkelejda Kasneci},
  journal={arXiv preprint arXiv:2102.02115},
  year={2021}
}

## Example

    
1. Train: ``python train.py -net sam -mod sam_adpt -exp_name *eye* -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 32 -dataset eye --data_path *../data*``
 

2. Evaluation: The code can automatically evaluate on the test set during traing, and you can also manually evaluate it by running val.py for.


Results will be saved at `` ./logs/`` in default.
