# SSTA-PRS
Selective Spatio-Temporal Aggregation Based Pose Refinement System, WACV 2021.
### [Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Yang_Selective_Spatio-Temporal_Aggregation_Based_Pose_Refinement_System_Towards_Understanding_Human_WACV_2021_paper.pdf)


## Refined pose data
- [Skeleton data](https://drive.google.com/file/d/1tJuGEZGgADgjinN7oT2qEMAeKi9CRj8E/view?usp=sharing) of Smarthome:

![ad](https://github.com/walker-a11y/SSTA-PRS/blob/master/demo/smarthome.png)

## SST-A toolbox
To obtain refined pose sequence, you need to:
1. Extract 2D poses of input video from 3 expert estimators (e.g., [LCRNet](https://thoth.inrialpes.fr/src/LCR-Net/), [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose), ...);
2. Save 2D pose results into 'xxx-pose1.npz', 'xxx-pose2.npz', 'xxx-pose3.npz', .... Make sure '.npz' has the dimension of nb_frames *  nb_joints * 2 and joints indexes are consistent;
3. Run the script to get refined pose by SST-A. Ouput will be save as 'output.npz'.
```shell script
python tools/ssta.py --pose1 <xxx-pose1> --pose2 <xxx-pose2> --pose3 <xxx-pose3> --outname <output> (--gt <filename-gt if have>)
```
## 3D visualization
For 3D visualization, please use [VideoPose3D](https://github.com/YangDi666/Video_3D_Pose_Estimation#i-have-2d-pose).

## Citation
If you find this code useful for your research, please consider citing our paper:
```bibtex
@InProceedings{Yang_2021_WACV,
    author = {Yang, Di and Dai, Rui and Wang, Yaohui and Mallick, Rupayan and Minciullo, Luca and Francesca, Gianpiero and Bremond, Francois},
    title = {Selective Spatio-Temporal Aggregation Based Pose Refinement System: Towards Understanding Human Activities in Real-World Videos},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month = {January},
    year = {2021}
}
```
