# SSTA-PRS
- Selective Spatio-Temporal Aggregation based Pose Refinement System, WACV 2021.

## Refined pose data
- [Skeleton data](https://drive.google.com/file/d/1tJuGEZGgADgjinN7oT2qEMAeKi9CRj8E/view?usp=sharing) of Smarthome:

![ad](https://github.com/walker-a11y/SSTA-PRS/blob/master/demo/smarthome.png)

## SST-A toolbox
- Get 2D poses for your video using 3 expert estimators (e.g. [LCRNet](https://thoth.inrialpes.fr/src/LCR-Net/), [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose), ...);
- Make the 2D pose results into '.npz' file (nb_frames * nb_joints * 2) and make sure the joints indexes are consistent;
- Run this script to get SST-A pose (saved as '.npz' file) :
```
python tools/ssta.py --pose1 <filename-pose1> --pose2 <filename-pose2> --pose3 <filename-pose3> --outname <output-filename> (--gt <filename-gt if have>)
```
## 3D reconstruction
- Use [VideoPose3D](https://github.com/YangDi666/Video_3D_Pose_Estimation#i-have-2d-pose) for 3D reconstruction and visualization.

## Citation
```
@InProceedings{Yang_2021_WACV,
    author    = {Yang, Di and Dai, Rui and Wang, Yaohui and Mallick, Rupayan and Minciullo, Luca and Francesca, Gianpiero and Bremond, Francois},
    title     = {Selective Spatio-Temporal Aggregation Based Pose Refinement System: Towards Understanding Human Activities in Real-World Videos},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {2363-2372}
}
```
