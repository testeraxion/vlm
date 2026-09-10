Top-level SLAM workspace.

Use this folder for external SLAM integrations (ORB-SLAM3/OpenVSLAM logs, wrappers, binaries).

Start by cloning the
[UZ-SLAMLab ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) repository (and
its Pangolin dependency) into the vendored `slam/` folder, where the rest of the
pipeline and the WSL build script expect them:

```bash
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git slam/ORB_SLAM3
git clone https://github.com/stevenlovegrove/Pangolin.git slam/Pangolin
```
