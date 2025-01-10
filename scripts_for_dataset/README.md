# Dataset Preparation

Here is the code for preparing the dataset for DiffPoseTalk training, sorry that I didn't have time to clean it up, so you may need to read the code to understand how to use it.

The steps are as follows:

1. First, use [SPECTRE](https://github.com/filby89/spectre) and [MICA](https://github.com/Zielon/MICA) to reconstruct the coefficients separately, and save the results in different folders. For example, the results of SPECTRE should be organized as follows:
    ```
    SPECTRE_coeffs
    ├── <person_1>
    │   ├── <video_1>.npz
    │   └── <video_2>.npz
    ├── <person_2>
    └── ...
    ```
    The `npz` file should contain at least the `shape`, `exp` and `pose` fields, with the shape of `(num_frames, dim_coeff)`.
2. Run `combine_mica_and_spectre.py` to combine the coefficients of the two methods. The combined coefficients will be saved in a new folder.
3. Use [6DRepNet](https://github.com/thohemp/6DRepNet) to reconstruct the head pose, and then run `fix_pose_prediction.py` to fix the pose prediction.
4. Run `make_combined_dataset.py` to generate the Lmdb database.
5. Run `calc_stats_random_sample.py` to generate the statistics of the dataset.
6. Run `generate_split_file.py` to generate the split file for training, validation and testing.
