### How to run the program
1. The command below will train the quaternions on all 9 datasets, save the results, plot the results, and build panorama images.
```shell
python main.py
```
2. The command below will train the quaternions on the specified datasets (dataset 1, 2, & 3 in this example), save the results, plot the results, and build panorama images.
```shell
python main.py --datasets 1 2 3
```

### Optional command line arguments
1. `--datasets`: List of datasets to train on.
  - __Default:__ `other_configs.yaml` -> "other_configs" -> "datasets"
2. `--plot_folder`: Folder to save the plots
  - __Default:__ `other_configs.yaml` -> "other_configs" -> "plot_figures_folder"
3. `--panorama_folder`: Folder to save the generated panorama image
  - __Default:__ `other_configs.yaml` -> "other_configs" -> "panorama_images_folder"
4. `--no_force_train`: If this is passed, then the program will try to find saved results (optimized quaternions, acceleration, etc) in the `results` folder. Then it will use these saved `np.ndarray` to plot the results and generate panorama images.
  - __Default:__ `False`
5. `--use_vicon`: If this is passed, then the program will use `vicon` dataset to generate panorama images.
  - __Default:__ `False`

### About `config.yaml` file
This is to be used by someone developing the program. The command line arguments should be sufficient for users to see results. The names of each config is self explanatory.
