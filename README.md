### Basic commands to run the program (Quick Start)
1. The command below will run on test mode, i.e: find the optimal quaternions on datasets 10 and 11, save the results, plot the results, and build panorama images.
```shell
python main.py test
```
2. The command below will train the quaternions on all 9 datasets, save the results, plot the results, and build panorama images.
```shell
python main.py train
```
3. The command below will train the quaternions on the specified datasets (dataset 1, 2, & 3 in this example), save the results, plot the results, and build panorama images.
```shell
python main.py train --datasets 1 2 3
```

### Using a different tracker
To use a different tracker, you can run the following command:
```shell
python main.py train --tracker <tracker_name>
```
where `<tracker_name>` is one of `pgd` (default, projected gradient descent), `ekf7` (7-state EKF), or `ekf4` (4-state EKF).

### Optional command line arguments for train mode
1. `--datasets` (only for training mode): List of datasets to train on.
  - __Default:__ All datasets (1 to 9).
2. `--plot_folder`: Folder to save the plots.
  - __Default:__ see `config.yaml` file.
3. `--panorama_folder`: Folder to save the generated panorama image.
  - __Default:__ see `config.yaml` file.
4. `--no_force_train` (only for training mode with `pgd` as the tracker): If this is passed as argument in the command line, then the program will try to find saved results (optimized quaternions, acceleration, etc) in the `results` folder (you can change this in `config.yaml`). Then it will use these saved `np.ndarray` to plot the results and generate panorama images. If the saved results are not found, then the program will train the quaternions.
  - __Default:__ `False`
5. `--use_vicon` (only for training mode): If this is passed as argument in the command line, then the program will use `vicon` dataset to generate panorama images.
  - __Default:__ `False`

### About `config.yaml` file
This is to be used by someone developing the program. The command line arguments should be sufficient for users to see results. The names of each config is self explanatory.
