# SERS_Analysis


Running a new experiment
---
1) Modify 'config/current_config.py'
2) Organize the data folder as demonstrated below.
3) Run 'main.ipynb'. You may comment out the first cell if you have already installed all required packages.
4) Investigate the results store under the result folder (the exact name of the result folder is also specified in 'config/current_config.py')

Exporting videos
---
Install ffmpeg 
- <code> sudo apt install ffmpeg </code>
  
Then, install [ffmpeg-python](https://github.com/kkroening/ffmpeg-python)
- <code> pip install ffmpeg-python </code>

Make changes to models/utils_models.py if you want to export a video with specific frame rate and resolution. Also, it is possible to export .gif files by uncommenting a few lines in models/utils_models.py.


Important files
---
- 'config/current_config.py': Store all parameters used to define experiments
- 'main.ipynb': Analyze experiment(s) using all the specified machine learning methods provided by the user in 'config/current_config.py'
  -   If 'data_split_method' is set to 
      - 'inter', a single .txt file from each class will be used as the test data for the current experiment, and the remaining .txt files will be used as the training data. The number of repetitions is automatically set to the minimum number of files per class.
      - 'intra', the spectra from all the files will be grouped together, and each spectra in the group will be randomly included the training and test sets with the proportion specified through the 'fraction_test_intra' variable. The number of repetitions is specified by users through the 'num_repetitions_intra' variable.
      - 'by-folders', all the files in './data_manual_split/train/' are used as training data and the files in './data_manual_split/test/' are used as test data.
  -   If 'experiment_name' is a string (not a list of strings) and is a valid experiment, it will perform the experiment. Otherwise, it will run all the experiments  under the data directory where each subfolder is treated as a valid experiment.


How to organize the data folder
---
If you set 'data_split_method' to 'intra' or 'inter', organize 'data_auto_split' as follows:
- Experiment 1
  - (Optional) raman_shifts.csv
  - Class1
    - file1.txt
    - file2.txt
    - ...
    - last_file.txt
  - Class2
    - file1.txt
    - file2.txt
    - ...
    - last_file.txt
  - ...
  - Class N1
    - file1.txt
    - file2.txt
    - ...
    - last_file.txt
- Experiment 2
  - (Optional) raman_shifts.csv
  - Class1
    - file1.txt
    - file2.txt
    - ...
    - last_file.txt
  - Class2
    - file1.txt
    - file2.txt
    - ...
    - last_file.txt
  - ...
  - Class N2
    - file1.txt
    - file2.txt
    - ...
    - last_file.txt
- ...

If you set 'data_split_method' to 'by-folders', organize 'data_manual_split' as follows:
- Experiment 1
  - (Optional) raman_shifts.csv
  - train
    - Class1
      - file1.txt
      - file2.txt
      - ...
      - last_file.txt
    - Class2
      - file1.txt
      - file2.txt
      - ...
      - last_file.txt
    - ...
    - Class N1
      - file1.txt
      - file2.txt
      - ...
      - last_file.txt
  - test
    - Class1
      - file1.txt
      - file2.txt
      - ...
      - last_file.txt
    - Class2
      - file1.txt
      - file2.txt
      - ...
      - last_file.txt
    - ...
    - Class N1
      - file1.txt
      - file2.txt
      - ...
      - last_file.txt
- Experiment 2
  - (Optional) raman_shifts.csv
  - train
    - Class1
      - file1.txt
      - file2.txt
      - ...
      - last_file.txt
    - Class2
      - file1.txt
      - file2.txt
      - ...
      - last_file.txt
    - ...
    - Class N1
      - file1.txt
      - file2.txt
      - ...
      - last_file.txt
  - test
    - Class1
      - file1.txt
      - file2.txt
      - ...
      - last_file.txt
    - Class2
      - file1.txt
      - file2.txt
      - ...
      - last_file.txt
    - ...
    - Class N1
      - file1.txt
      - file2.txt
      - ...
      - last_file.txt


Note
---
- Hyperparameter tuning is not currently available for CatBoost and LGBM. Also, the default hyperparameter lists are not extensive, so the performance will be suboptimal. Feel free to modify the lists (For example, search for 'hparams' in './models/performRF.py' and modify it to include more parameters to sweep).
- Supported ML method options are
  - pca
  - pc-lda
  - svm, pc-svm, lda-svm
  - rf, pc-rf, lda-rf
  - catboost, pc-catboost, lda-catboost
  - lgbm, pc-lgbm, lda-lgbm
  - logistic-reg, pc-logistic-reg, lda-logistic-reg


To-do
---
- Implement more-than-one-chip cross-validation for interchip experiments

Main server: NAI's Bob
---

License
---
This work is released under the MIT License.
