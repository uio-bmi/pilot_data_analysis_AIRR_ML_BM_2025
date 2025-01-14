## How to reproduce the findings of pilot data analysis involving one baseline ML method (logistic regression) and one widely-used, published ML method (Emerson method)

#### Notes specific to peer-review stage

- The datasets are not provided here owing to their large size. We do not intend to re-run all the pilot data analyses in this computational capsule, as the memory and CPU requirements are high and can span somewhere between 6 and 8 weeks depending upon the available computational resources. The goal of setting up this computational capsule is to install the required software and dependencies in containerized environment and demonstrate the reproducibility of the pilot data analysis. 
- Different versions of immuneML software were used for the two different ML methods: version 2.1.2 for logistic regression and version 2.2.6 for Emerson methods. These two versions were installed in separate computational capsules linked with the manuscript. The respective environments have to be used to reproduce the findings of each ML method. Note that this python package (AIRR-ML-BM-2025) is also required (and installed in both environments) to run the pilot data analysis.

#### To reproduce the findings

- This folder contains all the specification files required to reproduce the findings of pilot data analysis involving one baseline ML method (logistic regression) and one widely-used, published ML method (Emerson method) on data that will be used in Kaggle competition. The specification files used for Kaggle competition data, and data of different research questions (numbered Q1-Q7 in the manuscript) in the second phase of stratified performance analysis are organized in separate subdirectories. 
- In some cases of the experimental datasets, where the repertoire sizes and dataset properties are very different from the simulated datasets, different hyperparameter spaces are used for the experimental datasets versus the simulated datasets. In such cases, additional specification files are provided within the respective subdirectories. For instance, under `kaggle_data`, we provide `emerson_experimental.yaml` file that was used to run the Emerson method on experimental datasets.

- The commands below assume that there are three different directories containing (a) training data, (b) test data, and (c) one directory that contains both training and test repertoires. Although duplication, this was for the convenience of setting up the analysis with a particular configuration of immuneML software. 
- As long as all the repertoires are initially in a directory named `simulated_repertoires` (also for experimental datasets), there are a couple of utility functions that can be run as shown below to split the repertoires into separate `training` and `test` directories.

```bash
split_train_test_all_dirs -s /path/to/many/datasets
```
The above command will split `simulated_repertoires` directory into `training` and `test` directories for each dataset in the `/path/to/many/datasets` directory.

In special cases, where more control on the test dataset size and class balance is required, the following command can be used:

```bash
split_train_test_single_dir --dir /path/to/single/dataset --test_size 0.2 --balanced_test True
```

After activating the environment with respective immuneML version as described above, the following command can be run to reproduce the findings of pilot data analysis involving either logistic regression or Emerson method:

```bash
nohup profile_ml --super_path_sim_dirs /path/to/many/datasets/ --ml_yaml_file logistic_kmer.yaml --ml_method_name logistic --output_dir /path/to/pilot_analysis_results/ --n_parallel_jobs 3 --delay_minutes 10 >> stdout.txt 2>> stderr.txt &
```

The above command shows the example for logistic regression, but by replacing the `ml_yaml_file` with `emerson.yaml`, and `ml_method_name` with `emerson`, the Emerson method can be run. The `n_parallel_jobs` parameter can be adjusted based on the available computational resources. Note that Emerson method internally uses a highly memory-intensive operation involving CompAIRR tool, and thus spinning many parallel jobs using Emerson method is not advisable. The `delay_minutes` parameter is used to stagger the start of multiple jobs to avoid overloading the system. The `stdout.txt` and `stderr.txt` files will contain the output and error messages respectively.

Instead of running the ML methods on many datasets at once, the following command can be used to run the ML methods on a single dataset:

```bash
nohup immune-ml emerson_experimental.yaml /path/to/output/ >> stdout.txt 2>> stderr.txt &
```



The commands below are to be run for the analysis of interpretability of the ML models:

```bash
nohup emerson_interpretability --super_path_sim_dirs /path/to/many/datasets/ --ml_out_super_path /path/to/pilot_analysis_results/emerson_ml_output --n_threads 6 >> interpretability_stdout.txt 2>> interpretability_stderr.txt &
```

```bash
nohup logistic_interpretability --super_path_sim_dirs /path/to/many/datasets/ --ml_out_super_path /path/to/pilot_analysis_results/logistic_ml_output --n_threads 6 >> interpretability_stdout.txt 2>> interpretability_stderr.txt &
```