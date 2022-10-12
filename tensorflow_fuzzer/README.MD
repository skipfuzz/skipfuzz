

## Running SkipFuzz


### Setup 
1. git clone https://github.com/tensorflow/tensorflow 
2. All scripts (instrument_and_run_tests.py, insert_db.py, ...) can be positioned in the tensorflow/tensorflow directory

a docker container can be set up using the inputs from the input directory to fuzz tensorflow:


## Using docker (optional) 
From the docker_setup directory,

1. run `git clone https://github.com/tensorflow/tensorflow original_tensorflow`
2. run `git checkout <desired version number>`

3. run `patch` on `tensorflow.diff` on the tensorflow repository (in a original_tensorflow directory) (This removes some issues when trying to run test cases of tensorflow in a non-standard way)
4. build a docker image based on the Dockerfile (assumed to be named skipfuzz in the subsequent steps) (`docker build -t skipfuzz`)
5. docker run -v /home/<user>/<directory containing the tensorflow directory>/:/workspace --gpus all --name skipfuzz -it skipfuzz /bin/bash
6. In the docker container,pip install coverage scipy
7. assuming that the current working directory is in /workspace/tensorflow/tensorflow/, move the files in the input directory from this repository to the current working directory. Likewise, the scripts + target_api.txt + all_functions.txt can be moved there.

### Prepare inputs (step 1 in the paper)

In the `tensorflow/tensorflow` directory,

3. run `python instrument_and_run_tests.py`
4. run `python construct_dbs.py`
5. run `python insert_db.py`
6. run `python find_input_categories.py`
7. run `python select_strongest_categories.py`

### Active Learning-driven fuzzing (step 2)

The core functionability of SkipFuzz is the fuzzer:

9. Run `python fuzz.py`


