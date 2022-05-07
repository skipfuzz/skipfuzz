# Skipfuzz: Semantic-Input Annotation to Efficiently Fuzz TensorFlow

This repository contains the code accompanying the paper: "Semantic-Input Annotation to Efficiently Fuzz TensorFlow".

It contains:
1. the inputs used for fuzzing TensorFlow
2. scripts for running the tool


## Target API functions

The 385 API functions used in our experiments can be found in `target_api.txt`. 
For the larger run with more API functions, the API functions are in `target_api.txt.more`.

## Running SkipFuzz


### Setup 
1. git clone https://github.com/tensorflow/tensorflow 
2. All scripts (extract_inputs.py, insert_into_db.py, ...) can be positioned in the tensorflow/tensorflow directory

### Prepare inputs

3. run `python extract_inputs.py`
4. run `python crawl_tf_api.py`
5. run `python insert_into_db.py`
6. run `python insert_into_db2.py`
7. run `python find_property_categories.py`
8. run `python find_nonsubsuming_categories.py`

### Fuzz

The core functionability of SkipFuzz is the fuzzer:

9. Run `python fuzzer.py <epsilon>`, e.g. `python fuzzer.py 0.5`



Examples of the output of "Prepare inputs" can be found in the `inputs/` directory of this repository.
The categories are in `undominated_invariant_sets.txt`.

Therefore, if desired, a docker container can be set up using the inputs from the input directory to fuzz tensorflow:

From the docker_setup directory,
1. run `patch` on the `tensorflow.diff` and a tensorflow repository (in a `original_tensorflow` directory)
2. build a docker image based on the Dockerfile (assumed to be named `skipfuzz`)
3. `docker run -v /home/<user>/<directory containing the tensorflow directory>/:/workspace  --gpus all  --name skipfuzz -it skipfuzz /bin/bash`
4. In the docker container,`pip install coverage scipy`
5. Now, `python fuzzer.py 0.5` can be run

