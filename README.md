# Skipfuzz: Semantic-Input Annotation to Efficiently Fuzz TensorFlow

This repository contains the code accompanying the paper: "Semantic-Input Annotation to Efficiently Fuzz TensorFlow".

It contains:
1. the inputs used for fuzzing TensorFlow
2. scripts for running the tool


## Target API functions

The API functions used in our experiments can be found in `target_api.txt`. For thr larger run, the API functions are in `target_api.txt.more`.

## Running the scripts

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

9. Run `python fuzzer.py <epsilon>`, e.g. `python fuzzer.py 0.5`



Examples of the output of "Prepare inputs" can be found in the `inputs/` directory of this repository.
The categories are in `undominated_invariant_sets.txt`.
