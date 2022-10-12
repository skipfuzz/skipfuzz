

## Running SkipFuzz

Create a docker image based on the docker

`docker build -t  pytorch_skipfuzz`



### Prepare inputs (step 1 in the paper)

3. run `python instrument_and_run_tests.py`
4. run `python construct_dbs.py`
5. run `python insert_db.py`
6. run `python find_input_categories.py`
7. run `python select_strongest_categories.py`

### Active Learning-driven fuzzing (step 2)

The core functionability of SkipFuzz is the fuzzer:

9. Run `python fuzz.py`


