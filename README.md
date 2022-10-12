

SkipFuzz
=========

This repository accompanies the research paper "SkipFuzz: Active Learning-based Input Selection for
Fuzzing Deep Learning Libraries".



SkipFuzz applies active learning to fuzzing. It learns the possible input constraints of the deep learning library functions. This provides two key benefits:
1. It can produce a high proportion of _valid_ inputs
2. It can produce a high _diversity_ of inputs

Together, the two benefits allow SkipFuzz to find a good number of crashes and vulnerabilities (over 150 crashes).

The code for fuzzing tensorflow in in [tensorflow_fuzzer](tensorflow_fuzzer) and the code for fuzzing pytorch is in [pytorch_fuzzer](pytorch_fuzzer)


Input properties
=======================

The input properties used by SkipFuzz to distinguish different categories of inputs can be found at [list_of_properties.md](list_of_properties.md).
The input properties allow SkipFuzz to skip between inputs that are more likely to violate different input validation checks, and hence, allowing active learning.
SkipFuzz learns from invocations of different categories/classes of inputs to understand which input validation checks are performed by the library.





