

SkipFuzz
=========

This repository accompanies the research paper "SkipFuzz: Active Learning-based Input Selection for
Fuzzing Deep Learning Libraries".



SkipFuzz applies active learning to fuzzing. It learns the possible input constraints of the deep learning library functions. This provides two key benefits:
1. It can produce a high proportion of _valid_ inputs
2. It can produce a high _diversity_ of inputs

Together, the two benefits allow SkipFuzz to find a good number of crashes and vulnerabilities (over 150 crashes).


The instructions and code for fuzzing tensorflow in in [tensorflow_fuzzer](tensorflow_fuzzer) and the code for fuzzing pytorch is in [pytorch_fuzzer](pytorch_fuzzer)





Input properties
=======================

The input properties used by SkipFuzz to distinguish different categories of inputs can be found at [list_of_properties.md](list_of_properties.md).
The input properties allow SkipFuzz to skip between inputs that are more likely to violate different input validation checks, and hence, allowing active learning.
SkipFuzz learns from invocations of different categories/classes of inputs to understand which input validation checks are performed by the library.





List of CVEs
===========

* [CVE-2022-29204](https://github.com/tensorflow/tensorflow/security/advisories/GHSA-hx9q-2mx4-m4pg)
* [CVE-2022-29202](https://github.com/advisories/GHSA-cwpm-f78v-7m5c)
* [CVE-2022-29213](https://github.com/advisories/GHSA-5889-7v45-q28m)
* [CVE-2022-29193](https://github.com/advisories/GHSA-2p9q-h29j-3f5v)
* [CVE-2022-29207](https://github.com/tensorflow/tensorflow/security/advisories/GHSA-5wpj-c6f7-24x8)
* [CVE-2022-29205](https://github.com/tensorflow/tensorflow/security/advisories/GHSA-54ch-gjq5-4976)
* CVE-2022-35934
* CVE-2022-35935
* CVE-2022-35960
* CVE-2022-35952
* CVE-2022-35997
* CVE-2022-35998
* CVE-2022-35988
* CVE-2022-41901
* CVE-2022-41909
* CVE-2022-41908
* CVE-2022-41893
* CVE-2022-41890
* CVE-2022-41889
* CVE-2022-41888
* CVE-2022-41887
* CVE-2022-41884
* CVE-2022-35991





