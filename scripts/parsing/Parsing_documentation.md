# How to generate performance graphs

In this document we will explain how to generate performance graphs from the output of the performance tests.

We will run the performance tests, parse the results and generate the graphs using the `gnuplot` tool.

For this, we are going to assume all the commands are run from the root of the repository.

## Running the performance tests

First, we need to run the performance tests. These tests are located in the `scripts` directory. As we are measuring the performance in different HPC systems, we have different scripts to run the tests in each system. For example, to run the tests in the `Finnisterrae III` system, we can use the commands in the `scripts/ft3/exec` directory.

To run the tests for the `Finnisterrae III`, which is based on slurm, we can use the following command:

```bash
sbatch scripts/ft3/exec/<performance_test_name>.sh
```

This will submit the job to the queue and the results will be stored in the `outputs` directory.

## Parsing the results

Once the tests are finished, we need to parse the results. For this, we can use the `raw_output_to_dat.py` script located in the `scripts/parsing` directory. This script will understand the output of the performance tests and will generate a `.dat` file with the parsed results.

To use the script, we can use the following command:

```bash
python scripts/parsing/raw_output_to_dat.py <input_raw_file> <output_dat_file>
```

## Generating the graphs

Finally, we can generate the graphs using the `gnuplot` tool. We have different gnuplot templatesdepending on the kind of graph we want to generate and the kind of test that have been ran. 

These templates are also located in the `scripts/parsing` directory, and generally are parameterized with the name of the input and output files. 

For example, to generate a graph from the input `.dat` file `output/example.dat`, and generate a graph `output/example_graph.pdf` we can use the following command:

```bash
gnuplot -e "in_filename='output/example.dat'; out_filename='output/example_graph.pdf'" scripts/parsing/gnuplot.template
```




