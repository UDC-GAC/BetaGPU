import argparse
import statistics
import subprocess
import os

def filter_output(lines: list[str]) -> list[str]:
    filtered_lines = []
    found_separator = False
    for line in lines:
        if found_separator or line.strip() == "+------------------------------------+":
            found_separator = True
            filtered_lines.append(line)
    return filtered_lines

def parse_execution(execution_lines: list[str]) -> dict:

    ## Parse parameters
    parameters = dict(zip([line.strip('\t ').split(':')[0] for line in execution_lines if line.startswith("\t")],
                               [line.split(':')[1].strip('\t \n') for line in execution_lines if line.startswith("\t")]))

    ## Parse results
    executions = [float(line.split('\t')[-1]) for line in execution_lines if line.startswith("Itr[")]

    parameters["executions"] = executions
    parameters["ExecTime"] = statistics.median(executions)

    return parameters

def append_execution(execution: dict, full_executions: dict) -> dict:
    current_key = execution.get("Number of elements"), execution.get("Execution mode"), execution.get("Function name")

    already_inserted_execution = full_executions.get(current_key, None)
    if already_inserted_execution:
        already_inserted_execution["executions"].extend(execution.get("executions"))
        already_inserted_execution["ExecTime"] = statistics.median(already_inserted_execution.get("executions"))
        full_executions[current_key] = already_inserted_execution
    else:
        full_executions[current_key] = execution

def parse_executions(execution_lines: list[str]) -> list[dict]:
    executions = dict()
    end_idx = [i for i, item in enumerate(execution_lines) if item.startswith("Total time =")]
    current_start = 0
    while end_idx:
        current_execution_lines = execution_lines[current_start:end_idx[0]]
        current_start = end_idx[0]+1
        end_idx = end_idx[1:]

        execution = parse_execution(current_execution_lines)
        append_execution(execution, executions)

    return list(executions.values())

def read_input_file(filename: str) -> list[dict]:
    with open(filename, 'r') as in_file:
        lines = in_file.readlines()

    execution_lines = filter_output(lines)
    executions = parse_executions(execution_lines)
    return executions
    

def write_output_file(filename: str, executions: list[dict]) -> None:
    
    #Get all different "Number of element fields"
    elements = list(set([execution.get("Number of elements") for execution in executions]))
    elements.sort()

    #Get all different "Execution mode" fields
    execution_modes = list(set([execution.get("Execution mode") for execution in executions]))
    execution_modes.remove("Sequential")
    execution_modes.sort()


    header = "+"
    for execution_mode in execution_modes:
        header += f"\t{execution_mode}"

    exec_time_lines = []

    base_executions = [execution for execution in executions if execution["Execution mode"] == "Sequential"]

    for elementNum in elements:
        current_line = f"{elementNum}"

        execution = [execution for execution in base_executions if execution["Number of elements"] == elementNum][0]
        base_time = execution.get("ExecTime")
        current_executions = [execution for execution in executions if execution["Number of elements"] == elementNum and execution["Execution mode"] != "Sequential"]
        current_executions.sort(key=lambda x: x["Execution mode"])
        for current_execution in current_executions:
            current_execution["Speedup"] = base_time / current_execution.get("ExecTime")
            current_line += f"\t{current_execution.get('Speedup'):.2f}"
        
        exec_time_lines.append(current_line)

    with open(filename, 'w') as out_file:
        out_file.write(header + "\n")
        for line in exec_time_lines:
            out_file.write(line + "\n")

    
        

def main() -> None :
    parser = argparse.ArgumentParser(
        prog='RawOutputToCSV',
        description='parses raw output to dat'
    )
    parser.add_argument('-d', '--output_dat_folder', default='output/data/CF_iters/', help='The output folder for the dat files', required=False)
    parser.add_argument('-g', '--output_graph_folder', default='output/graphs/CF_iters/', help='The output folder for the graph files', required=False)
    parser.add_argument('-a', '--alphas', default='0.2,0.7,20.,300.', help='Coma separated list of alpha values to test')
    parser.add_argument('-b', '--betas', default='0.2,0.7,20.,300.', help="Coma separated list of beta values to test")
    parser.add_argument('-n', '--num_elements', default=10000,type=float, help="Number of elements to test")

    args = parser.parse_args()

    PROG_BINARY = "bin/benchmark_cont_frac"
    GNUPLOT_SCRIPT = "scripts/parsing/gnuplot_cont_frac_iterations.template"
    GNUPLOT_BINARY = "gnuplot"
    N = args.num_elements
    alphas = [float(alpha) for alpha in args.alphas.split(',')]
    betas = [float(beta) for beta in args.betas.split(',')]

    if len(alphas) == 0:
        raise ValueError("Alphas cannot be empty")
    
    if len(betas) == 0:
        raise ValueError("Betas cannot be empty")

    # Get num experiments
    num_experiments = len(alphas) * len(betas)
    current_experiment = 0
    digits = len(str(num_experiments))

    # For each experiment ...
    for alpha in alphas:
        for beta in betas:
            print(f"[{current_experiment:0{digits}d},{num_experiments}] Executing with alpha={alpha} and beta={beta}...")
            current_experiment += 1
            filename = f"CF_iters_A{alpha}_B{beta}.dat"
            ## Join dat folder with filename
            full_dat_filename = os.path.join(args.output_dat_folder, filename)
            ## Execute CPP program
            with open(full_dat_filename, 'w') as out_file:
                with open(os.devnull, 'w') as devnull:
                  subprocess.run([PROG_BINARY, '-n', str(N), '-a', str(alpha), '-b', str(beta)], stdout=out_file, text=True, stderr=devnull)
            ## Generate the graph by calling gnuplot
            ## Example for alpha 0.2 and beta 0.7:
            ## gnuplot -e "in_filename='output/data/cf_iters.dat'; out_filename='output/graphs/CF_iters_a0.2_b0.7.pdf'; alpha='0.2'; beta='0.7';" scripts/parsing/gnuplot_cont_frac_iterations.template
            subprocess.run([GNUPLOT_BINARY, f"-e", f"in_filename='{full_dat_filename}'; out_filename='{os.path.join(args.output_graph_folder, filename.replace('.dat', '.pdf'))}'; alpha='{alpha}'; beta='{beta}';", GNUPLOT_SCRIPT])
        
    print(f"[{current_experiment:0{digits}d},{num_experiments}] Done! All graphs have been generated!")

if __name__ == "__main__":
    main()