import argparse
import statistics

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

    parameters["ExecTime"] = statistics.median(executions)

    return parameters

def parse_executions(execution_lines: list[str]) -> list[dict]:
    executions = []
    end_idx = [i for i, item in enumerate(execution_lines) if item.startswith("Total time =")]
    current_start = 0
    while end_idx:
        current_execution_lines = execution_lines[current_start:end_idx[0]]
        current_start = end_idx[0]+1
        end_idx = end_idx[1:]

        execution = parse_execution(current_execution_lines)
        executions.append(execution)

    return executions

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
    print(base_executions)

    for elementNum in elements:
        current_line = f"{elementNum}"

        execution = [execution for execution in base_executions if execution["Number of elements"] == elementNum][0]
        base_time = execution.get("ExecTime")
        current_executions = [execution for execution in executions if execution["Number of elements"] == elementNum and execution["Execution mode"] != "Sequential"]
        current_executions.sort(key=lambda x: x["Execution mode"])
        for current_execution in current_executions:
            current_execution["Speedup"] = base_time / current_execution.get("ExecTime")
            current_line += f"\t{current_execution.get('Speedup')}"
        
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
    parser.add_argument('input_raw_file', help='The file produced by a sbatch execution of the exec script')
    parser.add_argument('output_dat_file', help='The output')

    args = parser.parse_args()

    input_data = read_input_file(args.input_raw_file)
    write_output_file(args.output_dat_file, input_data)

if __name__ == "__main__":
    main()