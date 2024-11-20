import re
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_and_plot_run_log(file_path):
    iterations = []
    losses = []
    accuracies_p = []
    accuracies_g = []
    accuracies_gt = []
    new_walks = []

    iter_pattern = r"Finished backprop iter (\d+)"
    loss_pattern = r"Loss: ([\d.]+)\."  # Note the added \. to catch the trailing period
    accuracy_pattern = r"Accuracy: <p> ([\d.]+)% <g> ([\d.]+)% <gt> ([\d.]+)%"
    new_walk_pattern = r"Iteration (\d+): new walk"

    with open(file_path, "r") as file:
        for line in file:
            iter_match = re.search(iter_pattern, line)
            if iter_match:
                iterations.append(int(iter_match.group(1)))

            loss_match = re.search(loss_pattern, line)
            if loss_match:
                losses.append(float(loss_match.group(1)))

            accuracy_match = re.search(accuracy_pattern, line)
            if accuracy_match:
                accuracies_p.append(float(accuracy_match.group(1)))
                accuracies_g.append(float(accuracy_match.group(2)))
                accuracies_gt.append(float(accuracy_match.group(3)))

            new_walk_match = re.search(new_walk_pattern, line)
            if new_walk_match:
                new_walks.append(int(new_walk_match.group(1)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(iterations, losses)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss over Iterations")

    ax2.plot(iterations, accuracies_p, label="p accuracy")
    ax2.plot(iterations, accuracies_g, label="g accuracy")
    ax2.plot(iterations, accuracies_gt, label="gt accuracy")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracies over Iterations")
    ax2.legend()

    # Add vertical lines for new walks
    # for walk in new_walks:
    #     ax1.axvline(x=walk, color='r', linestyle='--', alpha=0.5)
    #     ax2.axvline(x=walk, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def analyse_log_file(file):
    # Regular expressions to match IDs and Objs lines
    id_pattern = re.compile(r"IDs: \[([^\]]+)\]")
    obj_pattern = re.compile(r"Objs: \[([^\]]+)\]")
    iter_pattern = re.compile(r"Finished backprop iter (\d+)")
    step_pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}:")

    # Initialize data structures
    iteration_data = {}
    current_iteration = None
    id_to_obj_previous = {}
    current_step = 0

    with open(file, "r") as file:
        for line in file:
            # Check for iteration number
            iter_match = iter_pattern.search(line)
            if iter_match:
                current_iteration = int(iter_match.group(1))
                current_step = 0  # Reset step counter for new iteration
                continue  # Proceed to next line

            # Check for step line (assumed to start with timestamp)
            if step_pattern.match(line):
                current_step += 1

            # Extract IDs
            id_match = id_pattern.search(line)
            if id_match:
                ids = list(map(int, id_match.group(1).split(",")))
                continue  # IDs are followed by Objs, proceed to next line

            # Extract Objs
            obj_match = obj_pattern.search(line)
            if obj_match:
                objs = list(map(int, obj_match.group(1).split(",")))

                # Ensure current_iteration is set
                if current_iteration is None:
                    continue  # Skip if iteration is not identified yet

                # Store IDs and Objs for this iteration and step
                if current_iteration not in iteration_data:
                    iteration_data[current_iteration] = []
                iteration_data[current_iteration].append((current_step, ids, objs))

    # Now, process the data to find shifts with detailed information
    shifts = defaultdict(list)  # Key: iteration, Value: list of shift details
    id_to_obj_current = {}

    sorted_iterations = sorted(iteration_data.keys())

    for idx, iteration in enumerate(sorted_iterations):
        steps = iteration_data[iteration]
        # For each step in the iteration
        for step in steps:
            step_num, ids, objs = step
            # For each ID in the batch
            for batch_idx, (id_, obj) in enumerate(zip(ids, objs)):
                key = (batch_idx, id_)  # Identify by batch index and ID
                if key in id_to_obj_previous:
                    prev_info = id_to_obj_previous[key]
                    prev_obj = prev_info["obj"]
                    if obj != prev_obj:
                        # Environment has changed for this batch member
                        shifts[iteration].append(
                            {
                                "batch_idx": batch_idx,
                                "id": id_,
                                "prev_obj": prev_obj,
                                "new_obj": obj,
                                "prev_iteration": prev_info["iteration"],
                                "prev_step": prev_info["step"],
                                "current_iteration": iteration,
                                "current_step": step_num,
                            }
                        )
                # Update current mapping
                id_to_obj_current[key] = {"obj": obj, "iteration": iteration, "step": step_num}
        # After processing all steps in the iteration, update previous mapping
        id_to_obj_previous = id_to_obj_current.copy()
        id_to_obj_current.clear()

    # Output the iterations where shifts occurred with detailed information
    print("Environment shifts detected with detailed information:")
    with open("shifts_output.txt", "w") as output_file:
        for iteration in sorted(shifts.keys()):
            shift_list = shifts[iteration]
            if shift_list:
                output_file.write(f"\nIteration {iteration}: number of shifts = {len(shift_list)}\n")
                for shift in shift_list:
                    output_file.write(
                        f"  Batch index {shift['batch_idx']}, ID {shift['id']} changed from "
                        f"object {shift['prev_obj']} (Iteration {shift['prev_iteration']}, Step {shift['prev_step']}) "
                        f"to object {shift['new_obj']} (Iteration {shift['current_iteration']},\
                        Step {shift['current_step']})\n"
                    )


def plot_loss_with_switches(log_file_path, output_file_path, large_switch_threshold):
    # Initialize lists to store data
    iterations = []
    losses = []
    large_switch_iterations = []
    switch_counts = {}

    # Regular expressions to match lines in the log
    loss_pattern = re.compile(r"Loss: ([\d\.]+)")
    iteration_pattern = re.compile(r"Finished backprop iter (\d+)")
    # For the output file with switches
    switch_iteration_pattern = re.compile(r"Iteration (\d+): number of shifts = (\d+)")

    # Parse the training log file
    with open(log_file_path, "r") as log_file:
        current_iteration = None
        for line in log_file:
            # Check for iteration number
            iteration_match = iteration_pattern.search(line)
            if iteration_match:
                current_iteration = int(iteration_match.group(1))
                iterations.append(current_iteration)
                continue  # Move to the next line

            # Check for loss value
            loss_match = loss_pattern.search(line)
            if loss_match and current_iteration is not None:
                loss = float(loss_match.group(1)[:-1])
                losses.append(loss)
                continue  # Move to the next line

    # Parse the output file to get switch information
    with open(output_file_path, "r") as output_file:
        for line in output_file:
            # Check for switch iteration
            switch_iter_match = switch_iteration_pattern.match(line)
            if switch_iter_match:
                iteration = int(switch_iter_match.group(1))
                num_shifts = int(switch_iter_match.group(2))
                # Record iterations with shifts exceeding the threshold
                if num_shifts >= large_switch_threshold:
                    large_switch_iterations.append(iteration)
                    switch_counts[iteration] = num_shifts

    # Ensure the lists are aligned
    iterations = iterations[: len(losses)]

    # Plotting the loss over iterations
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, losses, label="Training Loss", color="blue")

    # Add markers for iterations with large switches
    for switch_iter in large_switch_iterations:
        if switch_iter in iterations:
            idx = iterations.index(switch_iter)
            plt.axvline(x=switch_iter, color="red", linestyle="--", alpha=0.5)
            # Optionally, add a text annotation for the number of shifts
            plt.text(
                switch_iter,
                losses[idx],
                f"{switch_counts[switch_iter]} shifts",
                rotation=90,
                va="bottom",
                ha="center",
                color="red",
                fontsize=8,
            )

    plt.title("Training Loss over Iterations with Large Batch Index Switches")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


parse_and_plot_run_log(
    "/Users/lukehollingsworth/Documents/PhD/SaxeLab/NeuralPlayground/NeuralPlayground/examples/agent_examples/begging_full/run.log"
)
# analyse_log_file('/Users/lukehollingsworth/Documents/PhD/SaxeLab/NeuralPlayground/NeuralPlayground/examples/agent_examples
# /test/run.log')
# plot_loss_with_switches('/Users/lukehollingsworth/Documents/PhD/SaxeLab/NeuralPlayground/NeuralPlayground/examples
# /agent_examples/test/run.log',
# '/Users/lukehollingsworth/Documents/PhD/SaxeLab/NeuralPlayground/NeuralPlayground/examples/agent_examples/
# test/output.txt', 50)
