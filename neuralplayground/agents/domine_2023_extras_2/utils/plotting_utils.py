import matplotlib.pyplot as plt
import numpy as np
import torch



def color_map(output):
    vmin = min(output)
    vmax = max(output)
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.jet, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm._A = []
    return sm

def plot_curves(curves, path, title, legend_labels=None, x_label=None, y_label=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title)

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    colormap = plt.get_cmap("viridis")

    # Use numpy linspace to create the values array
    values = np.linspace(0, 1, len(curves))

    # Map the values to colors using the chosen colormap
    colors = [colormap(value) for value in values]

    for i, curve in enumerate(curves):
        label = legend_labels[i] if legend_labels else None
        color = colors[i % len(colors)]

        # Convert PyTorch tensors to NumPy arrays for plotting
        if isinstance(curve, torch.Tensor):
            curve = curve.detach().numpy()

        ax.plot(curve, label=label, color=color)

    if legend_labels:
        ax.legend()

    plt.savefig(path)
    plt.show()
    plt.close()