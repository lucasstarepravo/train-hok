import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mcolors


def plot_training_pytorch(history, log_x=False, log_y=False):
    """
    Plots the training and validation loss with options for logarithmic x and y axes.

    Parameters:
        history (object or dict): Object containing 'training_loss' and 'val_loss' attributes
                                  or a dictionary with 'history' as a key.
        log_x (bool): Whether to use a logarithmic scale for the x-axis.
        log_y (bool): Whether to use a logarithmic scale for the y-axis.
    """
    try:
        training_losses = history.training_loss
        validation_losses = history.val_loss
    except AttributeError:
        training_losses, validation_losses = history['history']

    # Create a range of epochs for the x-axis
    epochs = range(1, len(training_losses) + 1)

    # Create the plot
    plt.figure()
    plt.plot(epochs, training_losses, 'b', label='Training Loss')
    plt.plot(epochs, validation_losses, 'r', label='Validation Loss')

    # Set title and labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Apply logarithmic scale if specified
    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')

    # Add legend and show the plot
    plt.legend()
    plt.show()


# This function will be used to plot 1 or 3 graphs, either only one graph with the % prediction error, or the one just
# mentioned + 2 graphs (i) showing the actual weight and the node positions and the other (ii) showing the predicted
# weights and positions
def plot_node_prediction_error(pred_l, actual_l, coor_subset, node='random', size=20, save_path=None):
    '''features is supposed to be the test set of features already scaled back and with coordinates, NOT distance'''
    '''This function only plots the following:
    - Absolute error
    - Predicted weights
    - Actual weights'''

    N = len(pred_l)
    if node == 'random':
        plot_i = np.random.randint(0, N)
    else:
        plot_i = int(N)

    features_node = coor_subset[plot_i, :]
    pred_l_node = pred_l[plot_i, :]
    actual_l_node = actual_l[plot_i, :]
    pred_l_node = pred_l_node.detach().numpy()
    error = pred_l_node - actual_l_node

    features_node = features_node.reshape(-1, 2)
    ref_node = features_node[0, :]
    neigh_nodes = features_node[1:, :]

    def save_or_show():
        plt.tight_layout()  # Adjust layout to minimize blank space
        if save_path:
            plt.savefig(save_path.format(plot_type=plot_type), bbox_inches='tight')  # Save with minimal blank space
        else:
            plt.show()
        plt.close()

    def add_grids():
        plt.grid(which='major', linestyle='-', linewidth=0.5, zorder=0)
        plt.grid(which='minor', linestyle='--', linewidth=0.3, zorder=0)
        plt.minorticks_on()

    def format_axes():
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        ax = plt.gca()
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    def format_colorbar(cbar):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        cbar.formatter = formatter
        cbar.ax.yaxis.set_major_formatter(formatter)
        cbar.update_ticks()

    # Absolute Error Plot
    plot_type = 'absolute_error'
    plt.scatter(ref_node[0], ref_node[1], c='pink', label='Reference Node', s=size, zorder=2)
    scatter = plt.scatter(neigh_nodes[:, 0], neigh_nodes[:, 1], c=abs(error), label=None, s=size, zorder=2)
    cbar = plt.colorbar(scatter)
    format_colorbar(cbar)  # Format the color bar
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    add_grids()
    format_axes()
    save_or_show()

    # Predicted Weights Plot
    plot_type = 'predicted_weights'
    plt.scatter(ref_node[0], ref_node[1], c='pink', label='Reference Node', s=size, zorder=2)
    scatter = plt.scatter(neigh_nodes[:, 0], neigh_nodes[:, 1], c=pred_l_node, label=None, s=size, zorder=2)
    cbar = plt.colorbar(scatter)
    format_colorbar(cbar)  # Format the color bar
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    add_grids()
    format_axes()
    save_or_show()

    # Actual Weights Plot
    plot_type = 'actual_weights'
    plt.scatter(ref_node[0], ref_node[1], c='pink', label='Reference Node', s=size, zorder=2)
    scatter = plt.scatter(neigh_nodes[:, 0], neigh_nodes[:, 1], c=actual_l_node, label=None, s=size, zorder=2)
    cbar = plt.colorbar(scatter)
    format_colorbar(cbar)  # Format the color bar
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    add_grids()
    format_axes()
    save_or_show()


def plot_c(x_axis, y_axis, optimal_c):
    # Calculate y values for the best fit line using the given optimal_c
    x_range = np.linspace(min(x_axis), max(x_axis), 100)
    y_best_fit = (1 / (optimal_c * x_range))**2

    # Plotting
    plt.figure(figsize=(10, 6))
    # Use 'x' marker for more precise indication of points
    plt.plot(x_range, y_best_fit, 'r-', label=f'Best Fit Line (c={optimal_c:.2f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of Initial Points with Optimal c')
    plt.legend()
    plt.grid(True)
    plt.show()
