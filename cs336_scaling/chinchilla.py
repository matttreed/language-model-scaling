import json
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_model_size():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(curr_dir, '../data/isoflops_curves.json')
    data = json.load(open(data_path))


    minimums = {}

    for datum in data:
        compute_budget = datum['compute_budget']
        parameters = datum['parameters']
        loss = datum['final_loss']

        if compute_budget not in minimums:
            minimums[compute_budget] = (parameters, loss)
        else:
            if loss < minimums[compute_budget][1]:
                minimums[compute_budget] = (parameters, loss)


    x_compute = np.array(list(minimums.keys()))
    y_size = np.array([value[0] for value in minimums.values()])

    sorted_indices = np.argsort(x_compute)
    x_compute_sorted = x_compute[sorted_indices]
    y_size_sorted = y_size[sorted_indices]

    log_x = np.log10(x_compute_sorted)
    log_y = np.log10(y_size_sorted)

    coefficients = np.polyfit(log_x, log_y, 1)
    poly = np.poly1d(coefficients)

    m = coefficients[0]
    b = coefficients[1]

    equation = f'y = {10**b:.2e} * x^{m:.2f}'

    print("Equation of the line:", equation)

    x_values = [1e23, 1e24]
    y_predicted = 10**poly(np.log10(x_values))
    print("Predicted y values for x =", x_values, ":", y_predicted)

    x_min_max = [2e18,2e24]

    plt.xscale('log')
    plt.yscale('log')
    plt.title("ISO-FLOPS Scaling Curve for Model Size")
    plt.scatter(x_compute_sorted, y_size_sorted, label='Minimum Loss Model Sizes for Each Compute Budget')
    plt.scatter(x_values, y_predicted, label='Extrapolated Model Sizes')
    plt.plot(x_min_max, 10**poly(np.log10(x_min_max)), color='red', label='Scaling Curve Fit')
    plt.xlabel('Compute Budget (FLOPs)')
    plt.ylabel('Size of Best Model (Parameters)')
    plt.legend()
    plt.show()


def plot_data_size():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(curr_dir, '../data/isoflops_curves.json')
    data = json.load(open(data_path))


    minimums = {}

    for datum in data:
        compute_budget = datum['compute_budget']
        parameters = datum['parameters']
        loss = datum['final_loss']

        if compute_budget not in minimums:
            minimums[compute_budget] = (parameters, loss)
        else:
            if loss < minimums[compute_budget][1]:
                minimums[compute_budget] = (parameters, loss)

    
    x_compute = np.array(list(minimums.keys()))
    y_data_size = np.array([x_compute[i]/ (6 * value[0]) for i, value in enumerate(minimums.values())])

    sorted_indices = np.argsort(x_compute)
    x_compute_sorted = x_compute[sorted_indices]
    y_data_size_sorted = y_data_size[sorted_indices]

    log_x = np.log10(x_compute_sorted)
    log_y = np.log10(y_data_size_sorted)


    coefficients = np.polyfit(log_x, log_y, 1)
    poly = np.poly1d(coefficients)

    m = coefficients[0]
    b = coefficients[1]

    equation = f'y = {10**b:.2e} * x^{m:.2f}'

    print("Equation of the line:", equation)

    x_values = [1e23, 1e24]
    y_predicted = 10**poly(np.log10(x_values))
    print("Predicted y values for x =", x_values, ":", y_predicted)

    x_min_max = [2e18,2e24]

    plt.xscale('log')
    plt.yscale('log')
    plt.title("ISO-FLOPS Scaling Curve for Data Size")
    plt.scatter(x_compute_sorted, y_data_size_sorted, label='Optimal Trained Tokens for Each Compute Budget')
    plt.scatter(x_values, y_predicted, label='Extrapolated Trained Tokens')
    plt.plot(x_min_max, 10**poly(np.log10(x_min_max)), color='red', label='Scaling Curve Fit')
    plt.xlabel('Compute Budget (FLOPs)')
    plt.ylabel('Number of Trained Tokens')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # plot_data_size()
    plot_model_size()

