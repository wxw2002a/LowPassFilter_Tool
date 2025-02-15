import numpy as np  # NumPy is used for numerical computations, such as handling arrays, mathematical operations, and polynomial fitting.
import pandas as pd  # Pandas is used for data analysis and manipulation, including reading CSV files into DataFrame objects.
import matplotlib.pyplot as plt  # Matplotlib is a plotting library for creating static, animated, and interactive visualizations in Python.

# A very small constant used to avoid division-by-zero errors or extremely small denominators in certain calculations.
EPSILON = 1e-18

def interpolate_quadratic_vertex(x_vals, y_vals):
    """
    Fits a second-degree polynomial (quadratic) to the provided data points (x_vals, y_vals),
    then computes and returns the x-coordinate of the vertex of that quadratic.

    :param x_vals: A list or array of x-coordinates.
    :param y_vals: A list or array of y-coordinates corresponding to x_vals.
    :return: The x-coordinate of the vertex of the fitted quadratic.
    :raises ValueError: If the leading (quadratic) coefficient is effectively zero,
        meaning a parabola cannot be formed uniquely.
    """
    # np.polyfit fits a polynomial of degree 2 to the given x_vals and y_vals,
    # returning the coefficients [a, b, c] for a*x^2 + b*x + c.
    p = np.polyfit(x_vals, y_vals, 2)
    
    # p[0] is the coefficient a. If |a| < EPSILON, we can't compute a meaningful vertex.
    if abs(p[0]) < EPSILON:
        raise ValueError("Quadratic coefficient is near zero; cannot find a unique vertex.")
    
    # The x-coordinate of the vertex of a quadratic a*x^2 + b*x + c is -b / (2a).
    return -p[1] / (2 * p[0])

def generate_signal(signal_type, n=100, t1=3, t2=0.005):
    """
    Generates different types of signals based on 'signal_type': 'step', 'ramp', or 'sin'.
    The function can be extended to support more signal types.

    :param signal_type: A string indicating which type of signal to generate ('step', 'ramp', 'sin').
    :param n: The sampling rate or number of samples per 1 unit of time (used in time discretization).
    :param t1: The end time for 'step' and 'ramp' signals (in seconds or arbitrary units).
    :param t2: A parameter used as threshold time for the step or ramp signals, 
               or as a small offset for demonstration (depends on signal type).
    :return: A NumPy array representing the generated signal.
    :raises ValueError: If 'signal_type' is not recognized or supported.
    """
    # Create a time array from 0 to t1 (inclusive), stepping by 1/n.
    t = np.arange(0, t1 + 1/n, 1/n)
    
    if signal_type == 'step':
        # Step signal: 0 before t2, 1 after t2 (including t2).
        # (t >= t2) returns a boolean array; astype(float) converts True -> 1.0, False -> 0.0
        return (t >= t2).astype(float)
    
    elif signal_type == 'ramp':
        # Ramp signal:
        #   from time 0 to t2, output increases linearly (like a ramp).
        #   after t2, it stays at a constant level.
        nx = round(t2 * n)    # The index at which to switch to a constant
        x_in = np.zeros_like(t)
        
        # For indices less than nx, we let x_in = t (a linear increase).
        x_in[:nx] = t[:nx]
        
        # For indices >= nx, the value remains the last ramp value (t[nx-1]).
        x_in[nx:] = t[nx - 1]
        return x_in
    
    elif signal_type == 'sin':
        # A simple sine wave signal:
        freq = 1
        # We create a time array at a 0.01 step from 0 to 100.01, then compute sine.
        t_sin = np.arange(0, 100.01, 0.01)
        return np.sin(2 * np.pi * freq * t_sin)
    
    else:
        raise ValueError(f"Unsupported signal_type: {signal_type}")

def read_csv_signal(filepath, column=1, max_samples=None):
    """
    Reads a CSV file from a given filepath and extracts a specific column of data.
    Optionally restricts the maximum number of samples returned.

    :param filepath: The path to the CSV file.
    :param column: The zero-based index of the column to extract.
    :param max_samples: If provided, restricts the returned data to this many samples.
    :return: A NumPy array of the extracted data values.
    """
    # Read the CSV file as a DataFrame. 'header=None' means the CSV has no header row.
    df = pd.read_csv(filepath, header=None)
    
    # Extract the specified column as a NumPy array.
    data = df.iloc[:, column].to_numpy()
    
    # If a maximum number of samples is given, slice the array.
    if max_samples is not None:
        data = data[:max_samples]
    
    return data

def R_calculation(x, y, p, T, h):
    """
    Calculates a quantity 'R' used in the iterative update in the lp_filter function.
    It applies an exponential weighting and an lp-norm-like transformation.

    :param x: The array (window) of input signal samples.
    :param y: The candidate filter output value for which we compute R.
    :param p: The norm-like parameter (Lp_norm_number).
    :param T: A constant used in the exponential.
    :param h: A step size for the exponent, used in np.exp(...) inside the function.
    :return: The computed R value, which is 0 if it is smaller than EPSILON.
    """
    # n is the length of the window x.
    n = len(x)
    
    # We construct an array of exponent factors: exp( (1 through n)*h / T ).
    exp_vals = np.exp(np.arange(1, n + 1) * h / T)
    
    # s1: The contribution of the last sample in x (x[-1]) compared to y, scaled by an exponential.
    s1 = (h * T / 2.0) * (abs(x[-1] - y)**(p - 1)) * np.sign(x[-1] - y) * exp_vals[-1]
    
    # s2: Sum of contributions of the rest of the samples in x, using exponential weights.
    diff = x[:-1] - y
    s2 = np.sum((np.abs(diff)**(p - 1)) * np.sign(diff) * exp_vals[:-1]) * (h / T)
    
    # Sum the two parts.
    s21 = s1 + s2
    
    # If s21 is very close to zero, return 0 to avoid numerical issues.
    return 0.0 if abs(s21) < EPSILON else (abs(s21)**(1 / (p - 1))) * np.sign(s21)

def lp_filter(
    Lp_norm_number,
    number=1,
    N=300,
    n=500,
    y_init=0,
    y_delta=0.01,
    y_beta=0.001,
    h=0.01,
    T=1
):
    """
    A custom filter that uses an Lp-norm-like iterative approach to track and filter an input signal.
    It attempts to adaptively find output y that minimizes (y - R_calculation(...))^2.

    :param Lp_norm_number: The p in the Lp norm. Typical values could be >= 1.
    :param number: An integer to select which input signal to use:
                   1 -> step, 2 -> ramp, 3/4/5 -> read from CSV, else -> sin.
    :param N: Number of samples of the output to produce (and to read from the input).
    :param n: The window size for the R_calculation function.
    :param y_init: The initial output value.
    :param y_delta: A small increment used to generate y_candidates around the current y.
    :param y_beta: A step used in the 'greedy' search to adjust y_out further.
    :param h: Step size used in exponent calculations for R.
    :param T: A time constant used in exponent calculations for R.
    :return: (y, x_in[:N])
        y     -> The filtered output array of length N.
        x_in  -> The input signal array (truncated to length N if necessary).
    """

    # Select the input signal based on 'number'.
    if number == 1:
        x_in = generate_signal('step')
    elif number == 2:
        x_in = generate_signal('ramp')
    elif number == 3:
        x_in = read_csv_signal(r"C:\Users\wangx\Desktop\importeddocument\Surge_x2x.csv", column=0, max_samples=10000)
         #user can use their own data
    elif number == 4:
        x_in = read_csv_signal(r"C:\Users\wangx\Desktop\importeddocument\scope_62_10k.csv", column=1)
         #user can use their own data
    elif number == 5:
        x_in = read_csv_signal(r"C:\Users\wangx\Desktop\importeddocument\Motor wires.csv", column=1, max_samples=5000)
         #user can use their own data
    else:
        # Default case, generate a sine signal if 'number' doesn't match 1-5
        x_in = generate_signal('sin')
    
    # If the generated (or read) signal is shorter than N, limit N.
    if len(x_in) < N:
        N = len(x_in)
    
    # Initialize the output array of length N, setting the first output to y_init.
    y = np.zeros(N, dtype=float)
    y[0] = y_init
    
    # Store the Lp norm parameter in a local variable for convenience.
    p = Lp_norm_number
    
    # We iterate over each sample from 1 to N-1 (since y[0] is already set).
    for step in range(1, N):
        
        # Create a window of size n from the input x_in.
        # If 'step' is less than n, we fill the window such that the beginning is zeros 
        # and the end includes x_in up to the current step.
        x_window = np.zeros(n)
        if step < n:
            x_window[n - step - 1:] = x_in[:step + 1]
        else:
            x_window = x_in[step - n + 1 : step + 1]
        
        # Get the previous output y. This is our 'center' point for exploration (y_delta steps).
        y2 = y[step - 1]
        
        # Define three candidate outputs around the current y2: (y2 - y_delta, y2, y2 + y_delta).
        y_candidates = np.array([y2 - y_delta, y2, y2 + y_delta])
        
        # For each candidate, compute R_calculation(...).
        R_vals = np.array([R_calculation(x_window, cand, p, T, h) for cand in y_candidates])
        
        # The cost function j is defined as (y_candidate - R_val)^2.
        j = (y_candidates - R_vals)**2
        
        # Use our quadratic interpolation to guess the 'best' y_out by fitting a parabola
        # to the (y_candidates, j) points, and then finding the vertex.
        y_out = interpolate_quadratic_vertex(y_candidates, j)
        
        # Define a helper function that computes |val - R_calculation(...)|.
        # We want to minimize this difference.
        def diff_fn(val):
            return abs(val - R_calculation(x_window, val, p, T, h))
        
        # Check the difference at the interpolated y_out.
        diff_old = diff_fn(y_out)
        
        # We'll try nudging y_out upwards (by y_beta), see if that reduces the difference.
        y_out_pos = y_out + y_beta
        diff_new = diff_fn(y_out_pos)
        
        if diff_new < diff_old:
            # If moving upwards in y decreases the difference, keep going in that direction
            while diff_new < diff_old:
                y_out = y_out_pos
                diff_old = diff_new
                y_out_pos += y_beta
                diff_new = diff_fn(y_out_pos)
        else:
            # Otherwise, try nudging y_out downwards.
            y_out_neg = y_out - y_beta
            diff_new = diff_fn(y_out_neg)
            while diff_new < diff_old:
                y_out = y_out_neg
                diff_old = diff_new
                y_out_neg -= y_beta
                diff_new = diff_fn(y_out_neg)
        
        # Store our updated y_out as the output for this step.
        y[step] = y_out
    
    # Return the filtered output y and the (truncated) input x_in.
    return y, x_in[:N]

def plot_lp_filter_by_number(num):
    """
    Plots different Lp filter outputs for a given 'num' (which selects the input signal).
    It compares the input signal (shown in gray) to the outputs from various p-values in lp_values.

    :param num: An integer specifying which input signal to plot (1, 2, 3, 4, 5, or other for 'sin').
    """
    # List of different p-values we want to test.
    lp_values = [2, 1.8, 1.6, 1.4, 1.2, 1.1, 1.02]
    
    # A set of distinct colors for plotting the input and each output.
    colors = ["#E0E0E0", "#CC0000", "#FF8000", "#FFFF00", "#00FF00", "#00FFFF", "#0080FF", "#FF00FF"]
    
    # Create a new figure for plotting.
    plt.figure()
    
    # Get one reference output from the filter with p=2, also retrieving the input signal.
    # '_' is used to capture the unused output from lp_filter (since we only need x_in).
    _, x_in = lp_filter(2, number=num)
    
    # Plot the input signal in gray (colors[0]).
    plt.plot(x_in, color=colors[0], label="Input")
    
    # For each p-value in lp_values, compute the filter output and plot.
    for i, lp_val in enumerate(lp_values):
        y_out, _ = lp_filter(lp_val, number=num)
        plt.plot(y_out, color=colors[i+1], label=f"lp={lp_val}")
    
    # Add plot title and labels.
    plt.title(f"Lp Filter Responses (number={num})")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    
    # Show legend.
    plt.legend(loc="best")
    plt.show()

def plot_all_separately():
    """
    Calls plot_lp_filter_by_number for a series of numbers (1 through 6),
    generating a separate plot for each input signal configuration.
    """
    for num in [1, 2, 3, 4, 5, 6]:
        plot_lp_filter_by_number(num)

# If this script is run as the main program, generate the plots for all signals.
if __name__ == "__main__":
    plot_all_separately()
