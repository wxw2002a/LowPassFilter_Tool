# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# A small constant to avoid division by zero in calculations.
EPSILON = 1e-18

def interpolate_quadratic_vertex(x_vals, y_vals):
    """
    Fit a quadratic polynomial (degree 2) to the provided data points (x_vals, y_vals)
    and return the x-coordinate of its vertex.
    
    This function uses numpy.polyfit to get the coefficients [a, b, c] of the quadratic:
        a*x^2 + b*x + c
    
    The vertex is computed as -b/(2*a). If the quadratic coefficient 'a' is too small,
    a ValueError is raised to indicate that a unique vertex cannot be determined.
    
    Parameters:
        x_vals : array-like
            The x-values for the data points.
        y_vals : array-like
            The y-values corresponding to x_vals.
    
    Returns:
        float: The x-coordinate of the vertex of the fitted quadratic polynomial.
    """
    p = np.polyfit(x_vals, y_vals, 2)  # Fit a quadratic polynomial.
    if abs(p[0]) < EPSILON:  # Check if the quadratic coefficient is near zero.
        raise ValueError("Quadratic coefficient is too small to determine a unique vertex.")
    return -p[1] / (2 * p[0])  # Compute the vertex x-coordinate.

def R_calculation(x, y, p, T, h):
    """
    Compute the R value used for updating the LP filter output.
    
    The algorithm applies an exponential weighting to a window of input samples (x).
    It computes two components:
      - s1: The contribution of the last sample, scaled by the step size and time constant.
      - s2: The sum of contributions of the remaining samples.
    The two are added to yield s21, which is then transformed using an Lp-like norm.
    
    Parameters:
        x : array-like
            A window (subset) of input samples.
        y : float
            A candidate output value for which R is calculated.
        p : float
            The Lp norm parameter.
        T : float
            The time constant used in the exponential weighting.
        h : float
            The step size used in computing the exponential factors.
    
    Returns:
        float: The computed R value. If the combined value s21 is very small,
               returns 0.0 to avoid numerical issues.
    """
    n = len(x)
    # Compute exponential weights: exp((i * h) / T) for each sample in the window.
    exp_vals = np.exp(np.arange(1, n + 1) * h / T)
    
    # s1 is the contribution from the last sample in the window.
    s1 = (h  / (2.0*T)) * (abs(x[-1] - y) ** (p - 1)) * np.sign(x[-1] - y) * exp_vals[-1]
    
    # s2 sums the contributions of the remaining samples.
    diff = x[:-1] - y
    s2 = np.sum((np.abs(diff) ** (p - 1)) * np.sign(diff) * exp_vals[:-1]) * (h / T)
    
    s21 = s1 + s2  # Combine both contributions.
    
    if abs(s21) < EPSILON:
        return 0.0  # Return 0 to avoid numerical instability.
    else:
        # Return the Lp-based transformation of s21.
        return (abs(s21) ** (1 / (p - 1))) * np.sign(s21)

def Lp_filter(xin, T, p, h=0.01, n=None, y_init=0, y_delta=0.01, y_beta=0.001):
    """
    Process the input step signal 'xin' using the LP filter algorithm and return the filtered output.
    
    The algorithm iterates over each sample in the input signal, using a sliding window of 
    size 'n' to compute a candidate output update based on the R_calculation. It then uses a
    local search strategy (adjusting upward and downward by y_beta) to refine the output.
    
    Parameters:
        xin : list or array-like
            The input signal (e.g., a step signal).
        T : float
            The time constant for exponential weight calculation.
        p : float
            The Lp norm parameter (e.g., 1.1, 1.02, 1.3, etc.).
        h : float, optional
            The step size used in the exponential calculation (default is 0.01). This represents the sampling time.
        n : int, optional
            The window size for processing. Defaults to 500 if not provided.
        y_init : float, optional
            The initial output value (default is 0).
        y_delta : float, optional
            The increment used to generate candidate outputs (default is 0.01).
        y_beta : float, optional
            The step size for local adjustment of the output (default is 0.001).
    
    Returns:
        tuple:
            y : numpy array
                The filtered output signal.
            xin : numpy array
                The input signal (converted to an array, may be truncated to match y's length).
    """
    xin = np.array(xin, dtype=float)  # Convert input list to a numpy array.
    N = len(xin)  # Total number of samples in the input.
    if n is None:
        n = 500  # Default window size.
    y = np.zeros(N, dtype=float)  # Initialize output array.
    y[0] = y_init  # Set the initial output value.

    # Process each sample starting from the second one.
    for step in range(1, N):
        # Create a window of 'n' samples ending at the current step.
        if step < n:
            # If there are not enough samples, pad the beginning with zeros.
            x_window = np.zeros(n)
            x_window[n - step - 1:] = xin[:step + 1]
        else:
            # Otherwise, use the last 'n' samples.
            x_window = xin[step - n + 1: step + 1]
        
        y_prev = y[step - 1]  # The previous output value.
        # Generate three candidate outputs: one lower, one equal, and one higher than the previous value.
        y_candidates = np.array([y_prev - y_delta, y_prev, y_prev + y_delta])
        
        # Compute the R value for each candidate.
        R_vals = np.array([R_calculation(x_window, cand, p, T, h) for cand in y_candidates])
        # Calculate the squared error between each candidate and its corresponding R value.
        j = (y_candidates - R_vals) ** 2
        
        try:
            # Use quadratic interpolation to estimate the optimal candidate.
            y_out = interpolate_quadratic_vertex(y_candidates, j)
        except ValueError:
            # If interpolation fails, choose the middle candidate.
            y_out = y_candidates[1]
        
        # Define a helper function that computes the absolute difference for a candidate.
        def diff_fn(val):
            return abs(val - R_calculation(x_window, val, p, T, h))
        
        diff_old = diff_fn(y_out)
        # Try adjusting upward: increment candidate by y_beta and check if improvement occurs.
        y_out_pos = y_out + y_beta
        diff_new = diff_fn(y_out_pos)
        if diff_new < diff_old:
            while diff_new < diff_old:
                y_out = y_out_pos
                diff_old = diff_new
                y_out_pos += y_beta
                diff_new = diff_fn(y_out_pos)
        else:
            # Otherwise, try adjusting downward.
            y_out_neg = y_out - y_beta
            diff_new = diff_fn(y_out_neg)
            while diff_new < diff_old:
                y_out = y_out_neg
                diff_old = diff_new
                y_out_neg -= y_beta
                diff_new = diff_fn(y_out_neg)
        # Update the output for the current step.
        y[step] = y_out

    return y, xin

def generate_step_signal(n=100, t1=3, t2=0.005):
    """
    Generate a step signal using the original parameters.
    
    Parameters:
        n : int, optional
            The sampling rate (samples per unit time). Default is 100.
        t1 : float, optional
            The end time of the signal. Default is 3.
        t2 : float, optional
            The threshold time at which the step occurs. Default is 0.005.
    
    Returns:
        numpy array: The generated step signal.
        
    The function creates a time vector from 0 to t1 with steps of 1/n, then returns a
    binary signal that is 0 before time t2 and 1 from time t2 onward.
    """
    t = np.arange(0, t1 + 1/n, 1/n)
    return (t >= t2).astype(float)

def test_number1():
    """
    Test function that exactly replicates the original "number1" case.
    
    This function uses the following fixed parameters:
      - Step signal generated with n=100, t1=3, t2=0.005.
      - T = 1.0.
      - lp_values = [2, 1.8, 1.6, 1.4, 1.2, 1.1, 1.02].
      - Filtering parameters: y_init=0, y_delta=0.01, y_beta=0.001, h=0.01.
      
    The input signal is plotted using the original color "#E0E0E0", and each filtered output
    is plotted using subsequent colors.
    """
    # Generate the default step signal using original parameters.
    xin = generate_step_signal()
    T_val = 1.0
    lp_values = [2, 1.8, 1.6, 1.4, 1.2, 1.1, 1.02]
    colors = ["#E0E0E0", "#CC0000", "#FF8000", "#FFFF00", "#00FF00", "#00FFFF", "#0080FF", "#FF00FF"]
    
    plt.figure(figsize=(8, 4))
    # Plot the input signal with the specified original input color.
    plt.plot(xin, color=colors[0], label="Input")
    # For each lp value, compute the filtered output and plot it.
    for i, lp_val in enumerate(lp_values):
        # Note: In test mode, filtering parameters remain unchanged.
        y_out, _ = Lp_filter(xin, T_val, p=lp_val, h=0.01)
        plt.plot(y_out, color=colors[i+1], label=f"lp={lp_val}")
    
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("LP Filter Responses (number1 - Step Signal)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("Choose mode:")
    print("1: Interactive mode (enter your own xin, T, lp, y_init, y_delta, y_beta, and h values)")
    print("2: Test mode (generate default step signal with original 'number1' parameters)")
    mode = input("Enter mode (1 or 2): ").strip()
    
    if mode == "2":
        test_number1()
    else:
        # Read input data. Supports comma-separated or whitespace-separated values.
        xin_str = input("Please enter the xin array (values separated by commas or whitespace):\n")
        if ',' in xin_str:
            xin_list = [float(x.strip()) for x in xin_str.split(',') if x.strip()]
        else:
            xin_list = [float(x.strip()) for x in xin_str.split() if x.strip()]
        if len(xin_list) == 0:
            print("Error: No valid numbers provided.")
            exit(1)
    
        try:
            T_val = float(input("Please enter T (time constant): "))
        except Exception:
            print("Error: T must be a valid number.")
            exit(1)
    
        try:
            lp_val = float(input("Please enter the lp value: "))
        except Exception:
            print("Error: lp value must be a valid number.")
            exit(1)
        
        # Prompt for additional custom parameters.
        try:
            y_init_str = input("Please enter y_init (initial output) [default=0]: ").strip()
            y_init = float(y_init_str) if y_init_str != "" else 0.0
        except Exception:
            print("Error: y_init must be a valid number.")
            exit(1)
        
        try:
            y_delta_str = input("Please enter y_delta (candidate step increment) [default=0.01]: ").strip()
            y_delta = float(y_delta_str) if y_delta_str != "" else 0.01
        except Exception:
            print("Error: y_delta must be a valid number.")
            exit(1)
        
        try:
            y_beta_str = input("Please enter y_beta (local adjustment step) [default=0.001]: ").strip()
            y_beta = float(y_beta_str) if y_beta_str != "" else 0.001
        except Exception:
            print("Error: y_beta must be a valid number.")
            exit(1)
        
        try:
            h_str = input("Please enter h (step size in exponential calculation) [default=0.01]: ").strip()
            h = float(h_str) if h_str != "" else 0.01
        except Exception:
            print("Error: h must be a valid number.")
            exit(1)
    
        # In interactive mode, generate a single filtered output using the custom parameters.
        y_out, _ = Lp_filter(xin_list, T_val, p=lp_val, h=h, y_init=y_init, y_delta=y_delta, y_beta=y_beta)
        plt.figure(figsize=(8, 4))
        plt.plot(np.array(xin_list), color="gray", label="Input")
        plt.plot(y_out, color='blue', label=f"Filtered Output (lp={lp_val})")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.title("Custom Step Response")
        plt.legend()
        plt.grid(True)
        plt.show()
