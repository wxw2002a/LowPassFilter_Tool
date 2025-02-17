LP Filter Step Response
This project implements an LP (Lp norm-based) filter to process a step signal. The algorithm computes a filtered output by considering all input samples during each update. Because the algorithm takes into account every sample in the input, it can work inefficiently with very long data sets.
Important Note:
•	Input Data Size:
It is recommended that the input data size be limited to around 200 samples for optimal performance. While the program does not enforce a limit on the size of the input data, using very long data arrays may result in slow performance due to the algorithm processing each sample.
Features
•	Interactive Mode:
In interactive mode, you can enter your own input signal (either comma-separated or whitespace-separated), time constant (T), lp value, and additional parameters: y_init, y_delta, y_beta, and h. These parameters control the filtering behavior.
•	Test Mode:
Test mode replicates the original "number1" case using fixed parameters:
o	Step signal generated with parameters: n=100, t1=3, t2=0.005
o	Filter parameters: T=1, lp_values = [2, 1.8, 1.6, 1.4, 1.2, 1.1, 1.02]
o	Filtering constants: y_init=0, y_delta=0.01, y_beta=0.001, h=0.01
In test mode, the input signal is plotted using the color #E0E0E0, and the filtered outputs for each lp value are plotted with corresponding colors.
Usage
1.	Run the Script:
Execute the script in an environment that supports graphical output (e.g., your local machine with a display).
2.	Select Mode:
o	Choose Mode 1 (Interactive Mode) to input your own data and parameters.
o	Choose Mode 2 (Test Mode) to run the default test case (original "number1" parameters).
3.	Data Input:
When entering data in interactive mode, if your input data does not include commas, you can separate values with spaces or newlines.
Performance Consideration
Warning:
Because the algorithm iterates over all input samples for each filtering step, very long input data arrays may result in decreased performance. For best performance, it is suggested to use around 200 samples. However, the program does not enforce a strict limit on the input size.



