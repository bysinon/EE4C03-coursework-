clear all;
load("ECG_database.mat");

%% Signal and Noise
reference_signal = 20 * mains_signal;  % PLI noise signal
standardData = Data1;  % Original ECG signal
d = standardData + reference_signal;  % Original signal plus PLI noise

fs = FS;  % Sampling frequency

%% Kalman Filter Parameters
f_pli = 50;  % Power line interference frequency (50 Hz)
T = 1 / fs;  % Sampling period
omega = 2 * pi * f_pli * T;  % Angular frequency

% State transition matrix A for tracking sine and cosine components of the PLI signal
A = [cos(omega), sin(omega); -sin(omega), cos(omega)];

H = [1, 0];  % Measurement matrix
Q = 0.01 * eye(2);  % Process noise covariance (assuming equal noise levels for sine and cosine components)
R = 0.1;  % Measurement noise covariance
x_hat = [0; 0];  % Initial state estimate, initialized to zero
P = eye(2);  % Initial error covariance matrix

kalman_filtered = zeros(LENGTH, 1);  % Filtered signal

%% Kalman Filter Algorithm
for n = 1:LENGTH
    % Prediction step
    x_hat_prior = A * x_hat;  % State prediction
    P_prior = A * P * A' + Q;  % Prediction error covariance
    
    % Update step
    K = P_prior * H' / (H * P_prior * H' + R);  % Kalman gain
    x_hat = x_hat_prior + K * (d(n) - H * x_hat_prior);  % Update state estimate
    P = (eye(2) - K * H) * P_prior;  % Update error covariance
    
    % Save the Kalman filtered signal (subtracting the estimated PLI noise)
    kalman_filtered(n) = d(n) - H * x_hat;
end

%% High-pass and Low-pass Filtering (Optional, further removal of baseline drift, etc.)
fc_high = 1;  % High-pass filter to remove DC drift and baseline wander
[b_hp, a_hp] = butter(2, fc_high/(fs/2), 'high');
ECG_hp_filtered_kalman = filtfilt(b_hp, a_hp, kalman_filtered);

fc_low = 40;  % Low-pass filter to remove high-frequency noise
[b_lp, a_lp] = butter(4, fc_low/(fs/2), 'low');
ECG_filtered2_kalman = filtfilt(b_lp, a_lp, ECG_hp_filtered_kalman);

%% LMS
% Parameters
mu = 0.01;
M = 10;     

t = (0:LENGTH-1) / FS;  % Time axis

% Initialize LMS filter
w = zeros(1, M);  % Filter weights
e = zeros(1, LENGTH);  % Error signal
input_buffer = zeros(1, M);

% LMS filtering algorithm
for n = M:LENGTH
    % Input to filter is the interference signal
    input_buffer = [reference_signal(n), input_buffer(1:end-1)];
    
    % Compute filter output (interference estimate)
    y = w * input_buffer';
    
    % Error signal (desired signal - interference estimate)
    e(n) = d(n) - y;
    
    % Update LMS filter weights with step size limitation and normalization
    w = w + 2 * mu * e(n) * input_buffer / (input_buffer * input_buffer' + eps);
end

% High-pass and low-pass filtering post-processing
fc_high = 1; 
[b_hp, a_hp] = butter(4, fc_high/(FS/2), 'high');  
ECG_hp_filtered = filtfilt(b_hp, a_hp, e); 

fc_low = 40;  
[b_lp, a_lp] = butter(4, fc_low/(FS/2), 'low'); 
ECG_filtered2_LMS = filtfilt(b_lp, a_lp, ECG_hp_filtered);  

%% RLS
M_1 = 1;             % Filter order
delta = 0.1;       % Small positive number to initialize P
lambda = 0.99;     % Forgetting factor
P_1 = (1/delta)*eye(M_1);
w_1 = zeros(M_1,1);

y = zeros(LENGTH,1);
e_1 = zeros(LENGTH,1);

for n = 1:LENGTH
    x_n = reference_signal(n);
    d_n = d(n);
    y_n = w_1' * x_n;
    e_n = d_n - y_n;

    k_n = (P_1 * x_n) / (lambda + x_n' * P_1 * x_n);
    w_1 = w_1 + k_n * e_n;
    P_1 = (1/lambda)*(P_1 - k_n * x_n' * P_1);

    y(n) = y_n;
    e_1(n) = e_n;
end

% High-pass and low-pass filtering post-processing
fc_high = 1; 
[b_hp, a_hp] = butter(4, fc_high/(FS/2), 'high');  
ECG_hp_filtered_RLS = filtfilt(b_hp, a_hp, e_1); 

fc_low = 40;  
[b_lp, a_lp] = butter(4, fc_low/(FS/2), 'low'); 
ECG_filtered2_RLS = filtfilt(b_lp, a_lp, ECG_hp_filtered_RLS);  

%% Plotting

figure;
subplot(4,1,1);
plot(d);
title('ECG corrupted by PLI');
xlabel('Samples (n)');
ylabel('Amplitude (mV)');

subplot(4,1,2);
plot(ECG_filtered2_kalman/100);
title('ECG signal After Kalman Filtering');
xlabel('Samples (n)');
ylabel('Amplitude (mV)');

subplot(4,1,3);
plot(ECG_filtered2_LMS/100);
title('ECG signal After LMS Filtering');
xlabel('Samples (n)');
ylabel('Amplitude (mV)');

subplot(4,1,4);
plot(ECG_filtered2_RLS/100);
title('ECG signal After RLS Filtering');
xlabel('Samples (n)');
ylabel('Amplitude (mV)');

%% Plot Error Curves

% Plot comparison of LMS and RLS error
figure;
plot(1:LENGTH, ECG_filtered2_RLS' - standardData, 'b', 'DisplayName', 'RLS');
hold on;
plot(1:LENGTH, ECG_filtered2_LMS - standardData, 'r', 'DisplayName', 'LMS');
hold on;
plot(1:LENGTH, ECG_filtered2_kalman' - standardData, 'g', 'DisplayName', 'Kalman');

xlabel('Samples (n)');
ylabel('Error Magnitude');
title('Error Magnitude vs. Sample Index for LMS, RLS, and Kalman');
legend;
hold off;

% Calculate and plot power spectral density (PSD) before and after filtering
figure;
subplot(4, 1, 1);
[psd_orig, f_psd] = pwelch(d, [], [], [], FS);
plot(f_psd, 10*log10(psd_orig), 'b');
title('PSD of Original PLI-ECG Signal');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

% PSD of filtered signal
subplot(4, 1, 2);
[psd_filt, f_psd] = pwelch(ECG_filtered2_kalman, [], [], [], FS);
plot(f_psd, 10*log10(psd_filt), 'r');
title('PSD of Kalman Filtered ECG Signal');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

subplot(4, 1, 3);
[psd_filt, f_psd] = pwelch(ECG_filtered2_LMS, [], [], [], FS);
plot(f_psd, 10*log10(psd_filt), 'r');
title('PSD of LMS Filtered ECG Signal');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

subplot(4, 1, 4);
[psd_filt, f_psd] = pwelch(ECG_filtered2_RLS, [], [], [], FS);
plot(f_psd, 10*log10(psd_filt), 'r');
title('PSD of RLS Filtered ECG Signal');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;
