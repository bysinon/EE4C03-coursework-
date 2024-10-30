clear all;
load("ECG_database.mat");

%% LMS
reference_signal = 0.01.*bwn; % Noise signal
standardData = Data1; % For comparison of filtering error later
d = reference_signal + standardData; % Original signal + error signal

% Parameters
mu = 0.01;  
M = 50;     

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

%% High-pass and Low-pass Filtering
fc_high = 1; 
[b_hp, a_hp] = butter(4, fc_high/(FS/2), 'high');  
ECG_hp_filtered = filtfilt(b_hp, a_hp, e); 

fc_low = 40;  
[b_lp, a_lp] = butter(4, fc_low/(FS/2), 'low'); 
ECG_filtered_LMS = filtfilt(b_lp, a_lp, ECG_hp_filtered);  

%% RLS
reference_signal = bwn; % Noise signal
standardData = Data1; % For comparison of filtering error later
d = reference_signal + standardData; % Original signal + error signal

M_1 = 1;             % Filter order
delta = 0.1;       % Small positive number to initialize P
lambda = 0.99;     % Forgetting factor
P = (1/delta) * eye(M_1);
w_1 = zeros(M_1, 1);

y = zeros(LENGTH, 1);
e_1 = zeros(LENGTH, 1);

for n = 1:LENGTH
    x_n = reference_signal(n);
    d_n = d(n);

    y_n = w_1' * x_n;

    e_n = d_n - y_n;

    k_n = (P * x_n) / (lambda + x_n' * P * x_n);

    w_1 = w_1 + k_n * e_n;

    P = (1/lambda) * (P - k_n * x_n' * P);

    y(n) = y_n;
    e_1(n) = e_n;
end

%% High-pass and Low-pass Filtering
fc_high = 1; 
[b_hp, a_hp] = butter(4, fc_high/(FS/2), 'high');  
ECG_hp_filtered_1 = filtfilt(b_hp, a_hp, e_1); 

fc_low = 40;  
[b_lp, a_lp] = butter(4, fc_low/(FS/2), 'low'); 
ECG_filtered_RLS = filtfilt(b_lp, a_lp, ECG_hp_filtered_1);  

%% Original Signal and Noise Signal 
reference_signal = bwn; 
standardData = Data1;  
d = standardData + reference_signal;

fs = FS;  % Sampling frequency
T = 1 / fs;  % Sampling period

%% Kalman Filter Parameters
% State transition matrix (position and velocity)
A = [1, T; 0, 1];  % State transition matrix
H = [1, 0];  % Measurement matrix, only observing position
Q = [0.001, 0; 0, 0.001];  % Process noise covariance, small noise
R = 1;  % Measurement noise covariance
x_hat = [0; 0];  % Initial state estimate [position, velocity]
P_1 = 0.01 * eye(2);  % Initial error covariance matrix

kalman_filtered = zeros(LENGTH, 1);  % Filtered signal

%% Kalman Filtering Algorithm
for n = 1:LENGTH
    % Prediction step
    x_hat_prior = A * x_hat;  % State prediction (position and velocity)
    P_prior = A * P_1 * A' + Q;  % Prediction error covariance
    
    % Update step
    K = P_prior * H' / (H * P_prior * H' + R);  % Kalman gain
    x_hat = x_hat_prior + K * (d(n) - H * x_hat_prior);  % Update state estimate
    P_1 = (eye(2) - K * H) * P_prior;  % Update error covariance
    
    % Save Kalman filtered signal (remove estimated baseline drift)
    kalman_filtered(n) = d(n) - H * x_hat;  % Remove estimated baseline drift
end

%% Bandpass Filtering (remove low and high-frequency noise)
fc_low = 1;  % Bandpass filter low cutoff frequency (1 Hz)
fc_high = 40;  % Bandpass filter high cutoff frequency (40 Hz)
[b_bp, a_bp] = butter(4, [fc_low, fc_high] / (fs / 2), 'bandpass');  % Design bandpass filter

% Apply bandpass filter
ECG_filtered_kalman = filtfilt(b_bp, a_bp, kalman_filtered);

%% Wavelet Transform

original_signal = Data1;
noisy_signal = original_signal + 0.01 .* bwn; % Add noise at 1% level

% Wavelet denoising parameters
wavelet = 'db4';
level = 4;

% Perform Wavelet Decomposition
[coeffs, lengths] = wavedec(noisy_signal, level, wavelet);

% Initialize the denoised coefficients array
coeffs_denoised = coeffs;

% Calculate Threshold and Apply Soft Thresholding
threshold = median(abs(coeffs(end - lengths(end) + 1:end))) / 0.6745 * sqrt(2 * log(length(noisy_signal)));
total_length = length(coeffs); % Get the total length of coeffs array

for i = 2:length(lengths)  % Start from the second layer to skip approximation coefficients
    start_idx = sum(lengths(1:i-1)) + 1;  % Start index for the current layer
    end_idx = sum(lengths(1:i));  % End index for the current layer
    
    % Ensure indices are within bounds
    if end_idx > total_length
        end_idx = total_length;
    end
    
    % Apply thresholding only if indices are valid
    coeffs_denoised(start_idx:end_idx) = wthresh(coeffs(start_idx:end_idx), 's', threshold);
end

% Reconstruct Denoised Signal
denoised_signal = waverec(coeffs_denoised, lengths, wavelet);

%% Plotting
% Plot time-domain signals before and after filtering
figure;
subplot(5,1,1);
plot(d);
title('ECG corrupted by BWN');
xlabel('Samples (n)');
ylabel('Amplitude (mV)');

subplot(5,1,2);
plot(ECG_filtered_kalman/100);
title('ECG signal After Kalman Filtering');
xlabel('Samples (n)');
ylabel('Amplitude (mV)');

subplot(5,1,3);
plot(ECG_filtered_LMS/100);
title('ECG signal After LMS Filtering');
xlabel('Samples (n)');
ylabel('Amplitude (mV)');

subplot(5,1,4);
plot(ECG_filtered_RLS/100);
title('ECG signal After RLS Filtering');
xlabel('Samples (n)');
ylabel('Amplitude (mV)');

subplot(5,1,5);
plot(denoised_signal/100);
title('ECG signal After Wavelet Transform');
xlabel('Samples (n)');
ylabel('Amplitude (mV)');

%% Plot Error Curve

% Plot comparison of LMS and RLS error
figure;
plot(1:LENGTH, ECG_filtered_RLS' - standardData, 'b', 'DisplayName', 'RLS');
hold on;
plot(1:LENGTH, ECG_filtered_LMS - standardData, 'r', 'DisplayName', 'LMS');
hold on;
plot(1:LENGTH, ECG_filtered_kalman' - standardData, 'y', 'DisplayName', 'Kalman');
hold on;
plot(1:LENGTH, denoised_signal - standardData, 'g', 'DisplayName', 'Wavelet');
xlabel('Samples (n)');
ylabel('Error Magnitude');
title('Error Magnitude vs. Sample Index for LMS, RLS, Kalman and Wavelet');
legend;
hold off;

%% Compute FFT of original and filtered signals
Y_orig = fft(d);          % FFT of original signal
Y_filt = fft(ECG_filtered_RLS);                 % FFT of filtered signal

% Generate frequency vector for plotting (up to Nyquist frequency)
f = (0:LENGTH/2-1) * (FS / LENGTH);        % Frequency vector, range from 0 to fs/2

% Compute magnitude spectrum (positive frequencies)
Y_orig_mag = abs(Y_orig(1:LENGTH/2)); % Magnitude spectrum of original signal
Y_filt_mag = abs(Y_filt(1:LENGTH/2)); % Magnitude spectrum of filtered signal

% Plot frequency spectrum before and after filtering
figure;
plot(f, Y_orig_mag, 'b', 'DisplayName', 'Noise-ECG'); hold on;
plot(f, Y_filt_mag, 'r', 'DisplayName', 'After Filtering');
legend('show');
xlabel('frequency(Hz)');
ylabel('Amplitudes');
title('Comparing noisy and filtered ECG signal in freq domain');
grid on;

