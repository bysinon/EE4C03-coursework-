clear all;
load("ECG_database.mat");

%% Original Signal and Noise Signal
reference_signal = 0.01 .* bwn; 
standardData = Data3;  
d = standardData + reference_signal;

fs = FS;  % Sampling frequency
T = 1 / fs;  % Sampling period

%% Kalman Filter Parameters
% State transition matrix (position and velocity)
A = [1, T; 0, 1];  % State transition matrix
H = [1, 0];  % Measurement matrix, observing only position
Q = [0.001, 0; 0, 0.001];  % Process noise covariance, small noise
R = 1;  % Measurement noise covariance
x_hat = [0; 0];  % Initial state estimate [position, velocity]
P = 0.01 * eye(2);  % Initial error covariance matrix

kalman_filtered = zeros(LENGTH, 1);  % Filtered signal

%% Kalman Filtering Algorithm
for n = 1:LENGTH
    % Prediction step
    x_hat_prior = A * x_hat;  % State prediction (position and velocity)
    P_prior = A * P * A' + Q;  % Prediction error covariance
    
    % Update step
    K = P_prior * H' / (H * P_prior * H' + R);  % Kalman gain
    x_hat = x_hat_prior + K * (d(n) - H * x_hat_prior);  % Update state estimate
    P = (eye(2) - K * H) * P_prior;  % Update error covariance
    
    % Save Kalman filtered signal (subtract estimated baseline drift)
    kalman_filtered(n) = d(n) - H * x_hat;  % Remove estimated baseline drift
end

%% Bandpass Filtering (Remove low and high-frequency noise)
fc_low = 1;  % Bandpass filter low cutoff frequency (1 Hz)
fc_high = 40;  % Bandpass filter high cutoff frequency (40 Hz)
[b_bp, a_bp] = butter(4, [fc_low, fc_high] / (fs / 2), 'bandpass');  % Design bandpass filter

% Apply bandpass filter
ECG_filtered2 = filtfilt(b_bp, a_bp, kalman_filtered);

%% Plotting

figure;
subplot(3,1,1);
plot(d);
title('Noisy ECG with Baseline Wander (Original Signal + Noise)');

subplot(3,1,2);
plot(kalman_filtered);
title('After Kalman Filtering (Baseline Wander Removed)');

subplot(3,1,3);
plot(ECG_filtered2);
title('After Bandpass Filtering (1-40 Hz)');


Y_orig = fft(d);          % FFT of original signal
Y_kalman = fft(ECG_filtered2);  % FFT of signal after Kalman filtering

f = (0:LENGTH/2-1) * (fs / LENGTH);   % Frequency vector

Y_orig_mag = abs(Y_orig(1:LENGTH/2));  % Magnitude spectrum of original signal
Y_kalman_mag = abs(Y_kalman(1:LENGTH/2));  % Magnitude spectrum of signal after Kalman filtering

figure;
plot(f, Y_orig_mag, 'b', 'DisplayName', 'Noisy ECG with PLI'); hold on;
plot(f, Y_kalman_mag, 'r', 'DisplayName', 'After Kalman Filtering');
legend('show');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Comparing Frequency Spectrum Before and After Kalman Filtering');
grid on;

% SNR Calculation
SNR = snr(standardData, standardData - ECG_filtered2');

% MSE Calculation
MSE = mean((standardData - ECG_filtered2') .^ 2);

% PRD Calculation
PRD = sqrt(mean((standardData - ECG_filtered2') .^ 2)) / sqrt(mean(standardData .^ 2)) * 100;

% Display Results
disp('Kalman Results:');
disp(['SNR: ', num2str(SNR), ' dB']);
disp(['MSE: ', num2str(MSE)]);
disp(['PRD: ', num2str(PRD), '%']);


%% Plot Error Magnitude vs. Sample Index
figure;
plot(abs(d), 'b', 'DisplayName', 'Noise-ECG'); hold on;
plot(abs(ECG_filtered2), 'r', 'DisplayName', 'After Filtering');
title('Plot of Error Magnitude vs. Sample Index');
xlabel('Sample Index');
ylabel('Error Magnitude');
grid on;


% Assuming filtered_ECG is the filtered signal, Fs is the sampling frequency
Fs = 500; % Replace with your actual sampling frequency

% Set window parameters
window_length = 512;  % Window length
noverlap = 256;       % Window overlap
nfft = 1024;           % Number of FFT points

figure;
% Plot spectrogram
spectrogram(ECG_filtered2, window_length, noverlap, nfft, Fs, 'yaxis');

% Set chart title and labels
title('Spectrogram of Filtered Kalman ECG Signal');
xlabel('Time (s)');
ylabel('Frequency (Hz)');

