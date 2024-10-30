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

% State transition matrix A for tracking the sine and cosine components of the PLI signal
A = [cos(omega), sin(omega); -sin(omega), cos(omega)];

H = [1, 0];  % Measurement matrix
Q = 0.01 * eye(2);  % Process noise covariance (assuming equal noise level for sine and cosine components)
R = 0.1;  % Measurement noise covariance
x_hat = [0; 0];  % Initial state estimate, initialized to zero
P = eye(2);  % Initial error covariance matrix

kalman_filtered = zeros(LENGTH, 1);  % Filtered signal

%% Kalman Filtering Algorithm
for n = 1:LENGTH
    % Prediction step
    x_hat_prior = A * x_hat;  % State prediction
    P_prior = A * P * A' + Q;  % Prediction error covariance
    
    % Update step
    K = P_prior * H' / (H * P_prior * H' + R);  % Kalman gain
    x_hat = x_hat_prior + K * (d(n) - H * x_hat_prior);  % Update state estimate
    P = (eye(2) - K * H) * P_prior;  % Update error covariance
    
    % Save Kalman filtered signal (subtract estimated PLI noise)
    kalman_filtered(n) = d(n) - H * x_hat;
end

%% High-pass and Low-pass Filtering (optional, for further baseline drift removal)
fc_high = 1;  % High-pass filter to remove DC and baseline drift
[b_hp, a_hp] = butter(2, fc_high / (fs / 2), 'high');
ECG_hp_filtered_kalman = filtfilt(b_hp, a_hp, kalman_filtered);

fc_low = 40;  % Low-pass filter to remove high-frequency noise
[b_lp, a_lp] = butter(4, fc_low / (fs / 2), 'low');
ECG_filtered2 = filtfilt(b_lp, a_lp, ECG_hp_filtered_kalman);

%% Plotting
figure;
subplot(3,1,1);
plot(d);
title('Noisy ECG with PLI (Original Signal + Mains Noise)');

subplot(3,1,2);
plot(kalman_filtered);
title('After Kalman Filtering (PLI Noise Removed)');

subplot(3,1,3);
plot(ECG_filtered2);
title('After High-pass and Low-pass Filtering');

%% Frequency Domain Comparison
Y_orig = fft(d);          % FFT of original signal
Y_kalman = fft(ECG_filtered2);  % FFT of signal after Kalman filtering

f = (0:LENGTH/2-1) * (fs / LENGTH);   % Frequency vector

Y_orig_mag = abs(Y_orig(1:LENGTH/2));  % Magnitude spectrum of original signal
Y_kalman_mag = abs(Y_kalman(1:LENGTH/2));  % Magnitude spectrum after Kalman filtering

figure;
plot(f, Y_orig_mag, 'b', 'DisplayName', 'Noisy ECG with PLI'); hold on;
plot(f, Y_kalman_mag, 'r', 'DisplayName', 'After Kalman Filtering');
legend('show');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Comparing Frequency Spectrum Before and After Kalman Filtering');
grid on;

SNR_orig = snr(d, reference_signal);
SNR_kalman = snr(kalman_filtered, mains_signal);

% Output SNR values
fprintf('SNR before Kalman filtering: %.2f dB\n', SNR_orig);
fprintf('SNR after Kalman filtering: %.2f dB\n', SNR_kalman);

% Compute and plot Power Spectral Density (PSD) before and after filtering
figure;
subplot(2, 1, 1);
[psd_orig, f_psd] = pwelch(d, [], [], [], FS);
plot(f_psd, 10*log10(psd_orig), 'b');
title('PSD of Original BWN-ECG Signal');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

% PSD of filtered signal
subplot(2, 1, 2);
[psd_filt, f_psd] = pwelch(ECG_filtered2, [], [], [], FS);
plot(f_psd, 10*log10(psd_filt), 'r');
title('PSD of Filtered ECG Signal');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

% Plot of error magnitude vs. sample index
figure;
plot(abs(d), 'b', 'DisplayName', 'Noise-ECG'); hold on;
plot(abs(ECG_filtered2), 'r', 'DisplayName', 'After Filtering');
title('Plot of Error Magnitude vs. Sample Index');
xlabel('Sample Index');
ylabel('Error Magnitude');
grid on;

% MSE and PRD
MSE_Kalman = mean((standardData - kalman_filtered').^2);
PRD_Kalman = 100 * sqrt(sum((standardData - kalman_filtered').^2) / sum(standardData.^2));
fprintf('Kalman MSE: %.6f\n', MSE_Kalman);
fprintf('Kalman PRD: %.2f%%\n', PRD_Kalman);

