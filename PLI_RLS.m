clear all;
load("ECG_database.mat");

reference_signal = 20 * mains_signal;
standardData = Data1; % For comparison of filtering error later
d = standardData + reference_signal;

% RLS Filter Parameters
M = 1;             % Filter order
delta = 0.1;       % Small positive number to initialize P
lambda = 0.99;     % Forgetting factor
P = (1 / delta) * eye(M);
w = zeros(M, 1);

y = zeros(LENGTH, 1);
e = zeros(LENGTH, 1);

for n = 1:LENGTH
    x_n = reference_signal(n);
    d_n = d(n);
    y_n = w' * x_n;
    e_n = d_n - y_n;

    k_n = (P * x_n) / (lambda + x_n' * P * x_n);
    w = w + k_n * e_n;
    P = (1 / lambda) * (P - k_n * x_n' * P);

    y(n) = y_n;
    e(n) = e_n;
end

% High-pass and Low-pass Filtering Post-processing
fc_high = 1; 
[b_hp, a_hp] = butter(4, fc_high / (FS / 2), 'high');  
ECG_hp_filtered = filtfilt(b_hp, a_hp, e); 

fc_low = 40;  
[b_lp, a_lp] = butter(4, fc_low / (FS / 2), 'low'); 
ECG_filtered2 = filtfilt(b_lp, a_lp, ECG_hp_filtered);  

%% Plotting

figure;
subplot(3,1,1);
plot(d);
title('PLI-ECG');

subplot(3,1,2);
plot(e);
title('1 After RLS filtering');

subplot(3,1,3);
plot(ECG_filtered2);
title('After High-pass and Low-pass filtering');

% Plot time-domain signals before and after filtering
figure;
plot(d, 'b', 'DisplayName', 'Noise-ECG'); hold on;
plot(ECG_filtered2, 'r', 'DisplayName', 'After Filtering');
legend('show');
xlabel('Index');
ylabel('Amplitudes');
title('Comparing noisy and filtered ECG signal in time domain');
grid on;

% Compute FFT of original and filtered signals
Y_orig = fft(d);          % FFT of original signal
Y_filt = fft(ECG_filtered2); % FFT of filtered signal

% Generate frequency vector (up to Nyquist frequency)
f = (0:LENGTH/2-1) * (FS / LENGTH);  % Frequency vector, range from 0 to fs/2

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

SNR_orig = snr(d, mains_signal);
SNR_filt = snr(e, mains_signal); % Only comparing the effect before high-pass and low-pass filtering

% Output specific SNR values
fprintf('SNR before filtering: %.2f dB\n', SNR_orig);
fprintf('SNR after filtering: %.2f dB\n', SNR_filt);

% Plot of error magnitude vs. sample index
figure;
plot(abs(d), 'b', 'DisplayName', 'Noise-ECG'); hold on;
plot(abs(ECG_filtered2), 'r', 'DisplayName', 'After Filtering');
title('Plot of Error Magnitude vs. Sample Index');
xlabel('Sample Index');
ylabel('Error Magnitude');
grid on;

% Assuming filtered_ECG is the filtered signal, and Fs is the sampling frequency
Fs = 500; % Replace with your actual sampling frequency

% Set window parameters
window_length = 512;  % Window length
noverlap = 256;       % Window overlap
nfft = 1024;          % Number of FFT points

spectrogram(d, window_length, noverlap, nfft, Fs, 'yaxis');

% Set chart title and axis labels
title('Spectrogram of Original ECG Signal');
xlabel('Time (s)');
ylabel('Frequency (Hz)');

figure;

% Plot spectrogram
spectrogram(ECG_filtered2, window_length, noverlap, nfft, Fs, 'yaxis');

% Set chart title and labels
title('Spectrogram of Filtered RLS ECG Signal');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
