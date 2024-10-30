clear all;
load("ECG_database.mat");

%% LMS
reference_signal = 0.01 .* bwn; % Noise signal
standardData = Data3; % For comparison of filtering error later
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
[b_hp, a_hp] = butter(4, fc_high / (FS / 2), 'high');  
ECG_hp_filtered = filtfilt(b_hp, a_hp, e); 

fc_low = 40;  
[b_lp, a_lp] = butter(4, fc_low / (FS / 2), 'low'); 
ECG_filtered2 = filtfilt(b_lp, a_lp, ECG_hp_filtered);  

%% plot

figure;
subplot(3,1,1);
plot(BWN_data);
title('BWN-ECG');

subplot(3,1,2);
plot(e);
title('1 After RLS filtering');

subplot(3,1,3);
plot(ECG_filtered2);
title('2 After High-pass Low-pass filtering');


% Plot time-domain signals before and after filtering
figure;
plot(BWN_data, 'b', 'DisplayName', 'Noise-ECG'); hold on;
plot(ECG_filtered2, 'r', 'DisplayName', 'After Filtering');
legend('show');
xlabel('Index');
ylabel('Amplitudes');
title('Comparing noisy and filtered ECG signal in time domain');
grid on;

% Compute FFT of original and filtered signals
Y_orig = fft(BWN_data);          % FFT of original signal
Y_filt = fft(ECG_filtered2);      % FFT of filtered signal

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
[psd_orig, f_psd] = pwelch(BWN_data, [], [], [], FS);
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


% SNR Calculation
SNR = snr(standardData, standardData - ECG_filtered2);

% MSE Calculation
MSE = mean((standardData - ECG_filtered2) .^ 2);

% PRD Calculation
PRD = sqrt(mean((standardData - ECG_filtered2) .^ 2)) / sqrt(mean(standardData .^ 2)) * 100;

% Display Results
disp('LMS Results:');
disp(['SNR: ', num2str(SNR), ' dB']);
disp(['MSE: ', num2str(MSE)]);
disp(['PRD: ', num2str(PRD), '%']);


% Assuming filtered_ECG is the filtered signal, and Fs is the sampling frequency
Fs = 500; % Replace with your actual sampling frequency

% Set window parameters
window_length = 512;  % Window length
noverlap = 256;       % Window overlap
nfft = 1024;          % Number of FFT points

figure;
% Plot spectrogram
spectrogram(ECG_filtered2, window_length, noverlap, nfft, Fs, 'yaxis');

% Set chart title and labels
title('Spectrogram of Filtered LMS ECG Signal');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
