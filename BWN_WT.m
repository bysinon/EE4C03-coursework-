% Load the original and noisy signals
% Assuming 'Data1' is the original ECG signal, and 'bwn' is the noise
load('ECG_database.mat'); % Replace with your actual filename if different
original_signal = Data3;
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

% Calculate performance metrics
% SNR Calculation
signal_power = mean(original_signal .^ 2);
noise_power = mean((original_signal - denoised_signal) .^ 2);
SNR = 10 * log10(signal_power / noise_power);

% MSE Calculation
MSE = mean((original_signal - denoised_signal) .^ 2);

% PRD Calculation
PRD = sqrt(mean((original_signal - denoised_signal) .^ 2)) / sqrt(mean(original_signal .^ 2)) * 100;

% Display Results
disp('Wavelet Denoising Results:');
disp(['SNR: ', num2str(SNR), ' dB']);
disp(['MSE: ', num2str(MSE)]);
disp(['PRD: ', num2str(PRD), '%']);

% Plot signals for visual comparison
figure;
subplot(3,1,1);
plot(original_signal);
title('Original ECG Signal');
xlabel('Sample');
ylabel('Amplitude');

subplot(3,1,2);
plot(noisy_signal);
title('Noisy ECG Signal (1% Noise Level)');
xlabel('Sample');
ylabel('Amplitude');

subplot(3,1,3);
plot(denoised_signal);
title('Wavelet Denoised Signal');
xlabel('Sample');
ylabel('Amplitude');
