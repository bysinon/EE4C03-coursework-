clear all;
load('ECG_database.mat');  

%%
% Plot PLI noise
figure;
subplot(2,1,1);
plot(mains_signal);
title('Power Line Interference Signal');
xlabel('Samples');
ylabel('Amplitude');

% Plot BWN noise
subplot(2,1,2);
plot(bwn);
title('Baseline Wander Noise Signal');
xlabel('Samples');
ylabel('Amplitude');

%%
% Define window size
windowSize = 500;  % Adjust window size based on data characteristics

% Calculate mean and variance of PLI
numWindowsPLI = floor(length(mains_signal) / windowSize);
pli_means = arrayfun(@(i) mean(mains_signal((i-1)*windowSize+1:i*windowSize)), 1:numWindowsPLI);
pli_vars = arrayfun(@(i) var(mains_signal((i-1)*windowSize+1:i*windowSize)), 1:numWindowsPLI);

% Calculate mean and variance of BWN
numWindowsBWN = floor(length(bwn) / windowSize);
bwn_means = arrayfun(@(i) mean(bwn((i-1)*windowSize+1:i*windowSize)), 1:numWindowsBWN);
bwn_vars = arrayfun(@(i) var(bwn((i-1)*windowSize+1:i*windowSize)), 1:numWindowsBWN);

% Plot statistical characteristics
figure;
subplot(2,2,1);
plot(pli_means);
title('PLI Mean Values');

subplot(2,2,2);
plot(pli_vars);
title('PLI Variance Values');

subplot(2,2,3);
plot(bwn_means);
title('BWN Mean Values');

subplot(2,2,4);
plot(bwn_vars);
title('BWN Variance Values');

