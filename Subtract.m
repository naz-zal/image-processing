%{
v1 = VideoReader('streetGray.mp4');

num_frames = v1.NumFrames;
height = v1.Height;
width = v1.Width;
num_training_frames = 240;

trainingFrames = zeros(height, width, num_training_frames);

for i = 1:num_training_frames

    frame = read(v1, i);
    trainingFrames(:, :, i) = frame(:, :, 1);

end

mean_frame = mean(trainingFrames, 3); 

test_frame = read(v1, 420);
test_gray = rgb2gray(test_frame);

mean_diff_test = abs(double(mean_frame) - double(test_gray));

threshold = 65;
binaryFrame = mean_diff_test > threshold;


% Use threshold less than and greater than 65 
% Moving on to Q2

Y = squeeze(trainingFrames(360, 640, :));
k_6 = 6;

iter = 300; % 200 iterations is just an example
GMModel_6 = fitgmdist(Y, k_6, 'RegularizationValue', 0.1, ...
    'Start', 'randSample', 'Options', statset('Display','off','MaxIter', iter, 'TolFun', 1e-6));

MOG_test_frame = read(v1, 420);
MOG_test_pixel = MOG_test_frame(360, 640); 
MOG_test_pixel = double(MOG_test_pixel); 

interval_6 = (MOG_test_pixel - 0.5):0.0001:(MOG_test_pixel + 0.5);
PDF_x_6 = pdf(GMModel_6, interval_6');
probability = trapz(interval_6, PDF_x_6);

min_threshold = 0.01; 

if probability < min_threshold
    pixelIndex_6 = 1; % pixel is classified as foreground
else
    pixelIndex_6 = 0; % pixel is classified as background
end

% With K = 1
k_1 = 1;
GMModel_1 = fitgmdist(Y, k_1, 'RegularizationValue', 0.1, ...
    'Start', 'randSample', 'Options', statset('Display','off','MaxIter', iter, 'TolFun', 1e-6));


interval_1 = (MOG_test_pixel - 0.5):0.0001:(MOG_test_pixel + 0.5);
PDF_x_1 = pdf(GMModel_1, interval_1');
probability_1 = trapz(interval_1, PDF_x_1);

if probability_1 < min_threshold
    pixelIndex_1 = 1; % pixel is classified as foreground
else
    pixelIndex_1 = 0; % pixel is classified as background
end

% With K = 3

k_3 = 3;
GMModel_3 = fitgmdist(Y, k_3, 'RegularizationValue', 0.1, ...
    'Start', 'randSample', 'Options', statset('Display','off','MaxIter', iter, 'TolFun', 1e-6));


interval_3 = (MOG_test_pixel - 0.5):0.0001:(MOG_test_pixel + 0.5);
PDF_x_3 = pdf(GMModel_3, interval_3');
probability_3 = trapz(interval_3, PDF_x_3);

if probability_3 < min_threshold
    pixelIndex_3 = 1; % pixel is classified as foreground
else
    pixelIndex_3 = 0; % pixel is classified as background
end
%}

% Question 5

all_Y = squeeze(trainingFrames(235:484, 490:789, :)); 
all_Y = reshape(all_Y, [], size(all_Y, 3));

MOG_test_pixels_main = MOG_test_frame(235:484, 490:789); 
MOG_test_pixel_main = double(MOG_test_pixels_main);
MOG_test_pixel_main = reshape(MOG_test_pixel_main, 1, []);

binaryFrameMain = zeros(75000, 1); 
iter_main = 400; 

for i = 1:75000
    
    GMModel_main = fitgmdist(all_Y(i, :)', k_6, 'RegularizationValue', 0.1, ...
        'Start', 'randSample', 'Options', statset('Display','off','MaxIter', iter_main, 'TolFun', 1e-6));

    interval_main = (MOG_test_pixel_main(i) - 0.5):0.0001:(MOG_test_pixel_main(i) + 0.5);
    PDF_x_main = pdf(GMModel_main, interval_main');
    probability_main = trapz(interval_main, PDF_x_main);

    if probability_main < min_threshold
        binaryFrameMain(i) = 1; % pixel is classified as foreground
    else
        binaryFrameMain(i) = 0; % pixel is classified as background
    end

end 





