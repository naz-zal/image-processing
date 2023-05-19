x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
y = [2.56, 1.14, 4.01, 6.02, 4.62, 7.48, 7.79, 9.40, 10.43, 9.23, 13.22, 13.94, 14.30, 15.32, 14.97];

x_mean = mean(x);
y_mean = mean(y); 

x_centered = x - x_mean;
y_centered = y - y_mean;

% Plotting the datapoints after centering the dataset
figure; 
plot(x_centered, y_centered, 'o'); 
xlabel("Centered X");
ylabel("Centered Y"); 

% Computing the covariance matrix
dataset = [x_centered', y_centered']; 

% Need to look more into covariance matrices but they seem to increase
% together with high large covariance?

cov_mat = cov(dataset);
[eigenvectors, eigenvalues] = eig(cov_mat); 

[~, index] = max(diag(eigenvalues)); 
V = eigenvectors(:, index); 

% Generate random data points in a 2-dimensional space
data = randn(200, 2);

% Center the dataset
mean_data = mean(data);
centered_data = data - mean_data;

% Plot the datapoints
plot(centered_data(:, 1), centered_data(:, 2), '.')
xlabel('Feature 1')
ylabel('Feature 2')

cov_data = cov(data);
[eigenvectors_data, eigenvalues_data] = eig(cov_data); 
