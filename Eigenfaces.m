%{
folders = dir('training');
folders = folders(~ismember({folders.name},{'.','..'}));
subFolders = folders([folders.isdir]);
face_matrix = zeros(10304, 320); 
count = 1;

for k = 1:length(subFolders)
    cur_dr = ['training\' subFolders(k).name];
    images = dir(cur_dr);
    images = images(~ismember({images.name},{'.','..'}));
    
    for i = 1:length(images)
        imshow(imread([cur_dr '\' images(i).name]));
        face = imread([cur_dr '\' images(i).name]); 
        face = face(:, :, 1); 
        face_reshaped = reshape(face, [], 1); 
        face_matrix(:, count) = face_reshaped; 
        count = count + 1; 
    end
end

% Calculating the mean image 
mean_image = zeros(10304, 1); 

for j = 1:10304
    column_mean = mean(face_matrix(j, :)); 
    mean_image(j, :) = column_mean; 
end

mean_face = reshape(mean_image, [112, 92]); 
imshow(uint8(mean_face)); 
imwrite(uint8(mean_face), 'mean.bmp');

X = zeros(10304, 320); 

for i = 1:320
    
    X(:, i) = face_matrix(:, i) - mean_image; 
    
end

n = 320; 
T = (1/n)*(X')*X; 


[T_eigenvectors, T_eigenvalues] = eig(T);

new_X = X*T_eigenvectors;


for i = 1:6 
    
    figure; 
    imshow(reshape(new_X(:,i),[112,92]), []); 
    
end 


for i = 315:320
    
    figure; 
    imshow(reshape(new_X(:,i),[112,92]), []); 
    
end 



[U, D, V] = svd(X); 

for i = 1:6
    
    figure;
    imshow(reshape(U(:,i),[112,92]), []);
    
end 

%}
cur_dr = 'training\s2';
images = dir(cur_dr);
images = images(~ismember({images.name},{'.','..'}));

face = imread([cur_dr '\' images(1).name]); 
training_face = face(:, :, 1); 
training_face_reshaped = reshape(training_face, [], 1);

W = (transpose(U(:,1:60))*(double(training_face_reshaped) - mean_image)); 
x_cap = U(:,1:60) * W + mean_image; 
imshow(reshape(x_cap,[112,92]), []);


W_120 = (transpose(U(:,1:120))*(double(training_face_reshaped) - mean_image)); 
x_cap_120 = U(:,1:120) * W_120 + mean_image; 
imshow(reshape(x_cap_120,[112,92]), []);
 







