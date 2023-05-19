%{
count = 1;
face_matrix = zeros(10304, 320);

folders = dir('Face\training');
folders=folders(~ismember({folders.name},{'.','..'}));
subFolders = folders([folders.isdir]);

for k = 1 : length(subFolders)
    cur_dr=['Face\training\' subFolders(k).name];
    images=dir(cur_dr);
    images=images(~ismember({images.name},{'.','..'}));
    for i=1 : length(images)
        face = imread([cur_dr '\' images(i).name]);
        face = face(:, :, 1); 
        face_reshaped = reshape(face, [], 1); 
        face_matrix(:, count) = face_reshaped; 
        count = count + 1;
    end
end

mean_image = zeros(10304, 1); 

for j = 1:10304
    column_mean = mean(face_matrix(j, :)); 
    mean_image(j, :) = column_mean; 
end

mean_face = reshape(mean_image, [112, 92]); 

X = zeros(10304, 320); 

for i = 1:320
    
    X(:, i) = face_matrix(:, i) - mean_image; 
    
end

[U, D, V] = svd(X); 

eigenvalues = diag(D).^2 / (size(X, 1) - 1);
% plot(1:320, eigenvalues); 

K = 15;
W = zeros(K, 320); 

folders = dir('Face\training');
folders=folders(~ismember({folders.name},{'.','..'}));
subFolders = folders([folders.isdir]);

count = 1; 
img_data = zeros(112*92, 8);
Wp_mean = zeros(1, 40);

for k = 1 : length(subFolders)
    cur_dr=['Face\training\' subFolders(k).name];
    images=dir(cur_dr);
    images=images(~ismember({images.name},{'.','..'}));
    
    for i=1 : length(images)
        face = imread([cur_dr '\' images(i).name]);
        face = face(:, :, 1); 
        training_face_reshaped = reshape(face, [], 1);
        W(:, count) = (transpose(U(:,1:K))*(double(training_face_reshaped) - mean_image)); 
        count = count + 1; 
        
        img_data(:, i) = reshape(face, [], 1);
    end
    
    Wp_mean(:, k) = mean(mean(img_data)); % 4.1
end

W_mean = mean(W, 1); 

arcti = imread('arctichare.png');
arcti = arcti(:, :, 1); 
arcti_resized = imresize(arcti, [112, 92]);

arcti_reshaped = reshape(arcti_resized, [], 1);
W_arctichare = (transpose(U(:,1:K))*(double(arcti_reshaped) - mean_image));

d_arcti = norm(W_mean - W_arctichare, 'fro');


test_image_one = imread('Face\testing\s1\10.png');
test_image_one = test_image_one(:, :, 1); 
test_image_one_reshaped = reshape(test_image_one, [], 1);
W_test = (transpose(U(:,1:K))*(double(test_image_one_reshaped) - mean_image));

d_test = norm(W_mean - W_test, 'fro');

% add the other 4 test images 

test_image_two = imread('Face\testing\s2\10.png');
test_image_two = test_image_two(:, :, 1); 
test_image_two_reshaped = reshape(test_image_two, [], 1);
W_test_two = (transpose(U(:,1:K))*(double(test_image_two_reshaped) - mean_image));

d_test_two = norm(W_mean - W_test_two, 'fro');


test_image_three = imread('Face\testing\s3\10.png');
test_image_three = test_image_three(:, :, 1); 
test_image_three_reshaped = reshape(test_image_three, [], 1);
W_test_three = (transpose(U(:,1:K))*(double(test_image_three_reshaped) - mean_image));

d_test_three = norm(W_mean - W_test_three, 'fro');


test_image_four = imread('Face\testing\s4\10.png');
test_image_four = test_image_four(:, :, 1); 
test_image_four_reshaped = reshape(test_image_four, [], 1);
W_test_four = (transpose(U(:,1:K))*(double(test_image_four_reshaped) - mean_image));

d_test_four = norm(W_mean - W_test_four, 'fro');


test_image_five = imread('Face\testing\s5\10.png');
test_image_five = test_image_five(:, :, 1); 
test_image_five_reshaped = reshape(test_image_five, [], 1);
W_test_five = (transpose(U(:,1:K))*(double(test_image_five_reshaped) - mean_image));

d_test_five = norm(W_mean - W_test_five, 'fro');

% Q4
training_face1 = imread('Face\training\s1\1.png');
training_face2 = imread('Face\training\s2\1.png');
training_face3 = imread('Face\training\s3\1.png');
training_face4 = imread('Face\training\s4\1.png');
training_face5 = imread('Face\training\s5\1.png');

training_face1 = reshape(training_face1(:, :, 1), [], 1); 
training_face2 = reshape(training_face2(:, :, 1), [], 1);
training_face3 = reshape(training_face3(:, :, 1), [], 1);
training_face4 = reshape(training_face4(:, :, 1), [], 1);
training_face5 = reshape(training_face5(:, :, 1), [], 1);

W_p1 = (transpose(U(:,1:K))*(double(training_face1) - Wp_mean(:, 1)));
W_p2 = (transpose(U(:,1:K))*(double(training_face2) - Wp_mean(:, 2)));
W_p3 = (transpose(U(:,1:K))*(double(training_face3) - Wp_mean(:, 3)));
W_p4 = (transpose(U(:,1:K))*(double(training_face4) - Wp_mean(:, 4)));
W_p5 = (transpose(U(:,1:K))*(double(training_face5) - Wp_mean(:, 5)));

reconstructed1 = U(:,1:K) * W_p1 + Wp_mean(:, 1);
reconstructed2 = U(:,1:K) * W_p2 + Wp_mean(:, 2);
reconstructed3 = U(:,1:K) * W_p3 + Wp_mean(:, 3);
reconstructed4 = U(:,1:K) * W_p4 + Wp_mean(:, 4);
reconstructed5 = U(:,1:K) * W_p5 + Wp_mean(:, 5);

subplot(1, 5, 1);
imshow(reshape(reconstructed1, [112,92]), []);
title("Face 1");

subplot(1, 5, 2);
imshow(reshape(reconstructed2, [112,92]), []);
title("Face 2");

subplot(1, 5, 3);
imshow(reshape(reconstructed3, [112,92]), []);
title("Face 3");

subplot(1, 5, 4); 
imshow(reshape(reconstructed4, [112,92]), []);
title("Face 4");

subplot(1, 5, 5);
imshow(reshape(reconstructed5, [112,92]), []);
title("Face 5");
%}

testing_face_matrix = zeros(10304, 80);
count = 1; 

folders = dir('Face\testing');
folders=folders(~ismember({folders.name},{'.','..'}));
subFolders = folders([folders.isdir]);

for k = 1 : length(subFolders)
    cur_dr=['Face\testing\' subFolders(k).name];
    images=dir(cur_dr);
    images=images(~ismember({images.name},{'.','..'}));
    for i=1 : length(images)
        face = imread([cur_dr '\' images(i).name]);
        face = face(:, :, 1); 
        face_reshaped = reshape(face, [], 1); 
        testing_face_matrix(:, count) = face_reshaped; 
        count = count + 1;
    end
end

testing_mean_image = zeros(10304, 1); 

for j = 1:10304
    column_mean = mean(testing_face_matrix(j, :)); 
    testing_mean_image(j, :) = column_mean; 
end

W_test = zeros(K, 80); 
count = 1; 

for k = 1 : length(subFolders)
    cur_dr=['Face\testing\' subFolders(k).name];
    images=dir(cur_dr);
    images=images(~ismember({images.name},{'.','..'}));
    
    for i=1 : length(images)
        face = imread([cur_dr '\' images(i).name]);
        face = face(:, :, 1); 
        testing_face_reshaped = reshape(face, [], 1);
        W_test(:, count) = (transpose(U(:,1:K))*(double(testing_face_reshaped) - testing_mean_image)); 
        count = count + 1; 
    end
end

dist_test_images = zeros(1, 40); 

for i = 1:40
    dist_test_images(:, i) = norm(W_test(:, 1) - Wp_mean(:, i), 'fro');
end

