function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

error_vector=zeros(1,64);

c_sigma_matrix=zeros(64,2);

c_vector = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vector = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

c_size=size(c_vector,2);
sigma_size=size(sigma_vector,2);

idx = 1;
for c_tmp =1:c_size
   
   for sigma_tmp=1:sigma_size
   
   model= svmTrain(X, y, c_vector(c_tmp), @(x1, x2) gaussianKernel(x1, x2, sigma_vector(sigma_tmp)));
   predictions = svmPredict(model, Xval);
   
   error_vector(idx) = mean(double(predictions ~= yval));
   c_sigma_matrix(idx,:)=[c_vector(c_tmp), sigma_vector(sigma_tmp)];
   idx= idx+1;
   end

end

%c_sigma_matrix

[min_error,min_error_index]=min(error_vector);

optimal_c_sigma=c_sigma_matrix(min_error_index,:);

C= optimal_c_sigma(:,1);
sigma = optimal_c_sigma(:,2);

% =========================================================================

end
