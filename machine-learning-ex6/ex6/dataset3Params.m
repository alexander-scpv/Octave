function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

C_vec = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 ];
sigma_vec = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 ];

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

prediction_error_min = 100;
for i = 1:size(sigma_vec,2)
  for j = 1:size(C_vec,2)
    sigma_temp = sigma_vec(i);
    C_temp = C_vec(j);
    model= svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
    predictions = svmPredict(model, Xval);
    prediction_error = mean(double(predictions ~= yval)*100);
    if  prediction_error < prediction_error_min
      C = C_temp;
      sigma = sigma_temp; 
      prediction_error_min = prediction_error;
    endif
endfor





% =========================================================================

end
