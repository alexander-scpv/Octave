function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%check_theta1 = size(Theta1)
%check_theta2 = size(Theta2)

X = [ones(m, 1) X];
%check_X_10=X(1:100,1:100)
%check_size_X=size(X)
part1 = sigmoid(Theta1*X');
%check_size_part1 = size(part1)
part1 = [ ones(1,columns(part1)) ; part1];
%check_part1_5=part1(1:26,1:5)
%check_size_part1 = size(part1)
part2 = Theta2*part1;
%check_size_part2=size(part2)
%check_theta2_5=Theta2(1:5,1:26)
%check_part2_10=part2(1:10,1:1000)
h = sigmoid(part2);
%check_h_10=h(1:10,1:10)
%check_size_h=size(h)
[max_values indices]  = max(part2);

p = indices';
%check_size_p = size(p)

%check_p=p





% =========================================================================


end
