function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
   %hidden_layer_size * (input_layer_size + 1)) = 25*401
   %hidden_layer_size =25
   %(input_layer_size + 1) =401
   %Theta1 becomes 25*401 elements into a (25,401) matrix 
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%Theta 2 (10,26) matrix
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Add ones to the X data matrix
X = [ones(m, 1) X];
a2 = sigmoid(Theta1*X');
%Add bias to a2
a2=[ones(1,size(a2,2)); a2];
h = sigmoid(Theta2*a2);

y_temp=zeros(size(y,1),num_labels);
for i =1:size(y_temp,1)
   y_temp(i,y(i))=1;
endfor

y=y_temp;
delta = 0;

delta = -y*log(h) - (1. - y)*log(1. - h);
delta =trace (delta);

J = (1/m)*delta;

Theta1_slice = Theta1(:,2:end);
Theta2_slice = Theta2(:,2:end);

   %check_Theta1_slice = size(Theta1_slice)
   %check_Theta2_slice = size(Theta2_slice)
part1 = trace(Theta1_slice*Theta1_slice');
part2=trace(Theta2_slice*Theta2_slice');


reg_term = (lambda/(2*m)) * ( part1 + part2);

J = J + reg_term;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

   %Theta1 becomes 25*401 elements into a (25,401) matrix 
   %Theta2 (10,26) matrix

delta3 = zeros(num_labels,1);
DELTA1 = zeros(hidden_layer_size,input_layer_size +1);
DELTA2 = zeros(num_labels,hidden_layer_size + 1); 


a1=X;
z2 = Theta1*X';
z3 = Theta2*a2;
a3 = sigmoid(z3);
delta3 = a3 -y';
delta2 = (Theta2' * delta3) .* a2.*(1-a2);
%We remove the first line of zeros, corresponding to the bias in the zero position 
delta2 = delta2(2:end,1:end);

DELTA1 = delta2*a1;   
DELTA2 = delta3*a2';

DVEC1=(1/m)*DELTA1;
DVEC2=(1/m)*DELTA2;

%check_reg1 = size((lambda/m) * Theta1)
%check_reg2 = size((lambda/m) * Theta2)

reg_term1 = ((lambda/m) * Theta1)(1:end,2:end);
reg_term2 = ((lambda/m) * Theta2)(1:end,2:end);

%check_reg1 = size(reg_term1)
%check_reg2 = size(reg_term2)

reg_term1 = [zeros(size(reg_term1,1),1) reg_term1];
reg_term2 = [zeros(size(reg_term2,1),1) reg_term2];

%check_reg1 = size(reg_term1)
%check_reg2 = size(reg_term2)

DVEC1= DVEC1 + reg_term1;
DVEC2= DVEC2 + reg_term2;

grad = [DVEC1(:);DVEC2(:)]; 

%printf("End grad\n")


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
%grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
