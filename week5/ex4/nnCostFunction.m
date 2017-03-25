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

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X=[ones(m,1) X];


tmp=[m,1];

for i=1:m

  vectorized_y=zeros([size(Theta2,1),1]);
  z2=X(i,:)*Theta1';
  a2=sigmoid(z2);
  
  a2=[1 a2];
  
  z3=a2*Theta2';
  
  hyp=sigmoid(z3);
  
  
  vectorized_y(y(i,:))=1;
  
  positive_log=log(hyp);
  negative_log=log(1-hyp);
  
  first_term=positive_log*vectorized_y;
  second_term=negative_log*(1.-vectorized_y);
   
  tmp(i)=first_term+second_term;
  
endfor

total_value=sum(tmp);
J=(-1/m)*total_value;


temp_Theta1=Theta1;
temp_Theta1(:,1)=0;

temp_Theta2=Theta2;
temp_Theta2(:,1)=0;


regularization=(lambda/(2*m))*(sum(sum(temp_Theta1.^2))+sum(sum(temp_Theta2.^2)));

J=J+regularization;


% Implementing backpropagation
D1=zeros(size(Theta1));
D2=zeros(size(Theta2));
  
for t=1:m
  logical_y=zeros([size(Theta2,1),1]);
  
  
  a_1=X(t,:)';
  
  z_2=Theta1*a_1;
  a_2=sigmoid(z_2);
  
  a_2=[1; a_2];
  
  z_3=Theta2*a_2;
  a_3=sigmoid(z_3);
   
  logical_y(y(t,:))=1; 
  

  delta_3=a_3-logical_y;
  
    
  delta_2=((Theta2(:,2:end))'*delta_3).*sigmoidGradient(z_2);
    

  D1=D1+(delta_2*a_1');
  D2=D2+(delta_3*a_2');
  
  

endfor


Theta1_grad=(1/m).*D1;
Theta2_grad=(1/m).*D2;



Theta1_tmp=(lambda/m).*Theta1(:,2:end);
Theta2_tmp=(lambda/m).*Theta2(:,2:end);


Theta1_tmp=[zeros(size(Theta1,1),1) Theta1_tmp];
Theta2_tmp=[zeros(size(Theta2,1),1),Theta2_tmp];

Theta1_grad=Theta1_grad+Theta1_tmp;
Theta2_grad=Theta2_grad+Theta2_tmp;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end