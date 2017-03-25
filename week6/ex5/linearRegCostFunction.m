function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hyp=X*theta;

error=hyp-y;

cost_function=(1/(2*m))*sum((error).^2);

theta_tmp=theta(2:end,:);


regularization= (lambda/(2*m))*sum(theta_tmp.^2);

J=cost_function+regularization;



unreg_grad=(1/m)*(X'*error);


reg_theta_tmp=(lambda/m).*[0;theta_tmp];

grad=unreg_grad+reg_theta_tmp;




% =========================================================================

grad = grad(:);

end
