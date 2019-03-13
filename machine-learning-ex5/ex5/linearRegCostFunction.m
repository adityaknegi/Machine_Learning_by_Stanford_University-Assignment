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




preiction=X*theta;
J=1/m/2*sum((preiction-y).^2) +lambda/2/m*(sum(theta(2:end).^2));

% X(:,1) IS 1


grad(1) =1/m*sum((preiction-y).*X(:,1));
for i = 2:size(theta)
    grad(i) =1/m*sum((preiction-y).*X(:,i))+lambda/m*theta(i);
endfor

 % #Note that you should not regularize the theta 0  term (theta(1))


	

% =========================================================================

grad = grad(:);

end
