function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
r =length(theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


value =-y'*(log(sigmoid(X*theta)))-(ones(m,1)-y)'*log(ones(m,1)-sigmoid(X*theta));
J=1/m*(value+(lambda/2)*sum(theta(2:r).^2));

#grad(1)=(1/m)*(X(1,:)*(sigmoid(X(1)*theta)-y(1,:)));
#grad(2:r)=(1/m)*(X(2:m,:)'*(sigmoid(X(2:m,:)*theta(2:r))-y(2:m))+lambda*theta(2:r));
#grad(1) =(1/m)*(X(1,:)'*(sigmoid(X(1,:)*theta(1)))-y(1));
#grad((2:r)=(1/m)*(X(2:m,:)'*(sigmoid(X(2:m,:)*theta(2:r))-y(2:m))+lambda*theta(2:r));
theta1=theta;
theta1(1)=0;
grad=(1/m)*(X'*(sigmoid(X*theta)-y)+lambda*theta1);

% =============================================================

end

