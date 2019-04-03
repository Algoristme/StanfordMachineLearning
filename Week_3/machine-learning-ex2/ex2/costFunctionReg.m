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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% lambda does not apply for theta(1) which is coeff of x0 since its dumm =1
hTheta = sigmoid(X*theta);
regularizationMat = (lambda/(2*m)) * (theta(2:length(theta)))' * theta(2:length(theta));
J = (1/m) * (-y' * log(hTheta) - (1-y)' * log(1-hTheta)) + regularizationMat;

% Initial theta(1) = 0
% https://www.coursera.org/learn/machine-learning/supplement/v51eg/regularized-logistic-regression
theta(1) = 0;

grad = ((1 / m) * (hTheta - y)' * X) + lambda / m * theta';

% =============================================================

end
