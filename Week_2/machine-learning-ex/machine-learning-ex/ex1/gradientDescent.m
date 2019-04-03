function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

thetaLen = length(theta);
tmp = zeros(2,1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    %Non vectorized code.
%     for j = 1:thetaLen
%         % ?(h ??(x(i)) - y(i))x j(i)
%         h = 0;
%         for i = 1 : m
%             h = h + (((theta(1)*X(i,1)+theta(2)*X(i,2)) - y(i)) * X(i,j));
%         end
%         tmp(j) = theta(j)-(1/m * alpha * h);
%     end
%     theta = tmp;

     % Vectorized code.
     % theta = theta - alpha * delta
     % delta = 1/m *(Sum(hFun(x(i)) - Sum(y(i))) * X(i)
     % predictionDiff = (Sum(hFun(x(i)) - Sum(y(i)))
     %X =  [x0 , x1]   Theta = [theta0]  zeros(2,1)
     %     [x0 , x2]           [theta1]
     % Sum(hFun(x(i)) = theta0*x0 + theta1*x1 + theta0*x0 + theta1*x2
     
     % https://github.com/gopaczewski/coursera-ml/blob/master/mlclass-ex1-005/mlclass-ex1/gradientDescent.m
     
     predictionDiff = (X * theta) - y ; % m X 1
     delta = 1/m * (predictionDiff' * X)' % ' ((n+1) x 1 vector)
     theta = theta - (alpha * delta); % ' ((n+1) x 1 vector)
     
     % Produce with X
     % (theta0*x0 + theta1*x1)(x0, x1)
     
     
     
     

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
