function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% J = (theta)' * X       -- to implement theta' X we need to take one column of X and then multiply with theta' operate on 1 column and then move on the next. 
hypothesis = X * theta;    
J = (1/(2*m)) * sum((hypothesis - y).^2);   % .^2 does element wise square which is what we also need.



% =========================================================================
%end
