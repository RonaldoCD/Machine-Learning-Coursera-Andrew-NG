function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
[m ,n] = size(X); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
s = zeros(m,1);
for i = 1:n
    s = s + theta(i) * X(:,i) + ;
end
J = 1 * sum((s - y).^2) / (2 * m);

% =========================================================================

end
