%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data3.txt');
p = size(data,2);
X = data(:, 1:p-1);
y = data(:, p);

plotData(X, y);

X_o = X(:,:);
% Add columns to the feature matrix X ---
X = [X X.^2];
%%%% 

m = length(y);
n = size(X,2);


% Print out some data points
% fprintf('First 10 examples from the dataset: \n');
% fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

%fprintf('Program paused. Press enter to continue.\n');
%pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 1500;

% Init Theta and Run Gradient Descent 
theta = zeros(n+1, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.

%h_area = 1650;
%h_bedrooms = 3;
%x_estimate = [1; (h_area - mu(1))/sigma(1); (h_bedrooms - mu(2))/sigma(2)];
%pop = 7;
%x_estimate = [1 ( pop - mu(1))/sigma(1)];
%x_estimate = [1 pop];

%price_gd = x_estimate * theta_n; % You should change this
%price_gd = price_gd * 10000;

% Plot the linear fit
hold on; % keep previous plot visible
plot(X_o, X * theta, '-', 'LineWidth', 2)
%plot(X(:,2), X_n*theta_n, '-', 'LineWidth', 2)
legend('Training data', 'Linear regression')

hold off % don't overlay any more plots on this figure

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');


% ============================================================

%fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
%         '(using gradient descent):\n $%f\n'], price_gd);

%fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data1.txt');
p = size(data,2);
X = data(:, 1:p-1);
y = data(:, p);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
%h_area = 1650;
%h_bedrooms = 3;
%x_estimate = [1; h_area; h_bedrooms];
pop = 7;
x_estimate = [1; pop];

price_ne = theta' * x_estimate; % You should change this
price_ne = price_ne * 10000;
% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price_ne);

