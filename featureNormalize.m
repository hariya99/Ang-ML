function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));       %size(X,2) gives the number of columns
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

num_of_features = size(X,2);   % num of columns in X gives the num of features or variables.

for feature = 1:num_of_features
    % mu is a row vector which holds mean value for each feature
    mu(:,feature) = mean(X(:,feature));           % X(:,feature) gives the whole column of the feature

    % sigma is a row vector which holds SD value for each feature
    sigma(:,feature) =  std(X(:,feature));

    X_norm(:,feature) = ( X(:,feature) - mu(:, feature)) / sigma(:,feature);

end
% ============================================================

end
