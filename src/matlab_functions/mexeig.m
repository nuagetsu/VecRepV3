function [P, lambda] = mexeig(X)
  [P,lambda] = eig(X);
  lambda = eig(X);
