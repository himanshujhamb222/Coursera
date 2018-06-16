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

hypothesis=X*(theta);
hypothesis=sigmoid(hypothesis);
sz=size(theta);
n=sz(1);
for i=1:m
  J=J+(-y(i)*log(hypothesis(i)) - (1-y(i))*log(1-hypothesis(i)));
end
reg=0;
for l=2:n
  reg=reg+(theta(l,1)^2);
  end
J=(J/m)+(lambda/(2*m))*reg;

%Gradient 
sum1=0;
  for i=1:m
    sum1=sum1+((hypothesis(i)-y(i))*X(i,1));
  end
grad(1,1)=(sum1/m);
for j=2:n
  sum1=0;
  for i=1:m
    sum1=sum1+((hypothesis(i)-y(i))*X(i,j));
  end
  grad(j,1)=(sum1/m)+(lambda/m)*theta(j,1);
  end





% =============================================================

end
