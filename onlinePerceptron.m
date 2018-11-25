% NAME
%   onlinePerceptron - Function following Rosenblatts perceptron
% FUNCTION
%   theta = onlinePerceptron(X,y,u,epoch)
% DESCRIPTION
%   Classifies linearly separable samples using the Perceptron cost
%   function
% INPUTS
%   X           (mat)       (MxN+1)     Training data points with ones
%                                       appended for the bias calculation.
%   y           (vector)    (Mx1)       Vector of training labels
%   u           (scalar)                Scalar indicating step size, usually 1 
%   epoch       (scalar)                Scalar indicating iterations. 
% OUTPUT
%   theta       (vector)    (N+1x1)     Vector indicating learnt parameters    
% AUTHOR
%   Rohit Kashyap , November 2018

function theta = onlinePerceptron(X,y,u,epoch)
%% Initialization
    counter = Inf;
    N = size(X,1);
    theta = zeros(size(X,2),1);
%% Run the algorithm
    while(counter~=0)
        counter = 0;
        for n = 1:N
            if(y(n)*X(n,:)*theta <= 0)
                theta = theta + u*y(n).*X(n,:)';
                counter = counter + 1;
            end
        end
        epoch = epoch - 1;
        if(epoch == 0)      % Terminate iff no linear separability          
            counter = 0;
        end
    end
end