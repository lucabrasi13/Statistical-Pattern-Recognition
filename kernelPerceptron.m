function f = kernelPerceptron(y,K,N,epoch)
% NAME
%   kernelPerceptron - Kernel Perceptron following the RKHS trick
% FUNCTION
%   f = kernelPerceptron(y,K,N,epoch)
% DESCRIPTION
%   Classifies linearly separable samples using Kernel Perceptron.
% INPUTS
%   K                    (mat)           (NxN)      Kernel Matrix
%   y                    (vector)       (Mx1)      Vector of training labels
%   N                   (scalar)                        Scalar indicating sample size
%   epoch          (scalar)                         Scalar indicating iterations. 
% OUTPUT
%   f                    (vector)       (Nx1)       Vector indicating decision.    
% AUTHOR
%   Rohit Kashyap , November 2018

    a = zeros(N,1); 
    counter = Inf;
    while(counter~=0)
        counter = 0;
        for i = 1:N
            temp1 = 0; temp2 = 0;
            for j = 1:N
                temp1 = temp1 + a(j)*y(j)*K(i,j);
                temp2 = temp2 + a(j)*y(j);
            end
            if(y(i)*(temp1+temp2)<=0)
                a(i) = a(i)+1;
                counter = counter+1;
            end
        end
        epoch = epoch - 1;
        if(epoch == 0)
            counter = 0;
        end
    end
    
    f(N,1) = 0;
    for i = 1:N
        temp1 = 0;temp2 = 0;
        for j = 1:N
            temp1 = temp1 + a(j)*y(j)*K(i,j);
            temp2 = temp2 + a(j)*y(j);
        end
        f(i) = temp1 + temp2;
    end
    f = sign(f);
end
