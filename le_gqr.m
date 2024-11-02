% 
% Abscissas and Weights of Legendre-Gauss Quadrature Rule
% 
% [A, W] = LE_GQR(N)
%   Return abscissas and weights of Legendre-Gauss quadrature rule of a 
%   given order
% 
%   [Input Argument]
%       n - Integer, order of the Gauss quadrature
% 
%   [Ouput Argument]
%       a - Vector, abscissas of the Gauss quadrature
%       w - Vector, weights of the Gauss quadrature
% 
% Details:
%   1. If n > 6, abscissas and weights will be caculated via legpts() in 
%      Chebyfun Toolbox


function [a, w] = le_gqr(n)
    switch n
        case 2
            a = [-sqrt(3)/3, sqrt(3)/3]';
            w = [1, 1]';
        case 3
            a = [-sqrt(15)/5, 0, sqrt(15)/5]';
            w = [5/9, 8/9, 5/9]';
        case 4
            a = [-sqrt(525+70*sqrt(30))/35, -sqrt(525-70*sqrt(30))/35, ...
                 sqrt(525-70*sqrt(30))/35, sqrt(525+70*sqrt(30))/35]';
            w = [(18-sqrt(30))/36, (18+sqrt(30))/36, (18+sqrt(30))/36, ...
                 (18-sqrt(30))/36]';
        case 5
            a = [-sqrt(245+14*sqrt(70))/21, -sqrt(245-14*sqrt(70))/21, ...
                 0, sqrt(245-14*sqrt(70))/21, sqrt(245+14*sqrt(70))/21]';
            w = [(322-13*sqrt(70))/900, (322+13*sqrt(70))/900, 128/225, ...
                 (322+13*sqrt(70))/900, (322-13*sqrt(70))/900]';
        case 6
            a = [-0.9324695142031520500580654697842, ...
                 -0.66120938646626448154108857124811, ...
                 -0.23861918608319690471297747080826, ...
                 0.23861918608319690471297747080826, ...
                 0.66120938646626448154108857124811, ...
                 0.9324695142031520500580654697842]';
            w = [0.17132449237917041218182134798553, ...
                 0.36076157304813871729010088529321, ...
                 0.46791393457269125910613638552604, ...
                 0.46791393457269125910613638552604, ...
                 0.36076157304813871729010088529321, ...
                 0.17132449237917041218182134798553]';
        otherwise
            [a, w] = legpts(n);
            w = w';  
    end
end