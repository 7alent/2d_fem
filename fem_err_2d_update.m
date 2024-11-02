
%% DO NOT EDIT THIS CODE!!!
%%% A copy (simplified version) of this code is fem_err_2d_simple()

% 
% Calculate the L2 or H1 Error of Two Function
% 
% ERR = FEM_ERR_2D_UPDATE(V, W, X_BOUND, ERR_TYPE, GQO)
%   Given 2 2-D functions in the form of a vector (function values on an 
%   equidistant grid) or a function expression, calulate the L2/H1 error 
%   between them
%   Consider Dirichlet boundary condition and suppose the domain of both 
%   functions is xy_bound = [xa, xb] x [ya, yb] 
%   The error is calculated by Gauss quadrature on each element if at least
%   one of the input functions (v and w) is a vector
% 
%   [Input Argument]
%       v, w - Vector or function handle, maybe TT-vector in the future
%       xy_ep - Vector, left and right endpoints of x and y intervals
%       err_type - Character, 'L2' or 'H1' for L2/H1 error resp.
%       gqr - Vector or cell, rules of Gauss quadrature of x/y, can be 
%             given as an 2 x 1 integer vector (orders of quadrature w.r.t. 
%             x and y) or a 2 x 1 cell (its values are 2-column matrices, 
%             whose columns are abscissas and weights of x/y resp.)
% 
%   [Ouput Argument]
%       err - Double, L2/H1 error of the 2 input functions
% 
% Details:
%   1. To calculte the L2/H1 norm of v/w, set w/v as 0
%   2. Gauss quadrature order (gqr) is recommended to be large (at least 20 
%      for a function handle input, especially when it oscillates severely)
% 
% Future Routines:
%   1. Deal with the case that x and y axes have different step length 
%      (i.e. number of elements on each row ~= number of elements on each 
%      column, N ~= M)


function err = fem_err_2d_update(v, w, xy_ep, err_type, gqr)  
    % Input check
    % Ensure v to be the function handle or longer vector in the 2 input
    % functions (v and w)
    % Ensure any vector to be a column vector
    if isa(v, 'function_handle') && isa(w, 'function_handle')
        input_type = 'Both are functions';
    elseif isa(v, 'function_handle') && ~isa(w, 'function_handle')
        input_type = 'Function and vector';
        if size(w, 1) == 1
            w = w';
        end
    elseif ~isa(v, 'function_handle') && isa(w, 'function_handle')
        input_type = 'Function and vector';
        if size(v, 1) == 1
            v = v';
        end
        [v, w] = deal(w, v);
    else
        input_type = 'Both are vectors';
        if size(v, 1) == 1
            v = v';
        end
        if size(w, 1) == 1
            w = w';
        end
        if size(v, 1) < size(w, 1)
            [v, w] = deal(w, v);
        end
    end
    

    % Gauss quadrature rule
    if ~iscell(gqr)
        [pa, pw] = le_gqr(gqr(1));
        [qa, qw] = le_gqr(gqr(2));
        gqr = {[pa pw], [qa qw]};
    else
        [pa, pw] = deal(gqr{1}(:, 1), gqr{1}(:, 2));
        [qa, qw] = deal(gqr{2}(:, 1), gqr{2}(:, 2));
    end
    

    % Endpoints
    [xa, xb, ya, yb] = deal(xy_ep(1), xy_ep(2), xy_ep(3), xy_ep(4));

    
    % Case I: Both v and w are function handles
    if strcmp(input_type, 'Both are functions')
        % Normalization
        norm_sq_v = norm_sq2(v, xy_ep, err_type, gqr);
        norm_sq_w = norm_sq2(w, xy_ep, err_type, gqr);
        v = @(x, y)(v(x, y)/sqrt(norm_sq_v));
        w = @(x, y)(w(x, y)/sqrt(norm_sq_w));

        % Caculate the error
        [v, w] = deal(@(x, y)(v(x, y)+w(x, y)), @(x, y)(v(x, y)-w(x, y)));
        err = sqrt(min(norm_sq2(v, xy_ep, err_type, gqr), ...
                       norm_sq2(w, xy_ep, err_type, gqr)));
    end
    

    % Case II: v/w and w/v are a function handle and a vector resp.
    if strcmp(input_type, 'Function and vector')
        % Normalization
        norm_sq_v = norm_sq2(v, xy_ep, err_type, gqr);
        norm_sq_w = norm_sq2(w, xy_ep, err_type, gqr);
        v = @(x, y)(v(x, y)/sqrt(norm_sq_v));
        w = w/sqrt(norm_sq_w);

        % Pre-calculation
        N = sqrt(size(w, 1))+1; % Number of elements on each row
        M = N;  % Number of elements on each column
        hx = (xb-xa)/N;
        hy = (yb-ya)/M;
        J = hx*hy/4; % Jacobian
        w = reshape(w, [N-1 M-1])';
        w = [zeros(1, N+1); ...
             zeros(M-1, 1), w, zeros(M-1, 1); ...
             zeros(1, N+1)];

        % Shape function and gradients
        f = @(pn, qm, p, q)((1+pn*p)*(1+qm*q)/4);
        if strcmp(err_type, 'H1')
            fp = @(pn, qm, p, q)((pn*(1+qm*q))/4*2/hx);
            fq = @(pn, qm, p, q)(((1+pn*p)*qm)/4*2/hy);
            vx = matlabFunction(diff(sym(v), 'x'), 'Vars', {'x','y'});
            vy = matlabFunction(diff(sym(v), 'y'), 'Vars', {'x','y'});
        end

        % Calculate the error
        norm_sq_v = 0;
        norm_sq_w = 0;
        for m = 1:M % Each row of elements
        for n = 1:N % Each column of elements
            xn = xa+(n-1)*hx; % Left endpoint of x in the element
            ym = ya+(m-1)*hy; % Left endpoint of y in the element
            ff = @(p, q)(w(m, n)*f(-1, -1, p, q) ...
                         +w(m, n+1)*f(1, -1, p, q) ...
                         +w(m+1, n)*f(-1, 1, p, q) ...
                         +w(m+1, n+1)*f(1, 1, p, q));
            rr = @(p, q)(v((p+1)*hx/2+xn, (q+1)*hy/2+ym));
            if strcmp(err_type, 'H1')
                wp = @(p, q)(w(m, n)*fp(-1, -1, p, q) ...
                             +w(m, n+1)*fp(1, -1, p, q) ...
                             +w(m+1, n)*fp(-1, 1, p, q) ...
                             +w(m+1, n+1)*fp(1, 1, p, q));
                wq = @(p, q)(w(m, n)*fq(-1, -1, p, q) ...
                             +w(m, n+1)*fq(1, -1, p, q) ...
                             +w(m+1, n)*fq(-1, 1, p, q) ...
                             +w(m+1, n+1)*fq(1, 1, p, q));
                vp = @(p, q)(vx((p+1)*hx/2+xn, (q+1)*hy/2+ym));
                vq = @(p, q)(vy((p+1)*hx/2+xn, (q+1)*hy/2+ym));
                integrand_v = @(p, q)((ff(p, q)+rr(p, q))^2 ...
                                      +(wp(p, q)+vp(p, q))^2 ...
                                      +(wq(p, q)+vq(p, q))^2);
                integrand_w = @(p, q)((ff(p, q)-rr(p, q))^2 ...
                                      +(wp(p, q)-vp(p, q))^2 ...
                                      +(wq(p, q)-vq(p, q))^2);
            else
                integrand_v = @(p, q)((ff(p, q)+rr(p, q))^2);
                integrand_w = @(p, q)((ff(p, q)-rr(p, q))^2);
            end
            for k = 1:size(gqr{1}, 1)
            for l = 1:size(gqr{2}, 1)
                norm_sq_v = norm_sq_v+integrand_v(pa(k), qa(l))...
                                      *J*pw(k)*qw(l);
                norm_sq_w = norm_sq_w+integrand_w(pa(k), qa(l))...
                                      *J*pw(k)*qw(l);
            end
            end
        end
        end
        err = sqrt(min(norm_sq_v, norm_sq_w));
    end

    
    % Case III: Both v and w are vectors
    if strcmp(input_type, 'Both are vectors')
        % Interpolation
        v = reshape(v, [sqrt(numel(v)) sqrt(numel(v))])';
        w = reshape(w, [sqrt(numel(w)) sqrt(numel(w))])';
        v = [zeros(1, size(v, 2)+2); ...
             zeros(size(v, 1), 1), v, zeros(size(v, 1), 1); ...
             zeros(1, size(v, 2)+2)];
        w = [zeros(1, size(w, 2)+2); ...
             zeros(size(w, 1), 1), w, zeros(size(w, 1), 1); ...
             zeros(1, size(w, 2)+2)];
        [xv, yv] = meshgrid(linspace(xa, xb, size(v, 2)), ...
                            linspace(ya, yb, size(v, 1)));
        [xw, yw] = meshgrid(linspace(xa, xb, size(w, 2)), ...
                            linspace(ya, yb, size(w, 1)));
        w = interp2(xw, yw, w, xv, yv, 'linear');
        
        % Normalization
        N = sqrt(size(w, 1))+1; % Number of elements on each row
        M = N; % Number of elements on each column
        hx = (xb-xa)/N;
        hy = (yb-ya)/M;
        J = hx*hy/4;
        norm_sq_v = 0;
        norm_sq_w = 0;
        f = @(pn, qm, p, q)((1+pn*p)*(1+qm*q)/4);
        if strcmp(err_type, 'H1')
            fp = @(pn, qm, p, q)((pn*(1+qm*q))/4*2/hx);
            fq = @(pn, qm, p, q)(((1+pn*p)*qm)/4*2/hy);
        end
        for m = 1:M % Each row of elements
        for n = 1:N % Each column of elements
            vv = @(p, q)(v(m, n)*f(-1, -1, p, q) ...
                         +v(m, n+1)*f(1, -1, p, q) ...
                         +v(m+1, n)*f(-1, 1, p, q) ...
                         +v(m+1, n+1)*f(1, 1, p, q));
            ww = @(p, q)(w(m, n)*f(-1, -1, p, q) ...
                         +w(m, n+1)*f(1, -1, p, q) ...
                         +w(m+1, n)*f(-1, 1, p, q) ...
                         +w(m+1, n+1)*f(1, 1, p, q));
            if strcmp(err_type, 'H1')
                vp = @(p, q)(v(m, n)*fp(-1, -1, p, q) ...
                             +v(m, n+1)*fp(1, -1, p, q) ...
                             +v(m+1, n)*fp(-1, 1, p, q) ...
                             +v(m+1, n+1)*fp(1, 1, p, q));
                vq = @(p, q)(v(m, n)*fq(-1, -1, p, q) ...
                             +v(m, n+1)*fq(1, -1, p, q) ...
                             +v(m+1, n)*fq(-1, 1, p, q) ...
                             +v(m+1, n+1)*fq(1, 1, p, q));
                wp = @(p, q)(w(m, n)*fp(-1, -1, p, q) ...
                             +w(m, n+1)*fp(1, -1, p, q) ...
                             +w(m+1, n)*fp(-1, 1, p, q) ...
                             +w(m+1, n+1)*fp(1, 1, p, q));
                wq = @(p, q)(w(m, n)*fq(-1, -1, p, q) ...
                             +w(m, n+1)*fq(1, -1, p, q) ...
                             +w(m+1, n)*fq(-1, 1, p, q) ...
                             +w(m+1, n+1)*fq(1, 1, p, q));
                integrand_v = @(p, q)(vv(p, q)^2+vp(p, q)^2+vq(p, q)^2);
                integrand_w = @(p, q)(ww(p, q)^2+wp(p, q)^2+wq(p, q)^2);
            else
                integrand_v = @(p, q)(vv(p, q)^2);
                integrand_w = @(p, q)(ww(p, q)^2);
            end
            for k = 1:size(gqr{1}, 1)
            for l = 1:size(gqr{2}, 1)
                norm_sq_v = norm_sq_v+integrand_v(pa(k), qa(l))...
                                      *J*pw(k)*qw(l);
                norm_sq_w = norm_sq_w+integrand_w(pa(k), qa(l))...
                                      *J*pw(k)*qw(l);
            end
            end
        end
        end
        v = v/sqrt(norm_sq_v);
        w = w/sqrt(norm_sq_w);

        % Calculate the error
        norm_sq_v = 0;
        norm_sq_w = 0;
        f = @(pn, qm, p, q)((1+pn*p)*(1+qm*q)/4);
        [v, w] = deal(v+w, v-w);
        N = sqrt(size(v, 1))-1; % Number of elements on each row
        M = N; % Number of elements on each column
        hx = (xa-xb)/N;
        hy = (ya-yb)/M;
        J = hx*hy/4;
        for m = 1:M  % Each row of elements
        for n = 1:N  % Each column of elements
            vv = @(p, q)(v(m, n)*f(-1, -1, p, q) ...
                         +v(m, n+1)*f(1, -1, p, q) ...
                         +v(m+1, n)*f(-1, 1, p, q) ...
                         +v(m+1, n+1)*f(1, 1, p, q));
            ww = @(p, q)(w(m, n)*f(-1, -1, p, q) ...
                         +w(m, n+1)*f(1, -1, p, q) ...
                         +w(m+1, n)*f(-1, 1, p, q) ...
                         +w(m+1, n+1)*f(1, 1, p, q));
            if strcmp(err_type, 'H1')
                vp = @(p, q)(v(m, n)*fp(-1, -1, p, q) ...
                             +v(m, n+1)*fp(1, -1, p, q) ...
                             +v(m+1, n)*fp(-1, 1, p, q) ...
                             +v(m+1, n+1)*fp(1, 1, p, q));
                vq = @(p, q)(v(m, n)*fq(-1, -1, p, q) ...
                             +v(m, n+1)*fq(1, -1, p, q) ...
                             +v(m+1, n)*fq(-1, 1, p, q) ...
                             +v(m+1, n+1)*fq(1, 1, p, q));
                wp = @(p, q)(w(m, n)*fp(-1, -1, p, q) ...
                             +w(m, n+1)*fp(1, -1, p, q) ...
                             +w(m+1, n)*fp(-1, 1, p, q) ...
                             +w(m+1, n+1)*fp(1, 1, p, q));
                wq = @(p, q)(w(m, n)*fq(-1, -1, p, q) ...
                             +w(m, n+1)*fq(1, -1, p, q) ...
                             +w(m+1, n)*fq(-1, 1, p, q) ...
                             +w(m+1, n+1)*fq(1, 1, p, q));
                integrand_v = @(p, q)(vv(p, q)^2+vp(p, q)^2+vq(p, q)^2);
                integrand_w = @(p, q)(ww(p, q)^2+wp(p, q)^2+wq(p, q)^2);
            else
                integrand_v = @(p, q)(vv(p, q)^2);
                integrand_w = @(p, q)(ww(p, q)^2);
            end
            for k = 1:size(gqr{1}, 1)
            for l = 1:size(gqr{2}, 1)
                norm_sq_v = norm_sq_v+integrand_v(pa(k), qa(l))...
                                      *J*pw(k)*qw(l);
                norm_sq_w = norm_sq_w+integrand_w(pa(k), qa(l))...
                                      *J*pw(k)*qw(l);
            end
            end
        end
        end
        err = sqrt(min(norm_sq_v, norm_sq_w));
    end


end


%--------------------------------------------------------------------------
% A built-in function for computing Gauss quadrature
function output = gq2(integrand, interval, rule)
    % Gauss quadrature rule
    if ~iscell(rule)
        [abscissa_p, weight_p] = le_gqr(rule(1));
        [abscissa_q, weight_q] = le_gqr(rule(2));
        rule = {[abscissa_p weight_p], [abscissa_q weight_q]};
    else
        [abscissa_p, weight_p] = deal(rule{1}(:, 1), rule{1}(:, 2));
        [abscissa_q, weight_q] = deal(rule{2}(:, 1), rule{2}(:, 2)); 
    end

    % Change of variables
    if ~all(interval == [-1 1 -1 1])
        range_p = interval(2)-interval(1);
        range_q = interval(4)-interval(3);
        integrand = @(p, q)(integrand((p+1)*range_p/2+interval(1), ...
                                      (q+1)*range_q/2+interval(3))...
                            *range_p*range_q/4);
    end

    % Gauss quadrature
    output = 0;
    for k = 1:size(rule{1}, 1)
        for l = 1:size(rule{2}, 1)
            output = output+integrand(abscissa_p(k), abscissa_q(l))...
                            *weight_p(k)*weight_q(l);
        end
    end
end


%--------------------------------------------------------------------------
% A built-in function for computing the square of L2 or H1 norm of a
% function given as a vector or a function handle
function output = norm_sq2(v, interval, norm_type, rule)
    if isa(v, 'function_handle')
        if strcmp(norm_type, 'L2')
            output = gq2(@(x, y)(v(x, y)^2), interval, rule);
        else
            vx = matlabFunction(diff(sym(v), 'x'), 'Vars', {'x','y'});
            vy = matlabFunction(diff(sym(v), 'y'), 'Vars', {'x','y'});
            output = gq2(@(x, y)(v(x, y)^2+vx(x, y)^2+vy(x, y)^2), ...
                         interval, rule);
        end
    else
        N = sqrt(size(v, 1))+1; % Number of elements on each row
        M = N; % Number of elements on each column
        x_range = interval(2)-interval(1);
        y_range = interval(4)-interval(3);
        hx = x_range/N;
        hy = y_range/M;
        J = hx*hy/4;
        v = reshape(v, [N-1 M-1])';
        v = [zeros(1, N+1); ...
             zeros(M-1, 1), v, zeros(M-1, 1); ...
             zeros(1, N+1)];
        f = @(pn, qm, p, q)((1+pn*p)*(1+qm*q)/4);
        if strcmp(norm_type, 'H1')
            fp = @(pn, qm, p, q)((pn*(1+qm*q))/4*2/hx);
            fq = @(pn, qm, p, q)(((1+pn*p)*qm)/4*2/hy);
        end
        output = 0;
        for m = 1:M % Each element on a row
        for n = 1:N % Each element on a column
            r = @(p, q)(v(m, n)*f(-1, -1, p, q) ...
                        +v(m, n+1)*f(1, -1, p, q) ...
                        +v(m+1, n)*f(-1, 1, p, q) ...
                        +v(m+1, n+1)*f(1, 1, p, q));
            rp = @(p, q)(v(m, n)*fp(-1, -1, p, q) ...
                         +v(m, n+1)*fp(1, -1, p, q) ...
                         +v(m+1, n)*fp(-1, 1, p, q) ...
                         +v(m+1, n+1)*fp(1, 1, p, q));
            rq = @(p, q)(v(m, n)*fq(-1, -1, p, q) ...
                         +v(m, n+1)*fq(1, -1, p, q) ...
                         +v(m+1, n)*fq(-1, 1, p, q) ...
                         +v(m+1, n+1)*fq(1, 1, p, q));
            if strcmp(norm_type, 'L2')
                integrand = @(p, q)(r(p, q)^2*J);
                output = output+gq2(integrand, [-1 1 -1 1], rule);
            else
                integrand = @(p, q)((r(p, q)^2+rp(p, q)^2+rq(p, q)^2)*J);
                output = output+gq2(integrand, [-1 1 -1 1], rule);
            end
        end
        end
    end
end