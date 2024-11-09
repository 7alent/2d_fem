% 
% Calculate L2 or H1 Error between Two Functions
% 
% ERR = FEM_ERR_2D_SIMPLE(V, W, X_BOUND, ERR_TYPE, GQO, VARARGIN)
%   Given 2 2-D functions in the form of a vector (function values on an 
%   equidistant grid) or a function expression, calulate the L2/H1 error 
%   between them
%   Consider Dirichlet boundary condition and suppose the domain of both 
%   functions is xy_bound = [xa, xb] x [ya, yb] 
%   The error is calculated by Gauss quadrature on each element if at least
%   one of the input functions (v and w) is a vector
%   This function is a simple version of fem_err_2d_update()
% 
%   [Input Argument]
%       v, w - Vector or function handle/sym, maybe TT-vector in the future
%       xy_ep - Vector, left and right endpoints of x and y intervals
%       err_type - Character or matrix, 'L2' or 'H1' for L2/H1 error resp.,
%                  or a positive definite matrix for positive-definite-
%                  matrix norm
%       gqr - Vector or cell, rules of Gauss quadrature of x/y, can be 
%             given as an 2 x 1 integer vector (orders of quadrature w.r.t. 
%             x and y) or a 2 x 1 cell (its values are 2-column matrices, 
%             whose columns are abscissas and weights of x/y resp.)
%       method - Character, optional, method to calculate the square of the
%                norm, only works when both v and w are vectors, 1/0 for 
%                'quadrature by element'/'quadratic form' methods resp., if
%                err_type is a matrix this argument will be reset as 0
% 
%   [Ouput Argument]
%       err - Double, L2/H1 error of the 2 input functions
% 
% Details:
%   1. To calculte the L2/H1 norm of v/w, set w/v as 0, but using codes
%      below is recommended
%               norm_v = sqrt(normsq_2d(v, xy_ep, 'L2', gqr));
%   2. Gauss quadrature order (gqr) is recommended to be large (at least 20 
%      for a function handle input, especially when it oscillates severely)
%   3. Sym inputs will be converted to function handles before calculation
% 
% Future Routines:
%   1. Deal with the case that x and y axes have different step length 
%      (i.e. number of elements on each row ~= number of elements on each 
%      column, N ~= M)


function err = fem_err_2d_simple(v, w, xy_ep, err_type, gqr, varargin)
    % Deal with sym input
    if isa(v, 'sym')
        v = matlabFunction(v, 'Vars', {'x','y'});
    end
    if isa(w, 'sym')
        w = matlabFunction(w, 'Vars', {'x','y'});
    end
    

    % Input check
    % Let v be the function handle or longer vector in v and w
    % Also ensure any vector to be a column vector
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
    

    % Optional input argument
    if strcmp(input_type, 'Both are vectors')
        if nargin == 6
            method = varargin{1};
        else
            method = 1;
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
        err = sqrt(min(normsq_2d(v, xy_ep, err_type, gqr), ...
                       normsq_2d(w, xy_ep, err_type, gqr)));
    end
    

    % Case II: v/w and w/v are a function handle and a vector resp.
    if strcmp(input_type, 'Function and vector')
        % Normalization
        norm_sq_v = normsq_2d(v, xy_ep, err_type, gqr);
        norm_sq_w = normsq_2d(w, xy_ep, err_type, gqr);
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
            ww = @(p, q)(w(m, n)*f(-1, -1, p, q) ...
                         +w(m, n+1)*f(1, -1, p, q) ...
                         +w(m+1, n)*f(-1, 1, p, q) ...
                         +w(m+1, n+1)*f(1, 1, p, q));
            vv = @(p, q)(v((p+1)*hx/2+xn, (q+1)*hy/2+ym));
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
                integrand_v = @(p, q)((ww(p, q)+vv(p, q))^2 ...
                                      +(wp(p, q)+vp(p, q))^2 ...
                                      +(wq(p, q)+vq(p, q))^2);
                integrand_w = @(p, q)((ww(p, q)-vv(p, q))^2 ...
                                      +(wp(p, q)-vp(p, q))^2 ...
                                      +(wq(p, q)-vq(p, q))^2);
            else
                integrand_v = @(p, q)((ww(p, q)+vv(p, q))^2);
                integrand_w = @(p, q)((ww(p, q)-vv(p, q))^2);
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
        
        % Determine error type
        if method
            if ~ischar(err_type)
                method = 0;
            end
        else
            if ischar(err_type)
                if strcmp(err_type, 'L2')
                    err_type = fem_mat_2d_boost(size(v, 1)-1, xy_ep, ...
                                                'Mass', gqr);
                elseif strcmp(err_type, 'H1')
                    err_type = fem_mat_2d_boost(size(v, 1)-1, xy_ep, ...
                                                'Laplace+Mass', gqr);
                else
                    error(['If norm_type is a character, ', ...
                           'it should be L2/H1']);
                end
            end
        end

        % Normalization
        norm_sq_v = normsq_2d(v, xy_ep, err_type, gqr, method);
        norm_sq_w = normsq_2d(w, xy_ep, err_type, gqr, method);
        v = v/sqrt(norm_sq_v);
        w = w/sqrt(norm_sq_w);

        % Calculate the error
        [v, w] = deal(v+w, v-w);
        norm_sq_v = normsq_2d(v, xy_ep, err_type, gqr, method);
        norm_sq_w = normsq_2d(w, xy_ep, err_type, gqr, method);
        err = sqrt(min(norm_sq_v, norm_sq_w));
    end


end