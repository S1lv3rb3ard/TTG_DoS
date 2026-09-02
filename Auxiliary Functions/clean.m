function X = clean(X)
    [mX, nX] = size(X);
    
    % Define state: s=issparse, r=isreal
    % Using a string key for readability
    state = [char('s' * issparse(X)), char('r' * isreal(X))];
    
    switch state
        case 'sr' % Sparse Real
            [r, c, v] = find(X);
            mask = abs(v) > eps;
            X = sparse(r(mask), c(mask), v(mask), mX, nX);
            
        case 's'  % Sparse Complex
            [r, c, v] = find(X);
            reV = real(v); imV = imag(v);
            reV(abs(reV) <= 1e-15) = 0;
            imV(abs(imV) <= 1e-15) = 0;
            vClean = reV + 1i*imV;
            mask = vClean ~= 0;
            X = sparse(r(mask), c(mask), vClean(mask), mX, nX);
            
        case 'r'  % Dense Real
            X(abs(X) <= eps) = 0;
            
        case ''   % Dense Complex (neither 's' nor 'r')
            reX = real(X); imX = imag(X);
            reX(abs(reX) <= eps) = 0;
            imX(abs(imX) <= eps) = 0;
            X = reX + 1i*imX;
    end
end