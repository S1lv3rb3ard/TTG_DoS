function Z = kronDiv(X, Y, side)
    % If side is right solves X = kron(Z, Y) for Z if side is left solves X = kron(Y, Z) for Z
    [mX, nX] = size(X);
    [mY, nY] = size(Y);
    mZ = mX/mY;
    nZ = nX/nY;
    if mod(mX, mY)~=0 || mod(nX, nY)~=0
        error('Dimensions of Y must evenly divide dimensions of X.')
    end
    if islogical(Y)
        sqNormY = nnz(Y);
    else
        sqNormY = sum(abs(nonzeros(Y)).^2);
    end
    if issparse(X)
        [rX, cX, vX] = find(X);
        if strcmpi(side, 'right')
            rZ = ceil(rX/mY);
            cZ = ceil(cX/nY);
        
            rY = mod(rX-1, mY)+1;
            cY = mod(cX-1, nY)+1;
        elseif strcmpi(side, 'left')
            rY = ceil(rX/mZ);
            cY = ceil(cX/nZ);

            rZ = mod(rX-1, mZ)+1;
            cZ = mod(cX-1, nZ)+1;
        else
            error('Side must be left or right.')
        end
        idxY = sub2ind([mY, nY], rY, cY);
        vY = Y(idxY);
        vProj = vX.*conj(vY);
        Z = sparse(rZ, cZ, vProj, mZ, nZ)/sqNormY;
    else
        Y = Y(:);
        if strcmpi(side, 'right')
            X = reshape(X, mY, mZ, nY, nZ);
            X = permute(X, [1, 3, 2, 4]);
        elseif strcmpi(side, 'left')
            X = reshape(X, mZ, mY, nZ, nY);
            X = permute(X, [2, 4, 1, 3]);
        else
            error('Side must be left or right.')
        end
        X = reshape(X, mY*nY, mZ*nZ);
        Z = (Y'*X)./sqNormY;
        Z = reshape(Z, mZ, nZ);
    end

    if islogical(X)
        Z = logical(Z);
    end
end