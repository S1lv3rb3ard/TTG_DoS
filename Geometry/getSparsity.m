function S = getSparsity(DoF, output)
    intraS = logical(speye(size(DoF, 1)));
    I = cell(1, 2);
    J = cell(1, 2);
    
    for i1 = 1:2
        r = find(DoF(:, 9) == i1);
        c = find(DoF(:, 9) == i1+1);
        colCheck = 2*(mod(i1+1, 3)+1)+(1:2);
        X = DoF(r, colCheck);
        Y = DoF(c, colCheck);
        % Match spectator vectors with a tolerance, not exact equality. getDoF computes
        % the same reciprocal vector by different routes (B*x0 for one layer, B*y for
        % the other), which can differ in the last bit. Exact matching then silently
        % dropped genuine interlayer pairs (about 9% of the pairs with |Q| <= 6.25 1/A
        % in a test), and 'clean' deleted the affected sites as if they were isolated.
        tol = 1e-9;   % 1/Angstrom; distinct spectator vectors differ by at least |b| ~ 2.9 1/A
        [uniqueY, ~, idxY] = uniquetol(Y, tol, 'ByRows', true, 'DataScale', 1);
        idxY = accumarray(idxY, (1:numel(c))', [], @(x) {x});
    
        [locInX, locUniqueX] = ismembertol(X, uniqueY, tol, 'ByRows', true, 'DataScale', 1);
        matchedCells = idxY(locUniqueX(locInX));
    
        I{i1} = repelem(r(locInX), cellfun(@numel, matchedCells));
        J{i1} = c(vertcat(matchedCells{:}));
    end
    finalI = [I{1}; I{2}; J{1}; J{2}];
    finalJ = [J{1}; J{2}; I{1}; I{2}];
    interS = sparse(finalI(:), finalJ(:), true, size(DoF, 1), size(DoF, 1));

    S = intraS | interS;
    if strcmp(output, 'block')
        numDoF = getNumDoF(DoF);
        S = mat2cell(S, numDoF, numDoF);
    end
end