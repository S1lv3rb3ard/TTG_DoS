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
        [uniqueY, ~, idxY] = unique(Y, 'rows');
        idxY = accumarray(idxY, (1:numel(c))', [], @(x) {x});
    
        [locInX, locUniqueX] = ismember(X, uniqueY, 'rows');
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