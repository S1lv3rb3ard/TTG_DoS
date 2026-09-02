function numDoF = getNumDoF(DoF)
    uniqueLayers = unique(DoF(:, 9));
    numLayers = length(uniqueLayers);
    numDoF = zeros(numLayers, 1);
    for i1 = 1:numLayers
        numDoF(i1) = nnz(DoF(:, 9)==uniqueLayers(i1));
    end
end