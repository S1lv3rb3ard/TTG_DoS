function [Shells, ShellWeights] = getShells(stack, tau)
    t{1} = [0.3302 0.23206 0.04969 -0.02499 0.00285 0.00204 -0.00014 -0.00029];
    t{2} = [-2.99251 -0.28983 0.02791 -0.00877 -0.01870 0.00621 -0.00256 -0.00018 -0.00033 -0.00264];
    [X1, X2] = ndgrid(-10:10);
    X = [X1(:) X2(:)]';
    ShellWeights = cell(1, 2);
    Shells = cell(1, 2);
    for i1 = 1:2
        sites = stack.A{2}*X+(i1==2)*stack.tau(:, 2);
        distance = round(vecnorm(sites(1:2, :), 2, 1)/stack.aG, 4);
        uniqueDistance = unique(distance);
        uniqueDistance = uniqueDistance(1:length(t{i1}));
        index = distance == uniqueDistance';
        ShellWeights{i1} = repelem(t{i1}, sum(index, 2)).';
        Shells{i1} = cell2mat(cellfun(@(x) sites(1:2, x)-(i1==2)*stack.tau(:, 2), num2cell(index, 2), 'uni', false)');
        tauMask = vecnorm(Shells{i1}, 2, 1)-tau<1e-15;
        Shells{i1} = Shells{i1}(:, tauMask);
        ShellWeights{i1} = ShellWeights{i1}(tauMask);
    end
end