function [Shells, ShellWeights] = getShells(stack, tauIntra)
% GETSHELLS  Intralayer neighbour groups of graphene (Jung & MacDonald 2013, Table II).
%
%   Shells{1}: A-A lattice vectors R, including the on-site term R = 0.
%   Shells{2}: A-B lattice vectors R, where the bond is R + tau_B.
%   ShellWeights{i}: the hopping amplitude of each vector (eV).
%
%   Groups are formed by BOND length: |R| for A-A, |R + tau_B| for A-B. A group is
%   kept whole when its bond length is below tauIntra (Angstrom). Whole groups keep
%   the C3 symmetry, so the monolayer Dirac point stays at K.
%   Default tauIntra = Inf keeps every tabulated group:
%   8 A-A values (on-site plus 7 groups) and 10 A-B groups.
%
%   The previous version compared |R| with tau for the A-B bonds, not the bond length
%   |R + tau_B|. With tau = 8 A it kept 2/3, 4/6, 2/6 and 2/6 of A-B groups 7-10 and
%   dropped A-A groups 6-7. That broke C3: the Dirac point moved 4.7e-4 1/A off K,
%   with a 5.1 meV splitting at K.
    if nargin < 2 || isempty(tauIntra), tauIntra = Inf; end
    t{1} = [0.3302 0.23206 0.04969 -0.02499 0.00285 0.00204 -0.00014 -0.00029];
    t{2} = [-2.99251 -0.28983 0.02791 -0.00877 -0.01870 0.00621 -0.00256 -0.00018 -0.00033 -0.00264];
    [X1, X2] = ndgrid(-10:10);
    X = [X1(:) X2(:)]';
    ShellWeights = cell(1, 2);
    Shells = cell(1, 2);
    for i1 = 1:2
        bonds = stack.A{2}*X+(i1==2)*stack.tau(:, 2);        % A-A: R;  A-B: R + tau_B
        distance = round(vecnorm(bonds, 2, 1)/stack.aG, 4);
        uniqueDistance = unique(distance);
        uniqueDistance = uniqueDistance(1:length(t{i1}));    % the tabulated groups
        keepGroup = uniqueDistance*stack.aG < tauIntra;      % whole groups only
        index = distance == uniqueDistance(keepGroup)';
        weights = t{i1}(keepGroup);
        ShellWeights{i1} = repelem(weights, sum(index, 2)).';
        Shells{i1} = cell2mat(cellfun(@(x) bonds(:, x)-(i1==2)*stack.tau(:, 2), ...
            num2cell(index, 2), 'uni', false)');
    end
end
