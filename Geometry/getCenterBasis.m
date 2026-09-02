function X = getCenterBasis(DoF, x)
mX = numel(x);
centerIdx = find(all(DoF(:, 3:8)==0, 2)).';
centerIdx = 2*(centerIdx-1)+[1; 2];
centerIdx = centerIdx(:);
X = sparse(centerIdx(x), 1:mX, 1, 2*size(DoF, 1), mX);
end