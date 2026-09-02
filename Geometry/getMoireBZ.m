function [MoireBZ, weights] = getMoireBZ(stack, N)
ranN = (0:N-1)/N;
[X1, X2] = ndgrid(ranN);
X = [X1(:) X2(:)].';
MoireBZ = cell(1, 3);
weights = zeros(1, 3);
for j = 1:3
    if j == 1
        C = stack.B{1}-stack.B{2};
    elseif j == 2
        C = stack.B{1}-stack.B{2};
    elseif j == 3
        C = stack.B{2}-stack.B{3};
    end
    weights(j) = abs(det(C))/(N^2);
    MoirePoints = C*X;
    MoireBZ{j} = [MoirePoints+stack.K(:, j), MoirePoints+stack.pK(:, j)];
end
end