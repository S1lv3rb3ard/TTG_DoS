function interSkeleton = getInterSkeletonTB(stack, DoF)
    S = getSparsity(DoF, 'block');
    for i1 = 1:2
        i2 = mod(i1, 3)+1;
        i3 = mod(i1+1, 3)+1;
        refI = find(DoF(:, 9)==i1); refJ = find(DoF(:, 9)==i2);
        [I, J] = find(S{i1, i2});
        I = refI(I); J = refJ(J);
        G = cell(3, 1);
        G{i1} = DoF(J, 2*i1+(1:2));
        G{i2} = DoF(I, 2*i2+(1:2));
        G{i3} = DoF(I, 2*i3+(1:2));
        interSkeleton.G{i1} = G{1}+G{2}+G{3};
        interSkeleton.P{i1}(:, 2) = exp(-1i*G{i2}*stack.tau(:, i2));
        interSkeleton.P{i1}(:, 3) = exp(1i*G{i1}*stack.tau(:, i1));
        interSkeleton.P{i1}(:, 4) = interSkeleton.P{i1}(:, 2).*interSkeleton.P{i1}(:, 3); %prod(interSkeleton.P{i1}(:, 2:3), 2);
        interSkeleton.P{i1}(:, 1) = 1;
        interSkeleton.r{i1} = 2*(I-1);
        interSkeleton.c{i1} = 2*(J-1);
    end
    interSkeleton.h = getInterlayerHopping(stack, 250, 15*stack.aG, 250, 250, 2*pi);
end