function intraSkeleton = getIntraSkeletonTB(stack, tau, DoF)
    [intraSkeleton.Shells, intraSkeleton.ShellWeights] = getShells(stack, tau);
    for i1 = 1:3
        i2 = mod(i1, 3)+1;
        i3 = mod(i1+1, 3)+1;
        mask = DoF(:, 9)==i1;
        intraSkeleton.r{i1} = 2*(find(mask)-1);
        intraSkeleton.R{i1} = rotd(stack.theta(i1));
        intraSkeleton.G{i1} = DoF(mask, 2*i2+(1:2))+DoF(mask, 2*i3+(1:2));
        intraSkeleton.G{i1} = intraSkeleton.G{i1}*intraSkeleton.R{i1};
    end
end