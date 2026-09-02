function H = getHamiltonianTB(stack, DoF, tau, delta)
    intraSkeleton = getIntraSkeletonTB(stack, tau, DoF);
    intraH =@(q) getIntralayerOperatorTB(q, stack, intraSkeleton);
    interSkeleton = getInterSkeletonTB(stack, DoF);
    interH =@(q) getInterlayerOperatorTB(q, tau, delta, DoF, interSkeleton);
    U = getUnitaryTransform(stack.tau(:), DoF); 
    H =@(q) intraH(q)+interH(q);
end