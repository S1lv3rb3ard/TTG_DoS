function H = getHamiltonianBM(stack, DoF)
    interH = getInterlayerOperatorBM(stack.K, DoF);
    intraH =@(q) getIntralayerOperatorBM(q, stack, DoF);
    H =@(q) clean(conj(interH+intraH(q-stack.K(:, 2))));
end