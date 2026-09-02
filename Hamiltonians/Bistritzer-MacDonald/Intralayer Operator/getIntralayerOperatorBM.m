function intraH = getIntralayerOperatorBM(q, stack, DoF)
    theta = deg2rad(stack.theta);
    %shiftK = stack.K-stack.K(:, 2);
    m = size(DoF, 1);
    [r, c, v] = deal(zeros(2*m, 1));
    offset = [1 2; 2 1];
    t = 0;
    vf =  6.582119569*0.8; % eVAs sqrt(3)/2 * 2.46 * 2.99251;
    for i1 = 1:3
        mask = DoF(:, 9) == i1;
        mValid = nnz(mask);
        I = 2*(find(mask)-1);
        k = q+DoF(mask, 1:2)';%-shiftK(:, i1);

        kPlus = k(1, :)+1i*k(2, :);
        kMinus = conj(kPlus);
        
        V{1} = -vf.*exp(-1i*theta(i1)).*kPlus;
        V{2} = -vf.*exp(1i*theta(i1)).*kMinus;
        for i2 = 1:2
            idx = t+(1:mValid);
            r(idx) = I+offset(1, i2);
            c(idx) = I+offset(2, i2);
            v(idx) = V{i2};
            t = t+mValid;
        end
    end
    intraH = sparse(r, c, v, 2*m, 2*m);
end