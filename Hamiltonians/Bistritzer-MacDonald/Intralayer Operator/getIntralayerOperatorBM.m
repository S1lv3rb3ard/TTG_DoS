function intraH = getIntralayerOperatorBM(q, stack, DoF, vf)
% GETINTRALAYEROPERATORBM  Rotated Dirac blocks of the continuum model.
%   q is measured from K_2 in the reflected coordinates of getHamiltonianBM.
%   vf (eV*Angstrom) defaults to 6.582119569*0.8 for backward compatibility; getHamiltonianBM
%   now passes the tight-binding value.
    if nargin < 4 || isempty(vf), vf = 6.582119569*0.8; end
    theta = deg2rad(stack.theta);
    %shiftK = stack.K-stack.K(:, 2);
    m = size(DoF, 1);
    [r, c, v] = deal(zeros(2*m, 1));
    offset = [1 2; 2 1];
    t = 0;
    for i1 = 1:3
        mask = DoF(:, 9) == i1;
        mValid = nnz(mask);
        I = 2*(find(mask)-1);
        k = q+DoF(mask, 1:2)';

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