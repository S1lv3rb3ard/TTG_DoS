function T = Chebyshev(P, scaledE, mode)
    [~, nX] = size(scaledE);
    T = zeros(P+1, nX, 'like', scaledE);
    if strcmpi(mode, 'gpu')
        T = gpuArray(T);
        scaledE = gpuArray(scaledE);
    end
    T(1, :) = 1; 
    T(2, :) = scaledE;
    scaledE = 2.*scaledE; %reshape(2.*scaledE, 1, mX, nX);
    for i3 = 3:P+1
        T(i3, :) = scaledE.*T(i3-1,:)-T(i3-2,:);
    end
    %T = permute(T, [3, 1, 2]);
end