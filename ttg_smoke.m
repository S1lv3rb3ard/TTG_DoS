function summary = ttg_smoke(repo,logFile)
%TTG_SMOKE CPU checks of TTG_DoS (Oct 7-8 changes), about a minute.
%   summary = ttg_smoke(repo, logFile), e.g. ttg_smoke(pwd, 'ttg_smoke_report.txt')
%   Run from a clean path: the repository is added with genpath(repo).
addpath(genpath(repo));
if exist(logFile,'file'), delete(logFile); end
diary(logFile);
cleanup = onCleanup(@() diary('off'));
fprintf('MATLAB %s, %s\n',version,datestr(now));
failures = {};
check = @(ok,msg) record(ok,msg);

stack = getStack(1.42*sqrt(3),[-1.4,0,2.8]);
minW = getMinW(stack);
W = 0.4324; L = 5*max(vecnorm(stack.B{2},2,1));
DoF = getDoF(stack,W,L,'clean');
DoFfull = getDoF(stack,W,L,'full');
fprintf('DoF: clean %d, full %d, minW %.4f\n',size(DoF,1),size(DoFfull,1),minW);

% Monolayer: whole groups -> Dirac at K, 3.17 meV, isotropic vF
[ED,vF] = getMonolayerDirac(stack);
fprintf('Monolayer: ED = %.3f meV, hbar vF = %.4f eV A\n',1e3*ED,vF);
check(abs(1e3*ED-3.17) < 0.02,'Dirac energy 3.17 meV');
check(abs(vF-5.577) < 2e-3,'hbar vF 5.577 eV A');

% Hamiltonian: Hermitian, time reversal
H = getHamiltonianTB(stack,DoF,6.25,0.25);
q = stack.K(:,2)+[0.004;-0.003];
Hq = H(q);
herm = norm(Hq-Hq','fro')/norm(Hq,'fro');
fprintf('Hermiticity error %.2e, size %d, nnz %d\n',herm,size(Hq,1),nnz(Hq));
check(herm < 1e-13,'H(q) Hermitian');
% Match on (G, layer): a vector (0,0,G3) occurs on layer 1 and on layer 2.
G = [DoF(:,3:8),DoF(:,9)]; [tf,loc] = ismembertol([-G(:,1:6),G(:,7)],G,1e-9,'ByRows',true,'DataScale',1);
fprintf('DoF closed under G -> -G (per layer): %d of %d sites matched\n',nnz(tf),numel(tf));
if all(tf)
    m = size(DoF,1); perm = zeros(2*m,1); perm(1:2:end) = 2*loc-1; perm(2:2:end) = 2*loc;
    P = sparse(1:2*m,perm,1,2*m,2*m);
    trErr = full(max(abs(H(-q)-P*conj(Hq)*P'),[],'all'));
    fprintf('Time reversal max|H(-q) - P conj(H(q)) P''| = %.2e\n',trErr);
    check(trErr < 1e-12,'time reversal');
else
    check(false,'DoF closed under G -> -G');
end

% Cross-valley identity behind the doubled weight: K' model = negated DoF
DoFm = DoF; DoFm(:,1:8) = -DoF(:,1:8);
Hm = getHamiltonianTB(stack,DoFm,6.25,0.25);
cvErr = full(max(abs(Hm(-q)-conj(Hq)),[],'all'));
fprintf('max |H_K''(-q) - conj(H_K(q))| = %.2e\n',cvErr);
check(cvErr < 1e-12,'K'' model at -q = conj(K model at q)');

% Sum rule: with L large enough every start site exists and sum w mu0 is the exact count
DoF8 = getDoF(stack,W,8*max(vecnorm(stack.B{2},2,1)),'clean');
X8 = getMoireBasis(stack,DoF8,W);
[~,w8] = getMoireBZ(stack,1); w8 = w8/(6*stack.V^2);
C = {stack.B{1}-stack.B{2},stack.B{1}-stack.B{2},stack.B{2}-stack.B{3}};
mass8 = 0; exact8 = 0;
for j = 1:3
    lim = floor(W*vecnorm(inv(C{j}),2,2));
    [n1,n2] = ndgrid(-lim(1):lim(1),-lim(2):lim(2));
    exact8 = exact8+w8(j)*2*nnz(vecnorm(C{j}*[n1(:) n2(:)]',2,1) < W);
    mass8 = mass8+w8(j)*size(X8{j},2);
end
fprintf('Sum rule at L = 8|b|: sum w mu0 = %.6f, exact site count %.6f\n',mass8,exact8);
check(abs(mass8-exact8) < 1e-12,'sum rule (all start sites present at L = 8|b|)');
lastwarn('');

% Start vectors: one layer per grid, all targets found (Oct 7 fix)
X = getMoireBasis(stack,DoF,W);
expectedLayer = [3,2,1];
for j = 1:3
    [r,~] = find(X{j});
    layers = unique(DoF(ceil(r/2),9)).';
    fprintf('Grid %d: %d start sites on layer(s) %s\n',j,size(X{j},2)/2,mat2str(layers));
    check(isequal(layers,expectedLayer(j)),sprintf('grid %d samples layer %d only',j,expectedLayer(j)));
end

% Shift identity spot check for grid 2 (layer-2 site (B1 n,0,0) vs centre at q + C12 n)
n = [1;0]; C12 = stack.B{1}-stack.B{2};
site = find(DoF(:,9) == 2 & all(abs(DoF(:,3:8)-[(stack.B{1}*n).',0,0,0,0]) < 1e-9,2));
centre = find(DoF(:,9) == 2 & all(DoF(:,3:8) == 0,2));
if ~isempty(site)
    s = 1/(1.25*norm(Hq,inf));
    q0 = stack.K(:,2)+C12*[0.3;0.4];
    e = @(i) sparse(2*i-1,1,1,2*size(DoF,1),1);
    muSite = ChebyshevMoments({s*H(q0)},{e(site)},24,'cpu');
    muCentre = ChebyshevMoments({s*H(q0+C12*n)},{e(centre)},24,'cpu');
    d = max(abs(muSite-muCentre));
    fprintf('Shift identity (layer 2, n = [1;0]), max moment difference = %.2e\n',d);
    check(d < 1e-6,'shift identity layer 2');
end

% Small DoS through getDoS (non-nested path), CPU, no parpool
[Q,weights,gridIdx] = getMoireBZ(stack,2); %#ok<ASGLU>
P = 200; s = 1/(1.25*norm(Hq,inf));
JobList = table; mu = [];
for X3 = 1:3
    for i = 1:size(Q{X3},2)
        mu(end+1,:) = ChebyshevMoments({s*H(Q{X3}(:,i))},{X{X3}},P,'cpu'); %#ok<AGROW>
        JobList = [JobList; table(1,X3,1,{Q{X3}(:,i).'},"TB",'VariableNames',{'t','X','N','Q','model'})]; %#ok<AGROW>
    end
end
JobList.mu = mu;
Job.list = JobList; Job.P = P; Job.scale = s;
Job.weights = {weights/(6*stack.V^2)};
E = linspace(-0.2,0.2,401);
Job = getDoS(Job,E,'cpu');
dos = Job.list.DoS;
mass = sum(Job.weights{1}.*arrayfun(@(X3) sum(mu(JobList.X == X3,1)),1:3));
fprintf(['DoS: finite %d, min %.3e, max %.3e; sum w mu0 = %.4f at L = 5|b| ' ...
    '(start sites missing, see the warning above)\n'],all(isfinite(dos)),min(dos),max(dos),mass);
check(all(isfinite(dos)),'DoS finite');

% Continuum model: Dirac points of layers 1 and 3 at K_j
Hbm = getHamiltonianBM(stack,DoF);
for j = [1,3]
    centreJ = find(DoF(:,9) == j & all(DoF(:,3:8) == 0,2));
    rows = 2*centreJ-1+(0:1);
    Hc = full(Hbm(stack.K(:,j))); Ec = eig(Hc(rows,rows));
    fprintf('BM: layer %d centre block at q = K_%d: |E| <= %.2e eV\n',j,j,max(abs(Ec)));
    check(max(abs(Ec)) < 1e-9,sprintf('BM Dirac point layer %d at K_%d',j,j));
end

lc = getLinecut(stack.K,10);
G12 = lc.q(:,21);
check(abs(norm(G12-stack.K(:,1))-norm(stack.K(:,2)-stack.K(:,1))) < 1e-12,'Gamma12 at moire-zone centre');

if isempty(failures)
    summary = 'all smoke checks passed';
else
    summary = sprintf('%d smoke check(s) failed: %s',numel(failures),strjoin(failures,'; '));
end
fprintf('\n===== %s =====\n',summary);

    function record(ok,msg)
        if ok
            fprintf('  PASS  %s\n',msg);
        else
            fprintf('  FAIL  %s\n',msg);
            failures{end+1} = msg;
        end
    end
end
