function [X,y,info] = CorMat3Mex_Wnorm(G,W,b,I,J,OPTIONS,y)
%%%%%%%%%% This code is designed to solve %%%%%%
%%     min    0.5||W^1/2*(X - G)*W^1/2 ||^2     
%%     s.t.   X_ij = b_ij   for (i,j) in (I,J)
%%            X >= tau*I    X is symmetric and positive semi-definite (tau can be zero)
%%
%%%%%%%%%%%%%%%% Based on the algorithm  in  %%%%%%%
%%% ``A Quadratically Convergent Newton Method for 
%%%  Computing the Nearest Correlation Matrix''
%%%  SIAM J. Matrix Anal. Appl. 28 (2006) 360--385.
%%%%%%%%%%%%% By Houduo Qi and Defeng Sun  %%%%%%%%%%
  
%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Input
%   G       the given symmetric correlation matrix
%   b       the right hand side of equality constraints
%   I       row indices of the fixed elements
%   J       column indices of the fixed elements
%   tau     the lower bound of the smallest eigenvalue of X*
%   W       the weight matrix for G
%
%   Output
%   X         the optimal primal solution
%   y         the optimal dual solution 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Last modified by Yan Gao and Defeng Sun on November 23, 2009;  10:00AM   
 

%%% set parameters 
tau    = 0;
tol    = 1.0e-6;    % termination tolerance
tolCG  = 1.0e-2;    % relative accuracy for CGs
maxit      = 200;
maxitsub   = 20;        % maximum num of Line Search in Newton method
maxitCG    = 200;       % maximum num of iterations in PCG
sigma      = 1.0e-4;    % tolerance in the line search of the Newton method
disp          = 1;      % display 
finalEig      = 1;      % =0 no need final eigendecomp 
const_hist    = 5;
const_sparse  = 2;      % check if sparse form for X should be explioted
progress_test = 1.0e-15;
%% get parameters from OPTIONS
if exist('OPTIONS')
    if isfield(OPTIONS,'tau');         tau        = OPTIONS.tau; end    
    if isfield(OPTIONS,'tol');         tol        = OPTIONS.tol; end
    if isfield(OPTIONS,'tolCG');       tolCG      = OPTIONS.tolCG; end
    if isfield(OPTIONS,'maxit');       maxit      = OPTIONS.maxit; end
    if isfield(OPTIONS,'maxitsub');    maxitsub   = OPTIONS.maxitsub; end
    if isfield(OPTIONS,'maxitCG');     maxitCG    = OPTIONS.maxitCG; end
    if isfield(OPTIONS,'disp');        disp       = OPTIONS.disp; end
    if isfield(OPTIONS,'finalEig');    finalEig   = OPTIONS.finalEig; end 
end

t0 = clock;
n = length(G);
k = length(b);

%%% reset parameters
for i = 1:k    % added on November 3, 2009.
    G(I(i),J(i)) = b(i);
    if I(i) ~= J(i)
        G(J(i),I(i)) = b(i);
    end
end
% G0 = G;
G  = G - tau*speye(n);
G  = (G + G')/2;
Ind = find(I==J);     
b(Ind) = b(Ind) - tau;
b0 = b;

W = real(W);
W = (W + W')/2; 
w = ones(n,1);
W_inv = eye(n);

% initialization
k1        = 0;
f_eval    = 0;
num_pcg   = 0;
prec_time = 0;
pcg_time  = 0;
eig_time  = 0;
% initial point
if ( nargin<7 )
    y = zeros(k,1);
end
x0 = y;
f_hist = zeros(const_hist,1);

if disp
    fprintf('\n ******************************************************** \n')
    fprintf( '         Semismooth Newton-CG Method for the W-norm case            ')
    fprintf('\n ******************************************************** \n')
    fprintf('\n The information of this problem is as follows: \n')
    fprintf(' Dim. of    sdp      constr  = %d \n',n)
    fprintf(' Num. of equality    constr  = %d \n',k)
end
%%% test if W is a diagonal matrix
diag_if = norm(W - diag(diag(W)),'fro');
if diag_if < 1.0e-12
    diag_index = 1;
    w = diag(W);
    if min(w) <= 0
    fprintf('Warning: W is not positive definite!')   
    return;
    end
else 
    diag_index =0;
end 
%%% compute W^(1/2) & W^(-1/2)
if diag_index == 0   % not a diagonal weight matrix
    t1         = clock;
    [P,lambda] = MYmexeig(W);
    eig_time   = eig_time + etime(clock,t1);
    if min(lambda) <= 0
        fprintf('Warning: W is not positive definite!')
        return;
    end
    tmp    = real(lambda.^(1/4));
    W_half = P*sparse(diag(tmp));
    W_half = W_half * W_half';
    tmp        = real(lambda.^(-1/4));
    W_half_inv = P*sparse(diag(tmp));
    W_half_inv = W_half_inv * W_half_inv';
    W_inv      = W_half_inv * W_half_inv;      
    % reset G
    G = W_half*G*W_half;
else
    w_half = w.^0.5;
    for i=1:n
        G(i,:) = w_half(i)*G(i,:);
    end
    for j=1:n
        G(:,j) = G(:,j)*w_half(j);
    end
end
G = (G + G')/2;

X = zeros(n,n);
for i = 1:k
    X(I(i),J(i)) = y(i);
end
X = 0.5*(X + X');
if diag_index == 0
    if k <= const_sparse*n 
        X = W_half_inv*sparse(X);
        X = X * W_half_inv;
    else
    X = W_half_inv*X*W_half_inv;
    end 
else
    for i=1:n
        X(i,:) =  X(i,:)/w_half(i);
    end
    for j=1:n
        X(:,j) =  X(:,j)/w_half(j);
    end
end
X = G + X;
X = (X + X')/2;

t1       = clock;
[P,lambda]    = MYmexeig(X);
eig_time = eig_time + etime(clock,t1);
if diag_index ==0
    WP = W_half_inv*P;
else
    WP = P;
    for i=1:n
        WP(i,:) = P(i,:)/w_half(i);
    end
end
[f0,Fy] = gradient(y,I,J,lambda,WP,b0);
f_eval  = f_eval + 1;
f = f0;
b = b0 - Fy;
norm_b = norm(b);
f_hist(1) = f;

val_G     = sum(sum(G.*G))/2;
% dual_val = val_G - f;
% fprintf('\n Initial Dual Objective Function value  ============= %d \n', dual_val)

if disp
    tt         = etime(clock,t0);
    [hh,mm,ss] = time(tt);
    fprintf('\n   Iter     NumCGs     StepLen     NormGrad        FunValue          time_used ')
    fprintf('\n   %2.0f        %s          %s           %5.4e        %8.7e          %d:%d:%d ',0,'-','-',norm_b,f,hh,mm,ss)
end

while ( norm_b>tol && k1 < maxit )
    
    Omega12 = omega_mat(lambda);
    
    % compute preconditioner
    t2 = clock;
    c  = precond_matrix(I,J,Omega12,WP);    
    prec_time = prec_time + etime(clock,t2);
    
    % preconditioned CG starts
    t3 = clock;
    [d,flag,relres,iterk] = pre_cg(b,I,J,tolCG,maxitCG,Omega12,WP,c,diag_index,W_inv,w);
    pcg_time = pcg_time + etime(clock,t3);
    num_pcg  = num_pcg + iterk;
   
    slope = (Fy-b0)'*d; 

    y = x0 + d;       
    X = zeros(n,n);
    for i = 1:k
        X(I(i),J(i)) = y(i);
    end
    X = 0.5*(X + X');
    
    if diag_index ==0
        if  k <= const_sparse *n
            X = W_half_inv*sparse(X);
            X = X * W_half_inv;
        else
            X = W_half_inv*X*W_half_inv;
        end
    else
        for i=1:n
            X(i,:) = X(i,:)/w_half(i);
        end
        for j=1:n
            X(:,j) =  X(:,j)/w_half(j);
        end
    end

    X = G + X;
    X = (X + X')/2;

    t1         = clock;
    [P,lambda] = MYmexeig(X);
    eig_time   = eig_time + etime(clock,t1);
    if diag_index == 0
        WP     = W_half_inv*P;
    else
        WP = P;
        for i=1:n
            WP(i,:) = WP(i,:)/w_half(i);
        end
    end    
    [f,Fy] = gradient(y,I,J,lambda,WP,b0);    
    f_eval = f_eval + 1;

    k_inner = 0;
    while( k_inner <= maxitsub && (f - f0 - sigma*0.5^k_inner*slope)/max(1,abs(f0)) > 1.0e-8 )
        y = x0 + 0.5^k_inner*d;   
        X = zeros(n,n);
        for i = 1:k
            X(I(i),J(i)) = y(i);
        end
        X = 0.5*(X + X');
        
        if diag_index ==0
            if k <= const_sparse *n
                X = W_half_inv*sparse(X);
                X = X * W_half_inv;
            else
                X = W_half_inv*X*W_half_inv;
            end
        else
            for i=1:n
                X(i,:) = X(i,:)/w_half(i);
            end
            for j=1:n
                X(:,j) =  X(:,j)/w_half(j);
            end
        end
       
        X = G + X;
        X = (X + X')/2;
        t1         = clock;
        [P,lambda] = MYmexeig(X);
        eig_time   = eig_time + etime(clock,t1);
        
        if diag_index ==0
            WP     = W_half_inv*P;
        else
            WP =P;
            for i=1:n
                WP(i,:) = WP(i,:)/w_half(i);
            end
        end
        
        [f,Fy]  = gradient(y,I,J,lambda,WP,b0);        
        k_inner = k_inner + 1;        
        f_eval  = f_eval + 1;
    end  
    
    k1 = k1+1;    
    x0 = y;
    f0 = f;
    b  = b0-Fy;
    norm_b = norm(b);
    
    if disp
    tt = etime(clock, t0);
    [hh,mm,ss] = time(tt);     
    fprintf('\n   %2.0d       %2.0d         %2.1e        %3.2e      %10.9e         %d:%d:%d ',k1,iterk,0.5^k_inner,norm_b,f, hh,mm,ss)
    end
    
    % slow convergence test
    if  (k1 < const_hist)
        f_hist(k1+1) = f;
    else
        for i = 1:const_hist-1
            f_hist(i) = f_hist(i+1);
        end
        f_hist(const_hist) = f;
    end  
    if ( k1 >= const_hist-1 && f_hist(1)-f_hist(const_hist) < progress_test )
        fprintf('\n Warning: Progress is too slow! :( ')
        break;
    end
    
    
end   % end for outer loop

% Optimal solution X*
Ip = find(lambda>1.0e-8);
r  = length(Ip);
 
if (r==0)
    X = zeros(n,n);
elseif (r==n)
    X = X;
elseif (r<=n/2)
    lambda1 = lambda(Ip);
    lambda1 = lambda1.^0.5;
    P1 = P(:,Ip);
    if r >1
        P1 = P1*sparse(diag(lambda1));
        X = P1*P1'; % Optimal solution X*
    else
        X = lambda1^2*P1*P1';
    end
else      
    lambda2 = -lambda(r+1:n);
    lambda2 = lambda2.^0.5;
    P2 = P(:,r+1:n);
    P2 = P2*sparse(diag(lambda2));
    X = X + P2*P2'; % Optimal solution X* 
end
X = (X+X')/2;

% rank of X* and corresponding multiplier Z*
r_X = r;
r_Z = length(find(abs(max(0,lambda)-lambda) > 1.0e-8));
% optimal primal and dual objective values
dual_obj = val_G - f;
prim_obj = sum(sum((X-G).*(X-G)))/2;

% recover X* to the original one
if diag_index == 0
    X = W_half_inv*X*W_half_inv;
else
    for i=1:n
        X(i,:) = X(i,:)/w_half(i);
    end
    for j=1:n
        X(:,j) =  X(:,j)/w_half(j);
    end
end
X = (X+X')/2;
X = X + tau*speye(n);
% objM = W_half*(X-G0)*W_half;
% prim_val0 = sum( sum(objM.*objM) )/2;
if finalEig
    if ( diag_index==1 && max(w)/min(w) - 1 <= 1.0e-12 ) %W=scalar*I
        lambda = lambda./max(w);
    else
        t1         = clock;
        [P,lambda] = MYmexeig(X);
        eig_time   = eig_time + etime(clock,t1);
        f_eval     = f_eval + 1;
    end
    info.P     = P;
    info.lam   = max(0,lambda);
    info.rank  = length( find(lambda > 1.0e-8) );
end
info.numIter = k1;
info.numEig  = f_eval;
info.numPcg  = num_pcg;
info.dualVal = dual_obj;

time_used = etime(clock,t0);
if disp
    fprintf('\n')
    fprintf('\n ================ Final Information ================= \n');
    fprintf(' Total number of iterations      = %2.0f \n',k1);
    fprintf(' Number of func. evaluations     = %2.0f \n',f_eval)
    fprintf(' Number of CG Iterations         = %2.0f \n',num_pcg)
    %fprintf(' Primal objective value          = %d \n', prim_val0)
    fprintf('\n The following results are for the transformed problem \n')
    fprintf(' Primal objective value          = %d \n', prim_obj)
    fprintf(' Dual objective value            = %d \n', dual_obj)
    fprintf(' Norm of gradient                = %3.2e \n', norm_b)
    fprintf(' Rank of  X* - (tau * I)         === %2.0d \n', r_X)
    fprintf(' Rank of optimal multiplier Z*   === %2.0d \n', r_Z)
    fprintf(' Computing time for preconditioner     = %3.1f \n',prec_time)
    fprintf(' Computing time for CG Iterations      = %3.1f \n',pcg_time)
    fprintf(' Computing time for eigen-decom        = %3.1f \n',eig_time)
    fprintf(' Total Computing time (secs)           = %3.1f \n',time_used)
    fprintf(' ====================================================== \n');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  end of the main program   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%  **************************************
%%  ******** All Sub-routines  ***********
%%  **************************************
%%% To change the format of time 
function [h,m,s] = time(t)
t = round(t); 
h = floor(t/3600);
m = floor(rem(t,3600)/60);
s = rem(rem(t,60),60);
return
%%% End of time.m


%%% mexeig decomposition
function [P,lambda] = MYmexeig(X)
[P,lambda] = mexeig(X);
P          = real(P);
lambda     = real(lambda);
if issorted(lambda)
    lambda = lambda(end:-1:1);
    P      = P(:,end:-1:1);
elseif issorted(lambda(end:-1:1))
    return;
else
    [lambda, Inx] = sort(lambda,'descend');
    P = P(:,Inx);
end
% % % Rearrange lambda and P in the nonincreasing order
% % if lambda(1) < lambda(end) 
% %     lambda = lambda(end:-1:1);
% %     P      = P(:,end:-1:1);
% % end
return
%%% End of MYmexeig.m



%%% To generate the essential part of the first-order difference of d
function Omega12 = omega_mat(lambda)
% We compute omega only for 1<=|idx|<=n-1
n       = length(lambda);
idx.idp = find(lambda>0);
idx.idm = setdiff([1:n],idx.idp);
r       = length(idx.idp);

if ~isempty(idx.idp)
    if (r == n)
        Omega12 = ones(n,n);
    else
        s  = n-r;
        dp = lambda(1:r);
        dn = lambda(r+1:n);
        
        Omega12 = (dp*ones(1,s))./(abs(dp)*ones(1,s) + ones(r,1)*abs(dn'));
        %Omega12 = max(1e-15,Omega12);
        %Omega = [ones(r) Omega12; Omega12' zeros(s)];
    end
else
    Omega12 = [];
end
return
%%% End of omega_mat.m 



%%% To generate F(y) 
function [f,Fy] = gradient(y,I,J,lambda,WP,b0)
n  = length(WP);
k  = length(y);
f  = 0.0;
Fy = zeros(k,1);

const_sparse = 2; 

I1 = find(lambda>0);
r  = length(I1);
if (r>0)       
%    if (r<n/2)
        lambda1 = lambda(I1);
        f = lambda1'*lambda1;

        lambda1 = lambda1.^0.5;
        WP1 = WP(:,I1);
        if r>1
            WP1 = WP1*sparse(diag(lambda1));
        else
            WP1 = lambda1*WP1;
        end
        WP1T = WP1';

        if (k<=const_sparse*n) %% sparse form
            i=1;
            while (i<=k)
                Fy(i) = WP1(I(i),:)*WP1T(:,J(i));
                i=i+1;
            end
        else %% dense form
            WP = WP1*WP1';
            i=1;
            while (i<=k)
                Fy(i) = WP(I(i),J(i));
                i=i+1;
            end
        end
%     else  % n/2<=r<=n
%         lambda2 = -lambda(r+1:n);
%         f = lambda'*lambda - lambda2'*lambda2;
% 
%         lambda2 = lambda2.^0.5;
%         WP2 = WP(:, r+1:n);
%         WP2 = WP2*sparse(diag(lambda2));
%         WP2T = WP2';
% 
%         if (k<const_sparse*n) %% sparse form
%             i=1;
%             while (i<=k)
%                 Fy(i) = X(I(i),J(i)) + WP2(I(i),:)*WP2T(:,J(i));
%                 i=i+1;
%             end
%         else %% dense form
%             WP = WP2*WP2T;
%             i=1;
%             while (i<=k)
%                 Fy(i) = X(I(i),J(i)) + WP(I(i),J(i));
%                 i=i+1;
%             end
%         end
%     end
end
 f = 0.5*f - b0'*y;
return
%%% End of gradient.m 


 %%% To generate the Jacobian product with x: F'(y)(x)
function Ax = Jacobian_matrix(x,I,J,Omega12,WP,diag_index,W_inv,w)
n      = length(WP);
k      = length(x);
[r,s]  = size(Omega12);

const_sparse = 2;
Ax = zeros(k,1);

if (r==0)
    Ax = 1.0e-10*x;
elseif (r==n)
    Z = zeros(n,n);
    for i = 1:k
        Z(I(i),J(i)) = x(i);
    end
    Z = 0.5*(Z + Z');

    if diag_index == 1 %W is diagonal
        for i=1:n
            Z(i,:) = Z(i,:)/w(i);
        end
        for j=1:n
            Z(:,j) =  Z(:,j)/w(j);
        end
        i = 1;
        while (i<=k)
            Ax(i) = Z(I(i),J(i));
            Ax(i) = Ax(i) + 1.0e-10*x(i);
            i = i+1;
        end
    else  %%% diag_index ==0
        if ( k<=const_sparse*n )
            Z = W_inv*sparse(Z);
            i=1;
            while (i<=k)
                Ax(i) = Z(I(i),:) * W_inv(:,J(i));
                Ax(i) = Ax(i) + 1.0e-10*x(i);    %add a small perturbation
                i=i+1;
            end
        else
            H = W_inv*Z*W_inv;
            i = 1;
            while (i<=k)
                Ax(i) = H(I(i),J(i));
                Ax(i) = Ax(i) + 1.0e-10*x(i);
                i = i+1;
            end
        end
    end

else % 0<r<n
    P1 = WP(:,1:r);
    P2 = WP(:,r+1:n);

    Z = zeros(n,n);
    for i = 1:k
        Z(I(i),J(i)) = x(i);
    end
    Z = 0.5*(Z + Z');

    if (k<=const_sparse*n)
        % sparse form
        if (r<n/2)
            %H = (Omega.*(WP'*sparse(Z)*WP))*WP';
            H1 = P1'*sparse(Z);
            Omega12 = Omega12.*(H1*P2);
            H = [(H1*P1)*P1' + Omega12*P2'; Omega12'*P1'];

            i=1;
            while (i<=k)
                Ax(i) = WP(I(i),:)*H(:,J(i));
                Ax(i) = Ax(i) + 1.0e-10*x(i);    %add a small perturbation
                i=i+1;
            end
        else % if r>=n/2, use a complementary formula.
            %H = ((E-Omega).*(WP'*sparse(Z)*WP))*WP';
            H2 = P2'*sparse(Z);
            Omega12 = ones(r,s)- Omega12;
            Omega12 = Omega12.*((H2*P1)');
            H = [Omega12*P2'; Omega12'*P1' + (H2*P2)*P2'];
            if diag_index == 1
                for i=1:n
                    Z(i,:) = Z(i,:)/w(i);
                end
                i=1;
                while (i<=k)
                    Ax(i) = Z(I(i),J(i))/w(J(i)) - WP(I(i),:)*H(:,J(i));
                    Ax(i) = Ax(i) + 1.0e-10*x(i);
                    i=i+1;
                end                                
            else
                Z = W_inv * sparse(Z);
                i=1;
                while (i<=k)
                    Ax(i) = Z(I(i),:)*W_inv(:,J(i)) - WP(I(i),:)*H(:,J(i));
                    Ax(i) = Ax(i) + 1.0e-10*x(i);
                    i=i+1;
                end
            end
            
        end

    else %dense form
        %Z = full(Z); to use the full form
        % dense form
        if (r<3*n/4)
            %H = WP*(Omega.*(WP'*Z*WP))*WP';
            H1 = P1'*Z;
            Omega12 = Omega12.*(H1*P2);
            H = P1*((H1*P1)*P1'+ 2.0*Omega12*P2');
            H = (H + H')/2;

            i = 1;
            while (i<=k)
                Ax(i) = H(I(i),J(i));
                Ax(i) = Ax(i) + 1.0e-10*x(i);
                i = i+1;
            end
        else % if r>=3*n/4, use a complementary formula.
            %H = -WP*( (E-Omega).*(WP'*Z*P) )*WP';
            H2 = P2'*Z;
            Omega12 = ones(r,s) - Omega12;
            Omega12 = Omega12.*(H2*P1)';
            H = P2*( 2.0*(Omega12'*P1') + (H2*P2)*P2');
            H = (H + H')/2;

            if diag_index ==1
                for i=1:n
                    Z(i,:) = Z(i,:)/w(i);
                end
                for j=1:n
                    Z(:,j) =  Z(:,j)/w(j);
                end
            else
                Z = W_inv *Z* W_inv;
            end
            H = Z - H;
            
            i = 1;
            while (i<=k)    %%% AA^* is not the identity matrix
                Ax(i) = H(I(i),J(i));
                Ax(i) = Ax(i) + 1.0e-10*x(i);
                i = i+1;
            end
        end
    end
    
end
return
%%% End of Jacobian_matrix.m  





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% PCG method %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This is exactly the algorithm given by Hestenes and Stiefel (1952)
%% An iterative method to solve A(x) =b  
%% The symmetric positive definite matrix M is a preconditioner for A.
%% See Pages 527 and 534 of Golub and va Loan (1996)
function [p,flag,relres,iterk] = pre_cg(b,I,J,tol,maxit,Omega12,WP,c,diag_index,W_inv,w_vec)
n = length(WP);
k1 = length(b);

% initial x0=0
r = b;             
n2b = norm(b);        % norm of b
tolb = max(tol,min(0.1,n2b))*n2b;       % relative tolerance
if n2b > 1.0e2
    maxit = min(1,maxit);
end

flag = 1;
iterk = 0;
relres = 1000;   % to give a big value on relres

p = zeros(k1,1);

z = r./c;        % z = M\r; here M =diag(c); if M is not the identity matrix 
rz1 = r'*z; 
rz2 = 1; 
d = z;

for k = 1:maxit
   if k > 1
       beta = rz1/rz2;
       d = z + beta*d;
   end  
   
   w = Jacobian_matrix(d,I,J,Omega12,WP,diag_index,W_inv,w_vec);   % w = A(d)
   if k1>n
       w = w + 1.0e-2*min(1.0, 0.1*n2b)*d;     % perturb it to avoid numerical singularity
   end
   denom = d'*w;
   iterk = k;
   relres = norm(r)/n2b;                 % relative residue = norm(r)/norm(b)
   if denom <= 0 
       % sssss = 0
       p = d/norm(d);       % d is not a descent direction
       break 
   else
       alpha = rz1/denom;
       p = p + alpha*d;
       r = r - alpha*w;
   end
   z = r./c; 
   if norm(r) <= tolb 
       iterk = k;
       relres = norm(r)/n2b;   % relative residue = norm(r)/norm(b)
       flag =0;
       break
   end
   rz2 = rz1;
   rz1 = r'*z;
end
return
%%% End of pre_cg.m 



%%% To generate the (approximate) diagonal preconditioner
function c = precond_matrix(I,J,Omega12,WP) 
n     = length(WP);
k     = length(I);
[r,s] = size(Omega12);
 
c = ones(k,1);

H = WP';
H = H.*H;
const_prec = 1;
if (r<n)
    if (r>0)        
        if (k<=const_prec*n)     % compute the exact diagonal preconditioner
            
            Ind = find(I~=J);
            k1  = length(Ind);
           if (k1>0)
                H1 = zeros(n,k1);
                for i=1:k1
                    H1(:,i) = WP(I(Ind(i)),:)'.*WP(J(Ind(i)),:)';
                end
            end
            
            if (r<n/2)                  
                H12  = H(1:r,:)'*Omega12;
                if(k1>0)
                    H12_1 = H1(1:r,:)'*Omega12;
                end
                
                d = ones(r,1);
                
                j=0;
                for i=1:k                   
                    if (I(i)==J(i))
                        c(i) = sum(H(1:r,I(i)))*(d'*H(1:r,J(i)));
                        c(i) = c(i) + 2.0*(H12(I(i),:)*H(r+1:n,J(i)));
                    else 
                        j=j+1;
                        c(i) = sum(H(1:r,I(i)))*(d'*H(1:r,J(i)));
                        c(i) = c(i) + 2.0*(H12(I(i),:)*H(r+1:n,J(i)));
                        c(i) = c(i) + sum(H1(1:r,j))*(d'*H1(1:r,j));
                        c(i) = c(i) + 2.0*(H12_1(j,:)*H1(r+1:n,j));
                        c(i) = 0.5*c(i);
                    end
                    if c(i) < 1.0e-8
                        c(i) = 1.0e-8;
                    end                      
                end   
                                
            else  % if r>=n/2, use a complementary formula
                Omega12 = ones(r,s)-Omega12;
                H12  = Omega12*H(r+1:n,:);
                if(k1>0)
                    H12_1 = Omega12*H1(r+1:n,:);
                end
                
                d =  ones(s,1);
                dd = ones(n,1);

                j=0;
                for i=1:k
                    if (I(i)==J(i))
                        c(i) = sum(H(r+1:n,I(i)))*(d'*H(r+1:n,J(i)));
                        c(i) = c(i) + 2.0*(H(1:r,I(i))'*H12(:,J(i)));
                        alpha = sum(H(:,I(i)));
                        c(i) = alpha*(H(:,J(i))'*dd) - c(i);
                    else
                        j=j+1;
                        c(i) = sum(H(r+1:n,I(i)))*(d'*H(r+1:n,J(i)));
                        c(i) = c(i) + 2.0*(H(1:r,I(i))'*H12(:,J(i)));
                        alpha = sum(H(:,I(i)));
                        c(i) = alpha*(H(:,J(i))'*dd) - c(i);

                        tmp = sum(H1(r+1:n,j))*(d'*H1(r+1:n,j));
                        tmp = tmp + 2.0*(H1(1:r,j)'*H12_1(:,j));
                        alpha = sum(H1(:,j));
                        tmp = alpha*(H1(:,j)'*dd) - tmp;
                        
                        c(i) = (tmp + c(i))/2;
                    end                    
                    if c(i) < 1.0e-8
                        c(i) = 1.0e-8;
                    end
                end                
            end
                                   
            
        else  % approximate the diagonal preconditioner
            HH1 = H(1:r,:);
            HH2 = H(r+1:n,:);

            if (r<n/2)
                H0 = HH1'*(Omega12*HH2);
                tmp = sum(HH1);
                H0 = H0 + H0'+ tmp'*tmp;
            else
                Omega12 = ones(r,s) - Omega12;
                H0 = HH2'*((Omega12)'*HH1);
                tmp  = sum(HH2);
                H0 = H0 + H0' + tmp'*tmp;
                tmp = sum(H);
                H0 = tmp'*tmp - H0;
            end

            i=1;
            while (i<=k)
                if (I(i)==J(i))
                    c(i) = H0(I(i),J(i));
                else
                    c(i) = 0.5*H0(I(i),J(i));
                end
                if  c(i) < 1.0e-8
                    c(i) = 1.0e-8;
                end
                i = i+1;
            end
        end        
    end  %End of second if
    
else % if r=n
    tmp = sum(H);
    H0  = tmp'*tmp;
    
    i=1;
    while (i<=k)
        if (I(i)==J(i))
            c(i) = H0(I(i),J(i));
        else
            c(i) = 0.5*H0(I(i),J(i));
        end
        if (c(i)<1.0e-8)
            c(i) = 1.0e-8;
        end
        i = i+1;
    end
end  %End of the first if
return
%%% End of precond_matrix.m 








 








