clear all




n = 300;
%%
%%-----------------------------------------
%%% Generate matrix G
%%-----------------------------------------
%%
LongCorr = 0.5; 
beta     = -0.05;
G = zeros(n,n);
for i = 1:n
    for j = 1:i        
        G(i,j) = LongCorr + (1-LongCorr)*exp(beta*abs(i-j));        
    end
end
G = G + G' - diag(diag(G));


%%% The rank constraint
r_rank = 50;




%%
%%-----------------------------------------
%%% The linear constraints
%%-----------------------------------------
%%                              
lh = 0;    % number of fixed off diagonal elements in each row
lh = min(lh,n-1);
%% I_e,J_e
%%%% for fixed  diagonal entries
I_d = [1:1:n]';
J_d = I_d;
%%%% for fixed off-diagonal entries
I_h = [];
J_h = [];
for i = 2:lh
    for j=1:i-1
    I_h = [I_h; i];
    J_h = [J_h; j];
    end
end
for i = lh+1:n
    r = rand(i-1,1);
    [r,ind] = sort(r);
    I_h = [I_h; i*ones(lh,1)];
    J_h = [J_h; i-ind(1:lh)];
end
k_h = length(I_h);
I_e = [I_d;I_h];
J_e = [J_d;J_h];
k_e = length(I_e);


%% to generate the bound e,l & u
%%%%%%% e
rhs = ones(n,1);  % diagonal elements
alpha0 =  1;
rhs = alpha0*rhs + (1-alpha0)*rand(n,1);
h = zeros(k_h,1);
e = [rhs;h];
%%%%%%% l



%%
%%-----------------------------------------
%%% Generate weight matrix H
%%-----------------------------------------
%%
H0 = sprand(n,n,0.5);
H0 = triu(H0) + triu(H0,1)'; % W0 is likely to have small numbers
H0 = (H0 + H0')/2;
H0 = 0.01*ones(n,n) + 99.99*H0; %%% H0 is in [0.01, 100]
H1 = rand(n,n);
H1 = triu(H1) + triu(H1,1)'; % W1 is likely to have small numbers
H1 = (H1 + H1')/2;
H  = 0.1*ones(n,n) + 9.9*H1; %%% H is in [.1,10]
%%%%%%%%%%%%%%%%%%%%% Assign weights H0  on partial elemens
s = sprand(n,1,min(10/n,1));
I = find(s>0);
d = sprand(n,1,min(10/n,1));
J = find(d>0);
if length(I)>0 && length(J)>0
    H(I,J) = H0(I,J);
    H(J,I) = H0(J,I);
end
H = (H + H')/2;
%%% scale H
Hscale = sum(sum(H))/n^2;
H      = H/Hscale;
 
Hmin = min(min(H))
Hmax = max(max(H))






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% for i = 1:k_h
%     G(I_h(i),J_h(i)) = 0;
%     G(J_h(i),I_h(i)) = 0;
% end
Ind = find(h==0);
for i = 1:length(Ind)
    G( I_h(Ind(i)),J_h(Ind(i)) ) = 0;
    G( J_h(Ind(i)),I_h(Ind(i)) ) = 0;
end 
G = G - diag(diag(G)) + eye(n);  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 ConstrA.e = e; ConstrA.Ie = I_e; ConstrA.Je = J_e;

 
 
 

 
 

fprintf('\n---------- Call PenCorr_lub_lub_HnormMajorDiag.m ----------------\n')
OPTIONS.tau    = 0;
OPTIONS.tolrel = 1.0e-5;
[X,INFOS] = PenCorr_HnormMajorDiag(G,H,ConstrA,r_rank,OPTIONS);





 




