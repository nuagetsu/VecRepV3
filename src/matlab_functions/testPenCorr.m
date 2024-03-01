clear all


n = 500;
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



%% I_e,J_e
%%%% for fixed  diagonal entries
I_e = [1:1:n]';
J_e = I_e;
k_e = length(I_e);


%% to generate the bound e,l & u
%%%%%%% e
e = ones(n,1);  % diagonal elements

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 ConstrA.e = e; ConstrA.Ie = I_e; ConstrA.Je = J_e;





fprintf('\n---------- Call PenCorr.m ----------------\n')
OPTIONS.tau    = 0;
OPTIONS.tolrel = 1.0e-5;
[X,INFOS] = PenCorr(G,ConstrA,r_rank,OPTIONS)







