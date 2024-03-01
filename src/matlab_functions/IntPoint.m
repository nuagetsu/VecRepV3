function [X,P,lambda,rank_X,rankErr,normInf,infoNum] = IntPoint(G,e,I_e,J_e,Rank,X,P,lambda,opt_disp)
%%%%%% Last modified on January 24, 2010.

%%% reset input
G   = (G + G')/2; 
k_e = length(e);
Ind    = find(I_e==J_e);
e_diag = e(Ind);

%%% set parameters
%tolrel  = 1.0e-8;   
tolinf   = 1.0e-6;
%tolrank = 1.0e-8;  %% no need to change
rankErr_stop = 1.0e-1;
maxit    = 5; 
maxitsub = 2;
const.disp1 = 20;
const.disp2 = 10;
infoNum.callCN     = 0;
infoNum.iterCN     = 0;
infoNum.CG         = 0;
infoNum.eigendecom = 0;
%%% penalty par  
rho       = 1.0e2;
rho_step  = 10;
rho_max   = 1.0e8;


t0 = clock;

%%% generate initial Z
%[P,lambda] = MYmexeig(X,0);   %% X is psd
rank_X   = length(find(lambda > 1.0e-8));
rankErr  = abs( sum(lambda) - sum(lambda(1:Rank)) ); 
rankErr0 = rankErr;
Z = mPCA(P,lambda,Rank,e_diag);
%%% test how good is mPCA(X).
infeas = zeros(k_e,1);
for i=1:k_e
    infeas(i) = e(i) - Z(I_e(i),J_e(i));
end
normInf = norm(infeas);
if normInf <= tolinf
    X = Z;
    [P,lambda] = MYmexeig(X,0);
    rank_X     = Rank;
    rankErr    = abs( sum(lambda) - sum(lambda(1:Rank)) );
    residue_X  = sum(sum((X-G).*(X-G)))^0.5;
    infoNum.eigendecom = infoNum.eigendecom + 1;
    fprintf('\n Initial RankErr  = %4.3e', rankErr)
    fprintf('\n Initial Residue  = %4.3e', residue_X)
    time_used = etime(clock, t0);
    fprintf('\n Time used to generate the initial point = %.1f',time_used)
    fprintf('\n *** This initial point is exactly CorNewton_mPCA! ***')
    return
end

%%% initial residue
residue_X = sum(sum((X-G).*(X-G)))^0.5;
residue_Z = sum(sum((Z-G).*(Z-G)))^0.5;
residue_XZ = sum(sum((X-Z).*(X-Z)))^0.5;
%%% initial function value
fc = 0.5*residue_X^2 + 0.5*residue_Z^2;
fc = fc + 0.5*rho*residue_XZ^2;
%fc_old = fc;
 
if opt_disp
fprintf('\n\n **************************************************** \n')
fprintf( '      The Iterative Majorization Method Initiated!!!      ')
fprintf('\n ******************************************************* \n')
end

opts.disp     = 0;        
break_level   = 0;
total_AltProj = 0;
for k1 = 1:maxit
    
    if opt_disp
    tt = etime(clock,t0);
    fprintf('\n ================')
    fprintf(' The %2.0dth level of penalty par. %6.5e',k1,rho)
    fprintf('  =========================')
    fprintf('\n ........Calling CaliMat1Mex_Wnorm')
    fprintf('\n CallNo.  NumIt  NumCGs    RankX    RankErr      Sqrt(2*FunVal)     Time')
    fprintf('\n %2.0fth      %s       %s       %3.0d     %3.2e     %9.8e    %.1f',...
        0,'-','-',rank_X,rankErr,sqrt(2)*fc^0.5,tt)
    end
       
    for itersub = 1:maxitsub
        
        C  = (X - G);
        G0 = G + rho*Z - C;
        G0 = G0/(1+rho);
        z = zeros(k_e,1);
        for i = 1:k_e
            z(i) = e(i) - G0(I_e(i),J_e(i));
        end
        %%% update X: projection onto the linear constraints
%         if itersub==1 
%            [X,z,info] = CorMat3Mex(G0,e,I_e,J_e,opts);
%         else   
            [X,z,info] = CorMat3Mex(G0,e,I_e,J_e,opts,z);             
%        end
        P          = info.P;
        lambda     = info.lam;
        rank_X     = info.rank;
        rankErr    = abs( sum(lambda) - sum(lambda(1:Rank)) ); 
        infoNum.callCN      = infoNum.callCN + 1;
        infoNum.iterCN      = infoNum.iterCN + info.numIter;
        infoNum.CG          = infoNum.CG + info.numPcg;       
        infoNum.eigendecom  = infoNum.eigendecom + info.numEig;
           
        
        residue_X  = sum(sum((X-G).*(X-G)))^0.5;
        residue_XZ = sum(sum((X-Z).*(X-Z)))^0.5;
        fc = 0.5*residue_X^2 + 0.5*residue_Z^2;
        fc = fc + rho*0.5*residue_XZ^2;
        %relErr = abs(fc_old - fc);
        %relErr = relErr/max(1, max(fc_old,fc));
        %fc_old = fc;
        %         if itersub > 1 && relErr <= tolrel
        %             break_level = 1;
        %             if rankErr < tolrank
        %                 break_level = 2;
        %             end
        %             tt = etime(clock,t0);
        %             fprintf('\n %2.0dth     %2.0f      %2.0f       %3.0d     %3.2e   %9.8e    %.1f',...
        %                 itersub, info.numIter, info.numPcg, rank_X, rankErr, sqrt(2)*fc^0.5, tt)
        %             break;
        %         end
        if rankErr <= rankErr_stop*rankErr0
            break_level = 1;
            tt = etime(clock,t0);
            fprintf('\n %2.0dth     %2.0f      %2.0f       %3.0d     %3.2e   %9.8e    %.1f',...
                itersub, info.numIter, info.numPcg, rank_X, rankErr, sqrt(2)*fc^0.5, tt)
            break;
        end
        
        %%% update Z: project onto the rank constraint
        C  = (Z - G);
        G0 = G + rho*X - C;
        G0 = G0/(1+rho);  % may not be psd.
        [P,lambda] = MYmexeig(G0,1);
        Z          = Projr(P,lambda,Rank);
        infoNum.eigendecom  = infoNum.eigendecom + 1;
        
%         residue_Z  = sum(sum((Z-G).*(Z-G)))^0.5;
%         residue_XZ = sum(sum((X-Z).*(X-Z)))^0.5;
%         fc = 0.5*residue_X^2 + 0.5*residue_Z^2;
%         fc = fc + rho*0.5*residue_XZ^2;
        
        infeas = zeros(k_e,1);
        for i=1:k_e
            infeas(i) = e(i) - Z(I_e(i),J_e(i));
        end
        normInf = norm(infeas); 
        if opt_disp
            if ( itersub <= const.disp1 || mod(itersub,const.disp2) == 0 )
                tt = etime(clock,t0);
                fprintf('\n %2.0dth     %2.0f      %2.0f       %3.0d     %3.2e     %9.8e    %.1f',...
                    itersub, info.numIter, info.numPcg, rank_X, rankErr, sqrt(2)*fc^0.5, tt)
                fprintf('normInf = %6.5e', normInf);
            end
        end
                
        if normInf <= tolinf             
            X          = Z;
            [P,lambda] = MYmexeig(X,0);
            infoNum.eigendecom  = infoNum.eigendecom + 1;
            rank_X     = Rank;
            rankErr    = abs( sum(lambda) - sum(lambda(1:Rank)) ); 
            residue_X  = sum(sum((X-G).*(X-G)))^0.5;
            break_level = 2;           
            break;
        end
         
    end   %end of subproblem
    
    if opt_disp
    tt = etime(clock,t0);
    [hh,mm,ss] = time(tt);
    fprintf('\n   Iter.   PenPar.   Rank(X)    RankError   ||X-G||         Time_used ')
    fprintf('\n   %2.0d     %3.2e    %2.0d      %3.2e     %9.8e     %d:%d:%d \n',...
        k1,rho,rank_X,rankErr,residue_X,hh,mm,ss)
    end
    
    total_AltProj = total_AltProj + itersub;
    if break_level == 1 
        fprintf('\n Alternating terminates at projection onto the linear constraints with small rankErr!')
        break;
    elseif  break_level == 2
        fprintf('\n Alternating terminates at projection onto rank constraints with small normInf!')
        fprintf('\n Norm of infeasibility = %4.3e', normInf)
        break;
    else
        %%% update rho
        rho = min(rho_step*rho, rho_max);       
    end
    
end
fprintf('\n Total No. of AltProj = %2.0d', total_AltProj)
fprintf('\n Initial RankErr  = %4.3e', rankErr)
fprintf('\n Initial ||X-G||  = %4.3e', residue_X)
time_used = etime(clock, t0);
fprintf('\n Time used to generate the initial point = %.1f',time_used)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% end of the main program %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%  **************************************
%%  ******** All Sub-routines  ***********
%%  **************************************

%%% To change the format of time 
function [h,m,s] = time(t)
t = round(t); 
h = floor(t/3600);
m = floor(rem(t,3600)/60);
s = rem(rem(t,60),60);
%%% End of time.m



%%% mexeig decomposition
function [P,lambda] = MYmexeig(X,order_abs)
[P,lambda] = mexeig(X);
P          = real(P);
lambda     = real(lambda);
if order_abs == 0  
    if issorted(lambda)
        lambda = lambda(end:-1:1);
        P      = P(:,end:-1:1);
    elseif issorted(lambda(end:-1:1))
        return;
    else
        [lambda, Inx] = sort(lambda,'descend');
        P = P(:,Inx);
    end
elseif order_abs == 1
    if issorted(abs(lambda))
        lambda = lambda(end:-1:1);
        P      = P(:,end:-1:1);
    elseif issorted(abs(lambda(end:-1:1)))
        return;
    else
        [lambda1, Inx] = sort(abs(lambda),'descend');
        %     for i=1:length(Inx_neg)
        %          tmp = find(Inx==Inx_neg(i));
        %          lambda1(tmp) = -lambda1(tmp);
        %     end
        lambda = lambda(Inx);
        P = P(:,Inx);
    end
end
return
%%% End of MYmexeig.m




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = mPCA(P,lambda,Rank,b)
%lambda>=0 and b>=0
n = length(lambda);
if nargin < 4
    b = ones(n,1);
end

if Rank>0
    P1       = P(:, 1:Rank);
    lambda1  = lambda(1:Rank);
    lambda1  = lambda1.^0.5;
    if Rank>1
        P1 = P1*sparse(diag(lambda1));
    else
        P1 = P1*lambda1;
    end
    pert_Mat = rand(n,Rank);
    for i=1:n
        s = norm(P1(i,:));
        if s<1.0e-12  % PCA breakdowns
            P1(i,:) = pert_Mat(i,:);
            s       = norm(P1(i,:));
        end
        P1(i,:) = P1(i,:)/s;
        P1(i,:) = P1(i,:)*sqrt(b(i));
    end
    X = P1*P1';
else
    X = zeros(n,n);
end
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function X  = Projr(P,lambda,r)
% n = length(lambda);
% X = zeros(n,n);
% if r>0
%     P1      = P(:,1:r);
%     lambda1 = lambda(1:r);
%     if r>1
%         lambda1 = lambda1.^0.5;
%         P1 = P1*sparse(diag(lambda1));
%         X  = P1*P1';
%     else
%         X = lambda1*P1*P1';
%     end
% end
% return
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = Projr(P,lambda,r)
n = length(lambda);
X = zeros(n,n);
if r>0
    P1      = P(:,1:r);
    lambda1 = lambda(1:r);
    for i=1:r
        P1(:,i) = lambda1(i)*P1(:,i);
    end
    X = P(:,1:r)*P1';
end
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%









