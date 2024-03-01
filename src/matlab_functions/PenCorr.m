function [X,INFOS] = PenCorr(G,ConstrA,Rank,OPTIONS)
%%%%%%%%%%%%% This code is designed to solve %%%%%%%%%%%%%%%%%%%%%
%%       min   0.5*<X-G, X-G>
%%             X_ij  =   e_ij     (i,j) in (I_e,J_e)
%%           rank(X) <= Rank
%%              X    >= tau*I       X is SDP (tau>=0 and may be zero)
%%%
%%%
%   Parameters:
%   Input
%   G           the given symmetric matrix
%   ConstrA:    the equality and inequality constraints
%   Rank:       the rank constraint of X
%   OPTIONS:    parameters in the OPTIONS structure
%
%   Output
%   X         the optimal primal solution
%   INFOS     the optimal dual solution to equality constraints
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Last modified on March 31, 2010.



%%
%%-----------------------------------------
%%% get constraints infos from constrA
%%-----------------------------------------
%%
e   = ConstrA.e;  I_e = ConstrA.Ie; J_e = ConstrA.Je;
k_e = length(e);  n = length(G);

tolrel = 1.0e-5;
%%
%%-----------------------------------------
%% get parameters from the OPTIONS structure.
%%-----------------------------------------
%%
if exist('OPTIONS')
    if isfield(OPTIONS,'tau');                tau               = OPTIONS.tau; end
    if isfield(OPTIONS,'tolrel');             tolrel            = OPTIONS.tolrel; end
    if isfield(OPTIONS,'tolrank');            tolrank           = OPTIONS.tolrank; end
    if isfield(OPTIONS,'tolsub');             tolsub            = OPTIONS.tolsub; end
    if isfield(OPTIONS,'tolPCA');             tolPCA            = OPTIONS.tolPCA; end
    if isfield(OPTIONS,'tolinfeas');          tolinfeas         = OPTIONS.tolinfeas; end
    if isfield(OPTIONS,'tolsub_rank');        tolsub_rank       = OPTIONS.tolsub_rank; end
    if isfield(OPTIONS,'maxit');              maxit             = OPTIONS.maxit; end
    if isfield(OPTIONS,'maxitsub');           maxitsub          = OPTIONS.maxitsub; end
    if isfield(OPTIONS,'use_CorNewtonPCA');   use_CorNewtonPCA  = OPTIONS.use_CorNewtonPCA ; end
    if isfield(OPTIONS,'use_InitialPoint');   use_InitialPoint  = OPTIONS.use_InitialPoint ; end
end
tau         = 0;
innerTolrel = tolrel;
tolsub      = max(innerTolrel, tolrel);
tolPCA      = max(innerTolrel, tolrel);
tolinfeas   = max(innerTolrel, tolrel);
tolsub_rank = tolsub;
tolrank     = 1.0e-8;  %%no need to change
maxit    = 500;
maxitsub = 100;
%use_CorNewtonPCA = 0;  %% set 0 always; set 1 to use CorNewtonPCA as the initial point
use_InitialPoint = 1;  %% =1 initial point from an alternating projection method
finalPCA         = 0;
residue_cutoff = 10;
if tolrel <= 1.0e-4;
   residue_cutoff = 100;
end

t0 = clock;

%%% reset inpust pars
G = G - tau*speye(n);
G = (G + G')/2;
Ind    = find(I_e==J_e);  % reset the diagonal part e
e(Ind) = e(Ind) - tau;
e_diag = e(Ind);
%%% constant pars
const.disp1 = 10;
const.disp2 = 10;
const.rank_hist    = 5;  const.rank_hist    = max(2, const.rank_hist);
const.rankErr_hist = 3;  const.rankErr_hist = max(2, const.rankErr_hist);
const.funcVal_hist = 2;  const.funcVal_hist = max(2, const.funcVal_hist);
const.residue_hist = const.funcVal_hist;  const.residue_hist = max(2, const.residue_hist);
rank_hist    = zeros(const.rank_hist,1);
rankErr_hist = zeros(const.rankErr_hist,1);
funcVal_hist = zeros(const.funcVal_hist,1);
residue_hist = zeros(const.residue_hist,1);
progress_rank1    = 1;
progress_rank2    = 1;
%progress_residue = 1.0e-4
progress_relErr   = 1.0e-5;
progress_rankErr  = 1.0e-3;
%%% penalty pars
c0_min = 1.0;
c0_max = 1e2;  %1e2;
alpha_min  = 1.2; %1.4
alpha_max  = 4.0;
c_max  = 1.0e8;
%%%
Totalcall_CN   = 0;
Totaliter_CN   = 0;
Totalnumb_CG   = 0;
Totalnumb_eigendecom  = 0;

fprintf('\n ******************************************')
fprintf('\n  The problem information: \n')
fprintf(' Dimension of SDP constr.  = %3.0f \n',n)
fprintf(' Fixed upper-diag entries  = %3.0f \n',k_e)
fprintf(' The required rank of X   <= %3.0f \n',Rank)
%%
%%---------------------------------------------------------------
%%% CorNewton3Mex preprocessing
%%---------------------------------------------------------------
%%
fprintf('\n ^^^^^^^^ Preprocessing by CorNewton3Mex ^^^^^^^^ ')
y = zeros(k_e,1);
for i=1:k_e
    y(i) = e(i) - G(I_e(i),J_e(i));
end
opts.disp = 0;
[X,y,info]   = CorMat3Mex(G,e,I_e,J_e,opts,y);
P      = info.P;
lambda = info.lam;
rank_X = info.rank;
Totalcall_CN = Totalcall_CN + 1;
Totaliter_CN = Totaliter_CN + info.numIter;
Totalnumb_CG = Totalnumb_CG + info.numPcg;
Totalnumb_eigendecom = Totalnumb_eigendecom + info.numEig;
residue_CorNewton = sum(sum((X-G).*(X-G)));
residue_CorNewton = residue_CorNewton^0.5;
rankErr_CorNewton = abs( sum(lambda) - sum(lambda(1:Rank)) );
if  ( rankErr_CorNewton <= tolrank )
    fprintf('\n The rankErr_CorNewton is already small enough!')
    fprintf('\n The rankErr_CorNewton  = %5.4e', rankErr_CorNewton)
    fprintf('\n Residue_CorNewton      = %9.8e', residue_CorNewton)
    time_used = etime(clock,t0);
    fprintf('\n Total computing time   = %.1f(secs) \n', time_used);
    if Rank < n
    fprintf('\n lambda(r) - lambda(r+1) === %d \n',lambda(Rank)-lambda(Rank+1));
    end
    INFOS.iter    = 0;
    INFOS.callCN  = Totalcall_CN;
    INFOS.itCN    = Totaliter_CN;
    INFOS.itCG    = Totalnumb_CG;
    INFOS.numEig  = Totalnumb_eigendecom;
    INFOS.rank    = rank_X;
    INFOS.rankErr = rankErr_CorNewton;
    INFOS.residue = residue_CorNewton;
    INFOS.time    = time_used;
    return;
end
fprintf('\n The rank of NCM        = %2.0d', rank_X)
fprintf('\n The rankErr_CorNewton  = %5.4e', rankErr_CorNewton)
fprintf('\n Residue_CorNewton      = %9.8e', residue_CorNewton)
residue_1  = residue_CorNewton;

%%
%%----------------------------------------------------------------------
%%% check how good is CorNewton_PCA
%%----------------------------------------------------------------------
%%
X1 = mPCA(P,lambda,Rank,e_diag);
residue_CorNewtonPCA = sum(sum((X1-G).*(X1-G)));
residue_CorNewtonPCA = residue_CorNewtonPCA^0.5;
residue_error = abs( residue_CorNewtonPCA - residue_CorNewton );
fprintf('\n Residue_CorNewtonPCA  = %9.8e',residue_CorNewtonPCA)
infeas = zeros(k_e,1);
for i=1:k_e
    infeas(i) = e(i) - X1(I_e(i),J_e(i));
end
NormInf_CorNewtonPCA = norm(infeas);
if ( residue_error/max(residue_cutoff, residue_CorNewtonPCA) <= tolPCA && NormInf_CorNewtonPCA <= tolinfeas)
    fprintf('\n CorNewton_PCA is good enough!')
    time_used = etime(clock,t0);
    fprintf('\n Total computing time   = %.1f', time_used);
    INFOS.iter    = 0;
    INFOS.callCN  = Totalcall_CN;
    INFOS.itCN    = Totaliter_CN;
    INFOS.itCG    = Totalnumb_CG;
    INFOS.numEig  = Totalnumb_eigendecom;
    INFOS.rank    = rank_X;
    INFOS.rankErr = 0;
    INFOS.residue = residue_CorNewtonPCA;
    INFOS.time    = time_used;
    return;
end


fprintf('\n\n ^^^^^^^^ Initial Guess ^^^^^^^^ ')
if use_InitialPoint %% use alternating projection method to generate initial point
    fprintf('\n use_InitialPoint!')
    opt_disp = 1;
    [X,P,lambda,rank_X,rankErr,normInf,infoNum] = IntPoint(G,e,I_e,J_e,Rank,X,P,lambda,opt_disp);

    %[X,P,lambda,rank_X,rankErr,infoNum] = InitialPoint(G,e,I_e,J_e,Rank,rankErr_CorNewton,X1);
    Totalcall_CN  = Totalcall_CN + infoNum.callCN;
    Totaliter_CN  = Totaliter_CN + infoNum.iterCN;
    Totalnumb_CG  = Totalnumb_CG + infoNum.CG;
    Totalnumb_eigendecom = Totalnumb_eigendecom + infoNum.eigendecom;

    residue_int = sum(sum((X-G).*(X-G)));
    residue_int = residue_int^0.5;
    residue_1   = residue_int;
else %% use CorNewton_PCA as initial point
    X = X1;
    [P,lambda] = MYmexeig(X);
    Totalnumb_eigendecom = Totalnumb_eigendecom + 1;
    residue_int = residue_CorNewtonPCA;
    residue_1   = residue_int;
end

%%% initialize U
P1 = P(:,1:Rank);
U  = P1*P1';
rankErr = abs( sum(lambda) - sum(lambda(1:Rank)) );


%%
%%---------------------------------------------------------------
%%% initial penalty parameter c
%%---------------------------------------------------------------
%%
if use_InitialPoint
    c0 = 0.50*(residue_int^2 - residue_CorNewton^2);
    c0 = 0.25*c0/max(1.0, rankErr_CorNewton - rankErr);
else  %use_CorNewtonPCA
    c0 = 0.50*(residue_CorNewtonPCA^2 - residue_CorNewton^2);
    c0 = 0.25*c0/max(1.0, rankErr_CorNewton);
end
%residue_error = abs(residue_int - residue_CorNewton);

%  if  residue_error/max(residue_cutoff, residue_int)< 0.95
%      c0 = c0_max;
%  end
if Rank <= 1
    c0 =  c0_max ;
end

if tolrel >= 1.0e-1;  %% less acurate, larger c
    c0 = 4*c0;
elseif tolrel >= 1.0e-2;  %% less acurate, larger c
    c0 = 2*c0;
% elseif tolrel >= 1.0e-3;  %% less acurate, larger c
%     c0 = 2*c0;
% elseif tolrel >= 1.0e-4;  %% less acurate, larger c
%     c0 = 2*c0;
end

% if tolrel <= 1.0e-6;  %% more acurate, smaller c
%     c0 = c0/2;
% end
c0 = max(c0, c0_min);
c0 = min(c0, c0_max);
c  = c0;


fprintf('\n\n ************************************** \n')
fprintf( '      The Penalty Method Initiated!!!      ')
fprintf('\n ************************************** \n')
fprintf('The initial rank     = %d \n', rank_X);
fprintf('The initial rankErr  = %d \n', rankErr);
fprintf('The initial ||X0-G|| = %9.8e \n', residue_1);

relErr_0    = 1.0e6;
break_level = 0;

k1       = 1;
sum_iter = 0;
while ( k1 <= maxit )

    subtotaliter_CN = 0;
    subtotalnumb_CG  = 0;
    subtotalnumb_eigendecom = 0;

    fc = 0.5*residue_1^2;
    fc = fc + c*rankErr;

    tt = etime(clock,t0);
    fprintf('\n ================')
    fprintf(' The %2.0dth level of penalty par. %3.2e',k1,c)
    fprintf('  =========================')
    fprintf('\n ........Calling CorNewton3Mex')
    fprintf('\n CallNo.  NumIt  NumCGs    RankX    RankErr      Sqrt(2*FunVal)     Time')
    fprintf('\n %2.0fth      %s       %s       %3.0d     %3.2e     %9.8e    %.1f',...
        0,'-','-',rank_X,rankErr,sqrt(2)*fc^0.5,tt)

    G0 = G + c*(U - eye(n));
    %const_primal = c*( sum(sum(U.*X)) - sum(lambda(1:Rank)) );
    %const_primal = const_primal + 0.5*(sum(sum(G.*G)) - sum(sum(G0.*G0)));

    if ( k1==1 || rankErr > tolrank )
        y  = zeros(k_e,1);
        for i = 1:k_e
            y(i) = e(i) - G0(I_e(i),J_e(i));
        end
    end

    for itersub = 1:maxitsub

        [X,y,info] = CorMat3Mex(G0,e,I_e,J_e,opts,y);
        P          = info.P;
        lambda     = info.lam;
        rank_X     = info.rank;
        major_dualVal = info.dualVal;
        rankErr    = abs( sum(lambda) - sum(lambda(1:Rank)) );
        Totalcall_CN             = Totalcall_CN + 1;
        subtotalnumb_CG          = subtotalnumb_CG + info.numPcg;
        subtotaliter_CN          = subtotaliter_CN + info.numIter;
        subtotalnumb_eigendecom  = subtotalnumb_eigendecom + info.numEig;
        fc         = sum(sum((X-G).*(X-G)));
        residue_1  = fc^0.5;
        fc         = 0.5*fc + c*rankErr;

        if ( itersub <= const.disp1 || mod(itersub,const.disp2) == 0 )
            tt = etime(clock,t0);
            fprintf('\n %2.0dth     %2.0f      %2.0f       %3.0d     %3.2e     %9.8e    %.1f',...
                itersub, info.numIter, info.numPcg, rank_X, rankErr, sqrt(2)*fc^0.5, tt)

%             fprintf('\n The primal value of the penalized problem = %5.4e',fc - const_primal)
%             fprintf('\n The dual value of the majorized problem   = %5.4e',major_dualVal)  %%% added on Jan 29, 2010.

            dispsub = 1;
        else
            dispsub = 0;
        end

        %%% rank history
        if  itersub <= const.rank_hist
            rank_hist(itersub) = rank_X;
        else
            for j = 1:const.rank_hist-1
                rank_hist(j) = rank_hist(j+1);
            end
            rank_hist(const.rank_hist) = rank_X;
        end
        %%% function value history
        if  itersub <= const.funcVal_hist
            funcVal_hist(itersub) = fc^0.5;
        else
            for j = 1:const.funcVal_hist-1
                funcVal_hist(j) = funcVal_hist(j+1);
            end
            funcVal_hist(const.funcVal_hist) = fc^0.5;
        end
        %%% residue history
        if sum_iter + itersub <= const.residue_hist
            residue_hist(sum_iter + itersub) = residue_1;
        else
            for j = 1:const.residue_hist-1
                residue_hist(j) = residue_hist(j+1);
            end
            residue_hist(const.residue_hist) = residue_1;
        end

        if rankErr <= tolrank
            tolsub_check = tolsub_rank;
        else
            tolsub_check = tolsub*max(10, min(100,rank_X/Rank));
        end
        if itersub >= const.funcVal_hist
            relErr_sub  = abs(funcVal_hist(1) - funcVal_hist(const.funcVal_hist));
            relErr_sub  = relErr_sub/max(residue_cutoff, max(funcVal_hist(1), funcVal_hist(const.funcVal_hist)));
        end
        if ( itersub >= const.funcVal_hist &&  relErr_sub <= tolsub_check )
            if dispsub == 0
                tt = etime(clock,t0);
                fprintf('\n %2.0dth     %2.0f      %2.0f       %3.0d       %3.2e   %9.8e    %.1f',...
                    itersub, info.numIter, info.numPcg, rank_X, rankErr, sqrt(2)*fc^0.5, tt)

%                fprintf('\n The primal value of the majorized problem = %5.4e',fc - const_primal)
%                fprintf('\n The dual value of the majorized problem   = %5.4e',major_dualVal)  %%% added on Jan 29, 2010.

            end
            break;
        elseif ( itersub >= const.rank_hist && abs( rank_hist(1) - rank_hist(const.rank_hist) ) <= progress_rank1...
                && rank_X - Rank >= progress_rank2 )
            %fprintf('\n Warning: The rank does not decrease in this level!')
            if dispsub == 0
                tt = etime(clock,t0);
                fprintf('\n %2.0dth     %2.0f      %2.0f       %3.0d       %3.2e   %9.8e    %.1f',...
                    itersub, info.numIter, info.numPcg, rank_X, rankErr, sqrt(2)*fc^0.5, tt)

%                 fprintf('\n The primal value of the majorized problem = %5.4e',fc - const_primal)
%                 fprintf('\n The dual value of the majorized problem   = %5.4e',major_dualVal)  %%% added on Jan 29, 2010.

            end
            break;
        end

        % update U, G0 and fc0
        P1  = P(:, 1:Rank);
        U   = P1*P1';
        G0  = G + c*(U - eye(n));
        const_primal = c*( sum(sum(U.*X)) - sum(lambda(1:Rank)) );
        const_primal = const_primal + 0.5*(sum(sum(G.*G)) - sum(sum(G0.*G0)));

    end   %end of subproblem

    sum_iter      = sum_iter + itersub;
    Totalnumb_CG  = Totalnumb_CG + subtotalnumb_CG;
    Totaliter_CN  = Totaliter_CN + subtotaliter_CN;
    Totalnumb_eigendecom = Totalnumb_eigendecom + subtotalnumb_eigendecom;
    fprintf('\n SubTotal %2.0f      %2.0f      %2.0f(eigendecom)',...
        subtotaliter_CN,subtotalnumb_CG,subtotalnumb_eigendecom);

    if sum_iter >= const.residue_hist
        relErr = abs(residue_hist(1) - residue_hist(const.residue_hist));
        relErr = relErr/max(residue_cutoff, max(residue_hist(1), residue_hist(const.residue_hist)));
    else
        relErr = abs(residue_hist(1) - residue_hist(sum_iter));
        relErr = relErr/max(residue_cutoff, max(residue_hist(1), residue_hist(sum_iter)));
    end
    tt = etime(clock,t0);
    [hh,mm,ss] = time(tt);
    fprintf('\n   Iter.   PenPar.   Rank(X)    RankError     relErr     ||X-G||         Time_used ')
    fprintf('\n   %2.0d     %3.2e    %2.0d      %3.2e     %3.2e   %9.8e     %d:%d:%d \n',...
        k1,c,rank_X,rankErr,relErr,residue_1,hh,mm,ss)


    if  k1 <= const.rankErr_hist
        rankErr_hist(k1) = rankErr;
    else
        for j=1:const.rankErr_hist-1
            rankErr_hist(j) = rankErr_hist(j+1);
        end
        rankErr_hist(const.rankErr_hist) = rankErr;
    end
    %%% termination test
    if ( relErr <= tolrel )
        if ( rankErr <= tolrank )
            fprintf('\n The rank constraint is satisfied!')
            break;
        elseif (  k1 >= const.rankErr_hist && abs(rankErr_hist(1) - rankErr_hist(const.rankErr_hist)) <= progress_rankErr )
            fprintf('\n Warning: The rank does not decrease any more! :( ')
            finalPCA = 1;
            break;
        end
    else
        %         if ( abs(residue_0 - residue_1)/max(1,max(residue_0,residue_1)) <= progress_residue )
        %             fprintf('\n Warning: The residue is decreasing slowly, quit! :( ')
        %             if ( rankErr > tolrank )
        %                 finalPCA = 1;
        %             end
        %             break;
        %         end
        if ( abs(relErr_0 - relErr)/max(1,relErr) <= progress_relErr )
            break_level = break_level + 1;
            if break_level == 3
                fprintf('\n Warning: The relErr is consecutively decreasing slowly, quit! :( ')
                if ( rankErr > tolrank )
                    finalPCA = 1;
                end
                break;
            end
        end
    end

    k1       = k1 + 1;
    relErr_0 = relErr;

    %%% update c
    if rank_X <= Rank
       c  = min(c_max, c);
    else
        if rankErr/max(1,Rank) > 1.0e-1
            c  = min(c_max, c*alpha_max);
        else
            c  = min(c_max, c*alpha_min);
        end
    end

end

%%
%%% check if y is the optimal dual Lagrange multiplier
%%
X_tmp      = G + diag(y);
X_tmp      = (X_tmp + X_tmp')/2;
[P0,lambda0] = mexeig(X_tmp);
%P0          = real(P0);
lambda0     = real(lambda0);
if issorted(abs(lambda0))
    lambda0 = lambda0(end:-1:1);
    %P0      = P0(:,end:-1:1);
elseif issorted(abs(lambda0(end:-1:1)))
else
    [lambda01, Inx] = sort(abs(lambda0),'descend');
    lambda0 = lambda0(Inx);
    %P0      = P0(:,Inx);
end
f = sum(lambda0(Rank+1:n).^2);
f = -f + y'*y;
f = 0.5*f;
dual_obj = -f;
fprintf('\n Dual function value    === %9.8e',dual_obj)
fprintf('\n Primal function value  === %9.8e \n', 0.5*residue_1^2)

%%
%%% final PCA correction
%%
if length(e_diag) == k_e && finalPCA
    X = mPCA(P,lambda,Rank,e_diag);
    %[P,lambda] = MYmexeig(X);
    %Totalnumb_eigendecom = Totalnumb_eigendecom + 1;
    rank_X  = Rank;
    rankErr = 0;
    residue_1  = sum(sum((X-G).*(X-G)))^0.5;
end
infeas = zeros(k_e,1);
for i = 1:k_e
    infeas(i) = e(i) - X(I_e(i),J_e(i));
end
NormInf = norm(infeas);

time_used = etime(clock,t0);
INFOS.iter    = k1;
INFOS.callCN  = Totalcall_CN;
INFOS.itCN    = Totaliter_CN;
INFOS.itCG    = Totalnumb_CG;
INFOS.numEig  = Totalnumb_eigendecom;
INFOS.rank    = rank_X;
INFOS.rankErr = rankErr;
INFOS.relErr  = relErr;
INFOS.infeas  = NormInf;
INFOS.residue = residue_1;
INFOS.time    = time_used;
fprintf('\n Final ||X-G||            ===== %9.8e', INFOS.residue);
fprintf('\n Primal function value    ===== %9.8e \n', 0.5*INFOS.residue^2);
%fprintf('\n MajorDual function value ===== %9.8e \n', major_dualVal);
%{

fid = fopen('result_PenCorr.txt','a+');
fprintf(fid,'\n ************ result_PenCorr ******************');
fprintf(fid,'\n  The problem information: \n');
fprintf(fid,' Dimension of SDP constr.  = %3.0f \n',n);
fprintf(fid,' Fixed upper-diag entries  = %3.0f \n',k_e);
fprintf(fid,' The required rank of X   <= %3.0f \n',Rank);
fprintf(fid,' ---------------------------------------------------- \n');
%fprintf(fid,'\n **************** Final Information ******************** \n');
fprintf(fid,' Num of pen level         = %d \n', INFOS.iter);
fprintf(fid,' Num of calling CN        = %d \n', INFOS.callCN);
fprintf(fid,' Total num of iter in CN  = %d \n', INFOS.itCN);
fprintf(fid,' Total num of CG          = %d \n', INFOS.itCG);
fprintf(fid,' Total num of eigendecom  = %d \n', INFOS.numEig);
fprintf(fid,' The rank of  X*          === %d \n', INFOS.rank);
fprintf(fid,' The rank error           === %3.2e \n', INFOS.rankErr);
fprintf(fid,' The rel func error       === %3.2e \n', INFOS.relErr);
fprintf(fid,' The infeasibility of X*  === %9.8e \n', INFOS.infeas);
fprintf(fid,' Final ||X-G||            ===== %9.8e \n', INFOS.residue);
fprintf(fid,' Primal function value    ===== %9.8e \n', 0.5*INFOS.residue^2);
fprintf(fid,' Computing time           ======= %.1f(secs) \n', INFOS.time);
fprintf(fid,' ********************************************************* \n');
fclose(fid);

%}
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
function [P,lambda] = MYmexeig(X)
[P,lambda] = mexeig(X);
P          = real(P);
lambda     = real(lambda);
%rearrange lambda in nonincreasing order
if issorted(lambda)
    lambda = lambda(end:-1:1);
    P      = P(:,end:-1:1);
elseif issorted(lambda(end:-1:1))
    return;
else
    [lambda, Inx] = sort(lambda,'descend');
    P = P(:,Inx);
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




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X  = Projr(P,lambda,r)
n = length(lambda);
X = zeros(n,n);
if r>0
    P1      = P(:,1:r);
    lambda1 = lambda(1:r);
    if r>1
        lambda1 = lambda1.^0.5;
        P1 = P1*sparse(diag(lambda1));
        X  = P1*P1';
    else
        X = lambda1*P1*P1';
    end
end
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% use an alternating projection method to generate an initial point
function [X,P,lambda,rank_X,rankErr,infoNum] = ...
    InitialPoint(G,e,I_e,J_e,Rank,rankErr_CorNewton,X1)
k_e = length(e);

maxit      = 20;
rank_ratio = 0.90;
tolinfeas  = 1.0e-6;
infoNum.callCN      = 0;
infoNum.iterCN      = 0;
infoNum.CG          = 0;
infoNum.eigendecom  = 0;

% if any(I_e-J_e)
%     use_mPCA = 0;
% else
%     use_mPCA = 1;
% end
use_mPCA = 1;
Ind = find(I_e==J_e);
e_diag = e(Ind);

c0    = 1.0e1;
cmax  = 1.0e6;
alpha = 2;
c     = c0;

opts.disp = 0;
for iter = 1:maxit
    if iter == 1
        Y = X1;
    else
        if use_mPCA
            Y = mPCA(P,lambda,Rank,e_diag);
        else
            Y = Projr(P,lambda,Rank);
        end
    end
    infeas = zeros(k_e,1);
    for i=1:k_e
        infeas(i) = e(i) - Y(I_e(i),J_e(i));
    end
    NormInf = norm(infeas);
    if ( NormInf <= tolinfeas )
        X = Y;
        [P,lambda] = MYmexeig(X);
        rank_X     = Rank;
        rankErr    = abs( sum(lambda) - sum(lambda(1:Rank)) );
        infoNum.eigendecom  = infoNum.eigendecom + 1;
        fprintf('\n Alternating terminates at projection onto rank constraint with good feasibility!')
        if iter==1
            fprintf('\n *** This initial point is exactly CorNewtonPCA! ***')
        end
        break;
    end

    G0 = (G + c*Y)/(1+c);
    %%% call CorNewton3Mex
    y = zeros(k_e,1);
    for i = 1:k_e
        y(i) = e(i) - G0(I_e(i),J_e(i));
    end
    [X,y,info]   = CorMat3Mex(G0,e,I_e,J_e,opts,y);
    P      = info.P;
    lambda = info.lam;
    rank_X = info.rank;
    infoNum.callCN      = infoNum.callCN + 1;
    infoNum.iterCN      = infoNum.iterCN + info.numIter;
    infoNum.CG          = infoNum.CG + info.numPcg;
    infoNum.eigendecom  = infoNum.eigendecom + info.numEig;
    rankErr = abs( sum(lambda) - sum(lambda(1:Rank)) );
    if ( rankErr <= rank_ratio*max(1,rankErr_CorNewton) )
        fprintf('\n Alternating terminates at projection onto linear constraints with small rank error!')
        break;
    end

    c = min( alpha*c, cmax );
end
fprintf('\n Num of iteration = %2.0d',iter)
fprintf('   RankErr          = %4.3e',rankErr)
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





