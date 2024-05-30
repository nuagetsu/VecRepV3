function [X,INFOS] = PenCorr_HnormMajorDiag(G,H,ConstrA,Rank,OPTIONS)
%%%%%%%%%%%%% This code is designed to solve %%%%%%%%%%%%%%%%%%%%%
%%       min  0.5 ||H o (X-G)||^2   where "o" is the Hadamard product symbol
%%             X_ij   = e_ij      (i,j) in (I_e,J_e)
%%           rank(X) <= Rank     
%%              X    >= tau*I      X is SDP (tau>=0 and may be zero)
%   Parameters:
%   Input
%   G:            the given symmetric correlation matrix
%   H:            the weight matrix for G
%   ConstrA:      the equality constraints
%   Rank:         the rank constraint to X
%   OPTIONS: 
%
%   Output
%   X         the optimal primal solution
%   INFOS     the final information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Last modified on June 25, 2010



%%
%%-----------------------------------------
%%% get constraints infos from constrA
%%-----------------------------------------
%%
e   = ConstrA.e; I_e = ConstrA.Ie; J_e = ConstrA.Je;
k_e = length(e); n = length(G);


tau      = 0;
maxit    = 200; 
maxitsub = 100;
tolrel   = 1.0e-5;  % default
tolrank  = 1.0e-8;  % no need to change
tolKKT   = 1.0e-5;
scale_data       = 0;  %% =1 scale H
use_InitialPoint = 1;  %% =1 initial point from an alternating projection method 
%%                                      
%%-----------------------------------------
%% get parameters from the OPTIONS structure. 
%%-----------------------------------------
%%
if exist('OPTIONS')
    if isfield(OPTIONS,'tau');                tau               = OPTIONS.tau; end
    if isfield(OPTIONS,'tolrel');             tolrel            = OPTIONS.tolrel; end
    if isfield(OPTIONS,'tolrank');            tolrank           = OPTIONS.tolrank; end
    if isfield(OPTIONS,'maxit');              maxit             = OPTIONS.maxit; end
    if isfield(OPTIONS,'maxitsub');           maxitsub          = OPTIONS.maxitsub; end
    if isfield(OPTIONS,'scale_data');         scale_data        = OPTIONS.scale_data; end
    if isfield(OPTIONS,'use_InitialPoint');   use_InitialPoint  = OPTIONS.use_InitialPoint ; end
end
innerTolrel = tolrel;
tolsub      = max(innerTolrel, 1.0*tolrel);  
tolsub_rank = tolsub;  
finalPCA = 0;
residue_cutoff = 10; %10
if tolrel <= 1.0e-4;
   residue_cutoff = 100; %100
end
%%% constant pars
const.disp1 = 10;
const.disp2 = 10;
const.rank_hist    = 10; %5;  const.rank_hist    = max(2, const.rank_hist);  
const.rankErr_hist = 10; %3;  const.rankErr_hist = max(2, const.rankErr_hist);  
const.funcVal_hist = 2; %10;4;  const.funcVal_hist = max(2, const.funcVal_hist);
const.residue_hist = const.funcVal_hist;  const.residue_hist = max(2, const.residue_hist); 
runhist.rank    = zeros(const.rank_hist,1);
runhist.rankErr = zeros(const.rankErr_hist,1);
runhist.funcVal = zeros(const.funcVal_hist,1);
runhist.residue = zeros(const.residue_hist,1);
progress_rank1    = 1;
progress_rank2    = 1;
%progress_residue  = 1.0e-4
progress_relErr   = 0.001*tolrel;
progress_rankErr  = 1.0e-3;
Totalcall_CN          = 0;
Totaliter_CN          = 0;
Totalnumb_CG          = 0;
Totalnumb_eigendecom  = 0;

%%% penalty pars
c0_min = 1.0;
c0_max = 1e3;   
alpha_min  = 1.2;  %1.4
alpha_max  = 4;
c_max      = 1.0e8;

t0 = clock;

%%% reset input pars
eye_n = speye(n);
G = G - tau*eye_n;   
G = (G + G')/2;         
Idx    = find(I_e==J_e);  % reset the diagonal part of e
e(Idx) = e(Idx) - tau;
e_diag = e(Idx);
ConstrA.e = e;
%%% scale H
H = (H + H')/2;    % make sure that H is symmetric
if ( scale_data )
    Hscale = sum(sum(H))/n^2;
    H      = H/Hscale;
end
H2H = H.*H;
% H_max  = max(max(H));
% H_max2 = H_max^2;


weightFlag = any(any(H-ones(n,n)));  % =0 if equal weights
if weightFlag == 0
    msg = sprintf('Equal weight case: call PenCorr.m! ');
    fprintf('\n %s ', msg);
    OPTIONS.tolrel = tolrel;
    [X,INFOS] = PenCorr(G,ConstrA,Rank,OPTIONS);
    return;
end

w_scalar = 1.4;  %% to control the overestimation about w
d     = diag(H);
if isfield(OPTIONS,'majorW')
    W     = OPTIONS.majorW;
    w_est = diag(W);
else
    w_est = max(H,[],2);
end

%   w_min0 = min(w_est)
%   w_max0 = max(w_est)

%% note that d<= w_est should be guaranteed
w_diff = max(0,w_est-d) + 1.0e-6; % e-6 to avoid numerical instability
Delta = max(0, H2H - d*d');
DW = d*w_diff' + w_diff*d';
DD = -DW + ( DW.^2 + 4*(Delta.*(w_diff*w_diff')) ).^(1/2);
DD = 0.5*(DD./(w_diff*w_diff'));
alpha = max(max(DD));
fprintf('alpha = %d',alpha)
%%%%% alpha is chosen such that [d+alpha*(w-d)_+]*[d+alpha*(w-d)_+]^T >= H2H
% if alpha <= 0.1
%     w_est = d + alpha*w_est;
% end
w_est = d + alpha*(w_diff - 1.0e-6);
%   w_min1 = min(w_est)
%   w_max1 = max(w_est)

w     = w_est/w_scalar;
Ind   = find(w==0);
w(Ind) = 1;
% w_min = min(w);
% w_max = max(w);

W = diag(w);
w_inv = w.^(-1);

% %%% form W
% if isfield(OPTIONS,'majorW')
%     W = OPTIONS.majorW;
%     w = diag(W);
%     w_est = w;
%     w = w_est/w_scalar;
%     W = diag(w);
%     w_inv = w.^(-1);
% else
%     if norm(H.*H-d*d','fro') <= 1.0e-6
%        
%         fprintf('H is of rank 1!')
%         w = d + 1.0e-6;
%         w_est = w;
%         w = w_est/w_scalar;
%         
%         w_inv = w.^(-1);
%         W     = diag(w);
%               
%     else       
%         w_est = max(H,[],2);
%         %     w0 = sum(H2H)'/n;
%         %     (w0./w)'
%         %     w0 = w0.^0.5;
%         %     w0 = w0*6;    %%%%%% needs to be checked %%%%
%         
%         w = w_est/w_scalar;
%         
%         w_min = min(w);
%         w_max = max(w);
%         if (w_max-w_min)/w_max <= 0.25
%             w = w_max*ones(n,1);
%         end
%         Ind = find(w==0);
%         w(Ind) = 1;
% 
%         
%         w_inv = w.^(-1);
%         W = diag(w);
%     end
% end






fprintf('\n ******************************************')
fprintf('\n  The problem information: \n')
fprintf(' Dimension of SDP constr.  = %3.0f \n', n)
fprintf(' Fixed upper-diag entries  = %3.0f \n', k_e)
fprintf(' The required rank of X   <= %3.0f \n', Rank)
%%
%%---------------------------------------------------------------
%%% CorNewton3Mex preprocessing
%%---------------------------------------------------------------
%%
fprintf('\n ^^^^^^^^ Preprocessing by CorMat3Mex_Wnorm ^^^^^^^^ ')
y = zeros(k_e,1);
for i = 1:k_e
    y(i) = e(i) - G(I_e(i),J_e(i));
end
opts.disp     = 0;
opts.finalEig = 1;
[X,y,info] = CorMat3Mex_Wnorm(G,W,e,I_e,J_e,opts,y);
P          = info.P;
lambda     = info.lam;
rank_X     = info.rank;
Totalcall_CN         = Totalcall_CN + 1;
Totaliter_CN         = Totaliter_CN + info.numIter;
Totalnumb_CG         = Totalnumb_CG + info.numPcg;
Totalnumb_eigendecom = Totalnumb_eigendecom + info.numEig;
objM = H.*(X-G);
residue_CorMatW = sum(sum(objM.*objM));
residue_CorMatW = residue_CorMatW^0.5;
rankErr_CorMatW = abs( sum(lambda) - sum(lambda(1:Rank)) );
if  ( rankErr_CorMatW <= tolrank )   
    %%% check KKT condition
    KKT = H2H - w*w';
    KKT = KKT.*(X-G);
    Err_KKT = sum(sum(KKT.*KKT))^0.5;
    Err_KKT = Err_KKT/max(w)^2;
    if ( Err_KKT <= tolKKT )  
    %fprintf('\n The rank of NCM %d    <= required rank %d \n', rank_X,Rank)
    fprintf('\n The initial calibration is good enough!')
    fprintf('\n The rank error  = %d', rankErr_CorMatW)
    fprintf('\n The KKT error   = %d', Err_KKT)
    fprintf('\n Residue_CorMatHdm      = %9.8e',residue_CorMatW)
     time_used = etime(clock,t0);
    fprintf('\n Total computing time   = %.1f(secs) \n', time_used);
    if Rank<n
        fprintf('\n lambda(r) - lambda(r+1) === %d \n',lambda(Rank)-lambda(Rank+1));
    end    
    INFOS.iter    = 0;
    INFOS.callCN  = Totalcall_CN;
    INFOS.itCN    = Totaliter_CN;
    INFOS.itCG    = Totalnumb_CG;
    INFOS.numEig  = Totalnumb_eigendecom;
    INFOS.rank    = rank_X;
    INFOS.rankErr = rankErr_CorMatW;
    INFOS.residue = residue_CorMatW;
    INFOS.time    = time_used;
    return;
    end
end
fprintf('\n The rank of NCM        = %d', rank_X)
fprintf('\n The rank error of NCM  = %d ', rankErr_CorMatW)  
fprintf('\n Residue_CorNewton      = %9.8e \n',residue_CorMatW)


fprintf('\n\n ^^^^^^^^ Initial Guess ^^^^^^^^ ')
%%% use alternating projection method to generate initial point
if use_InitialPoint
    fprintf('\n use_InitialPoint!')
    opt_disp = 1;
    [X,P,lambda,rank_X,rankErr,normInf,infoNum] = IntPoint_Hnorm(G,H,W,e,I_e,J_e,Rank,X,P,lambda,opt_disp);
   
    %[X,P,lambda,rank_X,rankErr,infoNum] = InitialPoint0(G,H_max2,e,I_e,J_e,Rank,rankErr_CorMatHdm,X1);
    Totalcall_CN  = Totalcall_CN + infoNum.callCN;
    Totaliter_CN  = Totaliter_CN + infoNum.iterCN;
    Totalnumb_CG  = Totalnumb_CG + infoNum.CG;
    Totalnumb_eigendecom = Totalnumb_eigendecom + infoNum.eigendecom;    
    objM = H.*(X-G);
    residue_int = sum(sum(objM.*objM));    
    residue_int = residue_int^0.5;
    residue_1   = residue_int;
else
    X = mPCA(P,lambda,Rank,e_diag);
    [P,lambda] = MYmexeig(X);
    Totalnumb_eigendecom = Totalnumb_eigendecom + 1;
    objM = H.*(X-G);
    residue_CorMatWPCA = sum(sum(objM.*objM))^0.5;
    residue_int = residue_CorMatWPCA;
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
    c0 = 0.50*(residue_int^2 - residue_CorMatW^2);
    c0 = 0.25*c0/max(1.0, rankErr_CorMatW - rankErr);
else  %use_CorNewtonPCA
    c0 = 0.50*(residue_CorMatWPCA^2 - residue_CorMatW^2);
    c0 = 0.25*c0/max(1.0, rankErr_CorMatW);    
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
if Rank <= 1
    c0 = max(c0,1.0e4);
end
c = c0;


fprintf('\n\n ************************************** \n')
fprintf( '      The Penalty Method Initiated!!!      ')
fprintf('\n ************************************** \n')
fprintf('The initial rank         = %d \n', rank_X);
fprintf('The initial rank error   = %d \n', rankErr);
fprintf('The initial ||Ho(X0-G)|| = %9.8e \n', residue_1);

relErr_0    = 1.0e6;
break_level = 0;

k1 = 1;
sum_iter = 0;
opts.disp     = 0;
opts.finalEig = 1;
while ( k1<=maxit )
    
    subtotaliter_CN = 0;
    subtotalnumb_CG  = 0;
    subtotalnumb_eigendecom = 0;
        
    fc = 0.5*residue_1^2;
    fc = fc + c*rankErr;
    fc_old = fc;
    
    C = H2H.*(X - G) + c*(eye_n - U);
    for i=1:n
        C(i,:) = w_inv(i)*C(i,:);
    end
    for j=1:n
        C(:,j) = C(:,j)*w_inv(j);
    end
    G0  = X - C;
    
    if ( k1==1 || rankErr > tolrank )
        y = zeros(k_e,1);
        for i=1:k_e
            y(i) = e(i) - G0(I_e(i),J_e(i));
        end
    end
    
    tt = etime(clock,t0);
    fprintf('\n ================')
    fprintf(' The %2.0dth level of penalty par. %6.5e',k1,c)
    fprintf('  =========================')
    fprintf('\n ........Calling CorNewton3Mex_Wnorm')
    fprintf('\n CallNo.  NumIt  NumCGs    RankX    RankErr      Sqrt(2*FunVal)     Time')
    fprintf('\n %2.0fth      %s       %s       %3.0d     %3.2e     %9.8e    %.1f',...
        0,'-','-',rank_X,rankErr,sqrt(2)*fc^0.5,tt)
    
    
    runhist.rank    = zeros(const.rank_hist,1);
    runhist.funcVal = zeros(const.funcVal_hist,1);
    for itersub = 1:maxitsub
       
        [X,y,info] = CorMat3Mex_Wnorm(G0,W,e,I_e,J_e,opts,y);  
        P          = info.P;
        lambda     = info.lam;
        rank_X     = info.rank;
        %rankErr    = abs( sum(diag(X)) - sum(lambda(1:Rank)) );
        rankErr    = abs( sum(lambda) - sum(lambda(1:Rank)) );
        Totalcall_CN             = Totalcall_CN + 1;
        subtotaliter_CN          = subtotaliter_CN + info.numIter;
        subtotalnumb_CG          = subtotalnumb_CG + info.numPcg;
        subtotalnumb_eigendecom  = subtotalnumb_eigendecom + info.numEig;
               
        objM = H.*(X-G);        
        fc   = sum(sum(objM.*objM));        
        residue_1  = fc^0.5;       
        fc         = 0.5*fc + c*rankErr; 
        
                
        if ( itersub <= const.disp1 || mod(itersub, const.disp2) == 0 )
            dispsub = 1;
            tt = etime(clock,t0);
            fprintf('\n %2.0dth     %2.0f      %2.0f       %3.0d     %3.2e     %9.8e    %.1f',...
                itersub, info.numIter, info.numPcg, rank_X, rankErr, sqrt(2)*fc^0.5,tt)
        else
            dispsub = 0;
        end

        %%% rank history
        if  itersub <= const.rank_hist
            runhist.rank(itersub) = rank_X;
        else
            for j = 1:const.rank_hist - 1
                runhist.rank(j) = runhist.rank(j+1);
            end
            runhist.rank(const.rank_hist) = rank_X;
        end
        %%% function value history
        if  itersub <= const.funcVal_hist
            runhist.funcVal(itersub) = sqrt(2)*fc^0.5;
        else
            for j = 1:const.funcVal_hist - 1
                runhist.funcVal(j) = runhist.funcVal(j+1);
            end
            runhist.funcVal(const.funcVal_hist) = sqrt(2)*fc^0.5;
        end
        %%% residue history
        if sum_iter + itersub <= const.residue_hist
            runhist.residue(sum_iter + itersub) = residue_1;
        else
            for j = 1:const.residue_hist - 1
                runhist.residue(j) = runhist.residue(j+1);
            end
            runhist.residue(const.residue_hist) = residue_1;
        end

        if fc > fc_old
            w_scalar = w_scalar/1.1;
            w_scalar = max(1,w_scalar);
            if w_scalar > 1
                w = w_est/w_scalar;
                w_min = min(w);
                w_max = max(w);
                if (w_max-w_min)/w_max <= 0.25
                    w = w_max*ones(n,1);
                end
                Ind = find(w==0);
                w(Ind) = 1;
                                
                W = diag(w);
                w_inv = w.^(-1);
            end
        end
        fc_old = fc;   
        
        
        
        if rankErr <= tolrank
            tolsub_check = tolsub_rank;       
        else
            tolsub_check = tolsub*max(10, min(100,rank_X/Rank)); %tolsub_check = tolsub*max(10, min(100,rank_X/Rank));
        end
        if itersub >= const.funcVal_hist
           relErr_sub  = abs(runhist.funcVal(1) - runhist.funcVal(const.funcVal_hist));
           relErr_sub  = relErr_sub/max(residue_cutoff, max(runhist.funcVal(1), runhist.funcVal(const.funcVal_hist)));
        end
        if ( itersub >= const.funcVal_hist &&  relErr_sub <= tolsub_check )
%             fprintf('\n The relErr_sub tolerance is achieved!')
%             fprintf('\n relErr_sub %8.7e <= tolsub_check %8.7e!',relErr_sub, tolsub_check)
            
            if dispsub == 0
                tt = etime(clock,t0);
                fprintf('\n %2.0dth     %2.0f      %2.0f       %3.0d       %3.2e   %9.8e    %.1f',...
                    itersub, info.numIter, info.numPcg, rank_X, rankErr, sqrt(2)*fc^0.5, tt)                
            end
            break;
        elseif ( itersub >= const.rank_hist && abs( runhist.rank(1) - runhist.rank(const.rank_hist) ) <= progress_rank1...
                && rank_X - Rank >= progress_rank2 )
            %fprintf('\n Warning: The rank does not decrease in this level!')
            if dispsub == 0
                tt = etime(clock,t0);
                fprintf('\n %2.0dth     %2.0f      %2.0f       %3.0d       %3.2e   %9.8e    %.1f',...
                    itersub, info.numIter, info.numPcg, rank_X, rankErr, sqrt(2)*fc^0.5, tt)                
            end
            break;
        end
               
        %%% update U, G0 and fc0
        P1 = P(:, 1:Rank);
        U  = P1*P1';
        
        C = H2H.*(X - G) + c*(eye_n - U);
        for j=1:n
            C(j,:) = w_inv(j)*C(j,:);
        end
        for j=1:n
            C(:,j) =  C(:,j)*w_inv(j);
        end
        G0  = X - C;
        
    end   %end of subproblem
    
    sum_iter      = sum_iter + itersub;    
    Totalnumb_CG  = Totalnumb_CG + subtotalnumb_CG;
    Totaliter_CN  = Totaliter_CN + subtotaliter_CN;
    Totalnumb_eigendecom = Totalnumb_eigendecom + subtotalnumb_eigendecom;
    fprintf('\n SubTotal %2.0f      %2.0f      %2.0f(eigendecom)',...
        subtotaliter_CN,subtotalnumb_CG,subtotalnumb_eigendecom);
    
    if sum_iter >= const.residue_hist
        relErr = abs(runhist.residue(1) - runhist.residue(const.residue_hist));
        relErr = relErr/max(residue_cutoff, max(runhist.residue(1), runhist.residue(const.residue_hist)));
    else
        relErr = abs(runhist.residue(1) - runhist.residue(sum_iter));
        relErr = relErr/max(residue_cutoff, max(runhist.residue(1), runhist.residue(sum_iter)));
    end
    tt = etime(clock,t0);
    [hh,mm,ss] = time(tt);
    fprintf('\n   Iter.   PenPar.   Rank(X)    RankErr     relErr     ||Ho(X-G)||         Time_used ')
    fprintf('\n   %2.0d     %3.2e    %2.0d      %3.2e     %3.2e   %9.8e     %d:%d:%d \n',...
        k1,c,rank_X,rankErr,relErr,residue_1,hh,mm,ss)
   
    
    if  k1 <= const.rankErr_hist
        runhist.rankErr(k1) = rankErr;
    else
        for i = 1:const.rankErr_hist - 1
           runhist.rankErr(i) = runhist.rankErr(i+1);
        end
        runhist.rankErr(const.rankErr_hist) = rankErr;
    end  
    %%% termination test
    if ( relErr <= tolrel )
        if ( rankErr <= tolrank )
            %fprintf('\n The rank constraint is satisfied!')
            break;
        elseif ( k1 >= const.rankErr_hist && abs(runhist.rankErr(1) - runhist.rankErr(const.rankErr_hist)) <= progress_rankErr )
            %fprintf('\n Warning: The rank does not decrease any more! :( ')
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
        if ( abs(relErr_0 - relErr)/max(1,relErr) <= progress_relErr && relErr <= 10*tolrel )
            break_level = break_level + 1;
            if break_level >= 3
                fprintf('\n Warning: The relErr is consecutively decreasing slowly, quit! :( ')
                if ( rankErr > tolrank )
                    finalPCA = 1;
                end
                break;
            end
        end
    end
   
    k1        = k1 + 1;
    relErr_0  = relErr;
    
    %%% update c
    if rank_X <= Rank
        c = min(c_max, c);
        %fprintf('\n The rank constraint is satisfied and keep c the same!')
        
        % w = w_est/1.7;   
        % w_inv = w.^(-1);
        % W = diag(w);
    else
        if rankErr/min(10,Rank) > 1.0e-1
            c = min(c_max, c*alpha_max);
        else
            c = min(c_max, c*alpha_min);
        end
    end
    
end

%% final PCA correction
if length(e_diag) == k_e && finalPCA
    X = mPCA(P,lambda,Rank,e_diag);
    %[P,lambda] = MYmexeig(X);
    %Totalnumb_eigendecom = Totalnumb_eigendecom + 1;
    rank_X  = Rank;
    rankErr = 0;
    objM = H.*(X-G);
    residue_1 = sum(sum(objM.*objM))^0.5;
end
infeas = zeros(k_e,1);
for i=1:k_e
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
fprintf('\n Final ||Ho(X-G)||        ===== %9.8e', INFOS.residue);
fprintf('\n Primal function value    ===== %9.8e', 0.5*INFOS.residue^2);
fprintf('\n Computing time           ======= %.1f(secs) \n', INFOS.time);

fid = fopen('result_PenCorr_HnormMajorDiag.txt','a+');
fprintf(fid,'\n ***** result_PenCorr_HnormMajorDiag *************');
fprintf(fid,'\n  The problem information: \n');
fprintf(fid,' Dim.  of    sdp        constr  = %d \n',n);
fprintf(fid,' Num. of fixed off-diag constr  = %d \n',length(e));
fprintf(fid,' The required rank      constr  = %d \n',Rank);
fprintf(fid,' ---------------------------------------------------- \n');
%fprintf(fid,'\n **************** Final Information ******************** \n');
fprintf(fid,' Num of pen level         = %d \n', INFOS.iter);
fprintf(fid,' Num of calling CN        = %d \n', INFOS.callCN);
fprintf(fid,' Total num of CG          = %d \n', INFOS.itCG);
fprintf(fid,' Total num of eigendecom  = %d \n', INFOS.numEig);
%fprintf(fid,' Total num of iter in CN  = %d \n', INFOS.itCN);
fprintf(fid,' The rank of  X*          === %d \n', INFOS.rank);
fprintf(fid,' The rank error           === %3.2e \n', INFOS.rankErr);
fprintf(fid,' The rel func error       === %3.2e \n', INFOS.relErr);
fprintf(fid,' The infeasibility of X*  === %9.8e \n', INFOS.infeas);
fprintf(fid,' Final ||Ho(X-G)||        ===== %9.8e \n', INFOS.residue);
fprintf(fid,' Primal function value    ===== %9.8e \n', 0.5*INFOS.residue^2);
fprintf(fid,' Computing time           ======= %.1f(secs) \n', INFOS.time);
fprintf(fid,' ********************************************************* \n');
fclose(fid);
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



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % use an alternating projection method to generate an initial point
% function [X,P,lambda,rank_X,rankErr,infoNum] = ...
%     InitialPoint0(G,H_max2,e,I_e,J_e,Rank,rankErr_CorMatHdm,X1)
% t0 = clock;
% 
% maxit        = 20;
% rank_ratio   = 0.90;
% tolinfeas    = 1.0e-6;
% 
% c0    = 10e0;
% cmax  = 1.0e6;
% alpha = 2;
% c     = c0;
% 
% n   = length(G);
% k_e = length(e);
% 
% if any(I_e-J_e)
%     use_mPCA = 0;
% else
%     use_mPCA = 1;
% end
% opts.disp     = 0;
% opts.finalEig = 1;
% 
% infoNum.callCN      = 0;
% infoNum.iterCN      = 0;
% infoNum.CG          = 0;
% infoNum.eigendecom  = 0;
% 
% for iter = 1:maxit
%     if iter==1
%         Y = X1;
%     else
%         if use_mPCA
%             Y = mPCA(P,lambda,Rank);
%         else
%             Y = Projr(P,lambda,Rank);
%         end
%     end
%     infeas = zeros(k_e,1);
%     for i=1:k_e
%         infeas(i) = e(i) - Y(I_e(i),J_e(i));
%     end
%     NormInf = norm(infeas);
%     if ( NormInf <= tolinfeas )
%         X = Y;
%         [P,lambda] = MYmexeig(X);
%         rank_X  = Rank;
%         rankErr = abs( sum(diag(X)) - sum(lambda(1:Rank)) );        
%         infoNum.eigendecom  = infoNum.eigendecom + 1;       
%         fprintf('\n Alternating terminates at projection onto rank constraint with good feasibility!')
%         if iter==1
%             fprintf('\n *** This initial point is exactly CorMat3MexW_PCA! ***')
%         end
%         break;
%     end
%      
%     G0 = (H_max2*G + c*Y)/(H_max2 + c);
%     %%% CorMat3Mex_Wnorm
%     y = zeros(k_e,1);
%     for i=1:k_e
%         y(i) = e(i) - G0(I_e(i),J_e(i));
%     end
%     [X,y,info] = CorMat3Mex_Wnorm(G0,eye(n),e,I_e,J_e,opts,y);
%     P      = info.P;
%     lambda = info.lam;
%     rank_X = info.rank;
%     infoNum.callCN      = infoNum.callCN + 1;
%     infoNum.iterCN      = infoNum.iterCN + info.numIter;
%     infoNum.CG          = infoNum.CG + info.numPcg;
%     infoNum.eigendecom  = infoNum.eigendecom + info.numEig;
%     rankErr = abs( sum(lambda) - sum(lambda(1:Rank)) );
%     if ( rankErr <= rank_ratio*max(1,rankErr_CorMatHdm) )
%         fprintf('\n Alternating terminates at projection onto linear constraints with small rank error!')
%         break;
%     end
% 
%     c = min( alpha*c, cmax );       
% end
% fprintf('\n Num of iteration = %2.0d',iter)
% fprintf('   RankErr          = %3.2e',rankErr)
% time_used = etime(clock, t0);
% fprintf('\n Time used to generating the initial point = %.1f',time_used)
% return
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





