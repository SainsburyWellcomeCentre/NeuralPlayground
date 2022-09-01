% This code reproduces figures 6 d: Mean Resultant vector lengths and
% correlation of directional firing rates for head direction cells;

% load headDirection_MVL.mat, headDirection_RatesA.mat, headDirection_RatesB.mat and headDirection_RatesAB.mat into the workspace.

% headDirection_MVL.mat: contains the mean resultant vector length (MVL)
% for 69 cells; rows correspond to single cells; column 1 -3:  MVL in A, B
% and AB respectively;

% headDirection_RatesA.mat: contains single cell firing rates binned into 64 directional bins (0-360
% degrees), rows correspond to 64 bins, columns correspond to single cells (n= 69 cells);

% headDirection_RatesB.mat and headDirection_RatesAB.mat have the same structure as
% above;

%% Fig. 6 d: Left panel: Mean resultant vector (MVL) in A, B, and AB

passHD = headDirection_MVL;

% Compute mean MVL in A, B and AB
meanMvlA = mean(passHD(:,1));% mean MVL in A
stdMvlA = std(passHD(:,1))/sqrt(size(passHD(:,1),1));

meanMvlB = mean(passHD(:,2));% mean MVL in B
stdMvlB = std(passHD(:,2))/sqrt(size(passHD(:,2),1));

meanMvlAB = mean(passHD(:,3));% mean MVL in AB
stdMvlAB = std(passHD(:,3))/sqrt(size(passHD(:,3),1));

% Plot single and average MVL
figure ('color','w');
plot([1 2 3],[passHD(:,1) passHD(:,2) passHD(:,3)],'k.','markers', 30)
line([0.8 1.2],[meanMvlA meanMvlA],'Color','r','LineWidth', 5 )
line([1.8 2.2],[meanMvlB meanMvlB],'Color','r','LineWidth', 5 )
line([2.8 3.2],[meanMvlAB meanMvlAB],'Color','r','LineWidth', 5 )
set(gca,'Xtick',1:3,'XTickLabel',{'A', 'B', 'AB'})
ylim([0 0.8])
ylabel('Mean vector length')
title(' Mean resultant vector legnth in A, B, and AB; n = 69 cells;')

%% Fig. 6 d: Right panel: Correlation of directional firing rates

allRatesA = headDirection_RatesA;
allRatesB = headDirection_RatesB;
allRatesAB = headDirection_RatesAB;

corrA_B = zeros(1, size(allRatesA,2));
corrA_AB = zeros(1, size(allRatesA,2));
corrB_AB = zeros(1, size(allRatesA,2));

% Go through all cells and correlate the firing rates
for mapNo = 1: size(allRatesA,2)
    
    corrA_B(mapNo)= corr(allRatesA(:,mapNo),allRatesB(:,mapNo));
    corrA_AB(mapNo)= corr(allRatesA(:,mapNo),allRatesAB(:,mapNo));
    corrB_AB(mapNo)= corr(allRatesB(:,mapNo),allRatesAB(:,mapNo));
    
end

meanA_B = mean(corrA_B);%mean correlation A vs B
semA_B = std(corrA_B)/sqrt(size(corrA_B,2));

meanA_AB = mean(corrA_AB); % mean correlation A vs AB
semA_AB = std(corrA_AB)/sqrt(size(corrA_AB,2));

meanB_AB = mean(corrB_AB); % mean correlation B vs AB
semB_AB = std(corrB_AB)/sqrt(size(corrB_AB,2));


hf = figure ('color','w');
plot([1 2 3],[corrA_B' corrA_AB' corrB_AB'],'k.','markers', 30)
line([0.8 1.2],[meanA_B meanA_B],'Color','r','LineWidth', 5 )
line([1.8 2.2],[meanA_AB meanA_AB],'Color','r','LineWidth', 5 )
line([2.8 3.2],[meanB_AB meanB_AB],'Color','r','LineWidth', 5 )
set(gca,'Xtick',1:3,'XTickLabel',{'A vs B', 'A vs AB', 'B vs AB'})
set(gca,'Ytick',-1:0.5:1,'XTickLabel',{'A vs B', 'A vs AB', 'B vs AB'})
ylim([0 1])
ylabel('Correlation')
title(' Correlation of directional firing rates; n = 69 cells;')

