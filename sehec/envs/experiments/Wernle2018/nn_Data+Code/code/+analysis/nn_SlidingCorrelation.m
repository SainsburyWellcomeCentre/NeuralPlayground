% Figure 1,c d: Sliding correlation between ratemaps in A¦B and AB

% Load ratemaps.mat and the meanSpacing.mat into the workspace;

% ratemaps.mat: rows corresponds to single cells; column 1 is the A¦B ratemap, column 2 is the AB ratemap
% meanSpacing.mat: each row corresponds to one cell and indicates the average
% grid spacing in A, B, and AB. This value is used to set the sliding
% correlation window size

%% Sliding correlation:
% Note: running this section may take a few minutes

% parameters
mapSize = [100 100];% ratemap size in bins ( 1 bin = 2cm)
boxCenter = (1: mapSize(1));% 1 : 100 center bins of the sliding correlation box

rm = ratemaps;

mapCorr = cell(1, size(ratemaps,1));

% Correlate for each cell the ratemaps in A¦B and AB
for mapNo = 1: size(rm,1)
    
    tmpMapA_B = rm{mapNo,1};% temporary map in A¦B
    tmpMapAB = rm{mapNo,2};% temporary map in AB
    
    
    % define size of the sliding correlation window: the site length corresponds to the cell`s mean grid spacing in A,B, and AB
    
    tmpSpacing = meanSpacing(mapNo);
    
    % check if the window size is odd which is necessary to include the center bin
    %if mod(tmpSpacing ,2)
    %  boxSize = tmpSpacing+1;
    % else
    %    boxSize = tmpSpacing ;
    % end
    
    boxSize = tmpSpacing  + ~mod(tmpSpacing ,2);% site length of the correlation box needs to be odd to include the bin on which the box is centerede.g. sliding box is 31 bins: 1 center bins +- 15 bins
    boxWing = floor (boxSize/2);% half of the site length of the correlation box
    
    boxCorr = zeros(mapSize);% refresh variable
    
    % Sliding correlation: Center the correlation box on all bins in the ratemap (`sliding correlation`)
    
    for rowNo = boxCenter(1) : boxCenter(end) % 1 : 100
        
        % define size of the correlation box in Y direction
        yRows = rowNo-boxWing : rowNo+boxWing ;
        yRows(yRows<1) = [];% If the bins extend over the ratemap delete them
        yRows(yRows>100)= [];
        
        for colNo = boxCenter(1) : boxCenter(end) % 1:100
            
            % define of the correlation box in X direction
            
            xColumns = colNo-boxWing : colNo+boxWing ;
            xColumns (xColumns <1)= [];% If the bins extend over the ratemap delete them
            xColumns (xColumns >100)=[];
            
            % extract ratemap values from bins that are covered by the current correlation box
            tmpBoxA_B = tmpMapA_B(yRows, xColumns);% temporary correlation box in A¦B
            tmpBoxAB = tmpMapAB(yRows, xColumns);% temporary correlation box in AB
            
            boxCorr(rowNo,colNo) = corr(tmpBoxA_B(:),tmpBoxAB(:));% spatial correlation between the two ratemap boxes
        end
    end
    
    mapCorr{mapNo} = boxCorr;% Store for each cell the sliding correlation values (100x100)
end
%% Figure 1 c: Plot the average sliding correlation heat map across all cells

mapSize = [100 100];% size of the sliding correlation heatmap

% Create a multidimensional matrix array
allMaps = reshape(cell2mat(mapCorr),mapSize(1),mapSize(2),size(mapCorr,2));

% Caculate the average sliding correlaton heatmap across all cells
allMapsMean = mean(allMaps,3);

% Plot the average heatmap
figure('color','w');
imagesc(flipud(allMapsMean)); % flip the matrix so A is on top of B
colormap jet

% Color coding from minimum to maximum value
cb = colorbar;
cb.Label.String = 'Correlation';
caxis([min(allMapsMean(:)) max(allMapsMean(:))]);
axis equal tight

% axis label
ax = gca;
ax.YTick = [size(allMapsMean,2)/4, size(allMapsMean,2)/4 * 3];
ax.YTickLabel = {'A' 'B'};
ax.XLabel.String = 'Bins';
ax.FontName = 'Arial';
ax.FontSize = 12;
title(['Fig. 1 c: Sliding Correlation A|B x AB all rats; cells = ' num2str(size(mapCorr,2))])


%% Figure 1 d: Calculate the average sliding correlation value for 1th percentile bands parallel (1x100 bins) or orthogonal (100 x 1 bins) to the inserted wall

mapSize = [100 100];% size of the sliding correlation heatmap

% Create a multidimensional matrix array
allMaps = reshape(cell2mat(mapCorr),mapSize(1),mapSize(2),size(mapCorr,2));

% Caculate the average sliding correlaton heatmap across all cells
allMapsMean = mean(allMaps,3);

tmpSliceMean = zeros(1, mapSize(1));
semSlice = zeros(1, mapSize(1));
tmpSliceMeanV = zeros(1, mapSize(1));
semSliceV = zeros(1, mapSize(1));

for pNo = 1: mapSize(1)
    
    tmpSlice = allMapsMean(pNo,1: mapSize(1));% parallel band (1x100 bins)
    tmpSliceV = allMapsMean(1: mapSize(1),pNo);% orthogonal band (100x1 bins)
    
    tmpSliceMean(pNo) = mean(tmpSlice);% mean
    semSlice(pNo) = std(tmpSlice)/sqrt(size(tmpSlice,2));% sem
    
    tmpSliceMeanV(pNo) = mean(tmpSliceV);
    semSliceV(pNo) = std(tmpSliceV)/sqrt(size(tmpSliceV,1));
end

% Plot the average values
x = 1:1:100;

mean_y = tmpSliceMean;
sem_y = semSlice;

meanV_y = tmpSliceMeanV;
semV_y = semSliceV;

% Parallel bands Data
hf = figure;
hf.Color = [1 1 1];
hf.Name = 'Errorbar patch';
ylim([0 0.8])

% Plot s.e.m
hp = patch;
hp.XData = [x, fliplr(x)];
hp.YData = [(mean_y  - sem_y), fliplr((mean_y  + sem_y))];
hp.FaceColor = [1 0 0];
hp.FaceAlpha = .2;
hp.EdgeColor = 'none';

% Plot mean
hl = line;
hl.XData = x;
hl.YData = mean_y;
hl.LineWidth = 1;
hl.Color = [1 0 0];

% Orthogonal bands Data
% Plot s.e.m
hp = patch;
hp.XData = [x, fliplr(x)];
hp.YData = [(meanV_y  - semV_y), fliplr((meanV_y  + semV_y))];
hp.FaceColor = [0 0 0];
hp.FaceAlpha = .2;
hp.EdgeColor = 'none';

% Plot mean
hl = line;
hl.XData = x;
hl.YData = meanV_y;
hl.LineWidth = 1;
hl.Color = [0 0 0];

ax = gca;
ax.YTick = 0:0.1:1;
ax.XTick = 0:10:100;
ax.XLabel.String = 'Band number';
ax.FontName = 'Arial';
ax.FontSize = 12;
legend('parallel bands','s.e.m.', 'orthogonal bands','s.e.m.','Location','southeast')

title(['Fig. 1 d: Average Correlation of 1th percentile bands all rats; cells = ' num2str(size(mapCorr,2))])
