% Fig. 4c,d: Development of  rate maps after wall removal in AB

% Load ratemapsDevelopment.mat, posA_B.mat, pos_AB and spkAB  into the workspace;

% ratemapsDevelopment.mat: rows corresponds to single cells; column 1 is
% the A¦B ratemap, column 2 is the AB ratemap (19 cells)

% posA_B.mat: position data for the trial in A¦B; rows correspond to single
% cells; col 1: time stamps (sampling 50Hz); col 2: x coordinate (cm); col
% 3: y coordinate (cm); col 4: speed Index (binary; threshold 5 cm/s);

% posAB.mat: position data for the trial in AB; same as posA_B

% spkAB.mat: spike timestamps for the trial in AB (sampling rate 48kHz); rows
% correspond to single cells;

%% Find minimum trial time that is common across all animals

samplesDuration = zeros(1, size(posAB,1));

for mapNo = 1 : size(posAB,1)
    
    mapT = posAB{mapNo}(:,1);% get position timestamps in the AB trial
    samplesDuration(mapNo) = max(mapT);%
    
end

duration = min(samplesDuration);% get the shortest trial time across cells

%% Find all positions duplicate positions in a certain interval, calculate firing rate, smooth it and increase the interval stepwise
rm = ratemapsDevelopment;

%Define time intervals
intStep = 60;% increase interval by 60 seconds
intEdges = 1:intStep:duration;
intEdges(1)= [];
intEdges(end+1) = duration;

% Store for center or distal parts the correlations between interval and
% reference ratemaps
envCorr = cell(1,2);

% Compute maps for only the center or distal part of the ratemap
for cwNo = 1:2
    
    % Define the limits of the ratemap
    corrLimitsX = 1:100;
    
    if cwNo == 1
        corrLimitsY = 25:75;% center bins
    else
        corrLimitsY = [1:24 76:100];% distal bins
    end
    
    
    mapCorr = cell(1, size(rm,1));
    % Go through all cells/maps
    for mapNo = 1: size(rm,1)
        
        % Get position data for the entire trial in AB
        xAB_r =  posAB{mapNo}(:,2); % x coordinate
        yAB_r =  posAB{mapNo}(:,3);% y coordinate
        tPosAB_r = posAB{mapNo}(:,1); % position timestamps
        
        % Speed Index for positions (binary; is 1 if the speed of the rat was above threshold of 5
        % cm/sec)
        idxSpeed_r = posAB{mapNo}(:,4);
        
        % Get spike timestamps for the entire trial in AB
        tSpkAB_r = spkAB{mapNo};
        
        % Produce the ratemap for the entire trial in AB
        % Match position timestamps to spiketime timestamps (positions are
        % sampled at 50Hz, spikes at 48kHz)
        matchIdxAB_r = knnsearch(tPosAB_r,tSpkAB_r);
        matchSpikeTimes = tPosAB_r(matchIdxAB_r);
        
        % Speed filter matched spiketimes
        speedSpikeIdx = idxSpeed_r(matchIdxAB_r);
        matchSpikeTimesSpeed =  matchSpikeTimes(speedSpikeIdx == 1);
        
        % Zero positions
        xAB_z = xAB_r + abs(min(xAB_r));
        yAB_z = yAB_r + abs(min(yAB_r));
        
        % Compute ratemaps for cumulative time intervals (increase
        % by 60s) only for common positions in A¦B and AB
        intCorr = zeros(size(intEdges,2),2);
        
        for intNo = 1: size(intEdges,2)
            
            % Define interval size
            tWindow = 0 : intEdges(intNo);
            
            tmpIdx_pos = tPosAB_r >= tWindow(1)& tPosAB_r <= tWindow(end);% find the position timestmaps that lie within the interval
            tmpIdx_spk = tSpkAB_r >= tWindow(1)& tSpkAB_r <= tWindow(end); % find spike timestamps that lie within interval
            
            % Select the data for time interval
            xAB = xAB_z(tmpIdx_pos);% x
            yAB = yAB_z(tmpIdx_pos);% y
            tPosAB = tPosAB_r(tmpIdx_pos); % pos timestamps
            
            idxSpeed = idxSpeed_r(tmpIdx_pos);% speed Idx
            
            tSpkAB = tSpkAB_r(tmpIdx_spk);% spike times
            
            % Produce the ratemap for the interval
            % Match pos timestamps to spike timestamps
            matchIdxAB = knnsearch(tPosAB,tSpkAB);
            matchSpkT = tPosAB(matchIdxAB);
            
            % Speed filter spikes
            speedSpikeIdx = idxSpeed(matchIdxAB);% binary vector for speed same size as tSpk
            matchSpikeSpkSpeed =  matchSpkT(speedSpikeIdx == 1);% speed filtered spiketimes , smaller than tSpk
            
            % Sort x,y positions (expressed in cm) into bins of the ratemap (1 bin = 2cm)
            binWidth = 2; % 1 bin = 2 cm
            limits = [1, 200, 1, 200];% ratemap limits in cm
            nBins = ceil((limits(2) - limits(1)) / binWidth);% ratemap limits
            edges = limits(1):binWidth:limits(1) + binWidth*nBins; % create nBinsX bins
            
            [~, ~, xx] = histcounts(xAB, edges);
            xx(xx == 0) = nan; % x == 0 indicates values outside edges range, could happen depending on the limitsX
            
            [~, ~, yy] = histcounts(yAB, edges);
            yy(yy == 0) = nan;
            
            % Count occurrences for each (x,y) timestamp, i.e. spike spread across positions
            n = histc(matchSpikeSpkSpeed,tPosAB);
            Nspikes = general.accumulate([xx yy], n, [nBins nBins]);
            
            % Occupancy map
            % Duration for each (X,Y) sample (clipped to maxGap)
            maxGap = 0.02;
            dt = [diff(tPosAB); 0];
            if sum(dt) == 0
                dt = [0; 0];
            end
            dt(end) = dt(end - 1);
            dt(dt>maxGap) = maxGap;
            
            % general.accumulate returns matrix MxN where M are columns and not rows, as opposite to
            % standard Matlab matrix notation.
            time = general.accumulate([xx yy], dt, [nBins nBins])'; % how much time animal spent at a specific bin
            time = round(time * 10e6) / 10e6; % remove floating numbers artefacts
            
            zRaw = Nspikes'./(time+eps);% instantaneous rates;
            
            % Smooth the ratemap
            zSmooth = general.smooth(zRaw,3.5);% 2D gaussian kernel with standard deviation of 3.5 bins in both directions
            
            % Replace not sampled positions with NaN
            selected = time == 0;
            zSmooth(selected) = NaN;
            zSmooth = zSmooth(corrLimitsY,corrLimitsX);%  center or distal
            
            % Get positions in A_B or AB that were not visited
            a_bIdx = isnan(rm{mapNo,1}(corrLimitsY,corrLimitsX));
            abIdx = isnan(rm{mapNo,2}(corrLimitsY,corrLimitsX));
            
            % Get the index for all not visited positions
            commonIdx = a_bIdx | abIdx;
            
            % Correlate the interval ratemap to the ratemap of the entire
            % trial in A¦B or AB
            
            rateCorr = zeros(1,2);
            for refNo = 1:2% A¦B,AB
                
                tmp_refMap = rm{mapNo,refNo};% extract reference ratemap
                tmp_refMap = tmp_refMap(corrLimitsY,corrLimitsX);% center or distal part of the ratemap
                
                tmp_refMap(commonIdx)= NaN;% set not visited positions (in the interval)to Nan in the reference map
                
                smoothIdx = isnan(zSmooth);% the smoothed interval map has also more not visited positions than the final smoothed map
                
                tmp_refMap(smoothIdx)= NaN;% set the reference maps to NaN
                zSmooth(commonIdx)= NaN;% set the smooth map to common NaN values
                
                % Correlate the two maps
                tmpCorr = corr(tmp_refMap(~isnan(tmp_refMap)),zSmooth(~isnan(zSmooth)));
                
                rateCorr(refNo) = tmpCorr;
                
            end
            
            % Store for each interval the correlation
            intCorr(intNo,:)= rateCorr;
            
        end
        % Store for each cell all interval correlations
        mapCorr{mapNo}= intCorr;
        
    end
    
    % Store correlations for center or distal parts
    envCorr{cwNo}= mapCorr;
end

%% Plot Fig. 4 c,d Correlation interval maps with A¦B or AB for central and distal parts of the box

names ={'Fig. 4c: Center AB', 'Fig. 4d: Distal AB'};

for cwNo = 1:2
    
    name = names{cwNo};
    corrEnv = envCorr{cwNo};
    
    % Compute the mean and s.e.m. across all cells for each interval
    allMaps = reshape(cell2mat(corrEnv),size(intEdges,2),size(rm,2),size(rm,1));% create multi dimensional matrix array
    intMean = nanmean(allMaps,3);
    intSem = std(allMaps,0,3,'omitnan')/sqrt(size(rm,1));%std(A,weighting, dim, nan)
    
    
    x = 1:1:size(intEdges,2);
    
    hf = figure;
    hf.Color = [1 1 1];
    errorbar(x,intMean(:,1),intSem(:,1),'color',[0.2 0.2 0.2],'LineWidth', 1)
    hold on
    errorbar(x,intMean(:,2),intSem(:,2),'color',[1 0 0],'LineWidth', 1)
    
    ylim([0 1])
    xlim([0 size(intEdges,2)+1])
    ylabel('Mean Correlation')
    xlabel('Minutes')
    title([name ' mean correlation'])
    legend('Interval map x A|B','Interval Map x AB','Location','southeast')
    ax = gca;
    ax.YTick = 0:0.1:1;
    ax.XTick = [1 5 10 15 (size(intEdges,2))];
    ax.FontName = 'Calibri';
    ax.FontSize = 15;
end
