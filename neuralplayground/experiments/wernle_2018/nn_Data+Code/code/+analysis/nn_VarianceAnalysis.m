% Figure 3b,c: Analysis of standard deviation of grid field distancees in A¦B and AB

% In this script
% (1)fields are detected in ratemaps A¦B and AB based on their shape and size,
% (2)for each field in A¦B or AB the distances to all neighbouring fields
% within 130% cell's average grid spacing in A,B, and AB is computed,the distances are normalized the cell's grid spacing and
% the standard deviation of field distances is extracted
% (3) Fig. 3 b,c is plotted

% Load ratemaps.mat, meanSpacing.mat and meanRadius into the workspace;

% ratemaps.mat: rows corresponds to single cells; column 1 is the A¦B ratemap, column 2 is the AB ratemap

% meanSpacing.mat: each row corresponds to one cell and indicates the
% average grid spacing in A, B, and AB expressed in bins. This value is used to detect fields
% within 60% of the cell's average grid spacing in A, B,and AB

% meanRadius.mat: each row corresponds to one cell and indicates the
% the cell's average grid radius (grid field radius of the field at the origin of the autocorrelogram) expressed in bins; This value is used to
% detect fields
%% FIELD DETECTION A!B and AB

% Detect grid peaks based on their size and shape with Matlab image processing toolbox
% Ratemaps are transformed into a greyscale image of connected regions or objects by applying the morphological image processing operations erosion and reconstruction (Matlab image processing toolbox). Erosion removed small objects in the original image A and resulted in an eroded marker image J.  The remaining objects in the eroded image J served to mark objects in in the original image A and to reconstruct them based on morphological greyscale reconstruction.
% Finally, the center bin of each object in the reconstructed image was determined and used to identify the x,y - location of peaks in the firing rate maps. If the center bin in the rate map had a higher mean firing rate than 1 Hz, it was counted as grid field.

rm = ratemaps;
nMaps = size(rm,1);
firingThreshold = 1;% after detecting the fields, use only fields which have a peak firing rate above 1 Hz;

% Store the coordinates for each field
fields = cell(nMaps,4);

% Go through fields in A¦B and AB
for envNo = 1:2
    
    mapID = envNo;
    k = envNo;
    
    if k == 1
        env = [1 2];% for A¦B store y,x field-coordinates in column 1 and 2
        name = 'A_B';
    else % k == 2
        env = [3 4];% for AB store y,x field-coordinates in column 3 and 4
        name = 'AB';
    end
    
    
    % Go through all cells
    for mapNo = 1: size(rm,1)
        
        tmpMap = rm{mapNo,mapID};% SELECT A_B or AB
        
        % Select the cell specific average grid field radius
        diskSize = meanRadius(mapNo);
        
        % Define size of the disk shaped structuring element
        se = strel('disk',diskSize);
        
        % Greyscale erode image
        Ie = imerode(tmpMap, se);
        
        % Greyscale reconstruct image
        Iobr = imreconstruct(Ie,tmpMap);
        
        % Detect maxima regions in the image (8-neighbouring pixels of constant intensity)
        regionalMaxMap = imregionalmax(Iobr);
        
        % Return connected components and characeristics with pixel Index list
        connectedRegions = bwconncomp(regionalMaxMap, 8);
        
        % Get field centers = center bin for the detected regions
        stats = regionprops(connectedRegions, 'Centroid');
        peaks = cat(1,stats.Centroid);
        peaks = round(peaks);
        
        % Transorm linear index to index in the 100 x 100 ratemap matrix
        peakIdx = sub2ind(size(tmpMap),peaks(:,2),peaks(:,1));% x,y
        
        % Get firing rate of peaks
        peakFiring = tmpMap(peakIdx);
        
        % Get only peaks above 1 Hz firing rate
        firingIdx = peakFiring > firingThreshold;
        
        % Get the field coordinates
        peaksX = peaks(firingIdx, 1);
        peaksY = peaks(firingIdx, 2);
        
        % Store the field coordinates for each cell in A¦B and AB
        fields{mapNo,env(1)} = peaksY;
        fields{mapNo,env(2)} = peaksX;
        
    end
    
end

%% Detect all neighbouring fields within 130% of the cell's average grid spacing in A,B,and AB

% Search within 130% of the cell's mean grid spacing in A, B, and AB
searchSpacing = 1.3;

% Columns for y,x coordinates in A¦B and AB
col = [1 2; 3 4];

% Standard deviation of field distances
mapDistStd = cell(nMaps,2);

% For each cell, coordinates for fields without neighbours
mapZeroY = cell(nMaps,2);
mapZeroX = cell(nMaps,2);

% For each cell, coordinates for fields with neighbours
mapY = cell(nMaps,2);
mapX = cell(nMaps,2);

% Go through environments A¦B and AB
for envNo = 1:2
    
    % Extract field coordinates y,x
    yFields_env = cell2mat(fields(:,col(envNo,1)));
    xFields_env = cell2mat(fields(:,col(envNo,2)));
    
    % Go through all cells
    for mapNo = 1: size(fields,1)
        
        yFields = fields{mapNo,col(envNo,1)};
        xFields = fields{mapNo,col(envNo,2)};
        tmpSpacing = meanSpacing(mapNo);% extract the cell's average grid spacing in A,B, and AB
        
        nFields = size(yFields,1);  % number of fields
        
        % Coordinates for fields with neighbours
        fieldY = zeros(1, nFields);
        fieldX = zeros(1, nFields);
        
        % Coordinates for fields without neighbouring fields
        fieldZY = zeros(1, nFields);
        fieldZX = zeros(1, nFields);
        
        distStd = zeros(1, nFields);
        
        % Go through all fields for the cell
        for fieldNo = 1: size(yFields,1)
            
            % get y and x for each field
            tmpY = yFields(fieldNo);
            tmpX = xFields(fieldNo);
            
            matchIdx = [];
            
            % Detect all neighbouring fields
            [nIdx,dist] = knnsearch([yFields xFields],[tmpY tmpX],'k',size(yFields,1));
            
            % Count only the fields that lie within 130% of the cell's average
            % grid spacing
            spacingIdx = dist < (tmpSpacing*searchSpacing);
            
            if sum(spacingIdx) == 1 % If no neighbouring field is detected - the first is always the field itself;
                
                % Store the coordinates of the field itself if it has less than 2 neighbours
                zeroY = tmpY;
                zeroX = tmpX;
                
                % Store coordinates of the field itself if it has more
                % than two neighbours
                vY = NaN;
                vX = NaN;
                
                % Store the distance to neighbouring fields in bins
                nDist = NaN;
                
                % Store the standard deviation of the distances to other
                % fields
                nDistStd = NaN;
                
            elseif sum(spacingIdx) == 2 % If there is only one neighbouring field detected
                
                % Store the coordinates of the field itself if it has less than 2 neighbours
                zeroY = tmpY;% store coordiantes
                zeroX = tmpX;
                
                % Store coordinates of the field itself if it has more
                % than two neighbours
                vY = NaN;
                vX = NaN;
                
                % Store the distance to the neighbouring field in bins
                nDist = dist(2); % exclude field itself
                
                % Store the standard deviation of the distances to other
                % fields
                nDistStd = NaN;
                
            else
                
                % Store coordinates of the field itself if it has more
                % than two neighbours
                vY = tmpY;
                vX = tmpX;
                
                % Store the coordinates of the field itself if it has less than 2 neighbours
                zeroY = NaN;
                zeroX = NaN;
                
                matchIdx = spacingIdx;
                
                % Extract distances in bins
                nDist = dist(matchIdx);
                nDist = nDist(2:end); % exclude field itself
                
                % Normalize distances to cell's average grid spacing in
                % A,B,and AB
                nDist = nDist/meanSpacing(mapNo);
                
                % Compute the standard deviation of the the normalized
                % distances
                nDistStd = std(nDist);
                
            end
            
            % Store for each field
            distStd(fieldNo) = nDistStd;% standard deviation of field distances normalized to the cell's average grid spacing
            
            % Coordinates for fields with neighbours
            fieldY (fieldNo) = vY;
            fieldX (fieldNo) = vX;
            
            % Coordinates for fields without neighbouring fields
            fieldZY (fieldNo) = zeroY;
            fieldZX (fieldNo) = zeroX;
            
            
            
        end
        
        % Store for each cell
        mapDistStd {mapNo,envNo} = distStd;
        
        mapZeroY{mapNo,envNo} = fieldZY;
        mapZeroX{mapNo,envNo} = fieldZX;
        
        mapY {mapNo,envNo} = fieldY;
        mapX {mapNo,envNo} = fieldX;
        
    end
end

%% Fig. 3b,c: Compute standard deviation of field distances and plot them in 12 x 12 matrix of 16x16cm windows

envVar = cell(1,2);

% Go through environment A¦B and then AB
for envNo = 1:2
    
    % Define windows of the matrix
    binWidth = 8;% window site length (1 bin = 2cm)
    mapSize = [97 97; 97 97];
    
    intervalsX = 1:binWidth:mapSize(envNo,2);% rows
    intervalsY = 1:binWidth:mapSize(envNo,1);% columns
    
    % Get the coordinates for each field
    tmpY = cell2mat(mapY(:,envNo)');
    tmpY(isnan(tmpY))=[];% delete fields without neighbours, they are counted seperately
    
    tmpX = cell2mat(mapX(:,envNo)');
    tmpX(isnan(tmpX))=[];
    
    % Get standard deviation of normalized field distanes
    tmpVar = cell2mat(mapDistStd(:,envNo)');
    tmpVar(isnan(tmpVar))= [];
    
    
    % Sort fields into 16x16 cm windows according to their position in A¦B or AB
    intVar = zeros(size(intervalsY,2)-1) ;
    idxY = [];
    for rowNo = 1:size(intervalsY,2)-1
        
        if rowNo == size(intervalsY,2)-1
            idxY = tmpY >= intervalsY(rowNo) & tmpY <= intervalsY(rowNo+1);% include first bin, and include last bin
        else
            idxY = tmpY >= intervalsY(rowNo) & tmpY < intervalsY(rowNo+1);
        end
        
        idxX = [];
        for colNo = 1: size(intervalsX,2)-1
            
            % if it is the last interval in X include the last bin as well
            if colNo == size(intervalsX,2)-1
                idxX = tmpX >= intervalsX(colNo) & tmpX <= intervalsX(colNo+1);% include first bin, and include last bin
            else
                idxX = tmpX>= intervalsX(colNo) & tmpX < intervalsX(colNo+1);
            end
            
            % If there is one or more field in this window
            varIdx = idxX == 1 & idxY == 1;
            
            % Store the average standard deviation
            if sum(logical(varIdx))== 0
                intervalVar = NaN;
            else
                intervalVar = nanmean(tmpVar(varIdx));% if there is more than one field extract the mean of the standard deviations
            end
            
            intVar(rowNo, colNo)= intervalVar;
        end
    end
    envVar{envNo} = intVar;
    
end

% Filter the matrix of standard deviations with a standard 2D Gaussian smoothing kernel with standard deviation of 2*ceil(2*SIGMA)+1 (default mode).
% (1 bin = 2cm);

sigma = 1;
gaussWin = 2*ceil(2*sigma)+1;

% Store the smoothed matrix
mapVarG = cell(1,2);

% Minimum/maximum values for plotting colorbars
minSmooth = zeros(1,2);
maxSmooth = zeros(1,2);

% for environments A¦B and AB
for envNo = 1:2
    
    % Extract the matrix values
    varEnv = envVar{envNo};
    
    % Set NaN to eps
    varMap = varEnv;
    varMap(isnan(varMap)) = eps;
    
    % Smooth the matrix with a 2D Gaussian kernel
    varMapG = imgaussfilt(varMap,sigma,'FilterDomain','spatial');
    
    % Get the maximum and minimum values for plotting the colorbars
    smoothMax = max(varMapG(:));
    maxSmooth(envNo) = smoothMax;
    
    smoothMin = min(varMapG(:));
    minSmooth(envNo) = smoothMin;
    
    % Store the smoothed matrix
    mapVarG{envNo} = varMapG;
    
end

% Plot Fig. 3b,c:  smoothed matrix of standard deviations in A¦B and AB


% Use the minimum and maximum of A¦B or AB to plot both matrices
smoothLimit = max(maxSmooth(1),maxSmooth(2));
smoothLimit1 = min(minSmooth(1),minSmooth(2));

figNames = {'b', 'c'};
names = {'A|B', 'AB'};

for envNo = 1:2
    
    hf = figure ('color','w');
    imagesc(flipud(mapVarG{envNo})); % flip the matrix so A is on top of B
    colormap jet
    axis tight equal
    
    % Color coding from minimum to maximum value
    caxis([smoothLimit1 smoothLimit]);
    
    ylabel('Bins')
    xlabel('Bins')
    c1 = colorbar;
    ylabel(c1,'Normalized standard deviation');
    
    title(['Fig.3' figNames(envNo) ' : Normalized standard deviation of field distances' names(envNo)])
    
end
