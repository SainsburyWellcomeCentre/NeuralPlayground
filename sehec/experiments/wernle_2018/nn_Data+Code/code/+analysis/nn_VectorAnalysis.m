% Figure 2 d,e,f: Vector analysis between ratemaps in A¦B and AB

% In this script
% (1)fields are detected in ratemaps A¦B and AB based on their shape and size,
% (2)vectors are computed between fields in A¦B And AB,
% (3)all vectors for all cells are plotted (Supplementary Fig. 4g)
% (4)angles and mean vector lengths for all vectors from all cells are
% computed and plotted and (Fig. 2d)
% (5)all vector distances from all cells are averaged and plotted (Fig. 2e)

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
firingThreshold = 1;% after detecting the fields, use only fields which have a peak firing rate above 1 Hz;

% Store the coordinates for each field
fields =  cell(size(rm,1),4);

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
%% Calculate vector distances between fields in A¦B and AB

% To estimate the translocation of individual firing fields, we computed a vector map of field movement for each cell in the following way.
% For each grid field, we computed a vector pointing from the field’s position in A or B to the nearest field in AB, within a radius corresponding to 60% of the cell’s average grid spacing in A, B, and AB.
% For each vector the length was computed in bins (1 bin = 2cm). If no neighboring field in AB could be identified, the field was not considered.
% Note that individual vectors originating from fields in A or B could point towards the same field in AB, indicating the merging of grid fields.
% To compare field movement across different grid cells and grid scales, vector lengths were normalized to each cell’s average grid spacing in A, B, and AB.


% Search within 60% of the cell's mean grid spacing in A, B, and AB
searchSpacing = 0.6;

% Store for each cell the the starting x,y-coordinates of the vectors for each cell/map
mapYStart = cell(1,size(rm,1));
mapXStart = cell(1,size(rm,1));

% Store for each cell the end x,y-coordinates of the vectors for each cell/map
mapY = cell(1,size(rm,1));
mapX = cell(1,size(rm,1));

% Store for each cell the vector distances
mapDist = cell(1,size(rm,1));% expressed in bins
mapDistNorm = cell(1,size(rm,1));% vector distances are normalized to the cell's average grid spacing in A,B and AB, expressed as percentage; This allows to pool vectors from cells with different grid spacing together and average them.

% Store for each cell the vector angles expressed in degrees
mapAng = cell(1,size(rm,1));



% Calculate for each cell the vector distances and angles
for mapNo = 1: size(rm,1)
    
    searchDist = meanSpacing(mapNo) *searchSpacing;% detect only fields within 60% grid spacing
    
    % Extract for the current cell fields in A¦B
    yFieldsA_B = fields{mapNo,1};
    xFieldsA_B = fields{mapNo,2};
    
    % Extract for the current cell fields AB
    yFieldsAB = fields{mapNo,3};% y
    xFieldsAB = fields{mapNo,4};% x
    
    nField = numel(yFieldsA_B); % number of fields in A|B
    
    % Refresh variables
    % Vector start coordinates
    yStart = zeros(1, nField);
    xStart = zeros(1, nField);
    
    % Vector nd coordinates
    yNear = zeros(1, nField);
    xNear = zeros(1, nField);
    
    % Vector distances
    distNear = zeros(1, nField);
    
    % Vector angles
    angNear = zeros(1, nField);
    
    
    % Go through all fields in A¦B of the current cell
    for fieldNo = 1: size(yFieldsA_B,1)
        
        % Get y and x for the current field
        tmpY = yFieldsA_B(fieldNo);
        tmpX = xFieldsA_B(fieldNo);
        
        % Refresh variable
        matchIdx = [];
        
        % Find for the current field in A¦B all neighbouring fields in AB
        [nIdx,dist] = knnsearch([yFieldsAB xFieldsAB],[tmpY tmpX],'k',size(yFieldsAB,1));
        
        % Select only the fields that lie within 60% of the grid spacing
        spacingIdx = dist < searchDist ;
        
        if sum(spacingIdx)== 0 % if no neighbouring field is detected
            
            % Store vector coordinates for start position in A¦B
            yS = NaN;% y coordinates
            xS = NaN;% x coordinates
            
            % Store vector coordinates for end position in AB
            yN = NaN;% y coordinates
            xN = NaN;% x coordinates
            
            % Store vector distance between fields
            nDist = NaN;
            nAng = NaN;
            
        elseif sum(spacingIdx)> 1 % if there is more than one  neighbouring field detected
            
            matchIdx = nIdx(spacingIdx);% extract the correct Indices
            
            % Store vector coordinates for starting position in A¦B
            yS = tmpY;
            xS = tmpX;
            
            % Store vector coordinates for end position of the
            % closest field in AB
            yN = yFieldsAB(matchIdx(1));% y coordinates
            xN = xFieldsAB(matchIdx(1));% x coordinates
            
            
            % Extract vector distance for the closest field
            % expressed in bins
            nDist = dist(1);% take the closest one (in knnsearch it is the first one)
            
            % Extract the angle of the vector expressed in degrees
            nAng = round(mod(atan2d(yN-tmpY, xN-tmpX), 360)) ;
            
        else
            
            % if there is only one detected
            matchIdx = nIdx(spacingIdx); % extract the correct Index
            
            % Store vector coordinates for starting position in A¦B
            yS = tmpY;% y coordinates
            xS = tmpX;% x coordinates
            
            % Store vector coordinates for end position of the closest field in AB
            yN = yFieldsAB(matchIdx); % y coordinates
            xN =xFieldsAB(matchIdx); % x coordinates
            
            
            % Extract the vector distance
            nDist = dist(spacingIdx)';% expressed in bins
            
            % Extract the angle of the vector expressed in degrees
            nAng = round(mod(atan2d(yN-tmpY, xN-tmpX), 360)) ;
            
        end
        
        % Store values for each field
        yNear(fieldNo) = yN;
        xNear(fieldNo) = xN;
        
        yStart(fieldNo) = yS;
        xStart (fieldNo) = xS;
        
        distNear(fieldNo) = nDist;
        angNear(fieldNo) = nAng;
        
        
    end
    
    % Store the field values for each cell/map
    
    mapY{mapNo} = yNear;
    mapX{mapNo} = xNear;
    
    mapYStart{mapNo} = yStart;
    mapXStart{mapNo} = xStart;
    
    mapDist{mapNo} = distNear';% measured in bins
    
    % Normalize vector distances by the cell's mean spacing in A, B, and AB in order to average across cells with different grid spacing
    mapDistNorm{mapNo} = distNear'./meanSpacing(mapNo);% expressed in percent average gridspacing
    
    mapAng{mapNo} = angNear';% expressed in degrees
    
    
end

%% Store coordiantes for vectors, distances and angles in a new matrix 'vectorData'

vsY = cell2mat(mapYStart);% start coordinate
vsX = cell2mat(mapXStart);

vY = cell2mat(mapY);% end coordinate
vX = cell2mat(mapX);

vDist = cell2mat(mapDist');% vector distance expressed in bins
vDistN = cell2mat(mapDistNorm');% vector distance normalized to the cell$s grid spacing in A,B, and AB expressed in percentage grid spacing

vAng = cell2mat(mapAng');% vector angles expressed in degrees

vectorData = [vsX' vsY' vX' vY' vDistN vDist vAng ];

% Exclude fields that do not have a neighbour (all NaN values) instead of setting to 0
exIdx = sum(isnan(vectorData),2)>1;
vectorData(exIdx,:) = [];

%% Plot all vectors from all cells: Supplementary Fig.4g

% Ratempe size in bins
mapSize = [100 100];

% Extract all fields A_B
yA_B = vectorData(:,2);
yA_B(isnan(yA_B)) = [];

xA_B = vectorData(:,1);
xA_B(isnan(xA_B)) = [];

% Extract all fields AB
yAB = vectorData(:,4);
yAB(isnan(yAB)) = [];

xAB = vectorData(:,3);
xAB(isnan(xAB)) = [];

% Plot all vectors
ux = xAB - xA_B;
vy = yAB - yA_B;

figure ('color','w');
xlim([0 100]);
ylim([0 100]);
axis equal tight
ax = gca;
ax.FontName = 'Arial';
ax.XLabel.String = 'Bins';
ax.XTick = 0:10:100;
ax.YTick = [mapSize(2)/4, mapSize(2)/4 * 3];
ax.YTickLabel = {'B' 'A'};
ax.FontSize = 10;
ax.DataAspectRatio = [1 1 1];

hold on;
quiver(xA_B,yA_B,ux,vy,'color', 'k', 'linewidth',0.1,'ShowArrowHead','on','AutoScale', 'off');
title(['Supp. Fig. 4g: All vectors A|B to AB from all rats; cells = '  num2str(size(rm,1))]);

%% Fig 2d: Distribution of translation vectors across blocks of 50 x 50 cm s

% To determine if fields were translocated in a preferred direction when the central wall was removed, vector maps were divided into blocks of 25 x 25 bins (1 bin = 2cm).
% For each band or block, the mean resultant vector was computed and a Rayleigh test was used to determine if the underlying distribution was uniform.

% Sort vectors from all rats and cells according to their starting position in A¦B into blocks of 25 x 25 bins (50 x 50 cm) and extract for each block all vector distances and angles

% Ratemape size
mapSize = [100 100];

% Extract all vectors coordinates, vector distancens and  angles
tmpDistMap = cell2mat(mapDistNorm');% Normalized vector distances includes NaN values!
tmpAngMap = cell2mat(mapAng');% Vector angles

% Extract vector starting coordinates in A¦B
yCoord = cell2mat(mapYStart)';
xCoord = cell2mat(mapXStart)';

quadrants = 25;% block size expressed in bins

% Define start and end bins of the the blocks
binsStart = 1:quadrants:mapSize(1);
binsEnd = 0:quadrants:mapSize(1);
binsEnd(1) = [];

nQuadrants = size(binsStart,2);% Number of blocks

% Store for each block vector distances and angles
quadrant_Dist = cell(nQuadrants);
quadrant_Ang = cell(nQuadrants);

% refresh variable
qDist = [];

% Go through all blocks
for rowNo = 1 : nQuadrants
    
    yRows =  binsStart(rowNo): binsEnd(rowNo);% define row numbers for current block
    
    for  colNo = 1 : nQuadrants
        
        xRows =  binsStart(colNo): binsEnd(colNo);% define column numbers for current block
        
        % Find all vectors for which the starting x,y-coordinates
        % fall into bins covered by the current block
        Idx = (yCoord >= yRows(1) & yCoord <= yRows(end)) & (xCoord >= xRows(1) & xCoord <=  xRows(end));
        
        if sum(Idx)== 0 % if there is no vector in the block
            
            % Store vector distance
            quadrant_Dist(rowNo, colNo) = NaN;
            
            % Store vector angle
            quadrant_Ang(rowNo, colNo) = NaN;
            
            
        elseif sum(Idx) == 1 % if there is one vector
            
            % Store vector distance
            quadrant_Dist(rowNo, colNo) = tmpDistMap(Idx);
            
            % Store vector angle
            quadrant_Ang(rowNo, colNo) = tmpAngMap(Idx);
            
            
        else % sum(Idx) > 1 % more there are more than one vector per quadrant
            
            qDist = tmpDistMap(Idx); % get specific quadrant values
            
            qAng = tmpAngMap(Idx);
            
            % Store for each block, the mean vector distance and
            % angular mean
            quadrant_Dist{rowNo, colNo} = qDist;
            quadrant_Ang{rowNo, colNo} = qAng;
            
            
        end
        
    end
end


%% Compute the average normalized vector distance, angle and mean resultatn vector lenght for each block

% Refresh variables
meanQuadrant_Dist = zeros(nQuadrants);
meanQuadrant_Ang = zeros(nQuadrants);
meanQuadrant_Mvl = zeros(nQuadrants);
meanQuadrant_P  = zeros(nQuadrants);
meanQuadrant_Z  = zeros(nQuadrants);

% go through all quadrants
for rowNo = 1 : nQuadrants
    
    for  colNo = 1 : nQuadrants
        
        % Compute average distance and s.e.m. expressed in percentage
        % grid spacing
        meanQvl =  nanmean(quadrant_Dist{rowNo,colNo});
        semQvl = nanstd(quadrant_Dist{rowNo,colNo})/sqrt(size(quadrant_Dist{rowNo,colNo},1));
        
        % Remove nan angles
        q_angles = quadrant_Ang{rowNo, colNo};
        q_angles(isnan(q_angles))=[];
        
        % Compute angular mean and s.d. expressed in degrees
        meanQAng = mod(rad2deg(general.circ_mean(deg2rad(q_angles(:)))),360);
        semQAng = mod(rad2deg(general.circ_std(deg2rad(q_angles(:)))),360);
        
        % Compute the mean resultant vector (MVL)
        qMvl = general.circ_r(deg2rad(q_angles(:)));
        
        % Perform Rayleigh test for uniformity of angular
        % distribution
        [pval, z] = general.circ_rtest(deg2rad(q_angles(:)));
        
        % Store values for each block
        meanQuadrant_Dist(rowNo, colNo) = meanQvl;%values are flipped up-site-down compared to the plotted matrix!
        meanQuadrant_Ang(rowNo, colNo) = meanQAng;
        meanQuadrant_Mvl(rowNo, colNo) = qMvl;
        meanQuadrant_P(rowNo, colNo) = pval;% values are flipped up-site-down compared to the plotted matrix!
        meanQuadrant_Z(rowNo, colNo) = z;
    end
end
%% Fig. 2d: Plot all blocks with single vectors (black) and superimposed mean resultant vectors (MVL, red)

rowNo = repmat(1:nQuadrants,nQuadrants,1)-.5;% create starting points for MVL's in the plot
colNo = rowNo';

hf = figure;
hf.Color = [1 1 1];
ax = gca;
ax.XTick = 1:nQuadrants;
ax.YTick = 1:nQuadrants;
ax.YTickLabel = {'B', '', 'A', ''};
ax.DataAspectRatio = [1 1 1];
ax.XLim = [0 nQuadrants];
ax.YLim = [0 nQuadrants];
grid on;
xlabel('Blocks');
title(['Fig. 2d: All vectors (black) and MVL (red) for all rat; cells = ' num2str(size(rm,1))]);%
hold on;

for qNo = 1: size(meanQuadrant_Mvl(:),1)
    
    
    
    % extract single vector lengths and angles to plot the vectors
    allVectors = quadrant_Dist{qNo};
    allAngles = quadrant_Ang{qNo};
    
    % Plot single vectors
    for vNo = 1 : size(allVectors,1)
        
        hl = line;
        hl.XData = [rowNo(qNo), rowNo(qNo)+((allVectors(vNo)/2) * cos(deg2rad(allAngles(vNo))))];
        
        hl.YData = [colNo(qNo), colNo(qNo)+((allVectors(vNo)/2) * sin(deg2rad(allAngles(vNo))))];
        
        hl.LineWidth = 1;
    end
    
    
    % Extract MVL for the current block and angles to plot the vector
    tmpMvl = meanQuadrant_Mvl(qNo)/2;% scale so it the vectors do not extend across the quadrant borders
    tmpAng = meanQuadrant_Ang(qNo);
    
    % Plot MVl on top
    hlmv = line;
    hlmv.XData = [rowNo(qNo), rowNo(qNo)+(tmpMvl * cos(deg2rad(tmpAng)))];
    
    hlmv.YData = [colNo(qNo), colNo(qNo)+(tmpMvl * sin(deg2rad(tmpAng)))];
    
    hlmv.LineWidth = 3;
    hlmv.Color = 'r';
    
    
    
end

%% Fig.2e: Compute the average vectorlength for a matrix of 12 x 12 windows (site length = 16 cm);

% Define windows
% window site length
binWidth = 8;% 1 bin = 2cm
mapSize = [97 97];

intervalsX = 1:binWidth:mapSize(2);% rows

if intervalsX(end) == mapSize(2)
    intervalsX(end)= intervalsX(end);
else
    intervalsX(end+1) = mapSize(1);
end

intervalsY = 1:binWidth:mapSize(1);% columns

if intervalsY(end) == mapSize(1)
    intervalsY(end)= intervalsY(end);
else
    intervalsY(end+1) = mapSize(1);
end


% Extract vector coordinates for starting position in A¦B
tmpY = cell2mat(mapYStart)';
tmpY(isnan(tmpY))=[];% if there is no neighbour don't count the field

tmpX = cell2mat(mapXStart)';
tmpX(isnan(tmpX))=[];% if there is no neighbour don't count the field


% Extract the normalized vector lengths
tmpVar = cell2mat(mapDistNorm');% normalized vectorlength
tmpVar(isnan(tmpVar))= [];


% Sort vectors according to their starting position in A¦B into
% windows

intVar = zeros(floor(mapSize./binWidth));% vectorlength
intVarNo = zeros(floor(mapSize./binWidth));% number of fields

idxY = [];% Index for window sorting
for rowNo = 1:size(intervalsY,2)-1
    
    if rowNo == size(intervalsY,2)-1
        idxY = tmpY >= intervalsY(rowNo) & tmpY <= intervalsY(rowNo+1);% include first bin, and include last bin
    else
        idxY = tmpY >= intervalsY(rowNo) & tmpY < intervalsY(rowNo+1);
    end
    
    idxX = [];
    for colNo = 1: size(intervalsX,2)-1
        
        if colNo == size(intervalsX,2)-1
            idxX = tmpX >= intervalsX(colNo) & tmpX <= intervalsX(colNo+1);% include first bin, and include last bin
        else
            idxX = tmpX>= intervalsX(colNo) & tmpX < intervalsX(colNo+1);
        end
        
        % Define logical Index indicating if there the current vector
        % is within the window
        varIdx = idxX == 1 & idxY == 1;
        
        if sum(logical(varIdx))== 0
            intervalVar = NaN;
            intervalVal = sum(logical(varIdx));
        else
            intervalVar = nanmean(tmpVar(varIdx));% if there are more than one field extract the mean of the distances
            intervalVal = sum(logical(varIdx)); % number of fields per window
        end
        
        % Store for each window the average vector length and number
        % of fields
        intVar(rowNo, colNo)= intervalVar;
        intVarNo(rowNo, colNo) = intervalVal;
    end
end

% Filter the matrix of vectorlengths with a standard 2D Gaussian smoothing kernel with standard deviation of 2*ceil(2*SIGMA)+1 (default mode).
% (1 bin = 2cm);

sigma = 1;
gaussWin = 2*ceil(2*sigma)+1;

% set NaN to eps
varMap = intVar;
varMap(isnan(varMap)) = eps;

% smooth the map
varMapG = imgaussfilt(varMap,sigma,'FilterDomain','spatial');

% get the maximum values for the colorbars
smoothMax = max(varMapG(:));
smoothMin = min(varMapG(:));
caxis([smoothMin smoothMax]);

% Plot the matrix of average vectorlenths

hf = figure ('color','w');

imagesc(flipud(varMapG)); % flip the matrix so A is on top of B
colormap jet

% Color coding from minimum to maximum value
axis tight equal
ax = gca;
ax.YTick = [size(varMap,2)/4, size(varMap,2)/4 * 3];
ax.YTickLabel = {'A', 'B'};
xlabel('Bins')
cb = colorbar;
ylabel(cb,'Norm Vectorlength');
title('Fig.2e: Average normalized vectorlengths for all rats; n = 128 cells; 1 bin = 16cm; ')
