% Figure 2f: Sliding cross correlation between ratemaps in A¦B and AB
% NOTE: this code will run 15 min or more. Cross-corrleations are very time
% intense computations. We compute for each cell 70x70 local corss-correlations
% , with 128 cells total.

% Load ratemaps.mat and the meanSpacing.mat into the workspace;

% ratemaps.mat: rows corresponds to single cells; column 1 is the A¦B ratemap, column 2 is the AB ratemap

% meanSpacing.mat: each row corresponds to one cell and indicates the
% average grid spacing in A, B, and AB expressed in bins. This value is
% used to normalize cross-correlation shift to the average grid spacing

%% Sliding cross-correlation: !!! THIS COMPUTATION IS VERY TIME INTENSE - appr. 10 min run (cross-correlations are very time intense in matlab)


rm = ratemaps;

% Define box size for the sliding cross correlation
boxSize = 31;% Entire site length including center bin
boxWing = 15;% half of the box Size

xCorr = cell(size(rm,1),1);

% Go through all cells/maps
for mapNo = 1: size(rm,1)
    
    % select ratemaps
    tmpMapA_B = rm{mapNo,1};% A¦B
    tmpMapAB = rm{mapNo,2};% AB
    
    % Define center bins of the boxes
    boxCenter = (1+boxWing : 100-boxWing);% bins 16 : 85 (+- boxWing of 15 bins)
    
    % initialize count variables to store crosscorr variables
    counti = 1;
    countj = 1;
    
    % Compute for all possible box locations the cross correlation between
    % the box in A¦B and the entire ratmap AB
    for rowNo = boxCenter(1) : boxCenter(end)
        
        ii = counti;
        
        % Define x direction bins for the current box
        yRows = rowNo-boxWing : rowNo+boxWing ;
        
        for colNo = boxCenter(1) : boxCenter(end) % in y direction
            
            jj = countj;
            
            % Define y direction bins for the current box
            xColumns = colNo-boxWing : colNo+boxWing ;
            
            % cut out tmp boxAB
            tmpBoxAB =tmpMapA_B(yRows, xColumns);
            
            % define center of box in ratemap coordinates
            boxCenter_rmY = rowNo;
            boxCenter_rmX = colNo;
            
            % Cross correlate the current box in A¦B with the entire ratemap in AB; box A¦B is always first aligned in upper left corner (1/1)
            
            ccm = normxcorr2(tmpBoxAB,tmpMapAB);% (template, image); normalized by std of boxarea; values -1 to 1; corr matrix size is template+image-1
            
            % Find amount of shift that is necessary to match the tmpBox to the
            % original location in the ratemap;
            % cross corr: shift = matrix coordinates - template size
            
            shiftMatchY = boxCenter_rmY - (boxWing+1);% left corner coordinate for cross corr matrix
            
            shiftMatchX = boxCenter_rmX - (boxWing+1);
            
            % Search for maxima within +- half the boxSize since the grid pattern is periodically repeating;
            % define shiftmatch in terms of cross corr corrdinates (+boxSize);
            % size should be 31x31
            
            ccmSearchYRows =  ((shiftMatchY + boxSize(1)) - boxWing : (shiftMatchY + boxSize(1)) + boxWing);
            
            ccmSearchXColumns  =  ((shiftMatchX + boxSize(1)) - boxWing : (shiftMatchX + boxSize(1)) + boxWing);
            
            % Extract only the search area from the entire
            % cross-correlation
            ccmSearch = ccm(ccmSearchYRows,ccmSearchXColumns);%
            
            % Find maximum in search map and linear Idx
            [ccmSearchMax,IdxMax]= max(ccmSearch(:));
            
            % get x,y coordinate for max with respect to ccmSearch
            [ccmSearch_shiftY_abs, ccmSearch_shiftX_abs] = ind2sub(size(ccmSearch),IdxMax);% 1:31 x 1: 31
            
            % The coordinates in the 31x31 box are the absolute amount of shift
            % Y: UP neg shift: 1:15; ! absolute shift amount from center: 15-shift; substract from ratemapY coordinate
            %    DOWN pos shift: 16:31; ! absolute shift from center is: shift -15; add to ratemapY coordinate
            % X: LEFT neg shift: 1:15; absolute: 15-shift; substract form ratemapX
            %    RIGHT pos shift: 16:31; aboslute: shift-15; add to ratemapX
            
            
            % Y shift
            if ccmSearch_shiftY_abs <= boxWing
                upShiftY = ccmSearch_shiftY_abs;
                upShiftY_abs = (boxWing+1) - upShiftY;% determine absolute shift with restpect to center of 31x31 which is 16/16 and 0/0
                shiftY_abs = upShiftY_abs;% save for vectorlength calculation
                
            else
                downShiftY = ccmSearch_shiftY_abs;
                downShiftY_abs = downShiftY - (boxWing+1);
                shiftY_abs = downShiftY_abs;
                
            end
            
            % X shift
            
            if ccmSearch_shiftX_abs <= boxWing
                leftShiftX = ccmSearch_shiftX_abs;
                leftShiftX_abs = ((boxWing+1) - leftShiftX);  % determine abs shift
                shiftX_abs = leftShiftX_abs; % store point
                
            else
                rightShiftX = ccmSearch_shiftX_abs;
                rightShiftX_abs = (rightShiftX - (boxWing+1));
                shiftX_abs = rightShiftX_abs;
                
            end
            
            
            % Compute the shift length (euclidean distance)
            tmpVlength = sqrt(shiftX_abs^2 + shiftY_abs^2);
            
            xCorr{mapNo}(ii,jj) = tmpVlength./meanSpacing(mapNo); % normalize shift length to the cell's average grid spacing in A,B, and AB
            
            countj = countj+1;
        end
        
        counti = counti+1;
        
        countj = 1;
    end
    
end

%% Plot the average

% Calculate average across all cells/maps
allDist = cat(3,xCorr{:});% normalized map to gridspacing
meanDistMap = mean(allDist,3);

hf = figure ('color','w');
imagesc(flipud(meanDistMap)); % flip the matrix so A is on top of B
colormap jet
cb = colorbar;
cb.Label.String = 'Normalized shift';
caxis([min(meanDistMap(:)) max(meanDistMap(:))]);
axis equal tight
ax = gca;
ax.YTick = [size(meanDistMap,2)/4, size(meanDistMap,2)/4 * 3];
ax.YTickLabel = {'A' 'B'};
ax.FontName = 'Calibri';
ax.FontSize = 12;
title('Fig. 2f: Average normalized shift for all rats; n = 128 cells');

