% This code reproduces figures 5 c,f and g;

% load diffAngularOffset.mat, diffDistanceOffset.mat, fieldOffsets20077.mat, fieldOffsets20285.mat and fieldOffsets20607.mat into the workspace.

% diffDistanceOffset.mat : contains data from 6 trials in 4 rats with a large number of simultaneously recorded cells with similar spacing and orientation.
% each row corresponds to one cell pair comparison; col 1 and 3 - difference in distance offset between A and part A in AB for the same cell pairs i,j (col 1) or a randmly chosen cell pair i,k (col 3);
% colum 2 and 4 same as for column 1 and 3 but for B. Distance offsets are
% expressed in bins ( 1 bin = 2cm).

% diffAngularOffset.mat: orientation offsets expressed in degrees, same
% structure as in diffDistanceOffset.mat

% fieldOffsets20077.mat: Average pairwise offset between fields of cell i
% and cell j expressed in cm; col 1 to 3
% pairwise offsets in A,B and AB respectively.

% fieldOffsets202085.mat and fieldOffsets20607.mat same structure as above
%% Fig. 5c: Difference in distance and orientation offset between same and random cell pairs for A part in A¦B vs. A part in AB, B part in A¦B vs. B part in AB

diffDist = diffDistanceOffset;
diffAng = diffAngularOffset;

names = {'Fig. 5c: Difference Distance Offset','Fig. 5c: Difference Orientation Offset'};
labels = {'Distance (cm)', 'Orientation (degrees)'};

differenceS = {};
differenceR = {};

k = 0;
for j = 1:2 % distance and angular offset
    
    k = j;
    
    if k == 1
        
        differenceS = diffDist(:,[1 3])*2;% same pairs; 1 bin = 2c
        differenceR = diffDist(:,[2 4])*2;% random pairs
        
        name = names{j};
        label = labels{j};
    else
        
        differenceS = diffAng(:,[1 3]);% same pairs
        differenceR = diffAng(:,[2 4]);% random pairs
        
        name = names{j};
        label = labels{j};
    end
    
    
    
    hf = figure ('color','w');
    
    if k == 1 % distance offset
        boxplot([differenceS(:,1) differenceR(:,1) zeros(size(differenceS(:,1))) differenceS(:,2) differenceR(:,2)],'Notch','off','Symbol','k.','Widths',0.5,'Labels',{'A same' 'A random' ' ' 'B same' 'B random'})
        ylim([-0.5 34])
        ax = gca;
        ax.YTick = 0:4:34;
        
    else % angular offset
        boxplot(rad2deg([differenceS(:,1) differenceR(:,1) zeros(size(differenceS(:,1))) differenceS(:,2) differenceR(:,2)]),'Notch','off','Symbol','k.','Widths',0.5,'Labels',{'A same' 'A random' ' ' 'B same' 'B random'})
        ylim([-5 200])
        
    end
    
    ax = gca;
    ax.FontName = 'Arial';
    ax.FontSize = 12;
    ylabel(label)
    
    h = findobj(gca,'Tag','Box');
    boxColors = [[0 0 0];[0 0 0];[0 0 0];[0 0 0];[0 0 0]];
    for jj = 1:length(h)
        
        patch(get(h(jj),'XData'),get(h(jj),'YData'),'y','FaceAlpha',.5,'FaceColor',boxColors(jj,:),'EdgeColor',boxColors(jj,:));
        
    end
    
    title([name ' all Rats; cell pairs ' num2str(size(differenceS(:,1),1))])
end

%% Fig. 5 f,g: Pairwise mean field offset


for envNo = 1:2
    
    j = envNo;
    
    if j == 1
        env = 'A';
        names = 'Fig.5 f: Field offset A vs. AB';
    elseif j == 2
        env = 'B';
        names = 'Fig.5 g: Field offset B vs. AB';
    end
    
    hf = figure ('color','w');
    for ratNo = 1:3
        
        k = ratNo;
        
        if k == 1
            allOffsets = fieldOffsets20607;% expressed in cm;
            groupcolor = [89/255 89/255 171/255];
            
        elseif k == 2
            allOffsets = fieldOffsets20077;
            groupcolor = [255/255 127/255 36/255];
            
        else
            allOffsets = fieldOffsets20285;
            groupcolor = [105/255 139/255 34/255];
            
        end
        
        plot(allOffsets(:,j)*2,allOffsets(:,3)*2, '.','color', groupcolor,'MarkerSize', 20)
        ax = gca;
        ax.XLabel.String = ([env ' offset (cm)']);
        ax.YLabel.String = 'AB offset (cm)';
        ax.YLim = [0 36];
        ax.XLim = [0 36];
        title(names);
        
        ax.YTick = 0:4:36;
        hold on
        
    end
    
end