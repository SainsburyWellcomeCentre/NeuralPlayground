### This text is from the readme of the original dataset available from Wenrle et al, you can use this clase to access the data directly ###

The attached code and data reproduce key figures of the paper ‘Integration of grid maps in merged environments’.
The folder ‘code’ contains three subfolders: 
+ analysis: contains the main codes to reproduce the figures; the beginning section of each         code indicates which data to load into the workspace and specifies the data structure
+ general: contains functions which are called by the main code in +analysis
+ helpers: contains subfunctions which are called by codes in +general
The code is compatible with Matlab 2015a or later versions. The following Matlab toolboxes are required: ‘Matlab Image Processing Toolbox’, ‘Matlab Statistics and Machine Learning Toolbox’, ‘Matlab Circular Statistics Toolbox’ and ‘Matlab geom2D toolbox’. 
The folder ’data’ contains subfolders with data for the key figures. 

The section below describes how to run and load data in order to produce key figures: 
Add the folder nn_Data+Code to your Matlab path.

Figure 1c, d Average sliding correlation heat map A|B × AB for all cells and rats.

Load ratemaps.mat and meanSpacing.mat into the workspace; 

ratemaps.mat contains ratemaps for 128 cells from 10 rats; rows correspond to single cells,         column 1: ratemap A¦B, column 2: ratemap AB;
meanSpacing.mat contains for each cell the average grid spacing in A, B, and AB; rows correspond to single cells;

Run the code nn_SlidingCorrelation.m to reproduce Fig. 1c, d.


Figure 2d, e and Supplementary Figure 4g Translocation of individual grid fields in the central zone after wall removal.

Load ratemaps.mat, meanSpacing.mat and meanRadius.mat into the workspace; 

ratemaps.mat contains ratemaps for 128 cells from 10 rats; rows correspond to single cells,            column 1: ratemap A¦B, column 2: ratemap AB;
meanSpacing.mat contains for each cell the average grid spacing in A, B, and AB; rows correspond to single cells;
meanRadius.mat contains the for each cell the average grid field radius which corresponds to the average radius of the peak at the origin of the cell’s autocorrelogram in A, B, and AB; rows correspond to single cells;

Run the code nn_VectorAnalysis.m to reproduce Supplementary Fig. 4g and Figure 2d, e.


Figure 2f Average sliding cross-correlation map for all cells and all rats.

Load ratemaps.mat and meanSpacing.mat into the workspace; 

ratemaps.mat contains ratemaps for 128 cells from 10 rats; rows correspond to single cells,           column 1: ratemap A¦B, column 2: ratemap AB;

Run the code nn_SlidingCrossCorrelation.m to reproduce Figure 2f; NOTE: cross-correlations are very time intense computations. We compute for each cell 70x70 local cross-correlations with 128 cells in total. The code will run 15 min or more to reproduce the figure. 


Figure 3b, c Continuity in the transition zone between the two original maps A and B emerged through increased equidistance of grid fields.

Load ratemaps.mat, meanSpacing.mat and meanRadius.mat into the workspace; 

ratemaps.mat contains ratemaps for 128 cells from 10 rats; rows correspond to single cells,      column 1: ratemap A¦B, column 2: ratemap AB;
meanSpacing.mat contains for each cell the average grid spacing in A, B, and AB; rows correspond to single cells;
meanRadius.mat contains the for each cell the average grid field radius which corresponds to the average radius of the peak at the origin of the cell’s autocorrelogram in A, B, and AB; rows correspond to single cells;

Run the code nn_VarianceAnalysis.m to reproduce Figure 3b, c top panels;


Figure 4c, d Grid maps in A and B merged rapidly into one map during the first trial in the merged environment AB.

Load ratemapsDevelopment.mat, posA_B.mat, pos_AB.mat and spkAB.mat into the workspace; 

ratemapsDevelopment.mat contains ratemaps for the first trial in AB (19 cells from 10 rats); rows correspond to single cells, column 1: ratemap A¦B, column 2: first trial ratemap AB;
posA_B.mat contains position data for the trial in A¦B; rows correspond to single cells; Column 1: position time stamps (sampling rate 50Hz); column 2: tracked x-coordinate (cm); column 3: tracked y-coordinate (cm); column 4: speed Index (binary; 1 if above the speed threshold of 5cm/s);
posAB.mat has the same structure as above.
spkAB.mat contains spike timestamps (sampling rate 48kHz); rows correspond to single cells;

Run the code nn_Development.m to reproduce Figure 4c, d;


Figure 5c, f, g Pairs of grid cells maintain their phase relationship after wall removal, suggesting that populations of grid cells merge coherently.

Load diffDistanceOffset.mat, diffAngularOffset.mat, fieldOffsets20077.mat, fieldOffsets20285.mat and fieldOffsets20607.mat into the workspace. 

diffDistanceOffset.mat contains data from 6 trials in 4 rats with a large number of simultaneously recorded cells with similar spacing and orientation. Each row corresponds to one cell pair comparison; column 1 and 3 indicate the difference in distance offset between A and part A in AB for the same cell pair i,j (col 1) or a randomly chosen cell pair partner i,k (col 3); column 2 and 4 same as columns 1 and 3 but for B; 
diffAngularOffset.mat has the same structure as diffDistanceOffset.mat.
fieldOffsets20077.mat contains the average pairwise offset between fields of cell i and cell j for one trial in rat 20077; rows correspond to cell pairs; column 1 – 3 indicate offsets in A, B, and AB respectively.
fieldOffsets20285.mat and fieldOffsets20607.mat structured as above.

Run the code nn_Coherence.m to reproduce Figure 5c, f, g;


Figure 6d Head direction cell responses in merged environments.

Load headDirection_MVL.mat, headDirection_RatesA.mat, headDirection_RatesB.mat, headDirection_RatesAB.mat into the workspace.

headDirection_MVL.mat contains the mean resultant vector lengths for 69 head direction cells; rows correspond to single cells;
headDirection_RatesA.mat contains single cell firing rates binned into 64 directional bins (0 to 360 degrees); rows correspond to 64 bins; columns correspond to single cells;
headDirection_RatesB.mat and headDirection_RatesAB.mat are structured as above;

Run the code nn_HeadDirectionCells.m to reproduce Figure 6d left and right panel.
