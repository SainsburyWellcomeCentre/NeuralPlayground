The sample includes conjunctive cells and head direction cells from layers III and V of medial entorhinal cortex and have  been published in 
Sargolini et al. (Science, 2006). 

The files are in matlab format. They include spike and position times for recorded cells from rats that were running in a 1 x 1 m
enclosure. The cells were recorded in the dorsocaudal 25% portion of the medial entorhinal cortex. Position is given for two LEDs 
to enable calculation of head direction.

The cell id is based on tetrode number and cell number (i.e: t2c7).

The file naming convention is as follow:

Rat number - session number _ cell id (i.e: 11084-03020501_t2c1).

Each session duration is normally 10 minutes, but some sessions are combination of 2 or more 10 minutes sessions, this is marked in 
the file name by using "+" between the session numbers (i.e: 11207-21060501+02_t6c1). Note that the 6 first digits in the session 
number is the date of the recording.

When loading the files into Matlab you get the following variables:

x1	Array with the x-positions for the first tracking LED.
y1	Array with the y-positions for the first tracking LED.
x2	Array with the x-positions for the second tracking LED.
y2	Array with the y-positions for the second tracking LED.
t	Array with the position timestamps.
ts	Array with the cell spike timestamps.

The position data have been smoothed with a moving mean filter to remove tracking jitter.


You can use the data for whatever you want but we take no responsibility for what is published!

Please refer to our web site where you obtained the data in your Methods section when you write up the results.

Best regards,

Raymond Skjerpeng and Edvard Moser
Kavli Institute for Systems Neuroscience
Centre for the Biology of Memory
Norwegian University of Science and Technology

Correspondence: Edvard Moser (edvard.moser@ntnu.no)
