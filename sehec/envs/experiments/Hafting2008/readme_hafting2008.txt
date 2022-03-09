These cells are from layers II and III of medial entorhinal cortex and have  been published in Hafting et al. (Nature, 2008). 

The files are in matlab format. They include spike and position times for a number of simultaneously recorded cells from rats that were running on a linear track (mostly 320 cm).
The cells were recorded in the dorsocaudal 25% portion of the medial entorhinal cortex. 

The cell id is based on tetrode number and cell number (i.e: t2c7).

The EEG file is included to enable analyses of theta phase relationships etc.


There are 3 file types available for each recording: EEG data, Position data and Spike data.

The naming convention for the files is as follow

Rat number - session number - file type (i.e: 11015-13120410_EEG)

The spike data file has the cell id as the file type (i.e: 11015-13120410_t5c1).


Each session duration is normally 10 minutes, but some sessions combine 2 or more 10 minutes sessions, this is marked in 
the file name by using "+" between the session numbers (i.e: 11265-13030610+11+12+13_POS). Note that the 6 first digits in the session 
number is the date of the recording. For some sessions there is more than 1 local EEG-signal (each from a different tetrode).

When loading the files into Matlab you get the following variables:

posx	Array with the x-positions for the tracking LED
posy	Array with the y-positions for the tracking LED
post	Array with the position timestamps.
ts	Array with the cell spike timestamps.
EEG	Array with the EEG-samples.

The position data have been smoothed with a moving mean filter to remove tracking jitter.

The EEG is sampled at 250 Hz, and the values are given in bits ranging from -128 to 127.
High resolution EEG sampled at 4800hz included in subfolder

All examples are shown in Supplementary Figures 2 and 3 in Hafting et al (2008).


You can use the data for whatever you want but we take no responsibility for what is published!

Please refer to our web site where you obtained the data in your Methods section when you write up the results.

Best regards,
Raymond Skjerpeng and Edvard Moser
Kavli Institute for Systems Neuroscience
Centre for the Biology of Memory
Norwegian University of Science and Technology

Correspondence: Edvard Moser (edvard.moser@ntnu.no)
