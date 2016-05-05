# FScategorization
Improving Audio Retrieval through Content and Metadata Categorization

We present here code for thresholding-based content ("Content" Folder) and LDA-based metadata categorization ("Tags" Folder)
in the context for freesound.

Content Categorization File Details: 

1. IDlist.json - List of 5635 sounds downloaded from freesound
2. no_analysis.txt - Contains a list of 387 sounds for which the content analysis descriptors were not computed when downloaded. We ignore these in our analysis
3. Run check.py for determining the loudness category. Replace the input directory with the path to the directory containing all the audio files.
4. You would require Essentia to run the code (http://essentia.upf.edu/) 
