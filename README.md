# CMPUT497TermProject

Shouyang Zhou, Sharon Hains and Sharif Bakouny 

We declare that we consulted with no other people and additionally did not use any external resources other than the resources referenced in the report.

References are included in the report.

To view our repo of this project, please go here: https://github.com/shouyang/CMPUT497TermProject

INSTRUCTIONS:
This project is split into two sections:
1. Benchmark evaluation of RAKE and TextRank
To run this, please run 'benchmark.py' as you normally would a Python file. 
You will likely need to download Python package 'lxml' and spaCy's english model file if not already installed.
This will take about 15-20 minutes to run.
Analysis results will be printed out, and no user input is required.
If any errors are presented finding pathways of the abstracts, please refer to the 'getAbstracts' function in loadData.py to change the pathway formatting as necessary. 

2. Stoplist generation
*We would not recommend running this as it takes 55-65 minutes on average execution to generate the stoplist.*
To run this, please run 'stoplistGeneration.py' as you normally would a Python file.
No user input is required.
Words that are included in the stoplist will be generated in a textfile called 'keywordAdjacencyStoplist.txt'.
Words that are excluded from the stoplist will be generated in a textfile called 'excludedFromStoplist.txt'.
Counts of the stopwords and excluded words are generated in the textfile called 'table1_3.txt'.
If any errors are presented finding pathways of the abstracts, please refer to the 'getAbstracts' function in loadData.py to change the pathway formatting as necessary.




