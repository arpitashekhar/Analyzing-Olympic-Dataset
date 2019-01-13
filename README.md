**Dataset**

We have used the “120 years of Olympic history” data-set from Kaggle.  This is a historical data-set on the modern Olympic Games, including all the Games from Athens 1896 to Rio 2016. There are 271116 rows and 15 columns. Each row corresponds to an individual athlete competing in an individual Olympic event. The columns are ID, Name, Sex, Age, Height, Team, Season, Medal, Event etc.


**Project Idea**

Over the past Olympics, there have been many contingents which has had sports-persons returning home without a medal for the corresponding sport in their tally. In this project, we aim to estimate the medal tally of each country so that we can make predictions about a country’s best chance of winning a medal in a sport. This would help a country better prepare its Olympics contingent. 
We have used Decision Tree Algorithm, Artificial Neural Network, Support Vector Machine and Long Short Term Memory (Time-Series) to train our model and predict the medal taaly.

**Software**

We have used python 3.64 for the development along with the following packages: 
(python 3.7 is not compatible with tensorflow backend used by keras)
numpy, pandas, seaborn, lstm, ann, svm, decisionTree, matplotlib, tensorflow, keras

To run the code:

1. Install the required packages using the command - "pip3 install \<package-name\>"
2. The main.py file can be run by - "python3 main.py" 
