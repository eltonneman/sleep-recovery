# sleep-recovery

**AUTHORS**: Rithik Raghuraman & Elton Neeman
**COURSE**:  DS 4420
**PROF**: Eric Gerber

### Project Overview:
This project is an extension of Gashi et al., who studied the sleep quality recognition using wrist-worn physiological sensors using general and personalized models. Gashi accomplished personalization by usinga hybrid train/test splitting method. In this project, it will be achieved through a hierarchical Beyesian model. 

We will be comparing two models:
1. **MLP (Population Model)**: trained on all participants except one, and tested on the left over participant. 
2. **Hierarchical Beyesian (Personalized Model)**: individual parameteres estimated per participant. 


### Data
We used the PMData Set, sourced from Kaggle, containing sports data over 5 months from 16 participants. 

**LINK:** https://www.kaggle.com/datasets/vlbthambawita/pmdata-a-sports-logging-dataset
