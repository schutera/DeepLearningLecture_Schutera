
============================================================================================================================================
BeeID - The bee reidentification data set
============================================================================================================================================

  .--.               .--.
 /    `.   o   o   .'    \
 \      \   \ /   /      /
 /\_     \ .-"-. /     _/\		WHAT IS THE SOCIETY WE WISH TO PROTECT?
(         V ^ ^ V         )		IS IT THE SOCIETY OF COMPLETE SURVEILLANCE FOR THE COMMONWEALTH?
 \_      _| 9_9 |_      _/		IS THIS THE WEALTH WE SEEK TO HAVE IN COMMON
  `.    //\__o__/\\    .'		OPTIMAL SECURITY 
    `._//\=======/\\_.'			AT THE COST OF MAXIMAL SURVEILLANCE?
     /_/ /\=====/\ \_\
       _// \===/ \\_
      /_/_//`='\\_\_\ hjw		- TOM STOPPARD 
        /_/     \_\

============================================================================================================================================
contact: mark.schutera@kit.edu / mark.schutera@mailbox.org
============================================================================================================================================


Data is originally from https://groups.oist.jp/bptu/honeybee-tracking-dataset

HONEYBEE TRACKING II
Here we placed dataset and supplemental information of the study Markerless tracking of an entire insect colony. 
As above, the dataset comprises detection and trajectory information from five beehive recordings at 10 fps 5 min segments.

Dataset files
For each recording (S1-S5) the compressed file contains video, detections and trajectory information:
S1 (0.6 GB) 
S2 (0.6 GB) 
S3 (0.6 GB) 
S4 (0.8 GB)
S5 (0.8 GB) 


You can install all required packages by: pip install -r requirements.txt
Our datamodulemaker.py will work all the magic to convert the HONEYBEE TRACKING II data set into the beeID reidentification data set.
The data set will the follow the specifications defined within the Open-ReID framework: https://cysu.github.io/open-reid/notes/data_modules.html

Accepting the default parameters will result in 90x90 px image samples, each trajectory translates into a bee_id. 
The trainval / test split is 80:20.
The gallery / query ratio is 10:1.

In case of questions, please do not hesitate to contact: mark.schutera@mailbox.org or mark.schutera@kit.edu
Feel free to use the dataset or code in your own work, and make sure to cite ours.

 
