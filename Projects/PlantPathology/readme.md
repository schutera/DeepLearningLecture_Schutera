1. Plant Pathology (Detecting Desease in Appleleafs)

  - **leaf_evaluation.py** 
  
      loads a model (.pb) and does predictions on images within a test directory, finally stores the predictions in a .csv file
  - **mediated_leaf_evaluation.py**
  
    leaf_evaluation.py with an implementation of a soft_probability approach, enabling to mediate from two pure classes onto the mixed class of 'multiple disease'. 
