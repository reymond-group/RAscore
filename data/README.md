# Retrosynthetic Accessibility (RA) score Data

* `rascore_training_data.zip` contains the datasets used to train the RAscore models. These contain SMILES and a binary value (0, 1) representing if a route was found.

* Raw AiZynthfinder data on the ChEMBL compounds using the USPTO policy can be found at the following link: https://cloud-new.gdb.tools/s/xxRkaNwQEyiN7xM \

The raw data is a .zip file containing a folder. The foler contains multiple .hdf files for which teh key is 'tables'. Some issues may be experienced loading some .hdf files due to the format of the route data and depth of the trees.