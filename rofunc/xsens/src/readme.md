Here is a brief description of the Python scripts provided:

`main_mvnx.py `

This is the main script. Use this script as a template on how to load the mvnx file and plot its data in Python: 
    * The function "main()" inside the file shows how to correctly use the scripts load_mvnx.py, mvn.py and mvnx_file_accessor.py in order to load the mvnx file. The function also plots the position data of the first segment.
    * You can also run this script from the command prompt with argument --mvnx_file (e.g main_mvnx.py --mvnx_file "testfile.mvnx").
    * Be aware that the script expects the dependencies listed in requirements.txt to be installed. You can install such dependencies by running the command "py -m pip install -r requirements.txt".

`load_mvnx.py`

This file contains the functions to open and read the file as an mvnx formatted XML file. See the function "main()" from the main_mvnx.py file to correctly make use of it.

`mvnx_file_accessor.py`

This file contains the class (data structure) to store the mvnx data parsed. It also provides an interface to retrieve the data. See the function "main()" from the main_mvnx.py file to correctly make use of it.

`mvn.py`

This file contains some definitions such as index and names of the different mvn parameters required by the other scripts.
