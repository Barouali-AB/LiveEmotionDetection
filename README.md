# LiveEmotionDetection
This text will present the content of the other files/folders in this repository.

Current directory:

	- bd_script.py:
	
		This is the real script that should be executed to have an intelligent emotioanl supporter.
		It is supposed to run in the backgroud
		
	- main.py :
	
		A modified version of the code, that adds showing the user the input of the camera and the
		currently detected emotion in a rectangle around the deected face.
		The time required to get a recommendation is altered too, and the time between image capture is low
		for demonstration purposes
		
	- recomm.ipynb :
	
		Note book used to analyse the work of the recommendation system (is not used by any other file)
		
	- recomm.py :
	
		The file containing the code used by the recommendation system. all the functions, the dataset import
		and dataset management. (is used by main.py and bd_script.py)
		
	- model_final_6.h5 :
	
		File containing the weights of the emotion detection model (detects 6 emotions) (is used by main.py and bd_script.py)
		
	- model_final_6.json :
	
		File containing a json representation of the emotion detection model (is used by main.py and bd_script.py)
		
Dataset:

	- quotes.csv :
	
		csv containing all the quotes we got from internet (preprocessed Dataset)
		
	- PreTraitement.ipynb :
	
		Note Book used to preprocess the data of quotes.csv
		
	- recomm.csv :
	
		csv containing the quotes that will later be used by the recommendation system (is used by recomm.py)
