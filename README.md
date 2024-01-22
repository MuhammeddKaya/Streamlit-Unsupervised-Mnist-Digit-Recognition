First run 'python3 -m venv venv venv' to set up a virtual environment
then run 'source venv/bin/activate' to activate the virtual environment
go to the main folder (cd Streamlit-Unsupervised-Mnist-Digit-Recognition) and run 'pip install -r ./docs/requrements.txt' to install the required packages. This will install the required packages
To run the code, go to the main folder and run 'streamlit run gui.py'. 
This code may run a bit late in the first phase because if there are no models, it retrains the models from scratch, if there are models, it uses these models. 
After the interface opens, you can draw the number and guess it. 
In total 3 different models were trained, 2 of them (k-means and Gaussian mixture) are unsupervised and one (random forest) is supervised. 
Unsupervised models show 32 samples from the cluster to which the picture you draw belongs. 
Supervised model predicts the picture directly
