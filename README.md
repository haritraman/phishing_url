Phishing URL Detection using Random Forest
This project is a machine learning model designed to detect malicious phishing URLs with high accuracy. It uses a lightweight Random Forest classifier trained only on lexical features (features extracted from the URL string itself). This approach makes the model fast and efficient, ideal for real-time applications.

The entire workflow, from data cleaning to model training and evaluation, is contained in the Copy_of_cyber.ipynb notebook.

Methodology
The project follows a standard machine learning pipeline:

Data Loading: Loads two CSV files:

majestic_million.csv: A list of benign, high-reputation URLs (Label 0).

verified_online.csv: A list of verified malicious phishing URLs from PhishTank (Label 1).

Preprocessing: The data is cleaned, deduplicated, and balanced by sampling an equal number of malicious and benign URLs (50,000 each) to prevent model bias.

Feature Engineering: A comprehensive function (extract_advanced_features) extracts 21 distinct lexical features from each URL string (e.g., url_length, num_dots, has_https, entropy).

Feature Selection: A preliminary Random Forest is trained on all 21 features to analyze their importance. Based on this, the Top 5 most predictive features are selected for the final model.

Data Splitting: The dataset (now with only the Top 5 features) is split into an 80% training set and a 20% testing set.

Scaling: The 5 features are normalized using StandardScaler (fit on the training data only) to improve model performance.

Model Training: The final RandomForestClassifier (with 100 trees) is trained on the scaled, 5-feature training set.

Evaluation: The model's performance is assessed on the 20% test set using a Classification Report, Confusion Matrix, and ROC AUC Score.

Saving: The trained model (random_forest_model.pkl) and the feature scaler (scaler.pkl) are saved to disk for future use.

Datasets
Benign: majestic_million.csv

Source: Majestic Million

Malicious: verified_online.csv

Source: PhishTank (A comparable open-source feed of verified phishing URLs).

🛠️ How to Run
Clone the repository:

Bash

git clone https://github.com/haritraman/phishing_url.git
cd phishing_url
Install dependencies: It's recommended to use a virtual environment.

Bash

pip install -r requirements.txt
<details> <summary>Click to see <b>requirements.txt</b></summary>

numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
gdown 
(Note: gdown is useful for downloading data from Google Drive if needed. tensorflow was in the original import list but not used for the final RF model.)

</details>

Add Data: This project requires the majestic_million.csv and verified_online.csv datasets. You must download them and place them in the location expected by the notebook (e.g., in your Google Drive, as seen in Cell 2 of the notebook).

Run the Notebook: Open and run the Copy_of_cyber.ipynb notebook in Google Colab or a local Jupyter environment.

📊 Model & Results
The final model is a RandomForestClassifier that uses only the Top 5 lexical features for prediction:

path_length

num_slashes

has_https

num_dots

url_length

Performance
The model's performance is evaluated on the 20% unseen test set. The key metrics generated in the notebook are:

Accuracy: Overall percentage of correct predictions.

Precision: Ability of the model to not label a benign URL as phishing.

Recall: Ability of the model to find all the phishing URLs.

F1-Score: The balanced mean of Precision and Recall.

ROC AUC Score: The model's ability to distinguish between the two classes.

The full Classification Report, Confusion Matrix, and ROC Curve are generated at the end of the notebook.