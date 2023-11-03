# fetch_works
Files:
1. data_daily.csv: given dataset
2. fetch_rewards_training.ipynb: ipynb that perform data analysis, data preprocessing and data training.
3. bestmodel.h5: saved trained model.
4. app.py: streamlit application
5. requirements.txt
6. README.md

Data Preprocessing:
1. The "# Date" column in the DataFrame is converted to a datetime format to facilitate date-based analysis. 
2. The code visualizes the data using matplotlib and seaborn, creating a line plot of Receipt_Count over time for a visual understanding of the data.

Data Aggregation:
1. Data is aggregated by month to get the total Receipt_Count for each month in the year 2021.
2. The aggregated data is normalized to have values between 0 and 1.

Data Preparation for Modeling
1. The code defines a function reshape_conversion(data) that takes a list of data, converts it to a NumPy array, reshapes it, and converts it to a PyTorch tensor.
2. The monthly data and corresponding Receipt_Count data are converted and reshaped for model input and output.

Model Definition
1. Neural Network Model:
The code defines an LSTM-based model using Keras. The model is built as a Sequential model. It contains two LSTM layers with a dropout layer between them. The model concludes with a Dense layer for prediction.
2. Model Compilation:
The model is compiled with a mean squared error loss function and the Adam optimizer with a learning rate of 0.001. Callbacks are defined, including early stopping to prevent overfitting, learning rate reduction, and model checkpoint to save the best model during training.


Model Training:
The code converts the data to TensorFlow tensors for model input. The model is trained using the fit method with the specified callbacks. The trained model is saved as "bestmodel.h5."

Model Evaluation:
Test data is prepared by converting and reshaping the final month's data and corresponding Receipt_Count data.
The model is evaluated using the test data, and the test loss is displayed.

