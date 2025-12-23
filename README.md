Stock Price Prediction
Creating the Dataset
• I used Google Colab to build the model.
• First, I loaded both datasets into the notebook.
• Then, I merged the datasets using an outer join to retain all available data.
• Next, I handled missing values by applying a rolling window of size 5 (equivalent to a
week in the stock market).
• Finally, I sorted the data in ascending order based on the date column to maintain
chronological consistency.
Preprocessing the Data
• I dropped the date column as it is not required for modeling.
• Then, I applied the Hampel filter to replace outliers with the median value of a window
size of 23 (~one month in stock market).
• Next, I split the data into:
o X: Contains stock prices and fundamental data.
o Y: Contains only stock prices.
• I scaled the data using MinMaxScaler for normalization.
• Then, I split the data into train (65%), validation (15%), and test (20%) sets.
• Finally, I created features and target values for each split, where the features contain
past 4 timesteps of data, which are used to predict the present value.
Model Building
To predict stock prices, I implemented a Long Short-Term Memory (LSTM) model using
TensorFlow/Keras.
Model Architecture:
• The model consists of three LSTM layers, each with 50 units to capture
temporal dependencies in stock price movements.
• The first two LSTM layers use return_sequences=True to pass outputs to the
next LSTM layer.
• The final LSTM layer is followed by a Dense layer with 1 neuron, which predicts
the stock price for the next timestep.
Compilation Details:
• Loss Function: mean_squared_error (MSE) is used to minimize prediction
errors.
• Optimizer: adam, an adaptive learning rate optimization algorithm, ensures
efficient training.
• Evaluation Metric: root_mean_squared_error (RMSE) is used to measure
prediction accuracy.
Evaluation Metrics and Results
After training the LSTM model, I evaluated its performance using key metrics, including
Root Mean Squared Error (RMSE) and R² Score. Additionally, I visualized model
performance using loss curves and actual vs. predicted stock prices.
1. Evaluation Metrics
• Root Mean Squared Error (RMSE)
RMSE measures the average error between actual and predicted stock prices.
A lower RMSE indicates better model performance.
• R² Score (Coefficient of Determination)
R² Score represents how well the model explains the variance in stock prices.
2. Results
Model Performance on Train, Validation, and Test Sets
Dataset RMSE R² Score
Train 26.97 1.00
Validation 67.52 0.99
Test 75.70 0.99
---
Challenges Faced --
1. Data Quality Issues – Missing values and outliers affected the dataset’s reliability,
requiring imputation and noise reduction techniques.
2. Feature Selection & Engineering – While experimenting with additional indicators like
Moving Averages and RSI, they degraded performance instead of improving it. The
model performed better with a minimalistic approach focusing on fundamental data,
stock price.
3. Optimal Time Step Selection – Choosing the right number of past time steps was
crucial for accurate predictions. A time step of 4 provided the best balance.
4. Hyperparameter Tuning – Finding the optimal LSTM layer size, number of neurons,
learning rate, and batch size required extensive experimentation. Manual tuning were
used to refine the model architecture.
5. Early Stopping Implementation – The model initially continued training even when
the validation loss stopped improving, leading to overfitting. Early stopping was
introduced to halt training when performance stagnated, ensuring better
generalization.
6. Model Evaluation & Interpretation – While RMSE and R² Score provided numerical
insights, visualizing actual vs. predicted stock prices was crucial for assessing trends
and performance.
