import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, Conv1D, MaxPooling1D, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.losses import Huber
import matplotlib.pyplot as plt
import optuna
import joblib
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import os
import logging
import holidays


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add this function to create lag features
def create_lag_features(data, lag_periods=[1, 7, 14, 30]):
    for lag in lag_periods:
        data[f'lag_{lag}'] = data['A_to_B Daily Passengers'].shift(lag)
    return data

def add_rolling_features(data, windows=[7, 14, 30]):
    for window in windows:
        data[f'rolling_mean_{window}'] = data['A_to_B Daily Passengers'].rolling(window=window).mean()
        data[f'rolling_std_{window}'] = data['A_to_B Daily Passengers'].rolling(window=window).std()
    return data


def add_holiday_features(data):
    # Use Malaysia's holidays
    malaysia_holidays = holidays.Malaysia()
    
    # Check if each date is a holiday in Malaysia
    data['is_holiday'] = data['Date'].isin(malaysia_holidays)
    
    # Calculate the days to the next holiday, default to a large number if no future holiday is found
    data['days_to_next_holiday'] = data['Date'].apply(
        lambda x: min((day - x).days for day in malaysia_holidays if day > x) if any(day > x for day in malaysia_holidays) else float('inf')
    )
    
    # Calculate the days from the last holiday, default to a large number if no past holiday is found
    data['days_from_last_holiday'] = data['Date'].apply(
        lambda x: min((x - day).days for day in malaysia_holidays if day < x) if any(day < x for day in malaysia_holidays) else float('inf')
    )
    
    return data


# Load and preprocess the data
def preprocess_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Month'] = data['Date'].dt.month
    data['DayOfYear'] = data['Date'].dt.dayofyear
    data['WeekOfYear'] = data['Date'].dt.isocalendar().week
    data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
    data['DayOfMonth'] = data['Date'].dt.day
    
    data = create_lag_features(data)
    data = add_rolling_features(data)
    data = add_holiday_features(data)
    data = handle_outliers(data)

    # Check which columns contain NA values
    na_columns = data.columns[data.isna().any()].tolist()
    print("Columns with NA values:", na_columns)
    
    # Calculate the number of days until the next New Year for each date
    data['days_until_new_year'] = data['Date'].apply(lambda x: (pd.Timestamp(year=x.year + 1, month=1, day=1) - x).days)
    
    # Handle NA values for each column
    for col in na_columns:
        if col.startswith('lag_') or col.startswith('rolling_'):
            # For lag and rolling features, we can forward fill or use the mean
            data[col] = data[col].fillna(method='ffill').fillna(data[col].mean())
        elif col in ['days_to_next_holiday', 'days_from_last_holiday']:
            # Use the days until the next New Year as a fallback for holidays
            data[col] = data[col].fillna(data['days_until_new_year'])
        else:
            # For other columns, use the mean or median
            data[col] = data[col].fillna(data[col].median())
    
    # Check if there are still any NA values
    remaining_na = data.isna().sum().sum()
    print(f"Remaining NA values: {remaining_na}")
    return data

# Extract features and target
def extract_features(data):
    X = data[['DayOfWeek', 'Month', 'DayOfYear', 'WeekOfYear', 'IsWeekend', 'DayOfMonth', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7', 
              'rolling_mean_14', 'rolling_std_14', 'rolling_mean_30', 'rolling_std_30']].values
    y = data['A_to_B Daily Passengers'].values.reshape(-1, 1)
    return X, y

# Create sequences
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

def create_model(trial, input_shape):
    seq_length, n_features = input_shape
    
    model = Sequential()
    
    # Add CNN layer (if used)
    if trial.suggest_categorical('use_cnn', [True, False]):
        filters = trial.suggest_int('conv_filters', 16, 128)
        kernel_size = trial.suggest_int('conv_kernel', 2, min(5, seq_length))
        model.add(Conv1D(filters=filters,
                         kernel_size=kernel_size,
                         activation='relu',
                         padding='same',  # Use 'same' padding to maintain temporal dimension
                         input_shape=(seq_length, n_features)))
        
        # Only add MaxPooling1D if the sequence length is sufficient
        if seq_length > 2:
            model.add(MaxPooling1D(pool_size=2))
    
    # Add LSTM layers
    n_lstm_layers = trial.suggest_int('n_lstm_layers', 1, 3)
    for i in range(n_lstm_layers):
        lstm_units = trial.suggest_int(f'lstm{i+1}_units', 32, 256)
        return_sequences = i < n_lstm_layers - 1
        if i == 0 and not trial.suggest_categorical('use_cnn', [True, False]):
            lstm_layer = LSTM(lstm_units, activation='tanh',
                              return_sequences=return_sequences, 
                              input_shape=(seq_length, n_features),
                              kernel_regularizer=l2(trial.suggest_float(f'l2_{i+1}', 1e-6, 1e-2, log=True)))
        else:
            lstm_layer = LSTM(lstm_units, activation='tanh',
                              return_sequences=return_sequences,
                              kernel_regularizer=l2(trial.suggest_float(f'l2_{i+1}', 1e-6, 1e-2, log=True)))
        
        model.add(lstm_layer)
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(trial.suggest_float(f'dropout{i+1}', 0.0, 0.5)))
    
    # Add Dense layers
    n_dense_layers = trial.suggest_int('n_dense_layers', 1, 2)
    for i in range(n_dense_layers):
        model.add(Dense(trial.suggest_int(f'dense{i+1}_units', 16, 128), 
                        kernel_regularizer=l2(trial.suggest_float(f'l2_dense_{i+1}', 1e-6, 1e-2, log=True))))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(trial.suggest_float(f'dropout_dense{i+1}', 0.0, 0.5)))
    
    model.add(Dense(1))
    
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss=Huber())
    return model

def create_model(trial, input_shape):
    seq_length, n_features = input_shape
    
    model = Sequential()
    
    # CNN layer (optional)
    if trial.suggest_categorical('use_cnn', [True, False]):
        filters = trial.suggest_int('conv_filters', 16, 256)
        kernel_size = trial.suggest_int('conv_kernel', 2, min(5, seq_length))
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same', input_shape=(seq_length, n_features)))
        if seq_length > 2:
            model.add(MaxPooling1D(pool_size=2))
    
    # LSTM/GRU layers
    n_recurrent_layers = trial.suggest_int('n_recurrent_layers', 1, 3)
    for i in range(n_recurrent_layers):
        units = trial.suggest_int(f'recurrent{i+1}_units', 32, 512)
        recurrent_type = trial.suggest_categorical('recurrent_type', ['LSTM', 'GRU'])
        bidirectional = trial.suggest_categorical('bidirectional', [True, False])
        
        # Regularization
        reg_type = trial.suggest_categorical('regularization', ['none', 'l1', 'l2', 'l1_l2'])
        if reg_type == 'none':
            regularizer = None
        elif reg_type == 'l1':
            regularizer = l1(trial.suggest_float('l1', 1e-6, 1e-2, log=True))
        elif reg_type == 'l2':
            regularizer = l2(trial.suggest_float('l2', 1e-6, 1e-2, log=True))
        else:
            l1_value = trial.suggest_float('l1', 1e-6, 1e-2, log=True)
            l2_value = trial.suggest_float('l2', 1e-6, 1e-2, log=True)
            regularizer = l1_l2(l1=l1_value, l2=l2_value)
        
        if recurrent_type == 'LSTM':
            layer = LSTM(units, return_sequences=(i < n_recurrent_layers - 1), kernel_regularizer=regularizer)
        else:
            layer = GRU(units, return_sequences=(i < n_recurrent_layers - 1), kernel_regularizer=regularizer)
        
        if bidirectional:
            layer = Bidirectional(layer)
        
        model.add(layer)
        model.add(LeakyReLU(alpha=trial.suggest_float(f'leaky_alpha_{i+1}', 0.01, 0.3)))
        model.add(Dropout(trial.suggest_float(f'dropout{i+1}', 0.0, 0.5)))
    
    # Dense layers
    n_dense_layers = trial.suggest_int('n_dense_layers', 1, 3)
    for i in range(n_dense_layers):
        model.add(Dense(trial.suggest_int(f'dense{i+1}_units', 16, 256)))
        model.add(LeakyReLU(alpha=trial.suggest_float(f'leaky_alpha_dense_{i+1}', 0.01, 0.3)))
        model.add(Dropout(trial.suggest_float(f'dropout_dense{i+1}', 0.0, 0.5)))
    
    model.add(Dense(1))
    
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss=Huber())
    return model


def cosine_decay_with_warmup(epoch, total_epochs, warmup_epochs=5, learning_rate_base=0.001, learning_rate_min=1e-6):
    if epoch < warmup_epochs:
        return learning_rate_base * ((epoch + 1) / warmup_epochs)
    else:
        return learning_rate_min + 0.5 * (learning_rate_base - learning_rate_min) * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))


def objective(trial, X_scaled, y_scaled, scaler_y):
    seq_length = trial.suggest_int('seq_length', 3, 30)
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
    
    if len(X_seq) < 10:
        return 1e6
    
    model = create_model(trial, (seq_length, X_seq.shape[2]))
    
    n_splits = min(5, len(X_seq) - 1)
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=max(1, len(X_seq) // 10))
    
    cv_scores = []
    for train_index, val_index in tscv.split(X_seq):
        X_train, X_val = X_seq[train_index], X_seq[val_index]
        y_train, y_val = y_seq[train_index], y_seq[val_index]
        
        max_epochs = trial.suggest_int('max_epochs', 100, 500)
        patience = trial.suggest_int('patience', 10, 50)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=patience//2, min_lr=1e-6)
        
        try:
            history = model.fit(
                X_train, y_train,
                epochs=max_epochs,
                batch_size=trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, lr_scheduler],
                verbose=0
            )
            
            val_pred = model.predict(X_val)
            val_pred = scaler_y.inverse_transform(val_pred)
            y_val_inv = scaler_y.inverse_transform(y_val.reshape(-1, 1))
            rmse = np.sqrt(np.mean((val_pred - y_val_inv)**2))
            cv_scores.append(rmse)
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            return 1e6
    
    return np.mean(cv_scores)

def predict_future(model, last_sequence, scaler_X, scaler_y, num_days, best_seq_length, last_date):
    future_predictions = []
    current_sequence = last_sequence.copy()

    for i in range(num_days):
        next_date = last_date + pd.Timedelta(days=i+1)
        
        # Prepare the input for the next prediction
        next_features = prepare_next_features(current_sequence, next_date)
        next_features_scaled = scaler_X.transform(next_features.reshape(1, -1))
        
        # Make prediction
        next_pred = model.predict(current_sequence.reshape(1, best_seq_length, -1))
        future_predictions.append(next_pred[0, 0])
        
        # Update the sequence for the next prediction
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = next_features_scaled
    
    # Inverse transform the predictions
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler_y.inverse_transform(future_predictions)
    
    return future_predictions
 
def prepare_next_features(current_sequence, next_date):
    # Extract relevant features from the current sequence and create new ones for the next date
    # This is a placeholder - you'll need to adapt this based on your actual feature set
    last_passengers = current_sequence[-1, -1]  # Assuming the last column is the passenger count
    
    next_features = np.array([
        next_date.dayofweek,
        next_date.month,
        next_date.dayofyear,
        next_date.isocalendar().week,
        int(next_date.dayofweek in [5, 6]),
        next_date.day,
        last_passengers,  # lag_1
        current_sequence[-7, -1] if len(current_sequence) >= 7 else last_passengers,  # lag_7
        np.mean(current_sequence[-7:, -1]),  # rolling_mean_7
        np.std(current_sequence[-7:, -1]),  # rolling_std_7
        np.mean(current_sequence[-14:, -1]) if len(current_sequence) >= 14 else np.mean(current_sequence[:, -1]),  # rolling_mean_14
        np.std(current_sequence[-14:, -1]) if len(current_sequence) >= 14 else np.std(current_sequence[:, -1]),  # rolling_std_14
        np.mean(current_sequence[-30:, -1]) if len(current_sequence) >= 30 else np.mean(current_sequence[:, -1]),  # rolling_mean_30
        np.std(current_sequence[-30:, -1]) if len(current_sequence) >= 30 else np.std(current_sequence[:, -1]),  # rolling_std_30
    ])
    
    return next_features

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Explained Variance': ev
    }
    
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    return metrics

def cross_validate(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        cv_scores.append(evaluate_model(y_test, y_pred))
    
    return cv_scores

from scipy import stats

def handle_outliers(data, column='A_to_B Daily Passengers', threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    outliers = z_scores > threshold
    
    logging.info(f"Number of outliers detected: {np.sum(outliers)}")
    
    # Option 1: Remove outliers
    # data_clean = data[~outliers].copy()
    
    # Option 2: Cap outliers
    data_clean = data.copy()
    lower_bound = data[column].mean() - threshold * data[column].std()
    upper_bound = data[column].mean() + threshold * data[column].std()
    data_clean.loc[data_clean[column] < lower_bound, column] = lower_bound
    data_clean.loc[data_clean[column] > upper_bound, column] = upper_bound
    
    return data_clean

# Main function to process each OD pair
def process_od_pair(od_data, origin, destination):
    logging.info(f"\nProcessing OD pair: {origin} to {destination}")
    
    # Preprocess the data
    processed_data = preprocess_data(od_data)
    
    # Check if we have enough data after preprocessing
    if len(processed_data) < 2:  # You might want to set a higher threshold
        logging.warning(f"Insufficient data for {origin} to {destination} after preprocessing. Skipping this pair.")
        return
    
    # Extract features and target
    X, y = extract_features(processed_data)
    
    # Check if we have features and target after extraction
    if X.shape[0] == 0 or y.shape[0] == 0:
        logging.warning(f"No features or target data for {origin} to {destination} after extraction. Skipping this pair.")
        return
    
    # Use RobustScaler instead of MinMaxScaler
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    try:
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    except ValueError as e:
        logging.error(f"Error scaling data for {origin} to {destination}: {str(e)}")
        return
    
    # Optuna study
    study = optuna.create_study(direction='minimize')
    
    def obj_wrapper(trial):
        return objective(trial, X_scaled, y_scaled, scaler_y)
    
    study.optimize(obj_wrapper, n_trials=100)
    
    best_seq_length = study.best_params['seq_length']
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, best_seq_length)
    
    # Train the best model
    best_model = create_model(study.best_trial, (best_seq_length, X_seq.shape[2]))
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: cosine_decay_with_warmup(epoch, total_epochs=300)
    )
    
    history = best_model.fit(
        X_seq, y_seq,
        epochs=300,
        batch_size=study.best_params['batch_size'],
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    # Evaluate the model
    y_true = scaler_y.inverse_transform(y_seq)
    predictions = scaler_y.inverse_transform(best_model.predict(X_seq))
    rmse, mae, mape = evaluate_model(y_true, predictions)
    
    detailed_error_analysis(y_true, predictions, processed_data['Date'])

    # Sanitize origin and destination names
    safe_origin = sanitize_filename(origin)
    safe_destination = sanitize_filename(destination)

    # Save results
    save_dir = f'model/lstm/{safe_origin}_to_{safe_destination}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Saving feature importance
    importance_df = analyze_feature_importance(best_model, X)
    logging.info(f"Top 5 important features:\n{importance_df.head()}")
    # Write the top 5 important features to a file within the specified directory
    file_path = os.path.join(save_dir, "feature_importance.txt")
    importance_df.head().to_string(file_path, index=False)
    logging.info(f"\nTop 5 important features written to {file_path}")

    # Saving model
    save_model(best_model, os.path.join(save_dir, 'lstm_model.keras'))
    joblib.dump(scaler_X, os.path.join(save_dir, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(save_dir, 'scaler_y.pkl'))
    
    with open(os.path.join(save_dir, 'best_params.json'), 'w') as f:
        json.dump(study.best_params, f)
    
    # Save evaluation results
    with open(os.path.join(save_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"RMSE: {rmse}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"MAPE: {mape}\n")
    
    # Make future predictions
    last_sequence = X_seq[-1]
    # Save future predictions
    future_predictions = predict_future(best_model, last_sequence, scaler_X, scaler_y, 7, best_seq_length, processed_data['Date'].iloc[-1])

    # Log original data range
    original_min = processed_data['A_to_B Daily Passengers'].min()
    original_max = processed_data['A_to_B Daily Passengers'].max()
    logging.info(f"Original data range: {original_min} to {original_max}")

    # Log prediction range
    pred_min = future_predictions.min()
    pred_max = future_predictions.max()
    logging.info(f"Prediction range: {pred_min} to {pred_max}")

    # Sanity check
    if pred_min < 0 or pred_max > original_max * 1.5:
        logging.warning(f"Predictions may be unreasonable. Please check the model and scaling.")

    # Save future predictions
    future_dates = pd.date_range(start=processed_data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=7)
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Passengers': future_predictions.flatten()
    })
    future_df.to_csv(os.path.join(save_dir, 'future_predictions.csv'), index=False)

    logging.info(f"Results saved in {save_dir}")

def sanitize_filename(name):
    # Replace colons and spaces with underscores, and remove any other invalid characters
    return ''.join(char if char.isalnum() or char in ('_', '-') else '_' for char in name)

def detailed_error_analysis(y_true, y_pred, dates):
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    error_df = pd.DataFrame({
        'Date': dates,
        'True': y_true.flatten(),
        'Predicted': y_pred.flatten(),
        'Error': errors.flatten(),
        'Abs_Error': abs_errors.flatten()
    })
    error_df['DayOfWeek'] = error_df['Date'].dt.dayofweek
    error_df['Month'] = error_df['Date'].dt.month
    error_df['Year'] = error_df['Date'].dt.year
    
    # Analyze errors by day of week
    dow_errors = error_df.groupby('DayOfWeek')['Abs_Error'].mean().sort_values(ascending=False)
    logging.info(f"Mean Absolute Error by Day of Week:\n{dow_errors}")
    
    # Analyze errors by month
    month_errors = error_df.groupby('Month')['Abs_Error'].mean().sort_values(ascending=False)
    logging.info(f"Mean Absolute Error by Month:\n{month_errors}")
    
    # Analyze errors by year
    year_errors = error_df.groupby('Year')['Abs_Error'].mean().sort_values(ascending=False)
    logging.info(f"Mean Absolute Error by Year:\n{year_errors}")
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(error_df['Error'], kde=True)
    plt.title('Error Distribution')
    plt.savefig('error_distribution.png')
    plt.close()
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.savefig('actual_vs_predicted.png')
    plt.close()
    
    logging.info("Error analysis plots saved as 'error_distribution.png' and 'actual_vs_predicted.png'")

def analyze_feature_importance(model, feature_names):
    # Get the weights of the first layer
    weights = model.layers[0].get_weights()[0]
    
    # Calculate the mean absolute weight for each feature
    feature_importance = np.mean(np.abs(weights), axis=1)
    
    # Create a dataframe with feature names and their importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Plot the feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    logging.info("Feature importance plot saved as 'feature_importance.png'")
    return importance_df

def plot_feature_importance(model: tf.keras.Model, feature_names: list):
    try:
        weights = model.layers[0].get_weights()[0]
    except (IndexError, AttributeError):
        logging.error("Model does not have accessible weights for feature importance.")
        return
    
    importance = np.mean(np.abs(weights), axis=1)

    if len(importance) != len(feature_names):
        logging.error("Length of feature importance does not match number of features.")
        return

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

def main():
    # Load the CSV file with multiple OD pairs
    data = pd.read_csv(r'data\testing\high-traffic-pairs.csv')
    
    # Group the data by Origin and Destination
    od_groups = data.groupby(['Origin', 'Destination'])
    
    # Process each OD pair
    for (origin, destination), group_data in od_groups:
        try:
            process_od_pair(group_data, origin, destination)
        except Exception as e:
            logging.error(f"Error processing {origin} to {destination}: {str(e)}")
            continue

if __name__ == "__main__":
    main()