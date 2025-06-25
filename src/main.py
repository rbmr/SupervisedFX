from datetime import datetime

from keras.models import Sequential, load_model
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from src.constants import MODEL_DIR
from src.data.data import ForexCandleData, Timeframe
from src.data.feature_engineer import FeatureEngineer


def get_data():
    df = ForexCandleData.load(
        "dukascopy",
        "EURUSD",
        Timeframe.H1,
        datetime(2020, 1, 1, 22),
        datetime(2024,12,31,21)
    ).df

    fe = FeatureEngineer()



def create_and_train_supervised_model(X_data, y_data):
    """
    Creates, trains, and saves a supervised MLP model for predicting target exposure.
    """
    print("--- Starting Supervised Model Training ---")

    # --- 1. Data Preparation ---
    # We shuffle the entire dataset to ensure the model doesn't learn spurious chronological patterns.
    print("Shuffling data to destroy temporal order...")
    X_shuffled, y_shuffled = shuffle(X_data, y_data, random_state=42)

    # Split the shuffled data into training and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(
        X_shuffled, y_shuffled, test_size=0.3, random_state=42
    )
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # --- 2. Model Architecture ---
    # Based on your request and the findings from your paper.
    input_dim = X_train.shape[1]

    model = Sequential([
        InputLayer(input_shape=(input_dim,), name="input_layer"),
        Dense(32, activation='relu', name="hidden_layer_1"),
        Dense(32, activation='relu', name="hidden_layer_2"),
        Dense(1, activation='relu', name="output_exposure")
    ])

    model.summary()

    # --- 3. Model Compilation ---
    model.compile(
        optimizer=Adam(learning_rate=3e-4),  # Same lr as your SAC baseline
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    # --- 4. Model Training ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_checkpoint = ModelCheckpoint(MODEL_DIR / model_name, monitor='val_loss', save_best_only=True)

    print("\nStarting model training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,  # Set a high number, EarlyStopping will handle the rest
        batch_size=256,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    print("\nTraining finished.")
    # The best model is already saved by ModelCheckpoint.
    # You can also load it back with: tf.

    return model