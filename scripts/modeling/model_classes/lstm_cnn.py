import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate

# Example function to build the LSTM-CNN model
def build_fire_model(time_steps, seq_features, static_features, lstm_units=64, dense_units=32):
    # LSTM branch for time-series input
    seq_input = Input(shape=(time_steps, seq_features), name="time_series_input")
    # LSTM layers to process the sequence
    x = LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(seq_input)
    x = LSTM(lstm_units//2, dropout=0.2, recurrent_dropout=0.2)(x)  # second LSTM layer
    # (The second LSTM returns a final output vector.)

    # Dense branch for static geographical input
    static_input = Input(shape=(static_features,), name="static_input")
    # A couple of dense layers to learn interactions of static features
    s = Dense(dense_units, activation='relu')(static_input)
    s = Dense(dense_units//2, activation='relu')(s)

    # Concatenate LSTM output and static feature output
    combined = Concatenate(axis=1)([x, s])
    # Final dense layers for classification
    combined = Dense(dense_units, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[seq_input, static_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# Example usage:
TIME_STEPS = 14   # using 14-day window for example
SEQ_FEATURES = 6  # ['2t','2d','10u','10v','sp','tp','lightning_count','...'] etc (for demo, assume 6 here)
STATIC_FEATURES = 8  # ['lat','lon','railway_count',... etc]

model = build_fire_model(TIME_STEPS, SEQ_FEATURES, STATIC_FEATURES)
model.summary()  # Print model architecture

# (Data preparation would go here: loading features and labels into numpy arrays, scaling them, etc.)
# Assume X_seq (num_samples, TIME_STEPS, SEQ_FEATURES) and X_static (num_samples, STATIC_FEATURES), and y (num_samples,) are prepared.

# Split data into train, val, test (for example using indices or sklearn train_test_split)
# X_seq_train, X_static_train, y_train = ...
# X_seq_val, X_static_val, y_val = ...
# X_seq_test, X_static_test, y_test = ...

# Train the model
history = model.fit(
    [X_seq_train, X_static_train], y_train,
    validation_data=([X_seq_val, X_static_val], y_val),
    epochs=50, batch_size=128,
    class_weight={0: 1, 1: 5},  # example: weight fire class higher to handle imbalance
    callbacks=[...]  # e.g., EarlyStopping on val_loss or val_f1 (if defined)
)

# After training, evaluate on test set
y_pred_prob = model.predict([X_seq_test, X_static_test])
y_pred = (y_pred_prob >= 0.5).astype(int)  # using 0.5 threshold (could be tuned for best F1)
# Compute evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
