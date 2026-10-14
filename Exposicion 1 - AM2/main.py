import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE    = 10000   
MAX_LEN       = 250     
EMBEDDING_DIM = 64      # Cada palabra = vector de 64 dimensiones
CELL_UNITS    = 64      
DENSE_NEURONS  = 32     # Neuronas en la capa oculta entre LSTM y la capa de salida
BATCH_SIZE    = 512     
EPOCHS        = 7

def plot_history(lstm_history, gru_history):
    _, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax_acc.plot(lstm_history.history['accuracy'],     label='LSTM Train')
    ax_acc.plot(lstm_history.history['val_accuracy'], label='LSTM Val', linestyle='--')
    ax_acc.plot(gru_history.history['accuracy'],      label='GRU Train')
    ax_acc.plot(gru_history.history['val_accuracy'],  label='GRU Val',  linestyle='--')
    ax_acc.set_title('Accuracy por Epoch')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()

    # Loss
    ax_loss.plot(lstm_history.history['loss'],     label='LSTM Train')
    ax_loss.plot(lstm_history.history['val_loss'], label='LSTM Val', linestyle='--')
    ax_loss.plot(gru_history.history['loss'],      label='GRU Train')
    ax_loss.plot(gru_history.history['val_loss'],  label='GRU Val',  linestyle='--')
    ax_loss.set_title('Loss por Epoch')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()
    print("\nGráfica guardada en training_history.png")

def train(model: Sequential, X_train, X_test, y_train, y_test):
    model.compile(
        optimizer='adam',               
        loss='binary_crossentropy',     # Pérdida para clasificación binaria
        metrics=['accuracy']            
    )

    model.summary()

    print("\nEntrenando...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,   # 20% del training set
        verbose=1
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    return { 'loss': loss, 'accuracy': accuracy, 'history': history }

def main():
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)


    print(f"Entrenamiento: {len(X_train)} reseñas")
    print(f"Prueba:        {len(X_test)} reseñas")
    print(f"Ejemplo de tokens:   {X_train[0][:10]}...") 

    # Truncar o rellenar con 0s al principio las reseñas
    X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding='pre', truncating='pre')
    X_test  = pad_sequences(X_test,  maxlen=MAX_LEN, padding='pre', truncating='pre')

    #Creacion del modelo LSTM
    lstm = Sequential([
        # input_dim  = tamaño del vocabulario
        # output_dim = tamaño del vector por palabra
        # input_length = largo de cada secuencia
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),

        # units = número de celdas de memoria.
        LSTM(units=CELL_UNITS),

        # Dense(DENSE_NEURONS, activation='relu'),
        Dropout(rate=0.2),

        # < 0.5 = negativo, >= 0.5 = positivo
        Dense(1, activation='sigmoid')
    ])
    #Creacion del modelo GRU
    gru = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        GRU(units=CELL_UNITS),
        # Dense(DENSE_NEURONS, activation='relu'),
        Dropout(rate=0.2),
        Dense(1, activation='sigmoid')
    ])

    lstm_metrics = train(lstm, X_train, X_test, y_train, y_test)
    gru_metrics  = train(gru,  X_train, X_test, y_train, y_test)

    print("#"*5, " Resultados ", "#"*5)
    print(f"Accuracy: LSTM = {lstm_metrics['accuracy']} | GRU = {gru_metrics['accuracy']}")
    print(f"Loss: LSTM = {lstm_metrics['loss']} | GRU = {gru_metrics['loss']}")

    plot_history(lstm_metrics['history'], gru_metrics['history'])

    

if __name__ == '__main__':
    main()