from data_preprocessing import load_and_preprocess_data
from model import fake_news_detection_model
from train_evaluate import train_model, evaluate_model
import tensorflow as tf

# Check for GPU availability
def check_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs found: {len(gpus)}")
        for gpu in gpus:
            print(f"GPU Name: {gpu.name}")
        try:
            # Set memory growth if a GPU is found
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error in setting GPU configurations: {e}")
    else:
        print("No GPU found, running on CPU.")


# Step 1: Load and preprocess data
file_path = '../data/Truth_Seeker_Model_Dataset.csv'
X_train, X_test, y_train, y_test, max_length, tokenizer = load_and_preprocess_data(file_path)

# Step 2: Check for GPU
check_gpu()

# Step 3: Create the model
vocab_size = len(tokenizer.get_vocab())
embedding_dim = 512
lstm_units = 64
num_heads = 8
dropout_rate = 0.2
model = fake_news_detection_model(vocab_size, embedding_dim, lstm_units, num_heads, dropout_rate, max_length)

# Step 4: Train the model
model = train_model(model, X_train, y_train, X_test, y_test)

# Step 5: Evaluate the model
evaluate_model(model, X_test, y_test)
