import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dropout, GlobalMaxPooling1D, Dense, Concatenate
from tensorflow.keras import regularizers

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"),
             tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def fake_news_detection_model(vocab_size, embedding_dim, lstm_units, num_heads, dropout_rate, max_tweet_length):
    inputs = Input(shape=(max_tweet_length,))
    embeddings = Embedding(vocab_size, embedding_dim)(inputs)

    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(embeddings)
    drop_out = Dropout(0.2)(lstm_out)

    transformer_block = TransformerBlock(embed_dim=embedding_dim, num_heads=num_heads, ff_dim=embedding_dim * 4)
    transformer_output = transformer_block(embeddings, training= True)

    concatenated = Concatenate()([drop_out, transformer_output])
    global_pooling = GlobalMaxPooling1D()(concatenated)
    global_pooling = Dropout(0.2)(global_pooling)

    outputs = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(global_pooling)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
