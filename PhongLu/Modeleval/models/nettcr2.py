"""Modern TensorFlow implementation of the official CDR3β NetTCR-2.0 CNN."""
from __future__ import annotations

import numpy as np
from tensorflow import keras

BLOSUM50 = {
    'A':(5,-2,-1,-2,-1,-1,-1,0,-2,-1,-2,-1,-1,-3,-1,1,0,-3,-2,0),'R':(-2,7,-1,-2,-4,1,0,-3,0,-4,-3,3,-2,-3,-3,-1,-1,-3,-1,-3),'N':(-1,-1,7,2,-2,0,0,0,1,-3,-4,0,-2,-4,-2,1,0,-4,-2,-3),'D':(-2,-2,2,8,-4,0,2,-1,-1,-4,-4,-1,-4,-5,-1,0,-1,-5,-3,-4),'C':(-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1),'Q':(-1,1,0,0,-3,7,2,-2,1,-3,-2,2,0,-4,-1,0,-1,-1,-1,-3),'E':(-1,0,0,2,-3,2,6,-3,0,-4,-3,1,-2,-3,-1,-1,-1,-3,-2,-3),'G':(0,-3,0,-1,-3,-2,-3,8,-2,-4,-4,-2,-3,-4,-2,0,-2,-3,-3,-4),'H':(-2,0,1,-1,-3,1,0,-2,10,-4,-3,0,-1,-1,-2,-1,-2,-3,2,-4),'I':(-1,-4,-3,-4,-2,-3,-4,-4,-4,5,2,-3,2,0,-3,-3,-1,-3,-1,4),'L':(-2,-3,-4,-4,-2,-2,-3,-4,-3,2,5,-3,3,1,-4,-3,-1,-2,-1,1),'K':(-1,3,0,-1,-3,2,1,-2,0,-3,-3,6,-2,-4,-1,0,-1,-3,-2,-3),'M':(-1,-2,-2,-4,-2,0,-2,-3,-1,2,3,-2,7,0,-3,-2,-1,-1,0,1),'F':(-3,-3,-4,-5,-2,-4,-3,-4,-1,0,1,-4,0,8,-4,-3,-2,1,4,-1),'P':(-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3),'S':(1,-1,1,0,-1,0,-1,0,-1,-3,-3,0,-2,-3,-1,5,2,-4,-2,-2),'T':(0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,2,5,-3,-2,0),'W':(-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1,1,-4,-4,-3,15,2,-3),'Y':(-2,-1,-2,-3,-3,-1,-1,-2,2,-1,-1,-2,0,4,-3,-2,-2,2,8,-1),'V':(0,-3,-3,-4,-1,-3,-3,-4,-4,4,1,-3,1,-1,-3,-2,0,-3,-1,5),
}

def encode(sequences, max_length: int) -> np.ndarray:
    result = np.zeros((len(sequences), max_length, 20), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for col, residue in enumerate(str(sequence)[:max_length]):
            result[row, col] = BLOSUM50[residue]
    return result

def build_model() -> keras.Model:
    """Matches official NetTCR-2.0 one-chain architecture: 5 CNN widths per input."""
    initializer = keras.initializers.GlorotNormal()
    cdr_in = keras.Input(shape=(30, 20), name="cdr3_beta")
    pep_in = keras.Input(shape=(9, 20), name="peptide")
    def tower(inputs):
        return keras.layers.Concatenate()([
            keras.layers.GlobalMaxPooling1D()(keras.layers.Conv1D(16, width, padding="same", activation="sigmoid", kernel_initializer=initializer)(inputs))
            for width in (1, 3, 5, 7, 9)
        ])
    features = keras.layers.Concatenate()([tower(cdr_in), tower(pep_in)])
    output = keras.layers.Dense(1, activation="sigmoid")(keras.layers.Dense(32, activation="sigmoid")(features))
    model = keras.Model([cdr_in, pep_in], output, name="nettcr2_cdr3beta")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])
    return model
