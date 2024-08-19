import numpy as np
import pandas as pd

def init_params(input_size):
    W1 = np.random.rand(100, input_size) * 0.1
    b1 = np.random.rand(100, 1)
    W2 = np.random.rand(10, 100) * 0.1
    b2 = np.random.rand(10, 1)
    W3 = np.random.rand(1, 10) * 0.1
    b3 = np.random.rand(1, 1)
    return W1.astype(np.float64), b1.astype(np.float64), W2.astype(np.float64), b2.astype(np.float64), W3.astype(np.float64), b3.astype(np.float64)

def leaky_relu(Z):
    return np.maximum(0.01 * Z, Z)

def leaky_relu_derivative(Z):
    return np.where(Z > 0, 1, 0.01)

def dropout(A, keep_prob=0.5, training=True):
    if training:
        mask = np.random.binomial(1, keep_prob, size=A.shape) / keep_prob
        return A * mask
    else:
        return A

def forward(W1, b1, W2, b2, W3, b3, X, keep_prob, training=True):
    Z1 = W1.dot(X) + b1
    A1 = leaky_relu(Z1)
    A1 = dropout(A1, keep_prob, training=training)
    Z2 = W2.dot(A1) + b2
    A2 = leaky_relu(Z2)
    A2 = dropout(A2, keep_prob, training=training)
    Z3 = W3.dot(A2) + b3
    A3 = Z3
    return Z1, A1, Z2, A2, Z3, A3

def backward(X, Y, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, lambd):
    m = Y.size
    dZ3 = A3 - Y.reshape(1, -1)
    dW3 = 1 / m * (dZ3.dot(A2.T) + lambd * W3)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = W3.T.dot(dZ3) * leaky_relu_derivative(Z2)
    dW2 = 1 / m * (dZ2.dot(A1.T) + lambd * W2)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * leaky_relu_derivative(Z1)
    dW1 = 1 / m * (dZ1.dot(X.T) + lambd * W1)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1.astype(np.float64), db1.astype(np.float64), dW2.astype(np.float64), db2.astype(np.float64), dW3.astype(np.float64), db3.astype(np.float64)

def update(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha, beta1, beta2, epsilon, t):

    if 'mW1' not in globals():
        mW1 = np.zeros_like(W1)
        mb1 = np.zeros_like(b1)
        mW2 = np.zeros_like(W2)
        mb2 = np.zeros_like(b2)
        mW3 = np.zeros_like(W3)
        mb3 = np.zeros_like(b3)

        vW1 = np.zeros_like(W1)
        vb1 = np.zeros_like(b1)
        vW2 = np.zeros_like(W2)
        vb2 = np.zeros_like(b2)
        vW3 = np.zeros_like(W3)
        vb3 = np.zeros_like(b3)

    mW1 = beta1 * mW1 + (1 - beta1) * dW1
    mb1 = beta1 * mb1 + (1 - beta1) * db1
    mW2 = beta1 * mW2 + (1 - beta1) * dW2
    mb2 = beta1 * mb2 + (1 - beta1) * db2
    mW3 = beta1 * mW3 + (1 - beta1) * dW3
    mb3 = beta1 * mb3 + (1 - beta1) * db3

    vW1 = beta2 * vW1 + (1 - beta2) * (dW1 ** 2)
    vb1 = beta2 * vb1 + (1 - beta2) * (db1 ** 2)
    vW2 = beta2 * vW2 + (1 - beta2) * (dW2 ** 2)
    vb2 = beta2 * vb2 + (1 - beta2) * (db2 ** 2)
    vW3 = beta2 * vW3 + (1 - beta2) * (dW3 ** 2)
    vb3 = beta2 * vb3 + (1 - beta2) * (db3 ** 2)

    mW1_corrected = mW1 / (1 - beta1 ** t)
    mb1_corrected = mb1 / (1 - beta1 ** t)
    mW2_corrected = mW2 / (1 - beta1 ** t)
    mb2_corrected = mb2 / (1 - beta1 ** t)
    mW3_corrected = mW3 / (1 - beta1 ** t)
    mb3_corrected = mb3 / (1 - beta1 ** t)

    vW1_corrected = vW1 / (1 - beta2 ** t)
    vb1_corrected = vb1 / (1 - beta2 ** t)
    vW2_corrected = vW2 / (1 - beta2 ** t)
    vb2_corrected = vb2 / (1 - beta2 ** t)
    vW3_corrected = vW3 / (1 - beta2 ** t)
    vb3_corrected = vb3 / (1 - beta2 ** t)

    W1 -= alpha * mW1_corrected / (np.sqrt(vW1_corrected) + epsilon)
    b1 -= alpha * mb1_corrected / (np.sqrt(vb1_corrected) + epsilon)
    W2 -= alpha * mW2_corrected / (np.sqrt(vW2_corrected) + epsilon)
    b2 -= alpha * mb2_corrected / (np.sqrt(vb2_corrected) + epsilon)
    W3 -= alpha * mW3_corrected / (np.sqrt(vW3_corrected) + epsilon)
    b3 -= alpha * mb3_corrected / (np.sqrt(vb3_corrected) + epsilon)

    return W1, b1, W2, b2, W3, b3



def get_predictions(A3):
    return A3

def get_mse(predictions, Y):
    return np.mean((predictions - Y) ** 2)

def train(merged_data, alpha, epochs, lambd, batch_size, lr_decay, lr_decay_every, keep_prob):
    input_size = merged_data.shape[1] - 1
    W1, b1, W2, b2, W3, b3 = init_params(input_size)
    m, n = merged_data.shape
    train_losses = []

    for epoch in range(epochs):
        train_data = merged_data[0:m].T
        Y_train = train_data[0]
        X_train = train_data[1:n]
        epoch_loss = 0

        if epoch % lr_decay_every == 0 and epoch > 0:
            alpha = alpha * lr_decay

        for i in range(0, Y_train.size, batch_size):
            X_batch = X_train[:, i:i+batch_size]
            Y_batch = Y_train[i:i+batch_size]
            Z1, A1, Z2, A2, Z3, A3 = forward(W1, b1, W2, b2, W3, b3, X_batch, keep_prob, True)
            dW1, db1, dW2, db2, dW3, db3 = backward(X_batch, Y_batch, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, lambd)
            W1, b1, W2, b2, W3, b3 = update(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha, 0.9, 0.999, 1e-8, epoch+1)
        epoch_loss /= (Y_train.size // batch_size)
        train_losses.append(epoch_loss)
        #if epoch % 100 == 0:
            #print('Epoch:', epoch, 'MSE:', get_mse(get_predictions(A3), Y_batch))
    return W1, b1, W2, b2, W3, b3, train_losses

def make_predictions(X, W1, b1, W2, b2, W3, b3):
    Z1, A1, Z2, A2, Z3, A3 = forward(W1, b1, W2, b2, W3, b3, X, 1, False)
    return get_predictions(A3)

def get_accuracy(predictions, Y):
    rounded_predictions = np.round(predictions.astype(np.float64)).astype(int)
    return np.mean(rounded_predictions == Y)

def main():
    data = pd.read_csv('train_data.csv')
    label = pd.read_csv('train_label.csv')

    merged_data = pd.merge(label, data, left_index=True, right_index=True)
    columns = ['BEDS', 'BATH', 'PRICE', 'PROPERTYSQFT', 'TYPE']

    merged_data = merged_data[columns]

    merged_data = pd.get_dummies(merged_data, columns=['TYPE'], prefix='TYPE')
    merged_data['TYPE_unknown'] = 0
    type_set = set(col for col in merged_data.columns if col.startswith('TYPE_'))


    for col in merged_data.columns:
        if col == 'PRICE' or col == 'BATH' or col == 'PROPERTYSQFT':
            median_val = merged_data[col].median()
            q1 = merged_data[col].quantile(0.25)
            q3 = merged_data[col].quantile(0.75)
            iqr = q3 - q1
            merged_data[col] = (merged_data[col] - median_val) / iqr

    merged_data_array = np.array(merged_data)
    m, n = merged_data_array.shape
    train_data = merged_data_array[0:m].T

    test = pd.read_csv('test_data.csv')
    test = test[columns[1:]]
    test = pd.get_dummies(test, columns=['TYPE'], prefix='TYPE')
    test['TYPE_unknown'] = 0
    columns_to_drop = []
    for col in test.columns:
        if col.startswith('TYPE_') and col not in type_set:
            columns_to_drop.append(col)
            test['TYPE_unknown'] = np.where(test[col] == 1, 1, test['TYPE_unknown'])

    test = test.drop(columns=columns_to_drop)

    total_columns = set(merged_data.columns) | set(test.columns)

    for col in total_columns:
        if col not in test.columns and col != 'BEDS':
            test[col] = 0

    for col in test.columns:
        if col == 'PRICE' or col == 'BATH' or col == 'PROPERTYSQFT': 
            median_val = test[col].median()
            q1 = test[col].quantile(0.25)
            q3 = test[col].quantile(0.75)
            iqr = q3 - q1

            test[col] = (test[col] - median_val) / iqr

    test = np.array(test).T
    W1, b1, W2, b2, W3, b3, train_losses = train(merged_data=merged_data_array, alpha=0.00008, epochs=4500, lambd=0.000001, batch_size=100, lr_decay=1.0, lr_decay_every=50, keep_prob=1.0)
    predictions_test = make_predictions(test, W1, b1, W2, b2, W3, b3)
    predictions_test = predictions_test.flatten()
    output_df = pd.DataFrame({'BEDS': np.round(predictions_test.astype(np.float64)).astype(int)})
    output_df.to_csv('output.csv', index=False)

if __name__ == '__main__':
    main()
