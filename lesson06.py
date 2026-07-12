## Introduction to Machine Learning
## Introduction to Machine Learning
## Introduction to Machine Learning
import numpy as np

## Prediting Revenue by Cohort and Industry
## Prediction Value Generator
class LinearRegression():
    ## Dims = Inputs = Number of intput feataures
    def __init__(self, dims, classes, learning_rate=0.02):
        self.weights = np.zeros((dims, classes))
        self.bias = 0.0
        self.learning_rate = 0.02

    def train(self, features, targets, epochs=500):
        for epoch in range(epochs):
            predictions = self.predict(features)
            weight_grad, bias_grad = self.gradients(
                features,
                targets,
                predictions
            )
            self.optimize(weight_grad, bias_grad)

            ## Every 10 epochs print the loss
            if epoch % 10 == 0:
                loss = self.loss(predictions, targets)
                print(f'{loss=:.2f}')

    ## Mean Error
    def gradients(self, features, targets, predictions):
        error = predictions - targets
        weight = features.T @ error / len(features)
        bias = error.mean(axis=0)
        return weight, bias

    ## Updates our paramaters so the model can leanr
    def optimize(self, weight, bias):
        self.weights -= weight * self.learning_rate
        self.bias -= bias * self.learning_rate

    def predict(self, inputs):
        return inputs @ self.weights + self.bias

    ## Calculate the error, so we can use this to train and correct the model
    def loss(self, prediction, targets):
        loss = prediction - targets
        return np.mean(loss ** 2)
        
    def __str__(self):
        return str(self.weights)#, self.bias

    def generate_data_rocks(self, samples):
        dims = self.weights.shape[0]
        X = np.random.uniform(-3, 3, size=(samples, dims))
        true_weights = np.full((dims, 1), 2.0)
        true_bias = 1.0
        y = X @ true_weights + true_bias
        return X, y

def main():
    number_of_features = 5
    number_of_classes = 1 ## True Labels
    model = LinearRegression(
        nunumber_of_features,
        number_of_classes,
    )
    print('model')
    print(model)
    print()

    features, targets = model.generate_data_rocks(2)
    print('features')
    print(features)
    print()

    print('targets')
    print(targets)
    print()

    predictions = model.predict(features)
    print('predictions')
    print(predictions)
    print()

    model.train(features, targets)

main()
