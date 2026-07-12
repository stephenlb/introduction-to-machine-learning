## Introduction to Machine Learning
## Introduction to Machine Learning
## Introduction to Machine Learning
import pandas
import numpy as np
import heater

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

    def load_data(self, path='./data/train.csv'):
        data = pandas.read_csv(path)
        max_age = 50
        ages = [(age > 0 and age or 20) / max_age for age in data['Age'].tolist()]
        genders = [gender == 'female' and 1 or -1 for gender in data['Sex'].tolist()]
        features = []
        for i in range(len(ages)):
            features.append([
                ages[i],
                genders[i],
            ])
        targets = [[survived] for survived in data['Survived'].tolist()]
        return np.array(features), np.array(targets)

    def generate_data_rocks(self, samples):
        dims = self.weights.shape[0]
        X = np.random.uniform(-3, 3, size=(samples, dims))
        true_weights = np.full((dims, 1), 2.0)
        true_bias = 1.0
        y = X @ true_weights + true_bias
        return X, y

def main():
    number_of_features = 2
    number_of_classes = 1 ## True Labels
    model = LinearRegression(
        number_of_features,
        number_of_classes,
    )
    print('model')
    print(model)
    print()

    features, targets = model.load_data()

    #features, targets = model.generate_data_rocks(40)
    print('features')
    print(features.shape)
    print(features[0:10])
    print()
    #heater.plot(features, theme='heatmap')

    print('targets')
    print(targets.shape)
    print(targets[10:15])
    print()

    predictions = model.predict(features)
    print('predictions')
    print(predictions[:5])
    print()

    model.train(features, targets)

    ## Test
    print("If you are 40 on the titantic, will you servive?")
    servived = model.predict([[5, -1],[10, -1],[20, -1],[40, -1]])
    print(servived)
    print(servived > 0.5)

main()
