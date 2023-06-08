import numpy as np

class Perceptron: 
    def __init__(self, num_inputs):
        self.weights = np.zeros(num_inputs)     
        self.bias = 0 
    
    def predict(self, inputs): 
        total = np.dot(inputs, self.weights) + self.bias 
        return 1 if total > 0 else 0

p = Perceptron(4)
inputs = np.array([1, 1, 1, 1])
print(p.predict(inputs))

print("Rate the importance of the following factors from 1 (low) to 10 (high):")
weight_weather = float(input("Good weather: "))
weight_day = float(input("Weekend day: "))
weight_friends = float(input("Friends available: "))
weight_rating = float(input("High movie rating: "))
p.weights = np.array([weight_weather, weight_day, weight_friends, weight_rating])
p.bias = -25

prediction = p.predict(inputs)
if prediction == 1:
    print("Go to the cinema!")
else: 
    print("Don't go to the cinema.")