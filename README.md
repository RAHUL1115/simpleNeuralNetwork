# Simple Neural Network
This is a simple neural network module in JavaScript

# CDN
use Simele Neural Network module in yor html files via cdn.
this Nural Network library is dependent on numjs.

```
<script src="https://cdn.jsdelivr.net/gh/nicolaspanel/numjs@0.15.1/dist/numjs.min.js"></script>
<script src="https://raw.githubusercontent.com/RAHUL1115/simpleNeuralNetwork/main/nn.js"></script>
```

# Initialize
new NeuralNetwork(number_of_inputs, array_of_hidden_layers, number_of_outputs, learning_rate)
### Example
```
let nn = new NeuralNetwork(2,[4],1,0.1)
```

# Training
Neural Network Training
```
for(let i = 0; i < 1000; i++){
  nn.train(input_array, Actual_output);
}
```

# Predection
Neural Network Prediction 
```
nn.query(input_array);
```
