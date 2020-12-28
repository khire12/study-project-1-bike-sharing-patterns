# Study-project-1-bike-sharing-patterns

This project work is a part of Udacity Deep learning Nano-degree.
<br>
### Task: To predict the how many bikes needed by the riders in the near future from the historical data. 
<br>
Implemented a simple neural network using NumPy.
The network architecture is similar to the architecture in the below diagram

<p align="center">
<img src="/first_neural_network/assets/neural_network.png" height="300" width="350">
</p>
  

Network Built has:
<br>
Input layer, 1 Hidden layer and one node in the output layer (for predicting the count of riders).
<br>The hidden layer has sigmoid activation function. There is no activation function in the output layer as we are predicting the count of bikes needed by the riders. 
The model is trained on 5000 epochs using simple batch gradient descent.

Markup: 1. Hyperparameters
            1. Hidden nodes: 12
            2. Output nodes: 1
            3. Batch: 128
            4. Iterations: 5000
            5. Learning rate: 1
        2. Results
            1. Training loss: 0.239 
            2. Validation loss: 0.415

<br>
<br>
Dataset used : Bike Sharing Dataset [1] developed by Hadi Fanaee-T at Laboratory of Artificial Intelligence and Decision Support (LIAAD), University of Porto
<br><br>
[1] Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.
