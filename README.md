# anns

Understanding Artificial Neural Networks

Started this after a long hiatus. 
Using online resources to understand the concept and implementation is hard when there is a lot of context switching.
Sticking to this one tonight.

##

## Building an ANN to predict Customer Churn

Stage 1 : DATA PRE-PROCESSING

        1.1 Import necessary ml libraries 
        1.2 Import Dataset
        1.3 Encode categorical data if any
        1.4 Do train-test split
        1.5 Do Feature Scaling
        
    
Stage 2: BUILDING THE ANN

        2.1 Import necessary nn libraries
        2.2 Initialize NN (define sequence of layers or define graph) 
        2.3 Compile the NN** 
        2.4 Fit the ANN to training set
        2.5 Check accuracy


Stage 3: MAKE PREDICTIONS

        3.1 Predict outcomes
        3.2 Construct confusion matrix
        3.3 Get accuracy
        3.4 make predictions
 

## Steps to train an ANN using Stochastic Gradient Descent (steps 2.2 and 2.3)

1. Randomly initialize weights to small numbers close to zero but not zero
2. Input first observation of dataset in the input layer, each feature in corresponding input node. 
   The number of input nodes = number of independent variables
3. Forward Propagation: from L to R the neurons are activated in a way that the impact of each neuron's actiavtion
   is limited by the weights. Propogate the activations until output is predicted
4. Compare predicted result to actual result. Measure generated error
5. Back Propagation: from R to L the error is back propagated. Update the weights according to how much they are 
   responsible for the error. The learning rate decide by how much the weights are updated
6. Repeat steps 1 to 5 and update the weights after a batch of observations (Batch Learning)
7. Redo more epochs
