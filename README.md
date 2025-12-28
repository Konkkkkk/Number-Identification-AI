# Number Identification AI

The project is a simple lab to implement an actual AI based on the expertise that I've learned so far by myself.
I simply want to learn the fundamental mechanisms of Neuron Networks and Deep Learning through this experiment.

## Functionality

Given an PNG image of size 28*28 of a handwritten number, the AI attempts to detect and recognize which number from 0 to 9 
the image contains. The AI is trained based on the public MNIST dataset, with an accuracy of approximately 97-98%.

## Usage

1. Create a `test` folder under the root directory `Number-Identification-AI`.
2. Place the image (PNG format, size 28*28) under the `test` folder.
3. Rename the image as `my_digit.png`
4. Run the script `train.py` first and then run the script `predict.py` to predict the number.