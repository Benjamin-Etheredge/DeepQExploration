This repo contains my personal experimentation with DeepQ methods.
Currently working on adapting to Atari environments, using docker, and using unit tests.

Dockerfile should contain all the needed stuff.



All Deep-Q logic is implemented in a TensorFlow graph. This was done for speed and as a learning experience. This makes an ugly graph. I'm currenlty working on refactoring to make the code easier to work with and the graph cleaner.

Below is an image of network from TensorBoard.

![](https://raw.githubusercontent.com/Benjamin-Etheredge/DeepQExploration/master/images/all_enhancements.PNG)
