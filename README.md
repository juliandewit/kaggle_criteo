Kaggle Criteo
=============

This is the software for the kaggle criteo challenge that ended up at the 4th place.
The algorithm is a GPU based "deep" neural network. Multiple models were trained using varied types of preprocessing and network architectures. The models were averaged into one final model.

##### The software
The software is written in c# .Net framework 4.5 using VS 2012.
GPU programming is done using [Cudafy.NET](https://cudafy.codeplex.com/) library. (CUDA 5.X)

The neural network is an ongoing project of mine and I'm still shaping and moving and renaming the code to make it sensible for me. The version in this solution is a subset of all the functionalities. Only the moving parts were put in this repository.

##### Compiling
If you want to compile the project you should install the NVIDIA Cuda SDK first.
Next to this you need VS2012. VS2013 *might* work. Although not completely necessary you can install the Cudafy.NET library.
With the cudafy tools you can check if everything is in working order. If so, compiling the project should not give major difficulties.

##### Running
First of all you need a 64bit windows OS.
The project was done on a workstation with 32Gb memory. Perhaps it will still run with a little bit less.
Propably 16Gb is not enough. The problem is that training is done with the complete dataset in memory for high throughput.
Last but not least you need a recent GPU board with cuda compute capability 3.




