##### Introduction
I am creating this project for fun. From a given set of images, the task is to create a panorama.

##### Step 1: DataPreparationPipeline 
A pipeline with 3 stages: load image, extract keypoints and compute descriptors. The output of one stage will be the input for another stage. To improve the runtime, each stage
is running on a separate thread. However, to control the number of threads lunched a Semaphore is used. 

All the details can found in [DataPreparationPipeline class](https://github.com/iacobmarian95/panorama_generation/include/panorama_generation/DataPreparationPipeline.h)


##### Step 2: Generate a panorama from a given set of images
http://matthewalunbrown.com/papers/ijcv2007.pdf
