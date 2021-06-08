##### Introduction
From a given set of images, the task is to create a panorama.

##### Step 1: DataPreparationPipeline 
A pipeline with 3 stages was created to preprocess the data: load image stage, extract keypoints stage and compute descriptors stage. The output of one stage will be the input for another stage.

To improve the runtime, each stage is running on a separate thread. However, to control the number of threads lunched, DataPreparationPipeline takes as a parameter
the number of threads to be lunched and uses a semaphore.
###### Runtime using 1 thread(on my machine) ===> 2.69s
###### Runtime using 4 threads(on my machine) ===> 1.58s

All the details can be found in [DataPreparationPipeline class](https://github.com/iacobmarian95/panorama_generation/blob/main/include/panorma_generation/DataPreparationPipeline.h)


##### Step 2: Generate a panorama from a given set of images
Paper: http://matthewalunbrown.com/papers/ijcv2007.pdf
