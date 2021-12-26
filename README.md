# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset

The dataset used in this project is brought from the famous [dogs-vs-cats competition](https://www.kaggle.com/c/dogs-vs-cats).

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning

The model chosen in this project is a pretrained version of resnet50 as it is one of the most commonly used convolutional neural networks. It is trained on more than a million images from the [ImageNet](https://www.image-net.org/) database.

The hyperparameters that are optimized include batch-size, learning-rate and the number of epochs the training has to run through.

Here is the screenshot of the hyperparameters tuning job:
![image](https://user-images.githubusercontent.com/41271840/147415338-fbfc0f05-5705-4bef-bcb4-85f1af7af1e1.png)


Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs
 
## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
