# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The dataset used in this project is brought from the famous [dogs-vs-cats competition](https://www.kaggle.com/c/dogs-vs-cats). Since the dataset doesn't categorize the images into categories based on the image content, I have written a script so that there would be two directories (one for dogs and another for cats). Once that is done, a portion of the training dataset was moved to a validation dataset.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning

The model chosen in this project is a pretrained version of resnet50 as it is one of the most commonly used convolutional neural networks. It is trained on more than a million images from the [ImageNet](https://www.image-net.org/) database.

The hyperparameters that are optimized include batch-size, learning-rate and the number of epochs the training has to run through.

Here is the screenshot of the hyperparameters tuning job:
![19145999-00c0938296f789a62f724188a8f53e19](https://user-images.githubusercontent.com/41271840/147415378-9578efeb-d6c7-45fd-8e66-2245b781c203.png)

The best training job and hyperparameters of it are queried like this.
#Get the best estimators and the best HPs
best_estimator = tuner.best_estimator()

#Get the hyperparameters of the best trained model
best_estimator.hyperparameters()

This gives the following output:

2021-12-25 08:31:18 Starting - Preparing the instances for training
2021-12-25 08:31:18 Downloading - Downloading input data
2021-12-25 08:31:18 Training - Training image download completed. Training in progress.
2021-12-25 08:31:18 Uploading - Uploading generated training model
2021-12-25 08:31:18 Completed - Training job completed
{'_tuning_objective_metric': '"average test loss"',
 'batch-size': '"256"',
 'epochs': '4',
 'lr': '0.00390315210654023',
 'sagemaker_container_log_level': '20',
 'sagemaker_estimator_class_name': '"PyTorch"',
 'sagemaker_estimator_module': '"sagemaker.pytorch.estimator"',
 'sagemaker_job_name': '"pytorch-training-2021-12-25-08-03-55-769"',
 'sagemaker_program': '"train_model_.py"',
 'sagemaker_region': '"us-east-1"',
 'sagemaker_submit_directory': '"s3://sagemaker-us-east-1-807116804612/pytorch-training-2021-12-25-08-03-55-769/source/sourcedir.tar.gz"'}

The best hyperparameters are: {'epochs': 4, 'batch-size': 256, 'lr': 0.00390315210654023}
 
## Debugging and Profiling
Model debugging and profiling in sagemaker is achieved in sagemaker by defining the rules and profiles in the script defining the PyTorch estimator. The respective hooks for the training and testing phases are defined in the train_model.py script.

### Results
By enabling the SageMaker Debugger and Profiler, I was able to see different debugging information while the training job was running and the resource consumption data like CPU and GPU usage etc. 

## Model Deployment
The deployed model can be queried by creating a Predictor object by specifying the endpointname ("imageclassifier") as a parameter and the calling the predict() method on it providing a data (sample image). The image has to be preprocessed a bit to be consumed by the predictor.

Screenshot of the endpoint in service.
![image](https://user-images.githubusercontent.com/41271840/147416678-b8a0452a-19d4-46d5-b591-9086aff81814.png)

