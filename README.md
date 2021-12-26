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
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
