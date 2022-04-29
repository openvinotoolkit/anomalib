 # Anomalib
 - Talk about use cases where anomaly based tasks are useful.
	- The value of merits of anomaly based tasks in solving the problem
		- Less data on the issues, more normal things are present
	- Example use cases in the industry:
		-	Defect detection, security, health care
	- Talk about how that translates to value derived for business
- How is it solved today? What are the alternatives?
	-	Alternative approaches
- How does anomaly based tasks make solving this problem more valuable
for businesses?
- What role does or can Anomalib play here?
- How can one use Anomalib?
- Summary

## Background

Anomaly detection is the process of identifying anomalous items in a stream of input data. An example of a real-world anomaly detection problem is industrial defect detection, where the aim is to identify anomalous products in the output of a production line. Anomaly detection problems are characterized by two main challenges:

First of all, the anomalous items in the dataset can be extremely scarce. Take for example an industrial process which manufactures an LED part at a 98% yield rate. That means for every 2 anomaly datapoints, there are 98 images where no defects occur. Generating a balanced dataset given the high manufacturing yields of today is an arduous task that could require up to tens of thousands of images to be collected just to have a hundred images of defects.

Second, there is usually no clear definition of the anomalous class. Instead, anything that deviates from the normal class should be considered anomalous. Two given defect types might be very different visually, but in the context of the anomaly detection problem, both fall within the anomalous category. This heteregeneous nature of the anomalous class makes it challenging for machine learning model to learn an implicit representation of abnormality. To make things even worse, not all defect types may be known beforehand. During the deployment of our model we might encounter a new defect type that we have never seen before. How do we teach the model to identify this defect type that we have no examples of and might not even know exists?

AI Researchers over the years have created a subset of Machine Learning Algorithms to address these inherent challenges related to Anomaly Detection datasets. The gist of these algorithms is that they require only good images for the training dataset. Meanwhile, the bad images are used in the validation dataset, aiding in quantifying the accuracy of the model. By learning the normality, Anomalib can detect abnormalities in domains where defects are unknown. Anomalib implements the most recent anomaly detection techniques and is continuously updated with the latest State-of-the-Art algorithms

## Draft

Anomaly detection stands in contrast to a normal classification task which depends on both normal and abnormal images to get a good performance. In addition, classification task requires that both the classes be present in equal proportion. This requirement does not always hold true and is also not preferred in certain scenarios. A good manufacturing unit would produce defect only rarely, datapoints for certain illnesses are rare when compared to those of the healthy, and it is preferable to not have examples of certain security threats when training a model to detect such threats.

Anomaly detection aim to address these challenges by providing algorithms which work well on such imbalanced data, and can be trained in an unsupervised manner so that they can identify unseen anomalies.

There are a few approaches used to detect anomalies.

## TODO


Anomalib provides a comprehensive solution to address these needs. It ships with 7 state-of-the-art algorithms and provides utilities such as hyperparameter optimization to tune them to any businesses' needs. It is modular and can be easily extended to include more algorithms. It supports OpenVINO export and inference so that the trained models can be deployed on Intel hardware.

By ensuring the presence of top algorithms in the library, businesses can be assured that their applications are being solved by the best models. Due to Anomalib's open source nature, and public contributions, customers can expect new features with higher velocity, and be assured that bugs and issues are resolved quickly. It's open source nature also ensures transparency and thus will increase the trust of our customers.

To get started with Anomalib, all one needs to do is clone the GitHub repository or install it directly via pip.

```bash

```
