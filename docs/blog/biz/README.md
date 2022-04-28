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

## Draft

Anomaly detection stands in contrast to a normal classification task which depends on both normal and abnormal images to get a good performance. In addition, classification task requires that both the classes be present in equal proportion. This requirement does not always hold true and is also not preferred in certain scenarios. A good manufacturing unit would produce defect only rarely, datapoints for certain illnesses are rare when compared to those of the healthy, and it is preferable to not have examples of certain security threats when training a model to detect such threats.

Anomaly detection aim to address these challenges by providing algorithms which work well on such imbalanced data, and can be trained in an unsupervised manner so that they can identify unseen anomalies.

There are a few approaches used to detect anomalies.

## TODO


Anomalib provides a comprehensive solution to address these needs. It ships with 8 state-of-the-art algorithms and provides utilities such as hyperparameter optimization to tune them any businesses' needs. It is modular and can be easily extended to include more algorithms. It supports OpenVINO export and inference so that the trained models can be deployed on Intel hardware.

With the ensuring the presence of top algorithms in the library, businesses can be assured that their applications are being solved by the best models. Due to Anomalib's open source nature, and public contributions, customers can expect new features with higher velocity, and be assured that bugs and issues are resolved quickly. It's open source nature also ensures transparency and thus will increase the trust of our customers.

To get started with Anomalib, all one needs to do is clone the GitHub repository or install it directly via pip.

```bash

```