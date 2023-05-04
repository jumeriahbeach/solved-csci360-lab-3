Download Link: https://assignmentchef.com/product/solved-csci360-lab-3
<br>
<ol>

 <li><strong>Problem Description</strong>: In this Lab, you will implement KNN with different distance metrics on a Breast Cancer data set. You can find the dataset at https://archive.ics.uci.edu/ml/datasets/Breast+Cancer</li>

</ol>

The goal is to predict from 9 features whether breast cancer patients had recurrence events or not. The features are all <em>categorical </em>features, in the sense that they have <em>levels</em>. For example, the levels of node-caps are yes and no. The levels of breastquad are left-up, left-low, right-up, right-low, and central. The dataset includes 201 instances of no recurrence and 85 instances of recurrence. You are given training data that has 160 instances of the no-recurrence class and 68 instances of the recurrence class.

Your task is to determine the so-called hyper-parameters of the algorithm, which are k and the distance metric used to determine the nearest neighbors.

<ol start="2">

 <li><strong>Data pre-processing</strong>: There are many steps that are needed for most datasets before a Machine Learning algorithm can be implemented on them. For this dataset:

  <ul>

   <li>First, replace missing values of each feature with the <em>mode </em>of that feature. For example, if the feature node-caps is missing in an instance, replace the missing value with whichever level of node-caps is more common in your data (yes or no?). In this data set, missing values are shown with ‘?’</li>

   <li>Dealing with categorical features: Machine Learning algorithms work with numeric values, so categorical features have to be converted to numeric features. Some categorical features are ordinal, which means there is an order among levels. An example of such a feature is inv-nodes. The smallest numbers of nodes were encoded to 0-2, while the largest were encoded to 36-39.</li>

  </ul></li>

</ol>

The ordinal features in this data set are: age, tumori-size, inv-nodes, and degmalig. Convert their levels to numbers. For example for inv-nodes, 0-2 should be converted to 1, 3-5 should be converted to 2, 6-8 should be encoded to 3, and so on.

On the other hand, some categorical features do not have any order. The features menopause, node-caps, breast, breast-quad, and irradiat are such features in this dataset. If such a feature has only two levels, its two levels will be converted to 0 and 1 in the data set. For example, node-cap=yes, becomes nodecap = 1. If the feature has more than one level, we replace it with one <em>auxilliary </em>binary feature for each level. For example, breast-quad has five levels: left-up, left-low, right-up, right-low, central. It will be replaced with five auxiliary features. Now, assume that the value of breast-quad is right-up for an instance. The values of those auxiliary features will be: left-up=0, left-low=0, right-up = 1, right-low=0, central=0. This way of converting the categorical features to numeric features is called one-hot encoding.

<ol start="3">

 <li><strong>Determining hyperparameters using a test set</strong>: We will use <em>l</em>-norm based distance:</li>

</ol>

in which <em>x<sub>j </sub></em>is the <em>j<sup>th </sup></em>numeric feature of instance <em>x </em>in the training set, and  is the <em>j<sup>th </sup></em>feature of the test point <em>x</em><sup>∗</sup>. We also use consider the edit distance. When <em>l </em>→ 0, <em>d<sub>l</sub></em>(<em>x,x</em><sup>∗</sup>) can be shown to be equal to the number of corresponding elements of <em>x </em>and <em>x</em><sup>∗ </sup>that are not equal. When <em>l </em>→ ∞ <em>d<sub>l</sub></em>(<em>x,x</em><sup>∗</sup>) can be shown to be equal to the maximum difference between elements of <em>x </em>and <em>x</em><sup>∗</sup>. Your code should use a function that a distance function distance(<em>L,x,x</em><sup>∗</sup>) that acts as follows:

distance(<em>L,x,x</em><sup>∗</sup>) =

<sup> </sup>edit distance = number of non-matching elements of <em>x </em>and <em>x</em><sup>∗                                                       </sup><em>L </em>= −1

<sup>                                                       </sup><em>/L</em>

<h1>L          ∈ {1,2,3,4,5,6} L = 1000</h1>

Note that the choice of <em>L </em>= −1 for the edit distance (whose code is given to you) is only symbolic to represent lim<em><sub>l</sub></em><sub>→0 </sub><em>d<sub>l</sub></em>(<em>x,x</em><sup>∗</sup>)

Consider <em>L </em>∈ {−1<em>,</em>1<em>,</em>2<em>,</em>3<em>,</em>4<em>,</em>5<em>,</em>6<em>,</em>1000} and <em>k </em>∈ {1<em>,</em>2<em>,</em>3<em>,…,</em>30}.

For each pair of (<em>k,L</em>), and for an instance <em>x</em><sup>∗ </sup>in the test set, determine the <em>k </em>nearest neighbors of <em>x</em><sup>∗ </sup>in the training set, using the distance metric distance(<em>L,x,x</em><sup>∗</sup>), and predict a label for <em>x</em><sup>∗ </sup>based on the majority of the labels of the nearest neighbors. Break possible ties in favor of the positive class, i.e., recurrence-events. Compare the majority vote label you predicted from the training set with the true label of <em>x</em><sup>∗</sup>. Next, calculate the percentage of misclassified points in your test set, i.e. the percentage of the labels in the test set that were predicted incorrectly. Summarize the results in a data structure and print them in a table. Find the <em>k </em>and <em>L </em>that result in the lowest misclassification rate in the test set.

Here is how your code will be graded:

<ul>

 <li>General soundness of code: 60 pts.</li>

 <li>Passing multiple test cases: 40 pts. The test cases will be based on different splits of the data into training and testing. Your code should return the correct (<em>k,L</em>). There might be more than one pair of (<em>k,L</em>) that have the lowest misclassification rate (or highest accuracy = 1-misclassification rate)</li>

</ul>

<ol start="4">

 <li>(20 pts <strong>Extra Credit</strong>, <strong>Weighted kNN</strong>):</li>

</ol>

Implement Weighted kNN, in which you should convert the labels of instances tino numeric labels (no-recurrence-events = 0 , re-currence-events=1), and do kNN like above, with the exception that instead of majority polling, you should use weighted majority polling, which calculates a score for each test point <em>x</em><sup>∗ </sup>based on its distances from its K-Nearest neighbors in the training set, <em>x</em><sup>(1)</sup><em>,x</em><sup>(2)</sup><em>,…,x</em><sup>(<em>k</em>)</sup>. Assume the numeric labels of those <em>k </em>nearest neighbors are <em>y</em><sub>1</sub><em>,y</em><sub>2</sub><em>,…,y<sub>k</sub></em>. Calculate the following score for <em>x</em><sup>∗</sup>:

[distance(<em>L,x</em><sup>(<em>i</em>)</sup><em>,x</em><sup>∗</sup>)]

[distance

in which you are giving more weight to the neighbors that are closer to <em>x</em><sup>∗</sup>. Here, we assume 0001. Then if <em>score</em>(<em>x</em><sup>∗</sup>) ≥ 0<em>.</em>5, you classify it in the positive (recurrenceevents) class, otherwise in the negative (no-recurrence-events). Everything else will be the same as vanilla kNN. Determine the best (<em>k,L</em>) based on test misclassification rate.