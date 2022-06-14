## Using machine learning to improve system uptime in a particle accelerator

### Summary

In my Capstone project for my data science master's at UVA, my team and I worked with Jefferson Labs to improve system uptime at the Oak Ridge particle accelerator by assessing the likelihood of impending machine issues. We applied several ML techniques on electrical signal data collected during machine operation. **The project was a success**, and we found that:

1. Signal data is indeed promising for differentiating between future "normal" and "fault" states in many cases

2. Some types of machine issues are more detectable than others

3. There is a lot more exciting work to be done on this problem

*Note: This repository is designed to showcase my personal contributions to my Capstone project for my Master's program. The focus will be on my individual work, with some reference to approaches explored by my colleagues. Please consult the full paper (contained in repository) for a more in-depth discussion of my work as well as that of my colleagues. Additionally, there will be a guide to navigating the repository at the end of this readme.

### Business Case

Particle accelerators have pushed the boundaries of scientific knowledge for decades. In order to do this, however, they require incredibly complex systems of machinery. With structures this large and intricate, issues are inevitable, and can bring down ROI in financial and scientific terms.

The folks at Oak Ridge have been working on improving accelerator uptime for years through mechanical improvements, but are still limited by their ability to manage and triage issues only after they have already happened. Our team's work was the first step in taking things further by moving from reactive to preemptive interventions.

We investigated two main questions:
1. *Is there enough information in the electrical signal data to predict upcoming issues?*
2. *What are the pros and cons of various techniques?*

We looked specifically at one component of the High Voltage Converter Module (HVCM) electrical system, which collects electrical signal data "pulses" as the accelerator runs. Several business needs shaped our work.

1. **False positive rate** - Any future implementation would constantly monitor HVCM signal data and to determine the likelihood of impending issues and shut down the machine if a certain likelihood threshold was met. As such, a good model would strike a balance between shutting down the machine to prevent faults and avoiding costly false positives.

2. **Uncertainty** - While "normal" machine function presents a consistent pattern through the signal data, different failure states have different phenotypes (some much further from "normal" than others), and even then are not always internally consistent. Additionally, a good model would need to account for previously unseen failure states. These considerations led our team to favor approaches which not only provided a point estimate of failure likelihood, but a distribution of uncertainty. 

*Note: Our clients are also interested in predicting the likelihood of the specific type of failure. This project represented the first exploratory stage in pre-emptive fault detection - as such, we focused only on the binary problem, with fault type prediction to follow in later iterations.*

### Data Description

The Oak Ridge team provided us with 208 labeled signal observations – 158 from the “normal” class and 50 from the “fault” class, all coming from one module of the HVCM.

For observations where a fault did occur, the metadata also describes the type of fault that occurred. These labels describe the state of the HVCM in the pulse immediately following the observation, in accordance with our goal of preemptive detection. These observations consist of 6100 timestamp values for each of the 32 HVCM components, each taken at a rate of 1 sample every 400 ns. We only found 19 of the 32 components to be useful, yielding a $6100 x 26$ matrix for each observation.

![image](https://user-images.githubusercontent.com/90712577/172864866-29cb5035-c765-4bc4-8d72-7b67fa241663.png)

*An example of normal machine functioning (left) and a fault state (right)*

### Approach

#### Model Selection
Each member of the team selected one class of models to apply to the problem. I went with [Gaussian Process Classification](http://krasserm.github.io/2020/11/04/gaussian-processes-classification/) (GPC). Although this is not the most commonly used technique, it fit our client's needs - GPCs work well with small data sets and provide a distribution of uncertainty rather than a point estimate. Selfishly, it also provided me with the opportunity to independently research and implement a technique not included in the curriculum.

Essentially, GPCs consider the response variable for all model fitting (i.e. training) data as well as prediction (testing) data to represent a single instance of an $N + M$ dimensional Gaussian ($N$ fitting observations, $M$ prediction observations). The covariance matrix of this Gaussian is then comprised of the similarity value between the feature spaces of each observation as calculated by a kernel function. This structure produces a mean and variance for a posterior Gaussian for each prediction observation conditioned on the seen target observations used during fitting and the known covariance matrix.

![image](https://user-images.githubusercontent.com/90712577/173613928-d38ca040-5526-4998-8b53-c03e94f86934.png)

*Standard GPC equation*

**If you're not a stats expert, the major takeaways are as follows:**

1. GPCs assume that observations with similar input spaces will have a similar output response (i.e. observed pulses with similar shapes should have similar outcomes)

2. GPCs condition their predictions for unseen observations based on the relationships of the input and output spaces of fitting observations (i.e. if a new pulse is similar to pulses seen before a "fault" state, the predicted likelihood of "fault" will be high)

3. GPCs not only predict how likely an imminent issue will be, but how confident the model is in that likelihood (i.e. if the new pulse is very similar to many observed pulses, prediction confidence will be high; if the pulse is unlike anything the model has seen, confidence will be low)

#### Implementation

Since GPCs scale poorly with dimensionality, it was not feasible to use the full $6100 \times 19$ feature matrix as the input space for the kernel function. Instead, I separated the 19 HVCM components into 19 different models, and used downsampling by a factor of $10$ on the pulse from each of these. This resulted in an input space of $610 \times 1$ for each observation. Instead of using the raw pulse, a [Fourier transformation](https://en.wikipedia.org/wiki/Fourier_transform) was used to provide a more informative input space.

The 19 models produced a fault likelihood distribution for each of the 26 HVCM components, and these were combined into a single distribution. This provides a nice results, since the higher the certainty in a single component, the more it will contribute to the final distribution. This way, the "important" components for differentiation naturally have a greater impact for each observation.

![image](https://user-images.githubusercontent.com/90712577/173622077-44f85970-f0d9-4954-94c6-208ff8677d12.png)

*Sample posterior fault likelihood distribution - per component (left) vs. aggregated (right)*

### Results, Takeaways, and Opportunities for Future Work

We evaluated each technique using 5-fold cross validation, and found that the GPC outperformed other methods on the limited data set, detecting over 60% of issues at the 5% false positive threshold.

![image](https://user-images.githubusercontent.com/90712577/173624680-da21cf7c-7252-40f3-abad-333aa7de9138.png)

*ROC curve for GPC, showing true positive vs. false positive rate*

Interestingly, I observed a wide discrepancy in prediction accuracy based on fault type - with some issues, there is a clear signal degredation leading up to breakdown, while in others, everything looks fine until it's too late. For example, the "TPS Fault" category was correctly identified 100% of the time with no false positives, while the "XMTR fault" category was only caught 7% of the time with the same threshold. This certainly warrants further investigation, especially from someone who knows a lot more about physics and the meaning and context of the various fault types. 

While the GPC method scales poorly as the number of observations increases, creating a balanced subset with more examples of rare fault types has potential to balance accuracy and performance. This has positive implications for improving prediction accuracy for hard-to-detect fault types, and for moving forward to the multiclass problem.

Another major area for future work lies in developing a more sophisticated approach for modeling the relationships between the 19 HVCM components. The current implementation treats these as independent, so a more intentional approach, hopefully involving physics expertise, is likely to further improve performance.
