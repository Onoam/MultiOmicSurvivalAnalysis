## Multi Omic Survival Analysis
### Introduction
For this project, we implement survival analysis algorithms for multi-omic cancer patient data from the TCGA dataset.
The data contains several genomic datasets for each of 8 types of cancers. Each dataset is an omic:
- MicroRNA expression 
- Gene expression
- DNA methylation

We implement a baseline algorithm by concatenating all datasets and learning a survival analysis predictor on it. Then we implement a multi-view learning algorithm that takes the different distributions in each omic into consideration when learning.

Finally, we implement a transfer learning algorithm, for learning on multiple cancers and predicting on the data of just one.

### Feature selection 
Because most of our models are computationally expensive, we put emphasis on proper feature selection.

As a first approach, we tried the “top-k variance” method, in which we kept the k features with highest variance within each omic (or in some cases, across all omics).  We chose k to be the highest value which still enabled reasonable computation time.

Unfortunately this method did not yield good model performance, often giving a concordance index of 0.4-0.6 on a holdout set (as well as in cross validation).  

Next, we trained a gradient boosted Cox-PH model (using the Scikit Survive library). The gradient boosted model trains many (100-10,000) small regression trees as base learners, with each tree predicting on the residual error of the previous ones.

The added bonus of this method is that the model performs feature selection as part of its training (by only using a small subset of features for each learner), and gives “feature-importance” values at the end of the learning process. We used this list to select features for our other models, by choosing the k features with highest importance values.

This method of selecting features yielded significantly improved results for all our models, as detailed in the next sections.

### Baseline

#### Methodology 
For each cancer type, we trained a CoxNet model by the following algorithm:
- Concatenate all three omics
- Remove 0-variance columns, and keep only:
  - Top-k most variance features 
  - Top-k otherwise selected features 
- Split data into 80% train, 20% holdout set
- Train a simple CoxPH  model (while trying different α values for L1 penalty) to obtain some initial trying values for α
-	For the top-k α values received in the previous step various L1/L2 penalty ratios:
-	Perform Random Search CV with all possible combinations of {alphas}X{L1/L2 ratio} on a CoxNet model, and keep the best estimator found 
-	Compute Cross Validation results for estimating C.I
-	Compute C.I on the holdout set 

#### Results 
Our gradient boosted model was trained and tested on the given folds:

Type	| CV C-index
---  | ---
brca	| 0.587
blca	| 0.515
hsnc	| 0.573
laml	| 0.629
lgg	| 0.885
luad	| 0.521


### Multi-view learning
#### Resources: 
•	Supervised graph clustering for cancer subtyping based on survival analysis and integration of multi-omic tumor data, researchgate
•	A Co-Regularization Approach to Semi-supervised Learning with Multiple Views. Link
We implement a multi-view learning approach, where each omic is a view. The simplest approach of concatenating the datasets ignores the different distributions in each omic, and may degrade performance. In the multi-view learning approach, two assumptions are utilized:
•	Compatibility: A predictor from each view would predict similar target values for most examples, as those of all other predictors.
•	Independence: The values of each view are independent (given the target value) of the value of all other views.
Therefore in this approach, we train a predictor all views simultaneously, along with a regularization term and a special co-regularization term. Our loss function then becomes:

#### Discussion
- Cox-net baseline achieved fairly good results in very minimal learning time (~1-2 minutes)
- Gradient boosting Feature selection performed a lot better than top variance feature selection


### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Onoam/MultiOmicSurvivalAnalysis/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
