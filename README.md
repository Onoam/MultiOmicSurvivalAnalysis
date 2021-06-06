## Multi Omic Survival Analysis
### Introduction
For this project, we implement survival analysis algorithms for multi-omic cancer patient data from the TCGA dataset.
The data contains several genomic datasets for each of 8 types of cancers. Each dataset is an omic:
- MicroRNA expression 
- Gene expression
- DNA methylation

We implement a baseline algorithm by concatenating all datasets and learning a survival analysis predictor on it. Then we implement a multi-view learning algorithm that takes the different distributions in each omic into consideration when learning.

Finally, we implement a transfer learning algorithm, for learning on multiple cancers and predicting on the data of just one.

##Feature selection 
Because most of our models are computationally expensive, we put emphasis on proper feature selection.

As a first approach, we tried the “top-k variance” method, in which we kept the k features with highest variance within each omic (or in some cases, across all omics).  We chose k to be the highest value which still enabled reasonable computation time.

Unfortunately this method did not yield good model performance, often giving a concordance index of 0.4-0.6 on a holdout set (as well as in cross validation).  

Next, we trained a gradient boosted Cox-PH model (using the Scikit Survive library). The gradient boosted model trains many (100-10,000) small regression trees as base learners, with each tree predicting on the residual error of the previous ones.

The added bonus of this method is that the model performs feature selection as part of its training (by only using a small subset of features for each learner), and gives “feature-importance” values at the end of the learning process. We used this list to select features for our other models, by choosing the k features with highest importance values.

This method of selecting features yielded significantly improved results for all our models, as detailed in the next sections.

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
