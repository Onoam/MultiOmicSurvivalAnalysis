## Multi Omic Survival Analysis
### Introduction
For this project, we implement survival analysis algorithms for multi-omic cancer patient data from the TCGA dataset.
The data contains several genomic datasets for each of 8 types of cancers. Each dataset is an omic:
- MicroRNA expression 
- Gene expression
- DNA methylation
We implement a baseline algorithm by concatenating all datasets and learning a survival analysis predictor on it. Then we implement a multi-view learning algorithm that takes the different distributions in each omic into consideration when learning.
For task 3, we implement a transfer learning algorithm, for learning on multiple cancers and predicting on the data of just one.

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
