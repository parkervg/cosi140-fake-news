# COSI-140 Fake News Project

Given a claim and a full-text verdict, have annotators determine which category/categories apply to the 'fake' claim.
  - Fake by contradictory quote
    - i.e. 'Donald Trump is not against marriage equality', disproven by referencing a quote where Trump states 'I'm opposed to gay marriage.'
  - Fake by contradictory qualitative data
    - i.e. 'No more than 1,000 Americans have died from COVID', disproven by referencing a table on the CDC website of ~540k deaths.
  - Fake by exaggeration
    - i.e. 'Joe Biden wants to immediately eliminate all fracking' stems from a valid truth proposition ('Joe Biden wants to shift toward clean energy'), but is false due to the statements' use of superlatives.
  - Fake by lack of evidence
    - i.e. those claims that have no direct contradiction, but are ungrounded in any type of reasonable(?) truth. 'Ostriches have learned to speak Russian.'
  - Fake by a dubious reference
    - Citing a statistic/research paper with questionable methodology
  - Bad Math
    - Not taking the whole picture into account

### Other Relevant Papers:

[5 Shades of Truth](https://www.researchgate.net/profile/Gerard-De-Melo-2/publication/328520326_Five_Shades_of_Untruth_Finer-Grained_Classification_of_Fake_News/links/5c6c8e354585156b570a963e/Five-Shades-of-Untruth-Finer-Grained-Classification-of-Fake-News.pdf?_sg%5B0%5D=5LaGTUBszztTcHV51rpPZ3VspLCduuXIzXqAYdThvKPU4JiHZ2P6_wCp4Vp2aSP0XpoFnoPLiRmNDrsxSmeCmg.oEcbpu7bq-k3ytZPxi6_V_Hpcxixt6sJwIlABdI7-uLsP8SY7Od1ldmYrGw9pCvexvqSkP-OpcdwfJ5z25Z3ZA&_sg%5B1%5D=VYjr4xbacZZP3D210viI1InCF_SXT6mK5V2HC-IPPv9uMaFynT_-8sv5CGd3Ljh9TxcJYu9C9rVT-LrBNez8_-k0hLqDdSHNprUwzi3UYqId.oEcbpu7bq-k3ytZPxi6_V_Hpcxixt6sJwIlABdI7-uLsP8SY7Od1ldmYrGw9pCvexvqSkP-OpcdwfJ5z25Z3ZA&_iepl=)
  - Still based on this arbitrary ordinal scale (1-5, 5 being 'most untrue')
  - Adds in categories of Factual, Propoganda, Hoax and Irony
    - Sort of a 'broader lens' than what we're proposing; classifying entire context of claim (might be possible with just URL) rather than types of contradictions in the claim itself.


### Using the Repo

To run the PyTorch models:
  - `python -m lib.LSTMClassifier`
  - `python -m lib.MLPClassifier`


### Results

![Label Distribution](visualizations/label_distribution.png?raw=true)
![Multilabel Instances](visualizations/multilabel_instances.png?raw=true)
![Label Correlation](visualizations/label_correlation.png?raw=true)



#### Naive Bayes feature_log_prob_ Exploration
Training one binary Naive Bayes classifier for each of the 7 classes yielded the followed feature log probabilities for the words. I just picked a couple that seemed to best reflect our intuition in understanding the fake news labels.


##### Evidence Lacking
- evidence
- government
- marijuana
- made
- thousands
- back
- border
- amendment
- mexico
- re
- women
- show
- china
- though
- using
- told

##### Quantitative Data
- increase
- population
- figure
- today
- made
- rates
- based
- though
- 10
- 2015
- point
- 2014
- office
- still
- 20
- average
- numbers
