# KutumLab-T035: DREAM Challenge TARGET 2035 Submission

This repository contains the code and documentation for the KutumLab-T035 team's submission to the [**DREAM Challenge: TARGET 2035**](https://www.synapse.org/Synapse:syn65660836/wiki/631410). The goal of this challenge is to predict small molecule binders for the protein WDR91.

## Methodology

Our approach is divided into two main steps, each employing different strategies for hit prediction.

### Step 1: Fingerprint-based Ensemble Model

In the first step, we developed a machine learning pipeline based on molecular fingerprints.

- **Feature Engineering**: We selected the top 500 most variable features from 9 different molecular fingerprint types. These were combined and scaled to create the final feature set.
- **Model Training**: A weighted XGBoost classifier was trained using a 5-fold cross-validation strategy to handle class imbalance and build robust models.
- **Prediction**: The final score for each compound was the average predicted probability from the five models in the ensemble.

For a detailed explanation, please see the [Step 1 Write-up](./STEP-1/README.md).

### Step 2: Max-Probability Ensemble and Embedding Closeness

For the second step, we submitted two distinct predictions:

1. **Max-Probability Ensemble**: This approach used the same 5-fold XGBoost ensemble from Step 1, but the final score was determined by the *maximum* predicted probability across the five models, representing the highest confidence prediction.
2. **Molformer Embedding Closeness**: This method used a pre-trained Molformer model to generate embeddings for test compounds and 14 known reference ligands. The final score was the maximum cosine similarity of a test compound to any of the reference ligands.

For a detailed explanation, please see the [Step 2 Write-up](./STEP-2/README.md).

## Repository Structure

```bash
.
├── LICENSE
├── README.md
├── STEP-1/               # Notebooks and write-up for Step 1 submission.
├── STEP-2/               # Notebooks and write-up for Step 2 submissions.
└── utils/                # Utility functions for data processing and analysis.
```

## Authors

The KutumLab-T035 team consists of the following members:

- Gautam Ahuja [1,2,*]
- Rik Ganguly [3,*]
- Zonunmawia [2,4]
- Bableen Kaur [2]
- Sagarika Toor [5]
- Vinita Sharma [5]
- Aakansha Rai [2]
- Rintu Kutum [1,2,3,6,#]

Affiliations:

- [*] Equal contribution
- [#] Corresponding author
- [1] Department of Computer Science, Ashoka University
- [2] Koita Centre for Digital Health - Ashoka (KCDH-A), Ashoka University
- [3] Mphasis AI & Applied Tech Lab at Ashoka, Ashoka University
- [4] International University of Applied Science, Bad Honnef, Germany
- [5] Department of Biology, Ashoka University
- [6] Trivedi School of Biosciences, Ashoka University

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
