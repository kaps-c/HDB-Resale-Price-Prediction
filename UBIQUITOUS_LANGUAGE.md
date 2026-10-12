# Project vocabulary

| Term | Meaning |
| --- | --- |
| Transaction | One recorded HDB resale row; identical attributes may describe different sales. |
| Target | The observed resale price in Singapore dollars (SGD). |
| Features | Property attributes and transaction month available to the model; never the target. |
| Training split | Transactions through December 2023 used to fit preprocessing and models. |
| Validation split | 2024 transactions used to compare configurations and stop neural training. |
| Test split | January–March 2025 transactions used for final historical evaluation. |
| Experiment | One model approach with a fixed configuration, split, seed, and recorded results. |
| Model bundle | Fitted preprocessing, model, and metadata needed to reproduce predictions. |
| MAE | Mean absolute prediction error, reported in SGD. |
| RMSE | Root mean squared prediction error in SGD; gives large errors more weight. |
| Feature-sequence LSTM | The original idea of treating feature columns as sequential inputs; not a time-series forecast. |
| Ridge | Linear regression with an L2 penalty that limits coefficient size. |
| Boosted trees | An additive sequence of decision trees correcting earlier prediction errors. |
| MLP | Multilayer perceptron: a feed-forward neural network for tabular inputs here. |
| Epoch | One pass through the training transactions. |
| R-squared | Error relative to predicting the evaluation-set mean; can be negative. |
