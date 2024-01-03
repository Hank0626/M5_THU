import pandas as pd

ratio = .2

dl = pd.read_csv('submission_dl.csv')

lgbm = pd.read_csv('submission_lgbm.csv')

dl.iloc[dl.shape[0]//2:, 1:] = ratio * dl.iloc[dl.shape[0]//2:, 1:] + (1 - ratio) * lgbm.iloc[dl.shape[0]//2:, 1:]

dl.to_csv('submission_ensemble', index=False)