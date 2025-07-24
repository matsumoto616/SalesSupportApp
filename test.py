#%%
import pandas as pd


# %%
df = pd.read_csv("./db/companies_archive_rev.csv")
df.to_csv("./db/companies_archive_rev.csv", encoding="utf-8-sig", index=False)
# %%
