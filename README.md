# 🌱 The AI-ESG Paradox in Europe

**🎓 MSc in Finance dissertation, Dublin City University**

> Do the AI and data tools behind ESG investing actually help investors, and what do they cost in emissions?

![Summary](AI-ESG%20Paradox%20Summary.png)

## 🔑 Key findings

- 📈 **The ESG-data industry is outgrowing the assets it serves.** MSCI's Sustainability & Climate revenue grew about 29% a year (2019-2024), against 5% for European assets under management.
- ⚖️ **No performance edge for ESG funds.** After adjusting for risk (Carhart four-factor model), Article 8/9 funds performed in line with conventional Article 6 funds: ahead in 2019-21, behind in 2022-24.
- 🤖 **No edge for "AI-intensive" funds either**, once funds that simply invest in AI stocks are set aside.
- 🌍 **Rising footprint.** Every major ESG-data provider became more carbon-efficient, yet emissions linked to MSCI's ESG business still rose 60%.
- 💸 **Investors bought the label.** Article 9 funds drew strong inflows until 2021, then steady outflows from 2022.

## 🗃️ Data

- **Bloomberg:** 3,917 Europe-domiciled equity funds (2,456 Article 8/9 and 1,461 Article 6), monthly returns, NAV and assets, 2019-2024; emissions of MSCI, S&P Global, Moody's, LSEG and Morningstar; MSCI segment revenue
- **Kenneth R. French Data Library:** developed-market Fama-French factors and momentum
- **EFAMA**, *Asset Management in Europe* (2023 and 2025 editions)
- **MSCI** SASB disclosures and **IEA** (2025), *Energy and AI*

## 🔬 Methods

Carhart four-factor alphas, a text-based AI Intensity Index built from fund descriptions, panel flow regressions with fund and month fixed effects, placebo tests and propensity-score matching, run in Python (pandas, NumPy, SciPy).

## 📂 Files

| File | Contents |
|---|---|
| 📄 `Suyash Sawant AI-ESG Thesis.pdf` | Full dissertation |
| 🐍 `Suyash Sawant AI-ESG Analysis.py` | Code that reproduces every table and figure |
| 📊 `Industry Dataset.xlsx` | EFAMA, MSCI, provider emissions and IEA data, with sources |
| 📈 `Risk Factors.csv` | Developed-market Fama-French factors and momentum, 2019-2024 |
| 🖼️ `AI-ESG Paradox Summary.png` | One-page summary of the results |

🔒 The two fund-level files (Fund Characteristics.csv and Fund Monthly Data.csv) are licensed Bloomberg data, so they are not published here. They are available on request for academic or review purposes.

## 📫 Contact
[LinkedIn](https://www.linkedin.com/in/suyashrch) · suyashrch@gmail.com
