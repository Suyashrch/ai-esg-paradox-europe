# The AI-ESG Paradox in Europe - data analysis
# Suyash Sawant, MSc in Finance, DCU (FBA1029 Dissertation)
#
# This script reproduces all the tables and figures in my dissertation. To run it, keep these four
# files in the same folder as the script:
#   Industry Dataset.xlsx       EFAMA, MSCI, ESG-data provider and IEA figures
#   Fund Characteristics.csv    the 3,917 equity funds (SFDR class, domicile, fees, description)
#   Fund Monthly Data.csv       monthly returns, NAV and fund assets in EUR, 2019-2024
#   Risk Factors.csv            developed-market Fama-French factors and momentum (Ken French's website)
# and run:  python "Suyash Sawant AI-ESG Analysis.py"
#
# Packages: pandas, numpy, scipy, matplotlib, openpyxl.
# Results are saved in an "output" folder (Tables.xlsx, tables.md, results.txt and the charts).
# It takes about a minute, mostly because of the placebo test.

import os, re, json, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RNG = np.random.default_rng(2026)

# ---------------------------------------------------------------
# Part 1: helper functions (regressions, fixed effects, alphas, matching)
# ---------------------------------------------------------------

# OLS with robust (HC1) standard errors, or clustered ones if a cluster variable is given
def ols(y, X, names, cluster=None):
    y=np.asarray(y,float); X=np.asarray(X,float); n,k=X.shape
    XtXi=np.linalg.pinv(X.T@X); b=XtXi@X.T@y; e=y-X@b
    if cluster is None:
        V=XtXi@((X*e[:,None]).T@(X*e[:,None]))@XtXi*n/(n-k); df=n-k
    else:
        g=pd.factorize(np.asarray(cluster))[0]; G=g.max()+1; S=np.zeros((G,k)); np.add.at(S,g,X*e[:,None])
        V=XtXi@(S.T@S)@XtXi*(G/(G-1))*((n-1)/(n-k)); df=G-1
    se=np.sqrt(np.clip(np.diag(V),0,None)); t=np.divide(b,se,out=np.zeros_like(b),where=se>0)
    p=2*stats.t.sf(np.abs(t),df); r2=1-(e@e)/(((y-y.mean())**2).sum())
    return pd.DataFrame({"coef":b,"se":se,"t":t,"p":p},index=names), r2, n

# removes fixed effects (fund and/or month) by repeatedly subtracting group means
def demean(d, cols, fe):
    out=d[cols].astype(float).copy()
    for _ in range(200):
        prev=out.values.copy()
        for f in fe: out=out-out.groupby(d[f]).transform("mean")
        if np.max(np.abs(out.values-prev))<1e-9: break
    return out

# caps extreme values at the 1st and 99th percentiles
def wins(s, lo=.01, hi=.99): return s.clip(s.quantile(lo), s.quantile(hi))

# runs the Carhart four-factor model for each fund and returns its annual alpha, Sharpe ratio etc.
def alphas(p, factors=("MKT","SMB","HML","MOM"), min_obs=36, start=None, end=None):
    rows=[]
    d0=p.dropna(subset=["exret",*factors])
    if start: d0=d0[d0.date>=start]
    if end: d0=d0[d0.date<=end]
    for f,d in d0.groupby("fund"):
        if len(d)<min_obs: continue
        X=np.column_stack([np.ones(len(d)),d[list(factors)].values])
        r,r2,n=ols(d.exret,X,["a",*factors])
        rows.append(dict(fund=f,alpha=12*r.loc["a","coef"],alpha_t=r.loc["a","t"],beta=r.loc["MKT","coef"],
                         r2=r2,months=n,sharpe=np.sqrt(12)*d.exret.mean()/d.exret.std(),
                         vol=np.sqrt(12)*d.ret.std(),ret_ann=((1+d.ret/100).prod()**(12/n)-1)*100))
    return pd.DataFrame(rows)

def fund_level(u,p):
    first=p.dropna(subset=["aum"]).sort_values("date").groupby("fund").first()
    a=alphas(p).merge(u,on="fund")
    a["ln_aum0"]=a.fund.map(np.log(first.aum)); a["age0"]=a.fund.map(first.age)
    return a

# cross-sectional regression across funds
def xsec(a, y, xs, cluster=None):
    d=a.dropna(subset=[y,*xs]).copy()
    X=np.column_stack([np.ones(len(d))]+[d[x] for x in xs])
    return ols(d[y],X,["const",*xs],cluster=None if cluster is None else d[cluster].fillna("na"))

# equal-weighted monthly returns of ESG and Article 6 funds
def ew_portfolios(p):
    g=p.dropna(subset=["ret"]).groupby(["month","esg"]).ret.mean().unstack()
    g.columns=["art6","esg"]; g["diff"]=g.esg-g.art6
    f=p.groupby("month")[["MKT","SMB","HML","MOM","RF"]].first()
    return g.join(f)

# placebo test: shuffle the variable randomly 1,000 times and see how often the effect is as big as the real one
def placebo(a, xvar, xs, reps=1000, cluster=None):
    base,_,_=xsec(a,"alpha",[xvar,*xs]); real=base.loc[xvar,"coef"]
    d=a.dropna(subset=["alpha",xvar,*xs]).copy(); draws=[]
    for _ in range(reps):
        d["_p"]=RNG.permutation(d[xvar].values); r,_,_=xsec(d,"alpha",["_p",*xs]); draws.append(r.loc["_p","coef"])
    draws=np.array(draws); return real, np.mean(np.abs(draws)>=abs(real)), draws

def logit(y,X):
    b=np.zeros(X.shape[1])
    for _ in range(100):
        pr=1/(1+np.exp(-X@b)); W=pr*(1-pr); H=X.T@(X*W[:,None])+1e-6*np.eye(X.shape[1])
        s=np.linalg.solve(H,X.T@(y-pr)); b+=s
        if np.max(np.abs(s))<1e-9: break
    return 1/(1+np.exp(-X@b))

# propensity-score matching: pair each treated fund with the most similar control fund
def psm(a, treat, y, covs, caliper=0.05):
    d=a.dropna(subset=[treat,y,*covs]).reset_index(drop=True)
    X=np.column_stack([np.ones(len(d))]+[(d[c]-d[c].mean())/(d[c].std() or 1) for c in covs])
    d["ps"]=logit(d[treat].values.astype(float),X); T=d[d[treat]==1]; C=d[d[treat]==0]
    diffs=[];mc=[]
    for _,r in T.iterrows():
        j=(C.ps-r.ps).abs().idxmin()
        if abs(C.ps[j]-r.ps)<=caliper: diffs.append(r[y]-C.loc[j,y]); mc.append(j)
    diffs=np.array(diffs); bal=[]
    for c in covs:
        sd=np.sqrt((T[c].var()+C[c].var())/2) or 1
        bal.append((c,100*(T[c].mean()-C[c].mean())/sd,100*(T[c].mean()-C.loc[mc,c].mean())/sd))
    return diffs.mean(), diffs.std(ddof=1)/np.sqrt(len(diffs)), len(diffs), len(T), bal

# ---------------------------------------------------------------
# Part 2: load the fund data and build the AI Intensity Index
# ---------------------------------------------------------------
def load():
    u = pd.read_csv("Fund Characteristics.csv").rename(columns={"Ticker": "fund", "Fund name": "Name", "Fund type": "fund_type",
        "Launch date": "Inception", "SFDR classification": "sfdr", "Fund size at screening (EUR m)": "aum_now_m",
        "Expense ratio (%)": "expense_ratio", "Management company": "mgmt_co", "Benchmark": "benchmark",
        "Objective category": "objective", "Fund description": "description", "Regional focus": "geo_focus"})
    u["esg"] = u.sfdr.isin(["Article 8", "Article 9"]).astype(int); u["art9"] = (u.sfdr == "Article 9").astype(int)
    u["etf"] = (u.fund_type == "ETF").astype(int); u["Inception"] = pd.to_datetime(u.Inception, errors="coerce")
    u["expense_ratio"] = pd.to_numeric(u.expense_ratio, errors="coerce")
    # AI Intensity Index: I search each fund's name and Bloomberg description for three groups of words
    # (explicit AI = 3 points, quant process = 2, rules-based/factor = 1) and scale the total to 0-10. See Appendix C.
    txt = (u.Name.fillna("") + " " + u.description.fillna("")).str.lower()
    cats = {"ai_explicit": (3, r"\bai\b|artificial intelligence|artf intell|machine learn|deep learn|neural net|natural language|big data|data science|alternative data"),
            "quant_process": (2, r"quantitative|\bquant\b|\bqnt\b|systematic|algorithm|model-driven|model-based|proprietary model|\bqi\b"),
            "rules_factor": (1, r"rules-based|smart beta|multi-factor|\bfactor\b|enhanced index|\benh\b|optimi[sz]ed|robot|automation")}
    raw = 0
    for k, (w, pat) in cats.items():
        u[k] = txt.str.contains(pat, regex=True).astype(int); raw = raw + w * u[k]
    u["AII"] = 10 * raw / 6; u["ai_any"] = (raw > 0).astype(int); u["ai_core"] = ((u.ai_explicit + u.quant_process) > 0).astype(int)
    u["ai_themed"] = txt.str.contains(r"\bai\b|artificial intelligence|artf intell|robot|automation|big data", regex=True).astype(int)
    u["has_desc"] = u.description.notna().astype(int)
    u["lux"] = (u.Domicile == "Luxembourg").astype(int); u["irl"] = (u.Domicile == "Ireland").astype(int)
    # monthly returns and assets, merged with the risk factors
    p = pd.read_csv("Fund Monthly Data.csv").rename(columns={"Ticker": "fund", "Month end": "date", "Total return (%)": "ret",
                                                         "NAV (EUR)": "nav", "Total assets (EUR m)": "aum"})
    p["date"] = pd.to_datetime(p.date); p["month"] = p.date.dt.to_period("M")
    p = p.drop_duplicates(["fund", "month"]).sort_values(["fund", "month"])
    p.loc[p.ret.abs() > 50, "ret"] = np.nan        # 12 bad returns caused by share splits or mergers
    p.loc[p.aum <= 0, "aum"] = np.nan
    f = pd.read_csv("Risk Factors.csv").rename(columns={"Month": "month", "Market minus risk-free (%)": "MKT", "Size, SMB (%)": "SMB",
                                                    "Value, HML (%)": "HML", "Risk-free rate (%)": "RF", "Momentum, WML (%)": "MOM"})
    f["month"] = pd.PeriodIndex(f.month.astype(str), freq="M")
    p = p.merge(f, on="month", how="left").merge(u, on="fund", how="inner")
    g = p.groupby("fund")
    p["aum_lag"] = g.aum.shift(1)
    p["flow"] = 100 * (p.aum - p.aum_lag * (1 + p.ret / 100)) / p.aum_lag      # net flow, as in Sirri and Tufano (1998)
    p["exret"] = p.ret - p.RF
    p["ret12_lag"] = g.ret.transform(lambda s: (1 + s / 100).rolling(12, min_periods=10).apply(np.prod, raw=True).shift(1) * 100 - 100)
    p["ln_aum_lag"] = np.log(p.aum_lag); p["age"] = (p.date - p.Inception).dt.days / 365.25
    p["flow_w"] = wins(p.flow); p["post22"] = (p.date >= "2022-01-01").astype(int)
    return u, p

NAVY, GOLD, GREY, LIGHT, RED = "#1F3864", "#A67C2E", "#7F7F7F", "#8EA9DB", "#C0504D"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 10})
OUT = "output"; os.makedirs(OUT, exist_ok=True)
T, LOG, K = {}, [], {}          # tables, log lines, key numbers for the text
def log(s): print(s); LOG.append(s)
def table(key, title, header, rows, notes): T[key] = dict(title=title, header=header, rows=rows, notes=notes)
def fp(p): return "<0.001" if p < 0.001 else f"{p:.3f}"
def st(p): return "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
def cagr(a, b, n): return (b / a) ** (1 / n) - 1
def figsave(fig, name, src):
    fig.text(.01, .01, src, fontsize=6, color=GREY); fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(f"{OUT}/{name}.png", dpi=220); plt.close(fig)

def regtable(models, order, labels):
    # turns a list of regressions into a table: coefficient with stars, t-stat underneath
    rows = []
    for v in order:
        r1, r2_ = [labels.get(v, v)], [""]
        for res, *_ in models:
            if v in res.index:
                r1.append(f"{res.loc[v,'coef']:.3f}{st(res.loc[v,'p'])}"); r2_.append(f"({res.loc[v,'t']:.2f})")
            else: r1.append(""); r2_.append("")
        rows += [r1, r2_]
    extra_keys = []
    for m in models:
        for k_ in m[3]:
            if k_ not in extra_keys: extra_keys.append(k_)
    for k_ in extra_keys: rows.append([k_] + [m[3].get(k_, "") for m in models])
    rows.append(["Observations"] + [f"{m[2]:,}" for m in models])
    rows.append(["R-squared"] + [f"{m[1]:.3f}" for m in models])
    return rows

# ---------------------------------------------------------------
# Part 3: industry and country tests (H1-H4, Chapter 6)
# ---------------------------------------------------------------
X = pd.ExcelFile("Industry Dataset.xlsx")
def sheet(name, cols):
    d = pd.read_excel(X, name, header=2).dropna(how="all").rename(columns=cols)
    k = list(cols.values())[0]
    return d[d[k].notna() & ~d[k].astype(str).str.startswith("Source")]
A1 = sheet("AuM by country 2024", {"Country": "country", "AuM, end-2024 (EUR bn)": "aum_eur_bn_2024", "AuM as % of GDP": "aum_gdp_pct_2024", "EU member (1 = yes)": "eu_member"})
SF = sheet("SFDR funds by country 2024", {"Country": "country", "Article 8 funds (EUR bn)": "article8_eur_bn", "Article 9 funds (EUR bn)": "article9_eur_bn", "Article 9 reported (1 = yes)": "article9_reported"})
EVO = sheet("European AuM over time", {"Year": "year", "European AuM (EUR tn)": "total_aum_eur_tn"}).astype({"year": int}).set_index("year")["total_aum_eur_tn"]
AL = sheet("Asset allocation", {"Year": "year", "Equities (%)": "equities_pct", "Bonds (%)": "bonds_pct", "Cash and money market (%)": "cash_mm_pct", "Other assets (%)": "other_pct"}).astype({"year": int})
MS = sheet("MSCI staff (SASB)", {"Year": "year", "Full-time employees": "employees_ft"}).astype({"year": int}).set_index("year")
SEG = sheet("MSCI revenue by segment", {"Year": "year", "Index (USD m)": "index_rev", "Analytics (USD m)": "analytics_rev", "All other / private assets (USD m)": "all_other_rev", "Sustainability & Climate (USD m)": "sust_climate_rev"}).astype({"year": int}).set_index("year")
PR = sheet("ESG data providers", {"Company": "company", "Year": "year", "Scope 1 (thousand t CO2e)": "scope1_kt", "Scope 2, market-based (thousand t CO2e)": "scope2_market_kt", "Scope 3 (thousand t CO2e)": "scope3_kt", "Revenue (m, reporting currency)": "revenue_m_local"}).astype({"year": int})
PR["company"] = PR.company.map({"MSCI": "MSCI US Equity", "S&P Global": "SPGI US Equity", "Moody's": "MCO US Equity", "LSEG": "LSEG LN Equity", "Morningstar": "MORN US Equity"})
SRC_A = "Sources: EFAMA, Asset Management in Europe (2023, 2025 editions); MSCI SASB tables; Bloomberg."
SRC_B = "Source: Bloomberg (fund data, pulled 2 Nov 2026); Kenneth R. French Data Library (developed-market factors)."

m = A1.merge(SF, on="country"); m = m[m.eu_member == 1].copy()
m["gdp_x"] = m.aum_gdp_pct_2024 / 100
h1 = m[m.article9_reported == 1].copy()
h1["art9_pct"] = 100 * h1.article9_eur_bn / (h1.article8_eur_bn + h1.article9_eur_bn)
m["esg_bn"] = m.article8_eur_bn + m.article9_eur_bn.fillna(0)
rows = [[r.country, f"{r.aum_eur_bn_2024:,.0f}", f"{r.gdp_x:.2f}", f"{r.article8_eur_bn:,.0f}",
         (f"{r.article9_eur_bn:,.2f}".rstrip("0").rstrip(".") if r.article9_reported else "n.r."),
         (f"{100*r.article9_eur_bn/(r.article8_eur_bn+r.article9_eur_bn):.1f}%" if r.article9_reported else "n.r.")]
        for r in m.sort_values("aum_eur_bn_2024", ascending=False).itertuples()]
table("A1", "Country sample: asset-management scale and SFDR-labelled fund assets, end-2024",
      ["Country", "Total AuM (EUR bn)", "AuM / GDP (x)", "Article 8 (EUR bn)", "Article 9 (EUR bn)", "Article 9 share"], rows,
      "EU member states that appear in both EFAMA (2025) Annex A.1 and Exhibit 4.9. AuM/GDP is a ratio (1.81 = assets equal to "
      "181% of GDP). n.r. = Article 9 assets not reported by EFAMA for that country; these countries are used in H2 (Article 8 only) "
      "but not in H1. Turkey, Switzerland, the UK and Norway are excluded because SFDR does not apply or data are incomplete. " + SRC_A)
K["n_h1"], K["n_h2"] = len(h1), len(m)
ol = stats.linregress(h1.gdp_x, h1.art9_pct); pr_, pp = stats.pearsonr(h1.gdp_x, h1.art9_pct); sr, sp = stats.spearmanr(h1.gdp_x, h1.art9_pct)
boot = []
for _ in range(5000):
    i = RNG.integers(0, len(h1), len(h1))
    if h1.gdp_x.iloc[i].nunique() > 1: boot.append(stats.linregress(h1.gdp_x.iloc[i], h1.art9_pct.iloc[i]).slope)
lo, hi = np.percentile(boot, [2.5, 97.5]); loo = []
for c in h1.country:
    s = h1[h1.country != c]; r, p = stats.pearsonr(s.gdp_x, s.art9_pct)
    loo.append([f"Excluding {c}", f"{r:.3f}", fp(p), f"{stats.linregress(s.gdp_x, s.art9_pct).slope:.2f}"])
table("A2", "H1: capital-market depth and the Article 9 ('dark green') share, end-2024",
      ["Test", "r / R2", "p-value", "Slope"],
      [["Pearson correlation", f"{pr_:.3f}", fp(pp), ""], ["Spearman rank correlation", f"{sr:.3f}", fp(sp), ""],
       ["OLS slope (pp per 1.0x AuM/GDP)", f"R2 = {ol.rvalue**2:.3f}", fp(ol.pvalue), f"{ol.slope:.2f} (SE {ol.stderr:.2f})"],
       ["Bootstrap 95% CI for the slope (5,000 draws)", "", "", f"[{lo:.2f}, {hi:.2f}]"]] + loo,
      f"n = {len(h1)} EU countries with reported Article 9 assets. Dependent variable: Article 9 share of Article 8+9 fund assets (%).")
K.update(h1_r=pr_, h1_p=pp, h1_sr=sr, h1_sp=sp, h1_b=ol.slope, h1_se=ol.stderr, h1_r2=ol.rvalue**2, h1_lo=lo, h1_hi=hi,
         h1_loo_min=min(float(x[2].replace("<", "")) for x in loo), h1_loo_max=max(float(x[2].replace("<", "")) for x in loo))
log(f"H1 r={pr_:.3f} p={pp:.3f} slope={ol.slope:.2f}")
el = stats.linregress(np.log(m.aum_eur_bn_2024), np.log(m.esg_bn)); tu = (el.slope - 1) / el.stderr
pu = 2 * stats.t.sf(abs(tu), len(m) - 2); m["share_esg"] = 100 * m.esg_bn / m.aum_eur_bn_2024
el2 = stats.linregress(np.log(m.gdp_x), np.log(m.share_esg))
table("A3", "H2: do SFDR-labelled fund assets scale with national asset-management size? (end-2024)",
      ["Statistic", "Estimate", "SE", "p-value"],
      [["Elasticity of Article 8+9 assets to total AuM", f"{el.slope:.3f}", f"{el.stderr:.3f}", fp(el.pvalue)],
       ["Test: elasticity = 1 (proportional scaling)", f"t = {tu:.2f}", "", fp(pu)],
       ["R-squared", f"{el.rvalue**2:.3f}", "", ""],
       ["Elasticity of the ESG share of AuM to AuM/GDP", f"{el2.slope:.3f}", f"{el2.stderr:.3f}", fp(el2.pvalue)],
       ["Median Article 8+9 assets as % of total AuM", f"{m.share_esg.median():.1f}%", "", ""]],
      f"OLS in logs, n = {len(m)} EU countries. Article 9 assets are added where EFAMA reports them. An elasticity above 1 means "
      "ESG fund assets are proportionally more concentrated in the largest asset-management markets.")
K.update(h2_b=el.slope, h2_se=el.stderr, h2_p=el.pvalue, h2_t1=tu, h2_p1=pu, h2_r2=el.rvalue**2, h2_med=m.share_esg.median(),
         h2_b2=el2.slope, h2_p2=el2.pvalue)
log(f"H2 elasticity={el.slope:.3f} se={el.stderr:.3f} p(=1)={pu:.3f}")
yrs = range(2019, 2025); sc = SEG["sust_climate_rev"]; tot = SEG[["index_rev", "analytics_rev", "all_other_rev", "sust_climate_rev"]].sum(axis=1)
hc = MS["employees_ft"]; rows = []
for y in yrs:
    rows.append([y, f"{EVO[y]:.1f}", f"{100*EVO[y]/EVO[2019]:.0f}", f"{hc[y]:,.0f}" if y in hc else "n/a",
                 f"{tot[y]:,.0f}", f"{sc[y]:,.0f}", f"{100*sc[y]/sc[2019]:.0f}", f"{100*sc[y]/tot[y]:.1f}%"])
rows.append(["CAGR 2019-2024", f"{100*cagr(EVO[2019],EVO[2024],5):.1f}%", "", f"{100*cagr(hc[2020],hc[2024],4):.1f}% (2020-24)",
             f"{100*cagr(tot[2019],tot[2024],5):.1f}%", f"{100*cagr(sc[2019],sc[2024],5):.1f}%", "", ""])
table("A4", "H3: European AuM vs the ESG-data layer (MSCI), 2019-2024",
      ["Year", "European AuM (EUR tn)", "AuM index (2019=100)", "MSCI FT employees", "MSCI revenue (USD m)",
       "Sustainability & Climate revenue (USD m)", "S&C index (2019=100)", "S&C share of revenue"], rows,
      "European AuM from EFAMA (2025) Exhibit 1.1 (year-end, includes market valuation effects). MSCI headcount from its SASB tables "
      "(2020 as at 30 September). Segment revenue comes from MSCI's segment reporting via Bloomberg; the segment was called 'ESG and Climate' "
      "before 2023. MSCI is a global firm, so its growth also reflects non-European demand.")
K.update(aum_cagr=cagr(EVO[2019], EVO[2024], 5), sc_cagr=cagr(sc[2019], sc[2024], 5), tot_cagr=cagr(tot[2019], tot[2024], 5),
         hc_cagr=cagr(hc[2020], hc[2024], 4), sc19=sc[2019], sc24=sc[2024], sc_sh19=sc[2019]/tot[2019], sc_sh24=sc[2024]/tot[2024],
         aum19=EVO[2019], aum24=EVO[2024], aum22=EVO[2022], aum21=EVO[2021])
log(f"H3 AuM CAGR {100*K['aum_cagr']:.1f}%  S&C CAGR {100*K['sc_cagr']:.1f}%")
rows = []
for col, lab in [("bonds_pct", "Bonds"), ("equities_pct", "Equities"), ("other_pct", "Other / alternatives"), ("cash_mm_pct", "Cash / money market")]:
    l = stats.linregress(AL.year, AL[col]); rows.append([lab, f"{AL[col].iloc[0]:.1f}%", f"{AL[col].iloc[-1]:.1f}%",
                                                         f"{l.slope:+.2f}", f"{l.rvalue**2:.3f}", fp(l.pvalue)])
    K[f"h4_{col}"] = (l.slope, l.pvalue, l.rvalue**2)
table("A5", "H4: European asset allocation, 2018-2024", ["Asset class", "2018", "2024", "Trend (pp/year)", "R2", "p-value"], rows,
      "OLS of each asset class share of total European AuM on a linear trend, n = 7 years. EFAMA (2023) Exhibit 4.2 for 2018-2022 "
      "and EFAMA (2025) Exhibit 4.2 for 2023-2024; the overlapping years 2020-2022 agree in both editions.")
table("A6", "Context: the scale of data-centre emissions (IEA, Energy and AI, 2025)", ["Metric", "Value"],
      [["Global fuel-combustion CO2 emissions, 2024", "35,000 Mt"], ["Data-centre indirect CO2 emissions, 2024", "180 Mt (0.5%)"],
       ["Data-centre share by 2030, Base Case", "1.0%"], ["Data-centre share by 2030, Lift-Off Case", "1.4% (emissions about 2.5x)"],
       ["Potential reductions from wider AI adoption, 2035", "up to 1,400 Mt (not on track)"]],
      "The emissions multiple and the share multiple differ because total combustion emissions also change by 2030.")

# figures for Chapter 6
fig, ax = plt.subplots(figsize=(6.2, 4)); ax.scatter(h1.gdp_x, h1.art9_pct, s=55, color=NAVY, zorder=3)
xs = np.linspace(0, h1.gdp_x.max() + .1, 50); ax.plot(xs, ol.intercept + ol.slope * xs, color=GOLD, label=f"OLS fit (R² = {ol.rvalue**2:.2f}, n = {len(h1)})")
for r in h1.itertuples(): ax.annotate(r.country, (r.gdp_x, r.art9_pct), xytext=((4, -11) if r.country == "Portugal" else (-8, 6) if r.country == "Poland" else (4, 3)), textcoords="offset points", fontsize=7.5)
ax.set_xlabel("Asset-management AuM / GDP (x)"); ax.set_ylabel("Article 9 share of Article 8+9 assets (%)")
ax.set_title("Capital-market depth and the 'dark green' share, end-2024", loc="left"); ax.legend(frameon=False)
figsave(fig, "fig1_depth_vs_art9", SRC_A)
s2 = SF[SF.country.isin(m.country) | SF.country.isin(["Sweden", "Finland"])].copy(); s2["a9"] = s2.article9_eur_bn.fillna(0)
s2 = s2.sort_values("article8_eur_bn").tail(12)
fig, ax = plt.subplots(figsize=(6.2, 4)); ax.barh(s2.country, s2.article8_eur_bn, color=LIGHT, label="Article 8")
ax.barh(s2.country, s2.a9, left=s2.article8_eur_bn, color=NAVY, label="Article 9")
ax.set_xlabel("EUR billions"); ax.set_title("SFDR Article 8 and 9 fund assets managed, end-2024", loc="left")
ax.legend(frameon=False, loc="lower right"); figsave(fig, "fig2_sfdr_by_country", SRC_A)
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.plot(list(yrs), [100 * EVO[y] / EVO[2019] for y in yrs], marker="o", color=NAVY, label="European AuM")
ax.plot(list(yrs), [100 * tot[y] / tot[2019] for y in yrs], marker="^", color=GREY, label="MSCI total revenue")
ax.plot(list(yrs), [100 * sc[y] / sc[2019] for y in yrs], marker="s", color=GOLD, lw=2, label="MSCI Sustainability & Climate revenue")
ax.axhline(100, color=GREY, lw=.6, ls=":"); ax.set_ylabel("Index (2019 = 100)")
ax.set_title("European AuM vs the ESG-data layer (2019 = 100)", loc="left"); ax.legend(frameon=False)
figsave(fig, "fig3_aum_vs_esgdata", SRC_A)
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.stackplot(AL.year, AL.bonds_pct, AL.equities_pct, AL.other_pct, AL.cash_mm_pct, labels=["Bonds", "Equities", "Other / alternatives", "Cash"],
             colors=[NAVY, LIGHT, GOLD, "#D9D9D9"])
ax.set_ylabel("% of total AuM"); ax.set_ylim(0, 100); ax.set_title("European asset allocation, 2018-2024", loc="left")
ax.legend(frameon=False, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(.5, -.1)); figsave(fig, "fig4_allocation", SRC_A)

# ---------------------------------------------------------------
# Part 4: fund-level tests (H5-H8, Chapter 7)
# ---------------------------------------------------------------
u, p = load(); a = fund_level(u, p); a.to_csv(f"{OUT}/fund_level.csv", index=False)
e = a[a.esg == 1].copy()
K.update(n_funds=u.fund.nunique(), n_esg=int(u.esg.sum()), n_a6=int((u.esg == 0).sum()), n_art9=int(u.art9.sum()),
         n_alpha=len(a), n_alpha_esg=len(e), n_obs=int(p.ret.notna().sum()))
grp = {"Article 6": u.sfdr == "Article 6", "Article 8": u.sfdr == "Article 8", "Article 9": u.sfdr == "Article 9"}
pr0 = p.dropna(subset=["aum"]).sort_values("date").groupby("fund").last()
rows = []
for g_, msk in grp.items():
    f_ = u[msk]; fa = a[a.sfdr == g_]
    rows.append([g_, f"{len(f_):,}", f"{f_.aum_now_m.median():,.0f}", f"{100*f_.etf.mean():.0f}%",
                 f"{100*(f_.Domicile=='Luxembourg').mean():.0f}% / {100*(f_.Domicile=='Ireland').mean():.0f}%",
                 (f"{f_.expense_ratio.median():.2f}" if f_.expense_ratio.notna().any() else "n.a."),
                 (f"{100*f_.ai_any.mean():.1f}%" if f_.has_desc.any() else "n.a."), f"{fa.ret_ann.mean():.1f}%", f"{fa.vol.mean():.1f}%"])
table("B1", "Fund sample: Europe-domiciled equity funds by SFDR classification, 2019-2024",
      ["Group", "Funds", "Median AuM (EUR m)", "ETF share", "Luxembourg / Ireland", "Median TER (%)", "Data-intensive (AII > 0)",
       "Mean annual return", "Mean volatility"], rows,
      "Open-end funds and ETFs; primary share class; equity focus; domiciled in ten EU countries; incepted before 2019; assets above "
      "EUR 50m at the screening date. Returns are total returns in EUR (dividends reinvested), monthly, January 2019 to December 2024. "
      "Expense ratios and fund descriptions were collected for Article 8/9 funds only. Source: Bloomberg fund screen, 2 November 2026.")
fac = ["MKT", "SMB", "HML", "MOM"]; w = ew_portfolios(p); Xw = np.column_stack([np.ones(len(w)), w[fac]])
port = {}
for c in ["esg", "art6", "diff"]:
    y = w[c] - (w.RF if c != "diff" else 0); r, r2, n = ols(y, Xw, ["a", *fac]); port[c] = (r, r2, n)
art9w = p[p.art9 == 1].groupby("month").ret.mean(); a8w = p[(p.esg == 1) & (p.art9 == 0)].groupby("month").ret.mean()
d98 = (art9w - a8w).dropna(); r98, r298, n98 = ols(d98.values, Xw[:len(d98)], ["a", *fac]); port["9-8"] = (r98, r298, n98)
rows = []
for g_ in ["Article 6", "Article 8", "Article 9"]:
    fa = a[a.sfdr == g_]
    rows.append([g_, f"{len(fa):,}", f"{fa.alpha.mean():.2f}", f"{fa.alpha.median():.2f}", f"{100*(fa.alpha_t>1.96).mean():.1f}%",
                 f"{100*(fa.alpha_t<-1.96).mean():.1f}%", f"{fa.sharpe.mean():.2f}", f"{fa.beta.mean():.2f}"])
table("B2", "Risk-adjusted performance by SFDR group: Carhart (1997) four-factor alphas",
      ["Group", "Funds", "Mean alpha (% p.a.)", "Median alpha", "Share alpha t > 1.96", "Share alpha t < -1.96", "Mean Sharpe", "Mean market beta"],
      rows + [["Equal-weighted portfolio: Article 8/9 minus Article 6", "", f"{12*port['diff'][0].loc['a','coef']:.2f}", f"t = {port['diff'][0].loc['a','t']:.2f}",
               "", "", "", f"{port['diff'][0].loc['MKT','coef']:.2f}"],
              ["Equal-weighted portfolio: Article 9 minus Article 8", "", f"{12*r98.loc['a','coef']:.2f}", f"t = {r98.loc['a','t']:.2f}",
               "", "", "", f"{r98.loc['MKT','coef']:.2f}"]],
      "Each fund's monthly excess return in euros is regressed on the developed-market Fama-French factors and momentum "
      "(funds with at least 36 months). Alphas annualised (x12). The portfolio rows regress the monthly return difference between "
      "equal-weighted portfolios on the same factors (72 months). " + SRC_B)
K.update(alpha_a6=a[a.sfdr == "Article 6"].alpha.mean(), alpha_a8=a[a.sfdr == "Article 8"].alpha.mean(), alpha_a9=a[a.sfdr == "Article 9"].alpha.mean(),
         port_diff=12 * port["diff"][0].loc["a", "coef"], port_diff_t=port["diff"][0].loc["a", "t"],
         port_98=12 * r98.loc["a", "coef"], port_98_t=r98.loc["a", "t"], beta_mean=a.beta.mean(), r2_mean=a.r2.mean())
ctrl = ["ln_aum0", "age0", "expense_ratio", "etf", "art9", "lux", "irl"]
lab = {"AII": "AI Intensity Index (0-10)", "ai_any": "Data-intensive (AII > 0)", "ai_core": "AI or quantitative process",
       "ai_explicit": "Explicit AI / ML wording", "ln_aum0": "ln(AuM), Jan 2019", "age0": "Fund age (years)", "expense_ratio": "TER (%)",
       "etf": "ETF", "art9": "Article 9", "lux": "Luxembourg-domiciled", "irl": "Ireland-domiciled", "esg": "Article 8/9 (ESG)",
       "const": "Constant", "ret12_lag": "Past 12-month return", "ln_aum_lag": "ln(AuM), lagged", "age": "Fund age (years)",
       "esg_post": "ESG x 2022-24", "art9_post": "Article 9 x 2022-24", "aii_post": "AII x 2022-24"}
mods = []
for v in ["AII", "ai_any", "ai_core", "ai_explicit"]:
    r, r2, n = xsec(e, "alpha", [v] + ctrl, cluster="mgmt_co"); mods.append((r, r2, n, {"Sample": "Art. 8/9"}))
    K[f"h5_{v}"] = (r.loc[v, "coef"], r.loc[v, "t"], r.loc[v, "p"])
ex = e[e.ai_themed == 0]; r, r2, n = xsec(ex, "alpha", ["AII"] + ctrl, cluster="mgmt_co"); mods.append((r, r2, n, {"Sample": "excl. AI-themed"}))
K["h5_excl"] = (r.loc["AII", "coef"], r.loc["AII", "t"], r.loc["AII", "p"])
table("B3", "H5: AI intensity and risk-adjusted performance among Article 8/9 funds",
      ["Variable", "(1)", "(2)", "(3)", "(4)", "(5)"],
      regtable(mods, ["AII", "ai_any", "ai_core", "ai_explicit"] + ctrl + ["const"], lab),
      "Dependent variable: annualised Carhart alpha (% p.a.). OLS; t-statistics in parentheses based on standard errors clustered by "
      "management company. Column (5) drops the funds that invest in the AI or robotics theme (their holdings, not their process, "
      "are AI-related). *** p<0.01, ** p<0.05, * p<0.10.")
log(f"H5 AII {K['h5_AII']}, explicit {K['h5_ai_explicit']}, excl themed {K['h5_excl']}")
mods = []; xs6 = ["esg", "art9", "ln_aum0", "age0", "etf", "lux", "irl"]
r, r2, n = xsec(a, "alpha", xs6); mods.append((r, r2, n, {"Dependent variable": "Alpha", "Funds": "All"}))
K["h6_esg"] = (r.loc["esg", "coef"], r.loc["esg", "t"], r.loc["esg", "p"]); K["h6_art9"] = (r.loc["art9", "coef"], r.loc["art9", "t"], r.loc["art9", "p"])
r, r2, n = xsec(a, "sharpe", xs6); mods.append((r, r2, n, {"Dependent variable": "Sharpe", "Funds": "All"}))
K["h6_sharpe"] = (r.loc["esg", "coef"], r.loc["esg", "t"], r.loc["esg", "p"])
a_noetf = a[a.etf == 0]; r, r2, n = xsec(a_noetf, "alpha", [x for x in xs6 if x != "etf"]); mods.append((r, r2, n, {"Dependent variable": "Alpha", "Funds": "Active (no ETFs)"}))
K["h6_active"] = (r.loc["esg", "coef"], r.loc["esg", "t"], r.loc["esg", "p"])
table("B4", "H6: Article 8/9 funds versus conventional (Article 6) funds", ["Variable", "(1)", "(2)", "(3)"],
      regtable(mods, xs6 + ["const"], lab),
      "Cross-section of funds. Alpha = annualised Carhart alpha (% p.a.); Sharpe = annualised Sharpe ratio. HC1-robust t-statistics in "
      "parentheses (management company is not available for Article 6 funds).")
log(f"H6 esg {K['h6_esg']}")
d = p.dropna(subset=["flow_w", "ret12_lag", "ln_aum_lag", "age"]).copy()
d["post21"] = (d.date >= "2022-01-01").astype(int); d["esg_post"] = d.esg * d.post21; d["art9_post"] = d.art9 * d.post21
mods = []
c1 = ["flow_w", "ret12_lag", "ln_aum_lag", "age", "esg", "art9"]; dm = demean(d, c1, ["month"])
r, r2, n = ols(dm.flow_w, dm[c1[1:]].values, c1[1:], cluster=d.fund); mods.append((r, r2, n, {"Fixed effects": "Month", "Sample": "All funds"}))
K["h7_esg"] = (r.loc["esg", "coef"], r.loc["esg", "t"], r.loc["esg", "p"]); K["h7_art9"] = (r.loc["art9", "coef"], r.loc["art9", "t"], r.loc["art9", "p"])
K["h7_perf"] = (r.loc["ret12_lag", "coef"], r.loc["ret12_lag", "t"])
c2 = ["flow_w", "esg_post", "art9_post", "ret12_lag", "ln_aum_lag"]; dm = demean(d, c2, ["fund", "month"])
r, r2, n = ols(dm.flow_w, dm[c2[1:]].values, c2[1:], cluster=d.fund); mods.append((r, r2, n, {"Fixed effects": "Fund + month", "Sample": "All funds"}))
K["h7_esgpost"] = (r.loc["esg_post", "coef"], r.loc["esg_post", "t"], r.loc["esg_post", "p"]); K["h7_a9post"] = (r.loc["art9_post", "coef"], r.loc["art9_post", "t"], r.loc["art9_post", "p"])
de = d[d.esg == 1].dropna(subset=["expense_ratio"]).copy(); de["aii_post"] = de.AII * de.post21
c3 = ["flow_w", "AII", "ret12_lag", "ln_aum_lag", "age", "expense_ratio", "etf", "art9"]; dm = demean(de, c3, ["month"])
r, r2, n = ols(dm.flow_w, dm[c3[1:]].values, c3[1:], cluster=de.mgmt_co.fillna("na")); mods.append((r, r2, n, {"Fixed effects": "Month", "Sample": "Art. 8/9"}))
K["h7_aii"] = (r.loc["AII", "coef"], r.loc["AII", "t"], r.loc["AII", "p"])
c4 = ["flow_w", "aii_post", "ret12_lag", "ln_aum_lag"]; dm = demean(de, c4, ["fund", "month"])
r, r2, n = ols(dm.flow_w, dm[c4[1:]].values, c4[1:], cluster=de.mgmt_co.fillna("na")); mods.append((r, r2, n, {"Fixed effects": "Fund + month", "Sample": "Art. 8/9"}))
K["h7_aiipost"] = (r.loc["aii_post", "coef"], r.loc["aii_post", "t"], r.loc["aii_post", "p"])
table("B5", "H7: investor flows, ESG labels and AI intensity (fund-month panel)", ["Variable", "(1)", "(2)", "(3)", "(4)"],
      regtable(mods, ["esg", "art9", "esg_post", "art9_post", "AII", "aii_post", "ret12_lag", "ln_aum_lag", "age", "expense_ratio", "etf"], lab),
      "Dependent variable: monthly net flow, % of lagged assets (Sirri and Tufano, 1998), winsorised at 1%/99%. Post-2021 = 2022-2024. "
      "Fixed effects removed by within transformation; t-statistics clustered by fund (columns 1-2) or management company (3-4). "
      "Time-invariant fund characteristics are absorbed by fund fixed effects in columns (2) and (4).")
log(f"H7 art9 {K['h7_art9']} art9_post {K['h7_a9post']} aii_post {K['h7_aiipost']}")
fy = d.groupby([d.date.dt.year, "sfdr"]).flow_w.mean().unstack(); K["flows_year"] = fy.round(3).to_dict()
PR["total_kt"] = PR[["scope1_kt", "scope2_market_kt", "scope3_kt"]].sum(axis=1); PR["int"] = 1000 * PR.total_kt / PR.revenue_m_local
ms_ = PR[PR.company == "MSCI US Equity"].set_index("year"); ms_["sc_kt"] = ms_.total_kt * sc / tot
rows = [[y, f"{ms_.loc[y,'scope1_kt']+ms_.loc[y,'scope2_market_kt']:.2f}", f"{ms_.loc[y,'scope3_kt']:.1f}", f"{ms_.loc[y,'total_kt']:.1f}",
         f"{ms_.loc[y,'int']:.1f}", f"{100*sc[y]/tot[y]:.1f}%", f"{ms_.loc[y,'sc_kt']:.2f}"] for y in yrs]
nm = {"MSCI US Equity": "MSCI", "SPGI US Equity": "S&P Global", "MCO US Equity": "Moody's", "LSEG LN Equity": "LSEG", "MORN US Equity": "Morningstar"}
peer = PR.pivot(index="company", columns="year", values="int")
rows2 = [[nm[c_], f"{peer.loc[c_,2019]:.1f}", f"{peer.loc[c_,2024]:.1f}", f"{100*(peer.loc[c_,2024]/peer.loc[c_,2019]-1):+.0f}%"] for c_ in nm]
table("B6", "H8: emissions of the ESG-data layer: MSCI and the attributable footprint of its Sustainability & Climate business",
      ["Year", "MSCI Scope 1+2 (kt CO2e)", "Scope 3 (kt)", "Total (kt)", "Intensity (t per USD m revenue)", "S&C share of revenue", "Attributable to S&C (kt)"], rows,
      "Scope 1, Scope 2 (market-based) and Scope 3 emissions in thousand tonnes of CO2e, from Bloomberg. Emissions attributable to S&C = total "
      "emissions x the segment's revenue share (a revenue-based allocation, since MSCI does not report emissions by segment).")
table("B6b", "Emissions intensity of the five largest listed ESG-data and ratings providers (tCO2e per million of revenue)",
      ["Provider", "2019", "2024", "Change"], rows2,
      "Scope 1 + Scope 2 (market-based) + Scope 3 from Bloomberg; revenue in reporting currency (LSEG in GBP, others USD). LSEG's 2021 "
      "jump in revenue reflects the Refinitiv acquisition. Scope 3 boundaries differ between companies, so levels are not strictly comparable; "
      "changes over time are more informative.")
K.update(sc_kt19=ms_.loc[2019, "sc_kt"], sc_kt24=ms_.loc[2024, "sc_kt"], msci_int19=ms_.loc[2019, "int"], msci_int24=ms_.loc[2024, "int"],
         msci_tot19=ms_.loc[2019, "total_kt"], msci_tot24=ms_.loc[2024, "total_kt"], msci_s3share24=ms_.loc[2024, "scope3_kt"] / ms_.loc[2024, "total_kt"],
         peer_med_change=float(np.median([peer.loc[c_, 2024] / peer.loc[c_, 2019] - 1 for c_ in nm])))
log(f"H8 S&C attributable {K['sc_kt19']:.2f} -> {K['sc_kt24']:.2f} kt; MSCI intensity {K['msci_int19']:.1f} -> {K['msci_int24']:.1f}")

# ---------------------------------------------------------------
# Part 5: robustness checks (Chapter 8)
# ---------------------------------------------------------------
rob = []
a3 = alphas(p, factors=("MKT", "SMB", "HML")).merge(u, on="fund")
first = p.dropna(subset=["aum"]).sort_values("date").groupby("fund").first()
for df_ in (a3,): df_["ln_aum0"] = df_.fund.map(np.log(first.aum)); df_["age0"] = df_.fund.map(first.age)
r, _, n = xsec(a3[a3.esg == 1], "alpha", ["AII"] + ctrl, cluster="mgmt_co"); r6, _, n6 = xsec(a3, "alpha", xs6)
rob.append(["Fama-French 3-factor alphas (no momentum)", f"{r.loc['AII','coef']:.3f} ({r.loc['AII','t']:.2f})", f"{r6.loc['esg','coef']:.3f} ({r6.loc['esg','t']:.2f})"])
for lab_, s_, e_ in [("Sub-period 2019-2021", "2019-01-01", "2021-12-31"), ("Sub-period 2022-2024", "2022-01-01", "2024-12-31")]:
    ap = alphas(p, start=s_, end=e_, min_obs=30).merge(u, on="fund"); ap["ln_aum0"] = ap.fund.map(np.log(first.aum)); ap["age0"] = ap.fund.map(first.age)
    r, _, _ = xsec(ap[ap.esg == 1], "alpha", ["AII"] + ctrl, cluster="mgmt_co"); r6, _, _ = xsec(ap, "alpha", xs6)
    rob.append([lab_, f"{r.loc['AII','coef']:.3f} ({r.loc['AII','t']:.2f})", f"{r6.loc['esg','coef']:.3f} ({r6.loc['esg','t']:.2f})"])
    K[f"rob_{s_[:4]}"] = (r.loc["AII", "coef"], r.loc["AII", "t"], r6.loc["esg", "coef"], r6.loc["esg", "t"])
aw = a.copy(); lo_, hi_ = aw.alpha.quantile([.05, .95]); aw["alpha"] = aw.alpha.clip(lo_, hi_)
r, _, _ = xsec(aw[aw.esg == 1], "alpha", ["AII"] + ctrl, cluster="mgmt_co"); r6, _, _ = xsec(aw, "alpha", xs6)
rob.append(["Alphas winsorised at 5%/95%", f"{r.loc['AII','coef']:.3f} ({r.loc['AII','t']:.2f})", f"{r6.loc['esg','coef']:.3f} ({r6.loc['esg','t']:.2f})"])
r, _, _ = xsec(e, "alpha", ["AII"] + ctrl + ["n_geo"] if False else ["AII"] + ctrl, cluster=None)
rob.append(["HC1 standard errors instead of clustering", f"{r.loc['AII','coef']:.3f} ({r.loc['AII','t']:.2f})", "as main"])
eg = e.copy(); dummies = pd.get_dummies(eg.geo_focus.fillna("Unknown"), prefix="g", drop_first=True).astype(float)
top = dummies.columns[dummies.sum() >= 20]; eg = pd.concat([eg, dummies[top]], axis=1)
r, _, _ = xsec(eg, "alpha", ["AII"] + ctrl + list(top), cluster="mgmt_co")
rob.append(["Adding geographic-focus fixed effects (ESG funds)", f"{r.loc['AII','coef']:.3f} ({r.loc['AII','t']:.2f})", "n.a."])
K["rob_geo"] = (r.loc["AII", "coef"], r.loc["AII", "t"], r.loc["AII", "p"])
r, _, _ = xsec(eg[eg.ai_themed == 0], "alpha", ["AII"] + ctrl + list(top), cluster="mgmt_co")
rob.append(["Geographic FE and excluding AI-themed funds", f"{r.loc['AII','coef']:.3f} ({r.loc['AII','t']:.2f})", "n.a."])
K["rob_geo_ex"] = (r.loc["AII", "coef"], r.loc["AII", "t"], r.loc["AII", "p"])
real, pperm, draws = placebo(e, "AII", ctrl, reps=1000); K["placebo"] = (real, pperm)
rob.append(["Placebo: AII randomly reassigned (1,000 draws)", f"permutation p = {pperm:.3f}", ""])
real6, pperm6, _ = placebo(a, "esg", ["art9", "ln_aum0", "age0", "etf", "lux", "irl"], reps=1000); K["placebo6"] = (real6, pperm6)
rob[-1][2] = f"permutation p = {pperm6:.3f}"
att, se_, nm_, nt, bal = psm(e, "ai_core", "alpha", ["ln_aum0", "age0", "expense_ratio", "etf", "art9", "lux", "irl"])
rob.append(["Propensity-score matching: AI/quant process vs matched funds (ATT)", f"{att:.3f} (t = {att/se_:.2f}; {nm_} of {nt} matched)", ""])
K["psm"] = (att, att / se_, nm_, nt, max(abs(b[2]) for b in bal), max(abs(b[1]) for b in bal))
att6, se6, nm6, nt6, bal6 = psm(a.assign(lux=a.lux, irl=a.irl), "esg", "alpha", ["ln_aum0", "age0", "etf", "lux", "irl"], caliper=0.02)
rob[-1][2] = f"{att6:.3f} (t = {att6/se6:.2f}; {nm6:,} of {nt6:,} matched)"
K["psm6"] = (att6, att6 / se6, nm6, nt6, max(abs(b[2]) for b in bal6))
table("B7", "Robustness: alternative models, samples and placebo tests",
      ["Test", "H5: coefficient on AII (t)", "H6: coefficient on ESG (t)"], rob,
      "Each row repeats the main model (Table 7.2, column 1 for H5; Table 7.3, column 1 for H6) with one change. "
      "Matching: one-to-one nearest neighbour with replacement on the propensity score (caliper 0.05 for H5, 0.02 for H6). "
      f"After matching, the largest standardised difference in covariates is {K['psm'][4]:.1f}% (H5) and {K['psm6'][4]:.1f}% (H6).")
log(f"Robust: placebo p={pperm:.3f}, PSM ATT={att:.3f} t={att/se_:.2f}")
kw = [["AI / machine learning (weight 3)", "AI, artificial intelligence, machine learning, deep learning, neural network, natural language, big data, data science, alternative data", f"{int(e.ai_explicit.sum())}"],
      ["Quantitative / algorithmic process (weight 2)", "quantitative, quant, systematic, algorithm(ic), model-driven, model-based, proprietary model, QI", f"{int(e.quant_process.sum())}"],
      ["Rules-based / factor / automation (weight 1)", "rules-based, smart beta, multi-factor, factor, enhanced index, optimised, robotics, automation", f"{int(e.rules_factor.sum())}"]]
table("C1", "AI Intensity Index: keyword dictionary", ["Category", "Terms searched in fund name and Bloomberg description", "Article 8/9 funds flagged"], kw,
      "AII = 10 x (sum of the weights of the categories present) / 6, so it ranges from 0 to 10. Matching is case-insensitive with word "
      "boundaries for short terms. AI-themed funds (holdings in AI, robotics or big-data companies) are flagged separately for robustness.")
K["aii_mean"] = e.AII.mean(); K["aii_pos"] = int((e.AII > 0).sum())

# ---------------------------------------------------------------
# Part 6: charts for Chapter 7 and saving the output files
# ---------------------------------------------------------------
cum = (1 + w[["esg", "art6"]] / 100).cumprod() * 100; cum.index = cum.index.to_timestamp()
fig, ax = plt.subplots(figsize=(6.2, 4)); ax.plot(cum.index, cum.esg, color=GOLD, lw=1.8, label="Article 8/9 funds (equal-weighted)")
ax.plot(cum.index, cum.art6, color=NAVY, lw=1.4, label="Article 6 funds (equal-weighted)"); ax.set_ylabel("Growth of EUR 100")
ax.set_title("Cumulative returns: ESG vs conventional equity funds, 2019-2024", loc="left"); ax.legend(frameon=False); figsave(fig, "fig5_cumulative", SRC_B)
fig, ax = plt.subplots(figsize=(6.2, 4)); data_ = [a[a.sfdr == g_].alpha.clip(-15, 15) for g_ in ["Article 6", "Article 8", "Article 9"]]
bp = ax.boxplot(data_, tick_labels=["Article 6", "Article 8", "Article 9"], showfliers=False, patch_artist=True, widths=.5)
for b_, c_ in zip(bp["boxes"], [NAVY, LIGHT, GOLD]): b_.set_facecolor(c_); b_.set_alpha(.8)
ax.axhline(0, color=GREY, lw=.7, ls=":"); ax.set_ylabel("Carhart alpha (% p.a.)")
ax.set_title("Distribution of four-factor alphas by SFDR group", loc="left"); figsave(fig, "fig6_alphas", SRC_B)
fig, ax = plt.subplots(figsize=(6.2, 4)); xs_ = np.arange(len(fy)); wdt = .27
for i_, (g_, c_) in enumerate(zip(["Article 6", "Article 8", "Article 9"], [NAVY, LIGHT, GOLD])):
    ax.bar(xs_ + (i_ - 1) * wdt, fy[g_], wdt, color=c_, label=g_)
ax.set_xticks(xs_); ax.set_xticklabels(fy.index); ax.axhline(0, color=GREY, lw=.7); ax.set_ylabel("Mean monthly net flow (% of assets)")
ax.set_title("Investor flows by SFDR group, 2019-2024", loc="left"); ax.legend(frameon=False); figsave(fig, "fig7_flows", SRC_B)
fig, ax = plt.subplots(figsize=(6.2, 4)); ax2 = ax.twinx()
ax.bar(list(yrs), [ms_.loc[y, "sc_kt"] for y in yrs], color=GOLD, alpha=.85, label="Emissions attributable to MSCI S&C (kt CO2e, left)")
ax2.plot(list(yrs), [ms_.loc[y, "int"] for y in yrs], color=NAVY, marker="o", label="MSCI intensity (t per USD m, right)")
ax.set_ylabel("kt CO2e"); ax2.set_ylabel("t CO2e per USD m revenue"); ax2.spines["right"].set_visible(True)
h_, l_ = ax.get_legend_handles_labels(); h2_, l2_ = ax2.get_legend_handles_labels(); ax.set_ylim(0, 9.5); ax2.set_ylim(0, 48); ax.legend(h_ + h2_, l_ + l2_, frameon=False, fontsize=7, loc="upper center")
ax.set_title("The ESG-data layer: falling intensity, rising footprint", loc="left"); figsave(fig, "fig8_esgdata_emissions", "Source: Bloomberg (MSCI emissions and segment revenue).")
fig = plt.figure(figsize=(8, 10)); fig.text(.06, .95, "The AI-ESG Paradox in Europe", fontsize=22, weight="bold", color=NAVY)
fig.text(.06, .915, "3,900 European equity funds, 2019-2024, and the ESG-data layer behind them", fontsize=11, color=GOLD)
kpis = [(f"{100*K['sc_cagr']:.0f}% vs {100*K['aum_cagr']:.0f}%", "growth p.a. of MSCI's ESG &\nclimate revenue vs European AuM"),
        (f"{K['port_diff']:+.1f}%", "four-factor alpha p.a., ESG\nminus conventional (not significant)"),
        (f"+{100*(K['sc_kt24']/K['sc_kt19']-1):.0f}%", "rise in emissions attributable to\nMSCI's ESG business, 2019-2024")]
for i, (big, small) in enumerate(kpis):
    fig.text(.06 + .31 * i, .845, big, fontsize=19, weight="bold", color=NAVY); fig.text(.06 + .31 * i, .805, small, fontsize=8, color="#444444")
a1 = fig.add_axes([.08, .46, .86, .29])
a1.plot(list(yrs), [100 * EVO[y] / EVO[2019] for y in yrs], marker="o", color=NAVY, label="European AuM")
a1.plot(list(yrs), [100 * sc[y] / sc[2019] for y in yrs], marker="s", color=GOLD, lw=2, label="MSCI Sustainability & Climate revenue")
a1.axhline(100, color=GREY, lw=.6, ls=":"); a1.set_title("The ESG-data layer is outgrowing the assets it serves (2019 = 100)", loc="left", fontsize=10); a1.legend(frameon=False, fontsize=8)
a2 = fig.add_axes([.08, .11, .86, .27]); a2.plot(cum.index, cum.esg, color=GOLD, lw=1.8, label="Article 8/9 funds"); a2.plot(cum.index, cum.art6, color=NAVY, lw=1.4, label="Article 6 funds")
a2.set_title("…while ESG funds earn the same risk-adjusted returns (growth of EUR 100)", loc="left", fontsize=10); a2.legend(frameon=False, fontsize=8)
fig.text(.06, .05, "Suyash Sawant · MSc Finance, Dublin City University", fontsize=9, color=NAVY)
fig.text(.06, .032, "Sources: EFAMA (2025); Bloomberg; MSCI SASB; Kenneth R. French Data Library.", fontsize=6.5, color=GREY)
fig.savefig(f"{OUT}/fig9_linkedin_summary.png", dpi=220); plt.close()

# save all tables and the key numbers
with pd.ExcelWriter(f"{OUT}/Tables.xlsx", engine="openpyxl") as xw:
    for k_, t_ in T.items():
        df = pd.DataFrame(t_["rows"], columns=t_["header"]); df.to_excel(xw, sheet_name=k_, startrow=2, index=False)
        ws = xw.sheets[k_]; ws["A1"] = f"Table {k_}: {t_['title']}"; ws.cell(row=len(df) + 5, column=1, value="Notes: " + t_["notes"])
with open(f"{OUT}/tables.md", "w") as f:
    for k_, t_ in T.items():
        f.write(f"### Table {k_}: {t_['title']}\n\n| " + " | ".join(t_["header"]) + " |\n|" + "---|" * len(t_["header"]) + "\n")
        for r_ in t_["rows"]: f.write("| " + " | ".join(str(x) for x in r_) + " |\n")
        f.write(f"\n*Notes:* {t_['notes']}\n\n")
json.dump({k_: (list(v) if isinstance(v, tuple) else v) for k_, v in K.items()}, open(f"{OUT}/key_numbers.json", "w"), indent=1, default=float)
json.dump(T, open(f"{OUT}/tables.json", "w"), default=str)
open(f"{OUT}/results.txt", "w").write("\n".join(LOG)); print("done", len(T), "tables")
