# Finance Time-Series Platform — Day-by-Day Plan & Progress

Tento dokument trackuje skutečný postup oproti původnímu 100denní plánu (Fáze A: Dny 1-50, Fáze B-D: Dny 51-100). Aktualizuje se průběžně — odškrtáváme hotové dny, upravujeme rozsah podle reálného tempa.

**Metoda:** 1h/den, productive struggle (nejdřív zkusit, pak hledat řešení), páteční retrieval recall.

---

## Fáze A — MVP (Týdny 1-10, Dny 1-50)

### Týden 1 — Setup & Bronze layer

- [x] **Den 1** — Azure infrastruktura: resource group, budget alerty (30/50/80/100%), Automation Account + Stop-AllResources runbook, Databricks workspace (Premium)
- [x] **Den 2** — Personal access token, Databricks CLI v1.4.0, GitHub repo, Python 3.14 + Git, `databricks bundle init`
- [x] **Den 3** — `databricks.yml` dev/prod targets (finance_dev/finance_prod catalogy), interaktivní cluster, Quarto blog + GitHub Pages, `publish-blog.ps1` automatizace
- [x] **Den 4** — ADLS Gen2 storage, Access Connector, Storage Credential, External Location, Unity Catalog catalogy vytvořené, `sources.yaml` (16 tickerů), `ingestion.py`, **první Bronze Delta table** (`finance_dev.bronze.ohlcv_daily`) ověřená end-to-end
- [ ] **Den 5** — `features.yaml` config, čištění dat (missing values, sparse/illiquid dny), **log returns** (+ teorie: stacionarita)
- [ ] **Den 6** — Rolling volatility (20/60d, backward-looking), Moving averages (20/50/200), RSI
- [ ] **Den 7** — Cross-asset features (korelace s VIX/sektorem), `features.py` modul, **první Silver Delta table**

### Týden 2 — Silver dokončení & Gold start

- [ ] **Den 8** — Validace Silver dat, unit testy pro `features.py` (pytest)
- [ ] **Den 9** — Páteční retrieval recall (týden 1-2 zpětně, z hlavy)
- [ ] **Den 10** — Gold layer návrh: agregace per ticker/den, příprava feature matrix pro clustering

### Týden 3 — Gold: Market regime clustering

- [ ] **Den 11** — K-means clustering — teorie (just-in-time), výběr features pro clustering
- [ ] **Den 12** — Implementace clusteringu, MLflow experiment tracking setup
- [ ] **Den 13** — Pojmenování/interpretace regimů (např. "low-vol uptrend", "high-vol selloff")
- [ ] **Den 14** — Validace regimů vizuálně (grafy), uložení do Gold table
- [ ] **Den 15** — Páteční retrieval recall

### Týden 4 — Gold: Klasifikační model

- [ ] **Den 16** — Time-based train/test split (teorie: look-ahead bias)
- [ ] **Den 17** — Random Forest / GBM model — první trénink
- [ ] **Den 18** — Metriky pro nebalancované třídy (precision/recall vs. accuracy)
- [ ] **Den 19** — MLflow Model Registry — staging/production verze
- [ ] **Den 20** — Páteční retrieval recall

### Týden 5 — Databricks Workflows orchestrace

- [ ] **Den 21** — DAB job definice: bronze→silver→gold jako jeden pipeline
- [ ] **Den 22** — Job clusters v `resources/*.yml`, scheduled denní run (dev)
- [ ] **Den 23** — Error handling, retry policy na úrovni jobu
- [ ] **Den 24** — Prod target deploy, ověření izolace dev/prod
- [ ] **Den 25** — Páteční retrieval recall

### Týden 6 — Lakebase / serving layer

- [ ] **Den 26** — Lakebase setup (nebo Postgres na Azure jako fallback, pokud nedostupné)
- [ ] **Den 27** — Sync "nejnovější regime + predikce per ticker" z Gold do Lakebase
- [ ] **Den 28** — Ověření serving table, výkon dotazů
- [ ] **Den 29** — Dokumentace rozdílu Delta (analytics) vs. Lakebase (operational serving)
- [ ] **Den 30** — Páteční retrieval recall

### Týden 7 — Databricks Apps / Dashboard

- [ ] **Den 31** — Databricks Apps setup (nebo lokální Streamlit + Databricks SQL connection jako fallback)
- [ ] **Den 32** — Dashboard: výběr tickeru, základní layout
- [ ] **Den 33** — Graf regime over time
- [ ] **Den 34** — Přehledový panel (aktuální regime + predikce per ticker)
- [ ] **Den 35** — Páteční retrieval recall

### Týden 8 — Dashboard dokončení & Terraform

- [ ] **Den 36** — Srovnání microcap vs. large-cap patterns v dashboardu
- [ ] **Den 37** — Terraform — základní IaC ukázka (storage, Unity Catalog metastore)
- [ ] **Den 38** — Polish dashboardu, UX vychytávky
- [ ] **Den 39** — GitHub Actions CI (volitelné) — pytest na push
- [ ] **Den 40** — Páteční retrieval recall

### Týden 9 — Testing & dokumentace

- [ ] **Den 41** — Rozšíření unit testů, edge cases (sparse data, chybějící tickery)
- [ ] **Den 42** — README: architektura, tech stack, jak spustit
- [ ] **Den 43** — Architecture diagram
- [ ] **Den 44** — Code review vlastního kódu, refactoring
- [ ] **Den 45** — Páteční retrieval recall

### Týden 10 — Finalizace MVP

- [ ] **Den 46** — End-to-end test celé pipeline od nuly
- [ ] **Den 47** — Příprava 10-15 min walkthrough prezentace
- [ ] **Den 48** — Live demo dry-run
- [ ] **Den 49** — Drobné opravy podle zpětné vazby
- [ ] **Den 50** — **MVP hotové.** Začátek sondování trhu (freelancermap, Upwork) může běžet souběžně od týdne 6-8.

---

## Fáze B — News & Sentiment (Týdny 11-14, Dny 51-64)

- [ ] News ingestion (Raw vrstva — zde poprvé dává smysl kvůli rate limitům API)
- [ ] VADER vs. FinBERT sentiment scoring — porovnání
- [ ] Join sentiment do Gold layer
- [ ] Price-sentiment divergence feature
- [ ] Re-trénink klasifikačního modelu s novým feature
- [ ] Rozšíření dashboardu o sentiment panel

## Fáze C — Fundamentals & Anomaly Detection (Týdny 15-17, Dny 65-77)

- [ ] Fundamentals ingestion přes yfinance (earnings, revenue)
- [ ] Earnings event flagy
- [ ] Isolation Forest anomaly detection
- [ ] Kategorizace anomálií
- [ ] Dashboard panel pro anomálie

## Fáze D — Network & Streaming (Týdny 18-20, Dny 78-100)

- [ ] Correlation/graph network mezi tickery (networkx)
- [ ] Centrality metriky jako features
- [ ] Structured Streaming — near-real-time news ingestion
- [ ] Finální feature store koncept
- [ ] End-to-end test celého rozšířeného systému
- [ ] Finální dashboard, dokumentace, prezentace

---

## Vedlejší cvičné stopy (mimo hlavní 100denní plán)

Tyto běží paralelně v samostatných konverzacích ve stejném projektu, nejsou součástí denního počítání Fáze A-D a neváží se na konkrétní dny.

- [ ] **Python trénink** — vlastní konverzace, učení Python syntaxe/konceptů přes vysvětlování `ingestion.py` blok po bloku (logging, dataclass, type hints, list comprehension, atd.), cílem je umět psát podobný kód samostatně, ne kopírovat
- [ ] **Star schema + SCD cvičení** — vlastní konverzace, samostatný syntetický dataset (brokerage/trading platforma: `dim_customer` se SCD Type 2, `dim_account`, `dim_security`, `dim_date`, `fact_trades`). Cíl: procvičit dimenze, surrogate vs. natural keys, Slowly Changing Dimensions. Záměrně oddělené od `finance_timeseries_platform` — jiný catalog (např. `practice_dwh`), jiný účel (data warehousing koncepty, ne lakehouse pipeline).

---

## Poznámky k odchylkám od plánu

- **Den 2 → Den 3 posun:** Den 2 doběhl o den později než plánováno, Den 3 zahrnoval navíc blog setup (nebyl v původním denním rozpisu, přidáno jako vedlejší iniciativa)
- **Den 4 navíc:** Plný universe tickerů (16, ne malý test set) — bylo to manuální sestavení ze známých jmen, ne finviz screener. **TODO:** ověřit/doplnit přes finviz.com (Market Cap < $300M, Sector: Technology/Healthcare AI) před tím, než seznam považujeme za finální.
- Rozpis Dnů 5+ výše je odhad založený na `project_spec.docx`, ne doslovný export z `finance_timeseries_plan.docx` (ten dokument nebyl nikdy nahrán/sdílen) — může se zpřesnit, pokud ten plán existuje a chceš ho dodat.
