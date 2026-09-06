# Car Price Prediction

Kompletan regresioni ML projekat za predviđanje cene polovnih automobila (`priceUSD`).

## 1. Cilj projekta

Cilj je da model na osnovu karakteristika automobila proceni njegovu okvirnu tržišnu cenu u USD.

Projekat prati ceo workflow:

**EDA → čišćenje → feature engineering → preprocessing → treniranje → evaluacija → poređenje → izbor finalnog modela**

## 2. Dataset

Originalni `cars.csv` sadrži **56,244 reda i 12 kolona**.

Ciljna promenljiva:
- `priceUSD`

Ulazne karakteristike:
- `make`
- `model`
- `year`
- `condition`
- `mileage(kilometers)`
- `fuel_type`
- `volume(cm3)`
- `color`
- `transmission`
- `drive_unit`
- `segment`

Dataset sadrži nedostajuće vrednosti, posebno u `volume(cm3)`, `drive_unit` i `segment`, što je obrađeno kroz preprocessing pipeline.

## 3. EDA

Jupyter notebook `notebooks/01_eda.ipynb` sadrži detaljnu analizu:

- dimenzije dataset-a;
- tipove podataka;
- broj jedinstvenih vrednosti;
- missing values;
- duplikate;
- deskriptivnu statistiku;
- distribuciju ciljne promenljive;
- boxplot cene;
- korelaciju numeričkih promenljivih sa cenom;
- odnose godine/kilometraže i cene;
- analizu kategorijskih promenljivih;
- proveru potencijalno nevalidnih vrednosti;
- vizuelizaciju novih feature-a;
- poređenje modela;
- analizu grešaka;
- actual vs. predicted grafikon;
- feature importance finalnog modela.

Notebook namerno sadrži **Markdown ćelije** koje objašnjavaju šta se radi, zašto se radi i kako se rezultat tumači.

## 4. Čišćenje podataka

`src/data_cleaning.py`:

- uklanja duplikate;
- standardizuje tekstualne kategorije;
- pretvara numeričke kolone u odgovarajući format;
- uklanja nevalidne/nelogične cene ispod 100 USD;
- uklanja godine van opsega 1950–2019;
- uklanja negativnu i ekstremnu kilometražu iznad 1,500,000 km;
- uklanja očigledno nevalidne zapremine motora iznad 10,000 cm³;
- ne briše legitimno nedostajuće kategorijske vrednosti, jer se one rešavaju u pipeline-u.

Posle čišćenja ostaje **55,812 redova**.

## 5. Feature engineering

`src/feature_engineering.py` kreira:

- `car_age` – starost automobila;
- `mileage_per_year` – prosečna kilometraža po godini;
- `engine_volume_liters` – zapremina motora u litrima;
- `is_newer_car` – indikator novijeg vozila;
- `is_high_mileage` – indikator velike kilometraže;
- `brand_model` – kombinacija marke i modela;
- `log_mileage` – log transformacija kilometraže;
- `log_engine_volume` – log transformacija zapremine;
- `age_squared` – nelinearni termin starosti;
- `mileage_age_interaction` – interakcija kilometraže i starosti.

## 6. Preprocessing

`src/data_preprocessing.py` koristi `ColumnTransformer`.

### Numeričke kolone

- `SimpleImputer(strategy="median")`
- `StandardScaler`

### Kategorijske kolone

- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`

Preprocessor se fituje samo na trening podacima kako bi se sprečilo data leakage.

## 7. Modeli

Upoređena su tri regresiona pristupa:

1. **Ridge Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**

Svi modeli koriste isti train/test split:

- 80% trening
- 20% test
- `random_state=42`

## 8. Rezultati

Rezultati se automatski čuvaju u `reports/model_comparison.csv`.

Aktuelni rezultati na test skupu:

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| **Random Forest** | **1,140.02** | **9,194,379.84** | **3,032.22** | **0.8727** |
| Decision Tree | 1,355.24 | 12,168,661.00 | 3,488.36 | 0.8315 |
| Ridge Regression | 1,877.55 | 17,091,859.00 | 4,134.23 | 0.7634 |

Vrednosti su zaokružene u README-u; kompletne vrednosti nalaze se u CSV izveštaju.

### Tumačenje

**MAE ≈ 1,140 USD** znači da finalni model na test skupu u proseku odstupa oko 1,140 USD od stvarne cene.

**R² ≈ 0.873** znači da model objašnjava približno 87% varijacije ciljne promenljive na ovom test skupu.

## 9. Finalni model

Na osnovu poređenja izabran je:

**Random Forest Regressor**

Finalni model se čuva kao:

```text
models/car_price_model.joblib
```

Konačna evaluacija:
- MAE: **1,140.02 USD**
- MSE: **9,194,379.84**
- RMSE: **3,032.22 USD**
- R²: **0.8727**

## 10. Struktura projekta

```text
car-price-prediction/
│
├── data/
│   ├── cars.csv
│   ├── cars_clean.csv
│   └── cars_features.csv
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_comparison.py
│
├── models/
│   └── car_price_model.joblib
│
├── reports/
│   ├── model_comparison.csv
│   ├── final_metrics.csv
│   └── example_predictions.csv
│
├── README.md
└── requirements.txt
```

## 11. Pokretanje projekta

Instalacija biblioteka:

```bash
pip install -r requirements.txt
```

Pokretanje čišćenja:

```bash
python src/data_cleaning.py
```

Feature engineering:

```bash
python src/feature_engineering.py
```

Poređenje modela:

```bash
python src/model_comparison.py
```

Treniranje finalnog modela:

```bash
python src/model_training.py
```

Evaluacija:

```bash
python src/model_evaluation.py
```

Pokretanje Jupyter-a:

```bash
jupyter notebook
```

Zatim otvoriti:

```text
notebooks/01_eda.ipynb
```

## 12. Važna napomena

Rezultati su dobijeni na jednom reproducibilnom hold-out test skupu (`random_state=42`). Model predstavlja procenu tržišne cene i nije garancija tačne prodajne cene svakog automobila.

## 13. Zaključak

Projekat pokriva zahtevani regresioni workflow: analiza podataka, identifikacija problema, čišćenje, feature engineering, preprocessing, treniranje više modela, evaluacija i obrazložen izbor finalnog modela.

Random Forest je izabran zato što je na istom test skupu ostvario najmanji MAE i RMSE i najveći R² među testiranim modelima.
