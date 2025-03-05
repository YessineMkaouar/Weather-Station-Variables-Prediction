# 🌦️ Weather Prediction Challenge

## 📌 Build Status

🚀 **Template Kit for RAMP Challenge**

---

## 📖 Introduction

This challenge focuses on **predicting weather conditions** based on historical meteorological data.

### **📍 Where does the data come from?**

This dataset is sourced from **ERA5**, providing surface weather data for **5 cities**:

- 🌆 **Berlin**
- 🌊 **Brest**
- 🏰 **London**
- ☀️ **Marseille**
- 🇫🇷 **Paris** (41 years of data, while others have 40 years)

### **🎯 What is the goal of this challenge?**

- The **Paris weather station** has **broken down**, so we will predict weather parameters for **Paris**, using data from other cities.
- Handle both **Analysis Variables** and **Forecast Variables**.
- Manage **time shifts** (Forecast variables are shifted by 3 hours compared to analysis variables).

### **❓ Why does it matter?**

- **Accurate weather forecasting** is crucial for aviation, agriculture, and climate research.
- **Data-driven insights** can enhance prediction models beyond traditional methods.
- **Advanced ML/DL models** can capture temporal and spatial weather dependencies.

---

## 🛠 Getting Started

### **📦 Install Dependencies**

To run a submission and the notebook, install the required dependencies using:

```bash
pip install -U -r requirements.txt
```

---

## 🗂️ Challenge Description

📌 **Get started on this RAMP challenge** with the dedicated **notebook** `starting_kit.ipynb` provided.

### **📥 How to Load the Data**

* **Each weather variable is stored in a separate NetCDF file per city** .
* **Xarray** can be used to load NetCDF files into Python and convert them into a Pandas DataFrame.

---


## 🏆 Test a Submission

All submissions should be placed inside the `submissions/` folder.

For instance for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the ramp-test command line:

```bash
ramp-test --submission my_submission
```

### **Get Help**

To learn more about the test command, run:

```bash
ramp-test --help
```

---

## 📌 Going Further

For more details about  **ramp-workflow** , visit the [official documentation]().

