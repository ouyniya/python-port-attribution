# 📊 Portfolio Attribution Analysis (Brinson Model) – Streamlit App

This project is a web-based interactive tool built with Streamlit for performing Brinson Attribution Analysis (1986 model). It helps portfolio managers and analysts break down portfolio performance into allocation, selection, and interaction effects by sector.


## 🧪 Demo

This project was made with ❤️ by ouyniya | nysdev.com

You can try the demo here:  
👉 https://your-demo-link.com


## 🚀 Features
- Upload or connect your portfolio and benchmark data.
- Calculate Brinson Attribution (1986) effects:

  📌 Allocation Effect

  🎯 Selection Effect
  
  🔗 Interaction Effect

  📈 Total Attribution

- Interactive stacked or grouped bar charts using Plotly.
- Filter and analyze by sector.
- Clean and intuitive web interface for non-technical users.



## 📊 Sample Data

Sample portfolio and benchmark files are provided in the data/ folder for demonstration purposes. You can also upload your own CSV files.



## 🛠️ Tech Stack
- Python 3.9+
- Streamlit
- Plotly
- Pandas, NumPy
- matplotlib



## 📄 Disclaimer

- This demo is provided for educational and evaluation purposes only. We do not guarantee the accuracy, completeness, or fitness for any particular purpose.
- In traditional Brinson attribution models, residual effect often arise due to compounding returns. These residuals can obscure a manager’s true skill or bias interpretations of outperformance or underperformance. 

  To address these, adjustments have been applied using appropriate methods such as the Frongello method (or similar).  

  📌 **Note:** The Frongello method is a refinement technique used in performance attribution to more accurately allocate returns, particularly when analyzing multi-period or time-linked performance.

  In this tool, we have applied the Frongello adjustment to ensure a more accurate representation of portfolio management decisions across time.



## ⚖️ License & Usage

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

For full details, see the [LICENSE](./LICENSE) file or visit [gnu.org/licenses/agpl-3.0](https://www.gnu.org/licenses/agpl-3.0).


**💬 Kind Request:**  
  📌 While commercial use is allowed, we kindly ask that you contact us via support@nysdev.com for permission if you plan to use it in a commercial setting.

