# 🌱 EcoRoute

### Real-time cost-efficient routing with carbon impact insights.

---

## 🚀 Overview

**EcoRoute** is a cost-first, carbon-aware routing application designed for **SME last-mile delivery fleets**.

It leverages real-time road and traffic data via **Mapbox APIs** and combines it with a custom optimisation and prediction layer to help businesses:

* 💰 Reduce delivery operating costs
* 🌍 Lower carbon emissions
* 📊 Make data-driven routing decisions

---

## 🧠 Key Idea

> **Mapbox provides routing. EcoRoute provides business decision-making.**

EcoRoute extends traditional route optimisation by translating routes into:

* Cost impact (£ per route)
* Carbon emissions (CO₂ per route)
* Annual savings projections

---

## 🏗️ System Architecture

EcoRoute is built as a layered system:

1. **Routing Layer (Mapbox)**

   * Real road-network routing
   * Traffic-aware travel time
   * Multi-stop route optimisation

2. **Optimisation Layer (EcoRoute)**

   * Cost-first route comparison
   * Payload-aware efficiency modelling
   * Business decision logic

3. **ML Prediction Layer**

   * Predicts CO₂ emissions
   * Learns from route conditions
   * Ready for real-world data training

---

## ⚙️ Features

* 🗺️ **Real-time routing** (Mapbox Directions + Optimization APIs)
* 💰 **Cost-first optimisation**
* 📦 **Payload-aware modelling (kg-based)**
* 🌍 **CO₂ estimation and reduction insights**
* 📊 **Annual cost & carbon savings projection**
* 🤖 **Lightweight ML prediction layer**
* 📈 **Interactive dashboard (Streamlit)**
* 🧭 **Route comparison + visualisation**

---

## 🧪 Demo Flow

1. Enter Mapbox API token
2. Select:

   * Vehicle type
   * Delivery payload (kg)
   * Delivery stops
3. Run analysis
4. View:

   * Cost savings
   * CO₂ reduction
   * Route comparison
   * Map visualisation

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Routing API:** Mapbox
* **ML:** Scikit-learn (Random Forest)
* **Data Handling:** Pandas
* **Visualisation:** Folium

---

## 🔐 Environment Setup

Create a `.env` file:

```text
MAPBOX_ACCESS_TOKEN=your_mapbox_token_here
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## 📊 Business Value

EcoRoute is designed for:

* 🚚 SME delivery fleets (5–50 vehicles)
* 🛒 Local logistics operators
* 🏥 Pharmacy / retail delivery services

### Value Proposition

> Reduce delivery cost first, while making carbon savings measurable.

---

## 🧠 Why EcoRoute?

| Capability                | Mapbox | EcoRoute |
| ------------------------- | ------ | -------- |
| Routing                   | ✅      | ✅        |
| Traffic-aware             | ✅      | ✅        |
| Cost optimisation         | ❌      | ✅        |
| CO₂ insights              | ❌      | ✅        |
| Business decision support | ❌      | ✅        |
| ML predictions            | ❌      | ✅        |

---

## 🚀 Future Improvements

* Real-time dynamic routing (continuous updates)
* Fleet telematics integration
* Driver behaviour analytics
* Reinforcement learning optimisation
* Multi-vehicle fleet scheduling
* SaaS platform for logistics operators

---

## 🏆 Hackathon Context

Built for:

**Innovate Suffolk – GreenTech 48-hour Start-up Hackathon**

Focus:

> Viable, scalable, cost-driven Greentech solution

---

## 👤 Author

**Uzor Nwokeaka**
AI/ML Engineer | Analytics Engineer

---

## 📌 License

This project is for demonstration and hackathon purposes.
