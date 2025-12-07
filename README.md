<h1 align="center">Criminal Suspect Localization Using CCTV-Based Face Recognition and Object Detection</h1>

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv10s-Object%20Detection-black?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Face%20Detection-MTCNN-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FaceNet-Face Recognition-cyan?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SVM-Classification-yellow?style=for-the-badge"/>
</p>

---

### **👥 Team Name: Gotham**

**Members:**
- Ardutra Agi Ginting (ardutraa40@gmail.com)  
- Muhammad Abyan Nurfajarizqi (muhammadabyan077@gmail.com)  
- Muhammad Hafidz Hidayatullah (hafidzhidayatullah1012@gmail.com)

**Origin:** Universitas Islam Indonesia

---

## 📌 **Project Overview**

> *"Identifying Suspect Locations Automatically Using Face Detection and Face Recognition"*

This project provides significant benefits in supporting law enforcement through the use of Face Detection and Face Recognition technology on CCTV to track the location of suspects more quickly and accurately. The system automates the monitoring process that was previously performed manually, reducing officer workload and improving tracking efficiency.
> ⚠️ **Important Note on NIK Usage:**  
> The NIK integration described here is a *conceptual design only*.  
> This project does **not** connect to real government databases.  
> Instead, a **mock database** of individual reference images is used to simulate the process. Actual national ID data remains under the authority of government institutions.

---

## 🖼️ **Pipeline**
<img width="245" height="851" alt="gotham drawio" src="https://github.com/user-attachments/assets/2d9152d4-2606-4ade-8caa-1405951c422a" />

---


### 🔧 **Technical Implementation**
### 1. **Face Detection – YOLOv10s**
- **Purpose:** Real-time face detection in CCTV feeds  
- **Advantages:** Lightweight, optimized for real-time inference  
- **Input:** CCTV video stream frame-by-frame  
- **Output:** Cropped face regions with bounding box coordinates  

### 2. **Face Recognition – FaceNet**
- **Purpose:** Generate 128-dimensional embedding vectors for each face  
- **Advantages:** State-of-the-art identity representation  
- **Process:** Converts detected faces into comparable embeddings  

### 3. **Classification – Support Vector Machine (SVM)**
- **Purpose:** Match detection using embedding comparisons  
- **Advantages:** Effective with high-dimensional data  
- **Output:** Match/No Match decision with confidence score  

### 4. **Database Integration**
- **NIK-based Search:** Reference face retrieval using mock NIK database  
- **Metadata Storage:** Timestamp, CCTV location, and detection evidence

---

## 🖥️ Web Application Features

| **Feature**      | **Description** |
|------------------|-----------------|
| **Home Page**    | System overview and capabilities |
| **CCTV Page**    | Real-time streaming from multiple cameras with location selection |
| **Search Page**  | NIK-based search for suspect identification |
| **History Page** | Complete detection records with filtering options |

---


### 🧾 **Result**
<img width="2096" height="1078" alt="IMG_4592" src="https://github.com/user-attachments/assets/7132f353-fb74-4382-8de8-35575db636a7" />
<img width="2096" height="1080" alt="IMG_4593" src="https://github.com/user-attachments/assets/b59f3f74-d0d5-40bd-bbde-47ac873c233a" />

---

