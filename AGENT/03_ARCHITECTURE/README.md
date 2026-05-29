# BDI KKU NMR AI System Blueprints & Model Architectures
=======================================================

This folder contains the engineering blueprints, system interface schemas, and structural definitions for our 4-stage convolutional deep learning architecture.

---

## 📁 File Directory & Descriptions

| File Name | Language / Type | Description / Purpose |
|:---|:---|:---|
| [`TECHNICAL_ARCHITECTURE.md`](TECHNICAL_ARCHITECTURE.md) | English (Markdown) | Comprehensive technical specification of the entire end-to-end GPU-accelerated pipeline. |
| [`pipeline_overview.md`](pipeline_overview.md) | English (Markdown) | Architectural overview diagram and explanation of the data stream from raw spectrograph coordinates down to clinical JSON reports. |
| [`advanced_model.md`](advanced_model.md) | English (Markdown) | Specifications of the advanced 4-stage modules (`SequenceAwareEncoder`, `SpectrumDecoder`, `LocalizedPatchEBM`, and Euler ODE alignment solver). |
| [`baseline_model.md`](baseline_model.md) | English (Markdown) | Document detailing our initial baseline models (PCA + Random Forests) used for bench comparison. |
| [`NMR_POC_Blueprint.md`](NMR_POC_Blueprint.md) | English (Markdown) | The interactive blueprint mapping Streamlit visualizations and REST API endpoints. |

---

## 🇹🇭 ภาษาไทย / Thai Version
โฟลเดอร์นี้รวบรวมแผนผังระบบสถาปัตยกรรมทางวิศวกรรม เอกสารอธิบายการรันข้อมูลระหว่างหน้าบ้าน-หลังบ้าน และรายละเอียดเชิงลึกของโครงข่ายประสาทเทียมระดับสูง (TRL 5+) เพื่อความสะดวกในการทบทวนแผนผังระบบไอที
