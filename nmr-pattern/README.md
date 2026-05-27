---
title: "ชุดข้อมูลสเปกตรัม NMR สำหรับตรวจรูปแบบสารเคมี"
track: "Phenome"
domain:
  - "Metabolomics"
  - "Compound Profiling"
source:
  - "สถาบันฟีโนมแห่งชาติ"
total_records: null
sampled_records: null
filetype:
  - "pdf"

---

# ชุดข้อมูลสเปกตรัม NMR สำหรับตรวจรูปแบบสารเคมี

## คำอธิบาย
ชุดข้อมูลสัญญาณ NMR (Nuclear Magnetic Resonance) แบบสองมิติ (2D) ที่มีประมาณ **20,000+ คอลัมน์ต่อตัวอย่าง** ข้อมูลนี้แสดงรูปแบบสัญญาณ (pattern) ของสารประกอบทางเคมี (compounds) ที่ตรวจพบในตัวอย่าง

สัญญาณ NMR มีลักษณะเป็น spectrum ที่แสดง chemical shift (ppm) เป็นแกนหลัก โดยแต่ละตัวอย่างมีข้อมูลประมาณ 20,000++ จุดข้อมูล (features) ซึ่งสอดคล้องกับตำแหน่งต่างๆ ใน spectrum

ข้อมูลนี้ใช้ในการระบุชนิดและตำแหน่งของสาร (เช่น กลูโคส) โดยต้องใช้ผู้เชี่ยวชาญในการตีความ

## ฟิลด์ข้อมูล
ข้อมูล NMR Pattern ไม่มีฟิลด์แบบตาราง แต่มีลักษณะเป็น 2D matrix: ตัวอย่าง × สัญญาณ (20,000+ features ต่อตัวอย่าง)

## โครงสร้างไฟล์
| ไฟล์ | รูปแบบ |
|------|--------|
| [`Domain_1_processed_NMR_spectrum.pdf`](Domain_1_processed_NMR_spectrum.pdf) | PDF (ตัวอย่าง) |

## โจทย์ / แนวทางวิเคราะห์
- **Compound Classification**: จำแนกรูปแบบสารแม่นยำจากสัญญาณ NMR
- **Pattern Recognition / ML**: ใช้ Machine Learning ระบุ compound อัตโนมัติ
- **Feature Selection**: เลือกตำแหน่งสัญญาณที่สำคัญที่สุด (ลดมิติ)
- **Biomarker Discovery**: หา biomarker จากสัญญาณเฉพาะ
- **Workflow Development**: เสนอ workflow ประยุกต์ใช้ในการคัดกรองโรค

## หมายเหตุ
- ข้อมูลนี้เป็นข้อมูลดิบจาก NMR spectrum
- ความซับซ้อนของสัญญาณที่ซ้อนทับกันเป็นความท้าทายหลัก
