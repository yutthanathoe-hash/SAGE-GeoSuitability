# SAGE-GeoSuitability V1.0.0 — รุ่นเผยแพร่บน GitHub

**ปรับรุ่นจาก V1.3.3.7 เป็น V1.0.0** เพื่อเตรียมเผยแพร่ซอร์สโค้ดบน GitHub

## สิ่งที่นำออก
- ช่อง `Polygon predictor layers` และ `Polygon field mapping CSV`
- โมดูลคำนวณ Predictor จาก Polygon และขั้นตอนประมวลผลที่เกี่ยวข้อง

**ยังคงรองรับ** Training/Test ที่เป็นจุดหรือ Polygon และ Study Area Polygon (AOI) ตามเดิม การตัด Polygon Predictor ออกไม่ใช่การตัดข้อมูลพื้นที่ศึกษาออก

## สิ่งที่รักษาไว้
- Multiple Raster Layers, Spatial CV, Model Comparison, Raster QA, OOD, SHAP, Independent Test และ Probability Calibration
- ค่าเริ่มต้น Independent Test Policy = 2 (อนุญาตเซลล์ซ้ำโดยยังไม่ยืนยันความเป็นอิสระเชิงเวลา)
- `Risk_Score.tif` และ `Risk_Level.tif`

## วิธีติดตั้ง
1. สำรองปลั๊กอินเดิม จากนั้นลบโฟลเดอร์ `sage_geosuitability` รุ่นเก่าออก
2. แตกไฟล์และคัดลอกโฟลเดอร์ `sage_geosuitability` ไปยังโฟลเดอร์ Python plugins ของ QGIS
3. เปิด QGIS ใหม่และเปิดใช้งานปลั๊กอิน
4. ตรวจสอบว่า Processing Toolbox แสดง **SAGE-GeoSuitability V1.0.0** และไม่มีช่อง Polygon Predictor/CSV

## ข้อควรระวัง
- เวอร์ชันนี้ผ่านการตรวจสอบโครงสร้างโค้ด แต่ยังต้องทดสอบการรันจริงใน QGIS ก่อนเผยแพร่อย่างเป็นทางการ
- เลือกสัญญาอนุญาตซอฟต์แวร์ (LICENSE) ก่อนเผยแพร่แบบ Open Source
- การตรวจสอบ SHAP และ Calibration ต้องดูรายงาน Audit ทุกครั้ง


## ขอบเขตการเผยแพร่

รุ่นนี้เป็นการเผยแพร่ซอฟต์แวร์ครั้งแรก ไม่ใช่การรับรองความถูกต้องทางวิทยาศาสตร์สำหรับทุกพื้นที่หรือทุกภัยพิบัติ ผลตัวอย่างมาจากกรณีศึกษาไฟป่าหนึ่งชุดเท่านั้น ตรวจสอบข้อจำกัดด้านการทับซ้อนของเซลล์ Train/Test และหน่วยความชันก่อนอ้างอิงผลวิจัย

**LICENSE:** เสนอ MIT License เป็นค่าเริ่มต้น เจ้าของโครงการควรยืนยันก่อนเผยแพร่สาธารณะ
