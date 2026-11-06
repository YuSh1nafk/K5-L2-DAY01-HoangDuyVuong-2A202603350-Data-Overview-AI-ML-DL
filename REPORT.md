# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về cả bức ảnh, cụ thể là từng vật thể nhỏ trên bức ảnh to, như cab ở đây được đánh score cao nhất -> model đánh giá là cab xuất hiện nhiều nhất trong ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- class list được tạo ra từ SAMPLES kèm checkpoint, evidence ở cell 2
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- class list chỉ là phân loại những vật thể đấy ra còn từng object như id, name hay taxonomy có nhiệm vụ của riêng nó class_id để gán id độc nhất cho mỗi vật thể trong bức ảnh, class_name là tên của vật thể đó còn taxonomy_name là model xử lý ảnh. 

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- guideline cần chỉ rõ lớp đấy liên quan đến chủ đề gì trong bức ảnh. Ví dụ như trong ảnh detect xe mà có 1 đàn gà giữa đường thì annotator chỉ chọn những gì liên quan tới xe

6. Vì sao không được lấy `score` của model làm ground truth?
- vì score của model được tính bằng “tương đối” trong khi đó ground truth là tiêu chuẩn vàng để đối chiếu. Ta không thể áp tính tương đối cho tiêu chuẩn vàng được

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- độ rộng của vật thể person rộng 113.58 pixel, cao 279.68 pixel ở vị trí góc bên trên trái [385.33, 69.24] đến góc dưới phải [498.92, 348.92]

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- khi threshold = 0.20 thì có 17 prediction, khi threshold = 0.60 thì có 6 prediction

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- khi hạ threshold người reviewer cần kiểm tra lại những object kỹ hơn dựa trên categories 

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Quy tắc box: ôm sát ảnh, tuyệt đối không chạm vào ảnh, khoảng trắng giữa viền box và ảnh lý tưởng nhất là 0px

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Lấy ví dụ như những cái spoon ở góc trên bên phải, cái spoon thứ 2, 4 bị spoon thứ 3 che khuất đi bởi box quá rộng. Trong trường hợp này em sẽ viết quy tắc trong guidline yêu cầu box phải ôm sát vật thể 

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-002 ; `class_name`=bowl ; `score`=0.735743 ; số điểm polygon=`polygon_point_count`=67 ; 3–5 điểm đầu của `polygon_xy`= [
        53.0,
        344.0
      ],
      [
        52.0,
        345.0
      ],
      [
        50.0,
        345.0
      ]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- box chỉ thể hiện 4 cạnh trong khi đó polygon thể hiện hang tram, nghìn các toạ độ x,y liền nhau, thể hiện sự gần đúng trong hình thể uốn lượn, tròn vuông tam giác như nào.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- instance_id là một id riêng của vật thể đó dựa trên tên sample_id + số độc nhất, class id là tên phân loại như bowl thì class id = 45 sẽ có nhiều vật thể bowl sử dụng class id 45

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Quy tắc vẽ biên mask: Bắt buộc phải phóng to bức ảnh tại mép vật thể để nhìn rõ ranh giới các điểm ảnh. Đường biên mask phải nằm chính xác trên dải pixel giao điểm giữa mép ngoài cùng của vật thể và vùng nền, tuyệt đối không vẽ tràn ra ngoài vùng nền. Không bao gồm shadow, quầng sáng hoặc hình phản chiếu của vật thể trên bề mặt khác. Không vẽ lấn vào bên trong thân vật thể. Phải tỉ mỉ đi lượn theo các chi tiết lồi lõm, răng cưa hoặc góc cạnh bám quanh viền thực tế.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Các spoon ở phía trên bên phải bị chạm nhau khiến cho không detect được hết tất cả các vật thể, trong trường hợp này em sẽ dùng guideline.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho cả ảnh theo taxonomy ImageNet-1K  |  Top-1 có thể không khớp trực giác cảnh giao thông (prediction ≠ GT)| Đọc guideline chọn đúng 1 lớp cho ảnh  |  Đúng taxonomy, đúng phạm vi cấp ảnh, không chép score thành nhãn|
| Phát hiện vật thể | Mỗi object: class + box xyxy | Hạ threshold ra thêm hộp nhiễu; object cắt mép | Vẽ box theo rule, đủ object trong guideline | Box có ôm sát không, thiếu/thừa object, ca che khuất |
| Instance segmentation | Mỗi instance: class + polygon | Biên hai vật chạm nhau / vùng mờ | Vẽ mask bám biên từng instance  | Mask có tràn object khác không, đủ instance cùng lớp  |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Chỉ dùng ảnh COCO công khai có sẵn trong notebook; không tải ảnh cá nhân, khuôn mặt, biển số hay dữ liệu nội bộ lên Colab/GitHub.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Báo cho lab coach hoặc giáo viên

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [X] `REPORT.md` (file này đã điền)
- [X] `day1_lab_outputs/classification_predictions.json`
- [X] `day1_lab_outputs/detection_predictions.json`
- [X] `day1_lab_outputs/segmentation_predictions.json`
- [X] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [X] `day1_lab_outputs/visuals/classification_top5.png`
- [X] `day1_lab_outputs/visuals/detection_predictions.png`
- [X] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [X] Đã nộp **link repository** trên LMS
