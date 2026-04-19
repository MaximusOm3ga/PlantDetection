from anom_classify_test import cnn_anom_det


#potato late blight
pot_path=r"C:\Users\sauri\Downloads\0396e413-e165-427a-ba72-04cbae7b8ab0___RS_LB 4751.JPG"
#potato early blight
pot_2_path=r"C:\Users\sauri\Downloads\0898bffc-57aa-4fdb-92d1-fd6a03d2a011___RS_Early.B 6946_180deg.JPG"
preds, conf=cnn_anom_det(pot_path)
preds2, conf2 = cnn_anom_det(pot_2_path)

print(f"Class: {preds}, Confidence: {conf}")
print(f"Class: {preds2}, Confidence: {conf2}")