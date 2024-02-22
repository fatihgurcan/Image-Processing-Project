import cv2
import dlib

# Dlib yüz algılama modelini ve gülme algılama modelini yükle
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\prog\shape_predictor_68_face_landmarks.dat")

# Video dosyasını yükle (video.mp4 yerine kullanmak istediğiniz video dosyasının adını yazın)
video_path = "video6.mp4"
cap = cv2.VideoCapture(video_path)

# Eğer video dosyası başlatılamazsa çıkış yap
if not cap.isOpened():
    print("Video dosyasi başlatilamadi.")
    exit()

# Hareket algılama için arkaplan modelini başlatma
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# Başlangıçta yüz ve gülümseme sayacı sıfır olarak başlar
total_faces = 0
smiling_faces = 0

# Daha önce tespit edilen yüzleri ve gülümseyen yüzleri izlemek için bir sözlük oluştur
detected_faces = {}
detected_smiling_faces = set()

while True:
    # Video dosyasından bir kare alın
    ret, frame = cap.read()

    # Video sonuna gelindiğinde çıkış yap
    if not ret:
        break

    # Hareket algılamak için arkaplanı çıkartın
    fgmask = fgbg.apply(frame)

    # Yüz algılama ve dikdörtgen çizme
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Her bir yüz için işlemleri gerçekleştir
    for face in faces:
        # Yüzü kare içine al
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Yüzü daha önce tespit edip etmediğimizi kontrol et
        face_id = None
        for fid, (prev_x, prev_y, prev_w, prev_h) in detected_faces.items():
            # Eğer mevcut yüz, daha önce tespit edilen yüzlerle örtüşüyorsa, aynı yüz olarak kabul et
            if (x >= prev_x and x <= prev_x + prev_w) or (prev_x >= x and prev_x <= x + w):
                if (y >= prev_y and y <= prev_y + prev_h) or (prev_y >= y and prev_y <= y + h):
                    face_id = fid
                    break

        if face_id is None:
            # Bu yüz daha önce tespit edilmemiş, yeni bir kimlik ata
            face_id = len(detected_faces) + 1
            detected_faces[face_id] = (x, y, w, h)

       # Gülme algılandığında yüzü farklı bir kare içine al
        shape = predictor(gray, face)
        mouth_points = [shape.part(i) for i in range(48, 68)] 
        # Ağız bölgesindeki noktalar
        mouth_left = min(mouth_points, key=lambda x: x.x).x
        mouth_right = max(mouth_points, key=lambda x: x.x).x
        mouth_top = min(mouth_points, key=lambda x: x.y).y
        mouth_bottom = max(mouth_points, key=lambda x: x.y).y

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Yüzü yeşil renkte çiz
        cv2.rectangle(frame, (mouth_left, mouth_top), (mouth_right, mouth_bottom), (255, 0, 0), 2) 

# Gülümsüyorsa ve daha önce tespit edilmediyse gülümseyen yüz sayacını artır
    if mouth_bottom - mouth_top > 10:
        if face_id not in detected_smiling_faces:
            smiling_faces += 1
            detected_smiling_faces.add(face_id)


    # Görüntüyü ekrana gösterme
    cv2.imshow('Frame', frame)
    cv2.imshow('Foreground Mask', fgmask)

    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Kullanılan kaynakları serbest bırakma
cap.release()
cv2.destroyAllWindows()

# Video işlemi bittikten sonra toplam yüz ve gülümseyen yüz sayısını yazdır
print("Toplam Yüz Sayısı:", len(detected_faces))
print("Gülümseyen Yüz Sayısı:", smiling_faces)
# Yüzde oranını hesapla
if len(detected_faces) > 0:
    smile_percentage = (smiling_faces / len(detected_faces)) * 100
    print("Gülümseme Oranı: {:.2f}%".format(smile_percentage))
else:
    print("Video içinde hiç yüz tespit edilemedi.")

