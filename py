from ultralytics import YOLO
import cv2
from gpiozero import LED

# --- SETUP ---
# LED di Pin GPIO 17
led = LED(17)

# Load model YOLO
print("Tunggu sekejap, sedang load model...")
model = YOLO("yolov8n.pt")

# SETUP KAMERA (PENTING!)
# Kita tambah cv2.CAP_V4L2 untuk paksa driver kamera Linux
# Kalau error, tukar nombor 0 jadi 1 atau -1
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Set saiz gambar
cap.set(3, 640)
cap.set(4, 480)

print("Sistem BERJAYA! Tekan 'q' untuk keluar.")

while True:
    # 1. Baca gambar
    ret, frame = cap.read()
    
    # Kalau tak dapat gambar, skip pusingan ini
    if not ret:
        print("Gagal baca kamera... mencuba lagi")
        continue

    # 2. YOLO Scan
    results = model(frame, stream=True, verbose=False)
    
    mouse_dikesan = False

    # 3. Proses setiap kotak
    # Perhatikan semua baris di bawah ini 'masuk ke dalam'
    for result in results:
        frame_kotak = result.plot()
        
        for box in result.boxes:
            class_id = int(box.cls[0])
            nama_barang = model.names[class_id]
            
            if nama_barang == "mouse":
                mouse_dikesan = True

    # 4. Logic LED
    if mouse_dikesan:
        led.on()
        print("JUMPA MOUSE!")
    else:
        led.off()

    # 5. Tunjuk video
    try:
        cv2.imshow("Kamera AI", frame_kotak)
    except:
        cv2.imshow("Kamera AI", frame)

    # 6. Keluar bila tekan 'q'
    # Baris ini WAJIB ada jarak (indent) supaya duduk dalam while
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
led.off()
print("Sistem tamat.")
