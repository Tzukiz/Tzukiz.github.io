from ultralytics import YOLO
import cv2
from gpiozero import LED

# --- SETUP ---
led = LED(17) # Pastikan cucuk di Pin Fizikal 11

print("Loading model... (Ini mungkin ambil masa sikit)")
model = YOLO("yolov8n.pt")

# SETUP KAMERA
# Kita set resolution rendah sikit (320x240) supaya laju
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(3, 320)  # Lebar
cap.set(4, 240)  # Tinggi

print("Sistem Laju Sedia! Tekan 'q' untuk keluar.")

# Variable untuk teknik 'Frame Skipping'
frame_count = 0
skip_rate = 3  # Scan AI setiap 3 frame (Lagi tinggi nombor, lagi smooth video)
mouse_dikesan = False # Simpan status terakhir

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    
    # --- TEKNIK LAJUKAN VIDEO ---
    # Kita cuma run AI bila frame_count boleh bahagi dengan skip_rate
    if frame_count % skip_rate == 0:
        
        # Run YOLO (stream=True lajukan proses)
        results = model(frame, stream=True, verbose=False)
        
        # Reset balik status sebelum check baru
        mouse_dikesan = False 
        
        for result in results:
            # Kita tak lukis kotak setiap masa sebab berat
            # Kita cuma nak logic LED je
            
            for box in result.boxes:
                class_id = int(box.cls[0])
                nama = model.names[class_id]
                
                if nama == "mouse":
                    mouse_dikesan = True
    
    # --- OUTPUT ---
    # Logic LED (Guna status detection terakhir)
    if mouse_dikesan:
        led.on()
        # Lukis tulisan MOUSE ringkas di skrin (lagi ringan dari lukis kotak)
        cv2.putText(frame, "MOUSE DIKESAN!", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        led.off()

    # Tunjuk video 'mentah'. Ini buat video nampak smooth sebab tak payah render kotak AI yg berat
    cv2.imshow("Kamera Laju", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
led.off()
