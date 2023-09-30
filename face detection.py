import tkinter as tk
from tkinter import messagebox
import cv2

# Initialize the camera
video_capture = cv2.VideoCapture(0)

# Load a pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize some variables
known_faces = []

# Functions for GUI buttons
def add_user():
    # Capture a frame and detect the face
    print("Please align your face properly and press 's' to capture.")
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_image = gray[y:y+h, x:x+w]
            known_faces.append(face_image)
            user_name = entry_name.get()
            messagebox.showinfo("Success", f"User {user_name} added successfully.")
            btn_add_user.config(state=tk.DISABLED)
            btn_recognize_user.config(state=tk.NORMAL)
            return

def recognize_user():
    # Capture a frame and recognize the face
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    recognized = False

    for (x, y, w, h) in faces:
        face_image = gray[y:y+h, x:x+w]

        for known_face in known_faces:
            result = cv2.matchTemplate(face_image, known_face, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, _ = cv2.minMaxLoc(result)

            if confidence > 0.8:
                recognized = True
                break

        if recognized:
            break

    if recognized:
        response = messagebox.askyesno("Welcome", f"Hello!\nDo you want to proceed?")
        if response:
            # Add your code to proceed with the recognized user
            pass
        else:
            # Add your code for not proceeding with the recognized user
            pass
    else:
        messagebox.showwarning("Unknown User", "Sorry, your face was not recognized.")

# Create GUI window
root = tk.Tk()
root.title("Face Recognition Social Media")

# Label and entry for user name
lbl_name = tk.Label(root, text="Enter Your Name:")
lbl_name.pack()
entry_name = tk.Entry(root)
entry_name.pack()

# Add User button
btn_add_user = tk.Button(root, text="Add Your Face", command=add_user)
btn_add_user.pack(pady=10)

# Recognize User button
btn_recognize_user = tk.Button(root, text="Recognize Your Face", command=recognize_user, state=tk.DISABLED)
btn_recognize_user.pack(pady=10)

# Exit button
btn_exit = tk.Button(root, text="Exit", command=root.quit)
btn_exit.pack()

# Run the GUI application
root.mainloop()

# Release the camera and close all windows
video_capture.release()
cv2.destroyAllWindows()
