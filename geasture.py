# importing the necessary packages

import cv2
import mediapipe as mp

# Explanation:
# cv2: OpenCV library for video capture and processing
# mediapipe: Library for hand tracking and landmark detection

# initializing the meadiapipe hands

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Explanation:
# mp_hands: Accesses the hands solution in MediaPipe
# hands: Initializes the hands module for hand detection
# mp_draw: Utility to draw hand landmarks on the frames


# intializing the video capture

cap=cv2.VideoCapture(0)

# Explanation:
# cap: Captures video from the default camera (usually the wecam)

#capture and process the each frame
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display

    frame = cv2.flip(frame,1)
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            
            #intializing the list to store landmarks coordinates
            landmark_list = []
            for id,lm in enumerate(hand_landmarks.landmark):
                # get the coordinates
                h,w,c = frame.shape
                cx,cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([cx,cy])

            if len(landmark_list) != 0:
                # Example logic for gesture recognition
                # # Open Hand (Palm) Gesture
                if landmark_list[4][1] < landmark_list[3][1] and landmark_list[8][1] < landmark_list[6][1]:
                    gesture = 'Palm detected'
                # pointing up gesture
                elif landmark_list[4][1] > landmark_list[3][1] and  landmark_list[8][1] < landmark_list[6][1]:
                    gesture = "Love you"
                else:
                    gesture = None


                if gesture:
                    cv2.putText(frame,gesture,(landmark_list[0][0] - 50, landmark_list[0][1] - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3,cv2.LINE_AA)

# Explanation:
 # ret, frame: Reads a frame from the video capture
 # cv2.flip(frame, 1): Flips the frame horizontally for amirror view
 # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB): Converts the frame from BGR to RGB
 # hands.process(frame_rgb): Processes the frame to detecthand landmarks PROJECT - H 9
 # mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS): Draws the detected hand landmarks on the frame

 # Dispaly the Frame
 
    cv2.imshow('Hand geasture Recognition',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Explanation:
# cv2.imshow('Hand Gesture Recognition', frame): Displays theframe with hand landmarks
# cv2.waitKey(1): Waits for 1 millisecond for a key press. If'q' is pressed, the loop breaks

# Release the Resources

cap.release()
cv2.destroyAllWindows()

# Explanation:
# cap.release(): Releases the webcam PROJECT - H 10
# cv2.destroyAllWindows(): Closes all OpenCV windows

