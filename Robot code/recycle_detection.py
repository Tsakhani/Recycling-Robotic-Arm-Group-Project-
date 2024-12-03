# Import necessary packages
import cv2
import math
import warnings
import numpy as np
import socket
from PIL import Image
import argparse
import time
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
import threading
import pigpio
import RPi.GPIO as GPIO

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Initialize counters and flags
cardboardCount, glassCount, metalCount, paperCount, plasticCount = 0, 0, 0, 0, 0
stop, done = 0, 1
inputDistance, inputAngle = ' 0', ' 90'

# Labels dictionary for different recycle materials
labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic'}

# Load the TensorFlow Lite model
model_path = "/home/pi/Downloads/model.tflite"  # Update with actual path
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument('-l', '--labels', required=False, default='default_labels.txt', help='Path to the labels file.')
ap.add_argument("-c", "--confidence", type=float, default=0.3, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Initialize the camera using libcamera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

time.sleep(2)  # Allow time for camera to warm up

def waist_servo_neutral():
    """Function to control servo movement."""
    # Pin configuration
    servo_pin = 23  # Define your servo GPIO pin

    # Setup
    GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering
    GPIO.setup(servo_pin, GPIO.OUT)

    # Initialize PWM on servo pin with 50Hz frequency
    pwm = GPIO.PWM(servo_pin, 50)
    pwm.start(0)  # Start with 0 duty cycle

    def set_angle(angle):
        """Set servo angle."""
        duty = 2 + (angle / 18)  # Calculate duty cycle from angle (approximation)
        GPIO.output(servo_pin, True)
        pwm.ChangeDutyCycle(duty)
        time.sleep(0.5)  # Allow time for the servo to move
        GPIO.output(servo_pin, False)
        pwm.ChangeDutyCycle(0)

    try:
       # Infinite loop to rotate back and forth
            # Move to 0 degrees (left)
            print("Moving to 0 degrees...")
            set_angle(0)
            time.sleep(5)  # Hold position for 5 seconds

            # Move to 90 degrees (neutral position)
            print("Moving to 90 degrees...")
            set_angle(90)
            time.sleep(2)  # Hold position for 30 seconds
    except KeyboardInterrupt:
        print("\nExiting loop and cleaning up.")
        pwm.stop()
        GPIO.cleanup()  # Cleanup all GPIO

def waist_servo_rotate():
    """Function to control servo movement."""
    # Pin configuration
    servo_pin = 23  # Define your servo GPIO pin

    # Setup
    GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering
    GPIO.setup(servo_pin, GPIO.OUT)

    # Initialize PWM on servo pin with 50Hz frequency
    pwm = GPIO.PWM(servo_pin, 50)
    pwm.start(0)  # Start with 0 duty cycle

    def set_angle(angle):
        """Set servo angle."""
        duty = 2 + (angle / 18)  # Calculate duty cycle from angle (approximation)
        GPIO.output(servo_pin, True)
        pwm.ChangeDutyCycle(duty)
        time.sleep(0.5)  # Allow time for the servo to move
        GPIO.output(servo_pin, False)
        pwm.ChangeDutyCycle(0)

    try:
            # Move to 180 degrees (right)
            print("Moving to 180 degrees...")
            set_angle(180)
            time.sleep(2)  # Hold position for 5 seconds

    except KeyboardInterrupt:
        print("\nExiting loop and cleaning up.")
        pwm.stop()
        GPIO.cleanup()  # Cleanup all GPIO


def shoulder_servo_neutral():
    """Function to control a servo using pigpio."""
    pi = pigpio.pi()  # Connect to local Pi (requires the daemon to be running)

    if not pi.connected:
        print("Unable to connect to pigpio daemon!")
        return

    SERVO_PIN = 22
    pi.set_mode(SERVO_PIN, pigpio.OUTPUT)

    try:
        # Infinite loop to rotate back and forth
            # Move to 90 degrees (neutral position)
            print("Moving to 90 degrees (neutral position)...")
            pi.set_servo_pulsewidth(SERVO_PIN, 2000)  # Neutral position (90 degrees)
            time.sleep(2)  # Hold position for 2 seconds
    finally:
        pi.set_servo_pulsewidth(SERVO_PIN, 0)  # Turn off PWM signal
        pi.stop()  # Cleanup and stop pigpio connection

def shoulder_servo_rotate():
    """Function to control a servo using pigpio."""
    pi = pigpio.pi()  # Connect to local Pi (requires the daemon to be running)

    if not pi.connected:
        print("Unable to connect to pigpio daemon!")
        return

    SERVO_PIN = 22
    pi.set_mode(SERVO_PIN, pigpio.OUTPUT)

    try:

            # Move forward to 135 degrees (45 degrees from neutral)
            print("Moving to 135 degrees...")
            pi.set_servo_pulsewidth(SERVO_PIN, 1650)  # 135 degrees (45 degrees forward)
            time.sleep(2)  # Hold position for 15 seconds
    finally:
        pi.set_servo_pulsewidth(SERVO_PIN, 0)  # Turn off PWM signal
        pi.stop()  # Cleanup and stop pigpio connection


def pinch_servo_neutral():
    """Function to control a servo using pigpio."""
    pi = pigpio.pi()  # Connect to local Pi (requires the daemon to be running)

    if not pi.connected:
        print("Unable to connect to pigpio daemon!")
        return

    SERVO_PIN = 13
    pi.set_mode(SERVO_PIN, pigpio.OUTPUT)

    try:
          # Infinite loop to rotate back and forth
            # Move to 90 degrees (neutral position)
            print("Moving to 90 degrees (neutral position)...")
            pi.set_servo_pulsewidth(SERVO_PIN, 1000)  # Neutral position (90 degrees)
            time.sleep(2)  # Hold position for 2 seconds
    finally:
        pi.set_servo_pulsewidth(SERVO_PIN, 0)  # Turn off PWM signal
        pi.stop()  # Cleanup and stop pigpio connection

def pinch_servo_rotate():
    """Function to control a servo using pigpio."""
    pi = pigpio.pi()  # Connect to local Pi (requires the daemon to be running)

    if not pi.connected:
        print("Unable to connect to pigpio daemon!")
        return

    SERVO_PIN = 13
    pi.set_mode(SERVO_PIN, pigpio.OUTPUT)

    try:

            # Move forward to 135 degrees (45 degrees from neutral)
            print("Moving to 135 degrees...")
            pi.set_servo_pulsewidth(SERVO_PIN, 500)  # 135 degrees (45 degrees forward)
            time.sleep(2)  # Hold position for 10 seconds
    finally:
        pi.set_servo_pulsewidth(SERVO_PIN, 0)  # Turn off PWM signal
        pi.stop()  # Cleanup and stop pigpio connection


def grip_servo_open():
    """Function to control a servo-based gripper."""
    # Pin configuration
    servo_pin = 12  # Define your servo GPIO pin

    # Setup
    GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering
    GPIO.setup(servo_pin, GPIO.OUT)

    # Initialize PWM on servo pin with 50Hz frequency
    pwm = GPIO.PWM(servo_pin, 50)
    pwm.start(0)  # Start with 0 duty cycle

    def set_angle(angle):
        """Set servo angle."""
        duty = 2 + (angle / 18)  # Calculate duty cycle from angle (approximation)
        GPIO.output(servo_pin, True)
        pwm.ChangeDutyCycle(duty)
        time.sleep(0.5)  # Allow time for the servo to move
        GPIO.output(servo_pin, False)
        pwm.ChangeDutyCycle(0)

    try:
          # Infinite loop to continuously open and close the gripper
            print("Opening gripper...")
            set_angle(60)  # Use a smaller angle for opening for testing
            time.sleep(2)  # Wait for a moment

    except KeyboardInterrupt:
        print("\nExiting loop and cleaning up.")
        pwm.stop()
        GPIO.cleanup()  # Cleanup all GPIO

def grip_servo_close():
    """Function to control a servo-based gripper."""
    # Pin configuration
    servo_pin = 12  # Define your servo GPIO pin

    # Setup
    GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering
    GPIO.setup(servo_pin, GPIO.OUT)

    # Initialize PWM on servo pin with 50Hz frequency
    pwm = GPIO.PWM(servo_pin, 50)
    pwm.start(0)  # Start with 0 duty cycle

    def set_angle(angle):
        """Set servo angle."""
        duty = 2 + (angle / 18)  # Calculate duty cycle from angle (approximation)
        GPIO.output(servo_pin, True)
        pwm.ChangeDutyCycle(duty)
        time.sleep(0.5)  # Allow time for the servo to move
        GPIO.output(servo_pin, False)
        pwm.ChangeDutyCycle(0)

    try:

            print("Closing gripper...")
            set_angle(0)  # Retain the close angle
            time.sleep(2)  # Wait for a moment

    except KeyboardInterrupt:
        print("\nExiting loop and cleaning up.")
        pwm.stop()
        GPIO.cleanup()  # Cleanup all GPIO


# Locking for servo operations
servo_lock = threading.Lock()
def move_arm(label):
    with servo_lock:
        grip_servo_open()
        pinch_servo_neutral()
        shoulder_servo_neutral()
        waist_servo_neutral()
        shoulder_servo_rotate()
        pinch_servo_rotate()
        grip_servo_close()
        shoulder_servo_neutral()
        pinch_servo_neutral()
        waist_servo_rotate()
        grip_servo_open()


# Start capturing frames and detection loop
while True:
    # Capture a frame
    image = picam2.capture_array()
    orig = image.copy()

    # Model inference with corrected dimensions
    _, height, width, _ = input_details[0]['shape']

    # Preprocess the image for TensorFlow Lite model
    frame_resized = cv2.resize(image, (width, height))
    frame_normalized = frame_resized.astype('float32') / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)

    # Model inference
    interpreter.set_tensor(input_details[0]['index'], frame_expanded)
    interpreter.invoke()
    results = interpreter.get_tensor(output_details[0]['index'])[0]

    # Display circles for arm's range of motion
    for r in [390, 365, 340, 315, 290]:
        cv2.circle(orig, (275, 445), r, (0, 0, 255), 3, 8, 0)

    # Process results for each detected class
    for label_id, score in enumerate(results):
        if score < args["confidence"]:
            continue

        label = labels.get(label_id, "Unknown")
        centerX, centerY, angle = 275, 445, 0  # Placeholder for center and angle

        # Classify detected material and trigger actions
        if label in labels.values():
            print(f"Detected {label}")
            move_arm(label)  # Move the robotic arm based on the material type

    # Display the resulting frame
    cv2.imshow("Object Detection", orig)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cv2.destroyAllWindows()
