import cv2
import numpy as np
import pyocr
import pyocr.builders
import mss
import mss.tools

# Path to the folder containing letter and number screenshots
SCREENSHOT_FOLDER = "letter_number_screenshots/"

def find_best_match(image, screenshots):
    best_match = None
    best_score = 0

    for name, screenshot in screenshots.items():
        result = cv2.matchTemplate(image, screenshot, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)

        if score > best_score:
            best_score = score
            best_match = name

    return best_match

def extract_screenshots():
    screenshots = {}

    # Load letter and number screenshots
    for name in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        screenshot_path = SCREENSHOT_FOLDER + name + ".png"
        screenshot = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
        screenshots[name] = screenshot

    return screenshots

def main():
    # Load letter and number screenshots
    screenshots = extract_screenshots()

    # Initialize PyOCR
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found. Please make sure you have OCR engines installed.")
        return
    ocr_tool = tools[0]  # Use the first available OCR tool

    # Set up the screen capture using mss
    with mss.mss() as sct:
        # Set the region of interest
        region = {"left": -1417, "top": -436, "width": 890, "height": 333}

        while True:
            # Capture the screen frame from the specified region
            frame = np.array(sct.grab(region))

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Perform OCR on the grayscale image
            ocr_text = ocr_tool.image_to_string(
                gray,
                lang='eng',  # Specify the language for OCR (change as needed)
                builder=pyocr.builders.TextBuilder()
            )

            # Process the recognized text
            if ocr_text:
                # Remove non-alphanumeric characters
                ocr_text = ''.join(filter(str.isalnum, ocr_text))

                # Find the best match for each character
                recognized_sequence = []
                for char in ocr_text:
                    best_match = find_best_match(screenshots[char.upper()], screenshots)
                    if best_match:
                        recognized_sequence.append(best_match)

                print("Recognized Sequence:", recognized_sequence)

            # Display the original frame
            cv2.imshow("Spectrogram", frame)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
