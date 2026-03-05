# config.py - All tunable parameters for the dashcam analyzer

import torch

# Device: auto-select GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model
YOLO_MODEL = "yolov8n.pt"       # Options: yolov8n/s/m/l/x.pt
CONF_THRESHOLD = 0.50           # Detection confidence threshold
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car=2, motorcycle=3, bus=5, truck=7

# Tracking
TRACKER_MAX_AGE = 30            # Max frames to keep lost track alive

# Lane detection ROI (ratio of frame height)
ROI_TOP_RATIO = 0.45            # Upper boundary of lane detection zone
ROI_BOTTOM_RATIO = 0.90         # Lower boundary of lane detection zone

# Canny edge detection
CANNY_LOW = 50
CANNY_HIGH = 150
GAUSSIAN_BLUR_KERNEL = (5, 5)

# Hough line transform
HOUGH_RHO = 1
HOUGH_THETA_DEG = 1             # degrees
HOUGH_THRESHOLD = 30
HOUGH_MIN_LINE_LEN = 30
HOUGH_MAX_LINE_GAP = 200
LANE_SLOPE_MIN = 0.4            # Minimum absolute slope to count as lane line
LANE_SLOPE_MAX = 3.0            # Maximum absolute slope

# Blinker detection
BLINKER_SIDE_RATIO = 0.15       # Fraction of bbox width used for left/right side crop
BLINKER_WINDOW_FRAMES = 30      # Sliding window size for blinker history
BLINKER_AMBER_HUE_LOW = 15      # HSV Hue lower bound for amber
BLINKER_AMBER_HUE_HIGH = 35     # HSV Hue upper bound for amber
BLINKER_SAT_LOW = 100
BLINKER_VAL_LOW = 100
BLINKER_PIXEL_RATIO_THRESHOLD = 0.03  # Min amber pixel ratio to count as ON
BLINKER_BLINK_MIN_TRANSITIONS = 1    # Min ON/OFF transitions to confirm blinking (lenient)

# Lane detection smoothing
LANE_SMOOTHING_FRAMES = 15      # Frames to average lane boundary positions over

# Lane change detection
LANE_CHANGE_MIN_FRAMES = 20     # Minimum frames crossing lane to count as change
LANE_CHANGE_CONFIRM_FRAMES = 8  # Frames vehicle must STAY in new lane to confirm
LANE_CHANGE_COOLDOWN_FRAMES = 90  # Cooldown after a lane change event

# Violation checker
BLINKER_PRE_CHANGE_SEC = 4      # Seconds before lane change to check for blinker

# Clip / output
CLIP_BUFFER_SEC = 5             # Seconds of video to buffer for pre/post clip
OUTPUT_DIR = "./output"
CLIP_DIR = "./output/clips"
LOG_FILE = "./output/violation_log.csv"

# Display
FONT_SCALE = 0.5
FONT_THICKNESS = 1
BOX_THICKNESS = 2
LANE_LINE_COLOR = (0, 255, 0)     # BGR green
VIOLATION_BOX_COLOR = (0, 0, 255) # BGR red
NORMAL_BOX_COLOR = (255, 165, 0)  # BGR orange
TEXT_COLOR = (255, 255, 255)

# ── Cut-in Detection ─────────────────────────────────────────────────────────
CUTIN_MIN_FRAMES_IN_EGO = 15      # frames vehicle must stay in ego lane to confirm cut-in
CUTIN_ENTRY_BUFFER_FRAMES = 10    # frames crossing boundary before we start counting
CUTIN_COOLDOWN_FRAMES = 120       # per-vehicle cooldown after a cut-in event
EGO_LANE_WIDTH_RATIO = 0.45       # fraction of frame width considered ego lane
CUTIN_MIN_LATERAL_SPEED = 0.015   # minimum lateral displacement per frame (ratio of frame_w)
CUTIN_FRONT_ZONE_RATIO = 0.70     # only track vehicles in the lower fraction of the frame

# ── Plate Recognition ────────────────────────────────────────────────────────
PLATE_CROP_EXPAND = 0.15          # fraction to expand vehicle bbox for plate search
PLATE_VOTE_WINDOW = 30            # frames for majority-vote text stabilization
PLATE_MIN_CONFIDENCE = 0.30       # EasyOCR minimum confidence threshold (최종 판단용)
PLATE_COLLECT_CONFIDENCE = 0.15   # EasyOCR minimum confidence for accumulation (수집용 — 낮게 유지)
PLATE_SHARPNESS_MIN = 50.0        # Laplacian variance minimum — blurry frames are skipped
PLATE_LANGUAGES = ["ko", "en"]    # OCR languages: Korean + English
# YOLO 번호판 검출 모델 경로 (None = 비활성화, 기본값)
# 로컬 .pt 파일 경로를 지정하면 활성화됨
# 예: PLATE_YOLO_MODEL = "C:/models/plate_yolov8n.pt"
# 공개 모델 참고: https://universe.roboflow.com 에서 license plate detection 모델 다운로드
PLATE_YOLO_MODEL = r"C:/Users/didgh/.cache/plate_detector/best.pt"
PLATE_YOLO_CONF = 0.25            # YOLO plate detector confidence threshold
PLATE_FRAME_SKIP = 5              # OCR를 N프레임마다 1번만 실행 (속도 개선)

# ── Web API ──────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
CLIP_BUFFER_SECS_WEB = 5.0        # pre/post buffer for cut-in clips (web jobs)
OUTPUTS_DIR = "outputs"           # root directory for all job outputs
