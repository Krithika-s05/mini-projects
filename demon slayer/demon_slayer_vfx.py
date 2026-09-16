#!/usr/bin/env python3
"""
===============================================================================
DEMON SLAYER BREATHING STYLE - REAL-TIME VFX ENGINE
===============================================================================

Requirements:
    pip install opencv-python mediapipe numpy

Run:
    python demon_slayer_vfx.py

Other modes:
    python demon_slayer_vfx.py --camera 1
    python demon_slayer_vfx.py --debug
    python demon_slayer_vfx.py --calibrate

Controls:
    Q / ESC    Quit
    D          Toggle debug mode
    C          Toggle HSV calibration window
    R          Reset VFX particles

Sword tracking:
    The default HSV range is configured for a bright GREEN prop.
    If your sword has another color, use:

        python demon_slayer_vfx.py --calibrate

    Then adjust the HSV sliders until only your sword appears white.

===============================================================================
"""

import argparse
import math
import random
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit(
        "\nMediaPipe is not installed.\n"
        "Run:\n"
        "    pip install mediapipe\n"
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "camera_index": 0,
    "width": 1280,
    "height": 720,

    # Default HSV range for GREEN sword.
    "sword_hsv_lower": np.array([40, 90, 90], dtype=np.uint8),
    "sword_hsv_upper": np.array([85, 255, 255], dtype=np.uint8),

    "sword_min_contour_area": 350,
    "sword_smoothing": 0.45,

    # Number of frames used for gesture majority voting.
    "gesture_history": 7,

    # Particle limit.
    "max_particles": 700,

    # Time before particles disappear.
    "particle_life": 1.5,

    # Combo system.
    "combo_window": 3.0,
}


# =============================================================================
# BREATHING STYLE INFORMATION
# =============================================================================

BREATHING_STYLES = [
    "Sun",
    "Water",
    "Flame",
    "Wind",
    "Stone",
    "Thunder",
    "Beast",
    "Serpent",
    "Mist",
    "Love",
    "Sound",
    "Insect",
    "Moon",
]


STYLE_COLORS = {
    "Sun": (0, 180, 255),
    "Water": (255, 200, 80),
    "Flame": (0, 60, 255),
    "Wind": (80, 255, 100),
    "Stone": (170, 170, 170),
    "Thunder": (220, 255, 255),
    "Beast": (0, 130, 255),
    "Serpent": (190, 40, 220),
    "Mist": (220, 220, 255),
    "Love": (180, 80, 255),
    "Sound": (0, 220, 255),
    "Insect": (220, 130, 255),
    "Moon": (255, 240, 180),
}


# Water -> Flame -> Sun unlocks Sun Ultimate.
COMBOS = {
    ("Water", "Flame", "Sun"): "Sun"
}


# =============================================================================
# MATH HELPERS
# =============================================================================

def distance(a, b):
    return float(np.linalg.norm(a - b))


def normalize(v):
    n = np.linalg.norm(v)

    if n < 1e-6:
        return np.zeros_like(v)

    return v / n


def perpendicular(v):
    return np.array([-v[1], v[0]], dtype=np.float32)


def rotate_vector(v, angle):
    c = math.cos(angle)
    s = math.sin(angle)

    return np.array(
        [
            v[0] * c - v[1] * s,
            v[0] * s + v[1] * c
        ],
        dtype=np.float32
    )


def clamp_point(p, width, height):
    return np.array(
        [
            np.clip(p[0], 0, width - 1),
            np.clip(p[1], 0, height - 1)
        ],
        dtype=np.float32
    )


# =============================================================================
# PARTICLE
# =============================================================================

@dataclass
class Particle:
    """
    Lightweight physics particle.

    position : current pixel position
    velocity : pixels per second
    color    : BGR color
    size     : particle radius
    life     : remaining life
    max_life : original life
    gravity  : downward acceleration
    drag     : velocity damping
    """

    position: np.ndarray
    velocity: np.ndarray
    color: Tuple[int, int, int]
    size: float
    life: float
    max_life: float
    gravity: float = 0.0
    drag: float = 0.0

    def update(self, dt):
        self.velocity *= max(0.0, 1.0 - self.drag * dt)
        self.velocity[1] += self.gravity * dt

        self.position += self.velocity * dt

        self.life -= dt

    @property
    def alive(self):
        return self.life > 0

    @property
    def alpha(self):
        return max(0.0, min(1.0, self.life / self.max_life))


# =============================================================================
# GESTURE RECOGNIZER
# =============================================================================

class GestureRecognizer:

    WRIST = 0

    TIPS = [4, 8, 12, 16, 20]
    MCPS = [2, 5, 9, 13, 17]

    PIPS = [3, 6, 10, 14, 18]

    # -------------------------------------------------------------------------
    # Gesture thresholds
    # -------------------------------------------------------------------------

    T = {
        "EXT_RATIO": 1.35,
        "FOLD_RATIO": 1.15,

        "PINCH_RATIO": 0.35,

        "MIST_SPREAD": 22.0,

        "SUN_TILT": 42.0,

        "WATER_SPREAD": 18.0,

        "LOVE_Z": -0.075,

        "MOON_MIN": 0.35,
        "MOON_MAX": 0.95,
    }

    def __init__(
        self,
        max_hands=1,
        detection_confidence=0.65,
        tracking_confidence=0.6
    ):

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def close(self):
        self.hands.close()

    # -------------------------------------------------------------------------

    def process(self, frame, debug=False):

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        results = self.hands.process(rgb)

        label = None
        features = None

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            if debug:

                self.mp_draw.draw_landmarks(
                    frame,
                    hand,
                    self.mp_hands.HAND_CONNECTIONS
                )

            features = self.extract_features(hand, w, h)

            label = self.classify(features)

        return label, features, frame

    # -------------------------------------------------------------------------

    def extract_features(self, hand, width, height):

        points = np.array(
            [
                [lm.x * width, lm.y * height]
                for lm in hand.landmark
            ],
            dtype=np.float32
        )

        z = np.array(
            [lm.z for lm in hand.landmark],
            dtype=np.float32
        )

        wrist = points[self.WRIST]

        palm_size = max(
            distance(wrist, points[9]),
            1.0
        )

        ratios = np.array(
            [
                distance(wrist, points[tip]) /
                max(distance(wrist, points[mcp]), 1.0)
                for tip, mcp in zip(self.TIPS, self.MCPS)
            ]
        )

        extended = ratios > self.T["EXT_RATIO"]
        folded = ratios < self.T["FOLD_RATIO"]

        semi = (~extended) & (~folded)

        thumb_index = distance(
            points[4],
            points[8]
        ) / palm_size

        thumb_pinky = distance(
            points[4],
            points[20]
        ) / palm_size

        # Palm orientation.
        orient = points[9] - wrist

        orient_angle = math.degrees(
            math.atan2(
                abs(orient[0]),
                abs(orient[1]) + 1e-6
            )
        )

        spread = self.calculate_spread(
            points,
            extended
        )

        return {
            "points": points,
            "z": z,
            "wrist": wrist,
            "palm_size": palm_size,
            "ratios": ratios,
            "extended": extended,
            "folded": folded,
            "semi": semi,
            "thumb_index": thumb_index,
            "thumb_pinky": thumb_pinky,
            "orientation": orient_angle,
            "spread": spread
        }

    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_spread(points, extended):

        ids = [
            (8, 5),
            (12, 9),
            (16, 13),
            (20, 17)
        ]

        directions = []

        for index, (tip, mcp) in enumerate(ids, start=1):

            if extended[index]:

                vector = normalize(
                    points[tip] - points[mcp]
                )

                directions.append(vector)

        if len(directions) < 2:
            return 0.0

        angles = []

        for i in range(len(directions) - 1):

            dot = np.clip(
                np.dot(
                    directions[i],
                    directions[i + 1]
                ),
                -1,
                1
            )

            angles.append(
                math.degrees(
                    math.acos(dot)
                )
            )

        return float(np.mean(angles))

    # -------------------------------------------------------------------------

    def classify(self, f):

        T = self.T

        ext = f["extended"]
        fold = f["folded"]
        semi = f["semi"]

        THUMB = 0
        INDEX = 1
        MIDDLE = 2
        RING = 3
        PINKY = 4

        # ---------------------------------------------------------------------
        # INSECT
        # Thumb + index touching, remaining fingers extended.
        # ---------------------------------------------------------------------

        if (
            f["thumb_index"] < T["PINCH_RATIO"]
            and ext[MIDDLE]
            and ext[RING]
            and ext[PINKY]
        ):
            return "Insect"

        # ---------------------------------------------------------------------
        # BEAST
        # Index + pinky extended.
        # ---------------------------------------------------------------------

        if (
            ext[INDEX]
            and ext[PINKY]
            and fold[MIDDLE]
            and fold[RING]
        ):
            return "Beast"

        # ---------------------------------------------------------------------
        # SERPENT
        # Thumb + pinky extended.
        # ---------------------------------------------------------------------

        if (
            ext[THUMB]
            and ext[PINKY]
            and fold[INDEX]
            and fold[MIDDLE]
            and fold[RING]
        ):
            return "Serpent"

        # ---------------------------------------------------------------------
        # SOUND
        # Thumbs up.
        # ---------------------------------------------------------------------

        if (
            ext[THUMB]
            and fold[INDEX]
            and fold[MIDDLE]
            and fold[RING]
            and fold[PINKY]
        ):
            return "Sound"

        # ---------------------------------------------------------------------
        # MOON
        # Four semi-curled fingers forming a C.
        # ---------------------------------------------------------------------

        if (
            semi[INDEX]
            and semi[MIDDLE]
            and semi[RING]
            and semi[PINKY]
            and
            T["MOON_MIN"]
            <
            f["thumb_pinky"]
            <
            T["MOON_MAX"]
        ):
            return "Moon"

        # ---------------------------------------------------------------------
        # STONE
        # Peace sign.
        # ---------------------------------------------------------------------

        if (
            ext[INDEX]
            and ext[MIDDLE]
            and fold[RING]
            and fold[PINKY]
            and not ext[THUMB]
        ):
            return "Stone"

        # ---------------------------------------------------------------------
        # THUNDER / LOVE
        # Index finger only.
        # ---------------------------------------------------------------------

        if (
            ext[INDEX]
            and fold[MIDDLE]
            and fold[RING]
            and fold[PINKY]
        ):

            index_z = f["z"][8] - f["z"][0]

            if index_z < T["LOVE_Z"]:
                return "Love"

            return "Thunder"

        # ---------------------------------------------------------------------
        # MIST
        # Four fingers extended and separated.
        # ---------------------------------------------------------------------

        if (
            ext[INDEX]
            and ext[MIDDLE]
            and ext[RING]
            and ext[PINKY]
            and not ext[THUMB]
            and f["spread"] >= T["MIST_SPREAD"]
        ):
            return "Mist"

        # ---------------------------------------------------------------------
        # OPEN HAND
        # Sun / Water / Wind.
        # ---------------------------------------------------------------------

        if (
            ext[THUMB]
            and ext[INDEX]
            and ext[MIDDLE]
            and ext[RING]
            and ext[PINKY]
        ):

            if f["orientation"] < T["SUN_TILT"]:
                return "Sun"

            if f["spread"] < T["WATER_SPREAD"]:
                return "Water"

            return "Wind"

        # ---------------------------------------------------------------------
        # FLAME
        # Closed fist.
        # ---------------------------------------------------------------------

        if not ext.any():

            return "Flame"

        return None


# =============================================================================
# GESTURE STABILIZER
# =============================================================================

class GestureStabilizer:

    def __init__(self, size=7):

        self.history = deque(maxlen=size)

    def update(self, label):

        self.history.append(label)

        valid = [
            x for x in self.history
            if x is not None
        ]

        if not valid:
            return None

        return Counter(valid).most_common(1)[0][0]

    def reset(self):

        self.history.clear()


# =============================================================================
# COMBO DETECTOR
# =============================================================================

class SequenceDetector:

    def __init__(self, combos, window=3.0):

        self.combos = combos
        self.window = window

        self.buffer = []
        self.previous = None

    def update(self, label):

        now = time.time()

        self.buffer = [
            (g, t)
            for g, t in self.buffer
            if now - t <= self.window
        ]

        if label is not None and label != self.previous:

            self.buffer.append(
                (label, now)
            )

            self.buffer = self.buffer[-6:]

        self.previous = label

        sequence = tuple(
            g for g, _ in self.buffer
        )

        for combo, result in self.combos.items():

            n = len(combo)

            if (
                len(sequence) >= n
                and sequence[-n:] == combo
            ):

                self.buffer.clear()
                self.previous = None

                return result

        return None

    def text(self):

        return " > ".join(
            g for g, _ in self.buffer
        )


# =============================================================================
# SWORD TRACKER
# =============================================================================

class SwordTracker:

    def __init__(
        self,
        hsv_lower,
        hsv_upper,
        min_area=350,
        smoothing=0.45
    ):

        self.lower = hsv_lower
        self.upper = hsv_upper

        self.min_area = min_area
        self.smoothing = smoothing

        self.previous_hilt = None
        self.previous_tip = None

        self.tip_speed = 0.0

        self.last_mask = None

    # -------------------------------------------------------------------------

    def set_hsv(self, lower, upper):

        self.lower = lower
        self.upper = upper

    # -------------------------------------------------------------------------

    @staticmethod
    def clean_mask(mask):

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mask = cv2.erode(
            mask,
            kernel,
            iterations=1
        )

        mask = cv2.dilate(
            mask,
            kernel,
            iterations=2
        )

        mask = cv2.GaussianBlur(
            mask,
            (5, 5),
            0
        )

        return mask

    # -------------------------------------------------------------------------

    def update(self, frame):

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        mask = cv2.inRange(
            hsv,
            self.lower,
            self.upper
        )

        mask = self.clean_mask(mask)

        self.last_mask = mask

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:

            self.tip_speed = 0

            return None

        contour = max(
            contours,
            key=cv2.contourArea
        )

        if cv2.contourArea(contour) < self.min_area:

            self.tip_speed = 0

            return None

        rect = cv2.minAreaRect(contour)

        box = cv2.boxPoints(rect).astype(
            np.float32
        )

        # Find the long axis.
        edge1 = np.linalg.norm(
            box[0] - box[1]
        )

        edge2 = np.linalg.norm(
            box[1] - box[2]
        )

        if edge1 > edge2:

            a = (box[0] + box[3]) / 2
            b = (box[1] + box[2]) / 2

        else:

            a = (box[0] + box[1]) / 2
            b = (box[2] + box[3]) / 2

        # Determine hilt/tip based on previous tip.
        if self.previous_tip is not None:

            if distance(a, self.previous_tip) < distance(
                b,
                self.previous_tip
            ):
                tip = a
                hilt = b

            else:
                tip = b
                hilt = a

        else:

            # Initial direction:
            # choose lower endpoint as hilt.
            if a[1] > b[1]:
                hilt = a
                tip = b
            else:
                hilt = b
                tip = a

        # ---------------------------------------------------------------------
        # Temporal smoothing
        # ---------------------------------------------------------------------

        if self.previous_hilt is not None:

            hilt = (
                self.smoothing * hilt
                +
                (1 - self.smoothing)
                * self.previous_hilt
            )

        if self.previous_tip is not None:

            tip = (
                self.smoothing * tip
                +
                (1 - self.smoothing)
                * self.previous_tip
            )

        # ---------------------------------------------------------------------
        # Tip velocity
        # ---------------------------------------------------------------------

        now = time.time()

        if self.previous_tip is not None:

            dt = max(
                now - getattr(
                    self,
                    "_previous_time",
                    now
                ),
                1e-3
            )

            self.tip_speed = (
                distance(
                    tip,
                    self.previous_tip
                )
                / dt
            )

        self._previous_time = now

        self.previous_hilt = hilt.copy()
        self.previous_tip = tip.copy()

        return (
            tuple(hilt.astype(int)),
            tuple(tip.astype(int))
        )


# =============================================================================
# PROCEDURAL DRAWING HELPERS
# =============================================================================

def alpha_overlay(base, overlay, alpha):

    cv2.addWeighted(
        overlay,
        alpha,
        base,
        1.0,
        0,
        dst=base
    )


def draw_glow_line(
    frame,
    p1,
    p2,
    color,
    thickness=2,
    glow=12
):

    layer = np.zeros_like(frame)

    cv2.line(
        layer,
        p1,
        p2,
        color,
        glow,
        cv2.LINE_AA
    )

    layer = cv2.GaussianBlur(
        layer,
        (0, 0),
        8
    )

    alpha_overlay(
        frame,
        layer,
        0.55
    )

    cv2.line(
        frame,
        p1,
        p2,
        color,
        thickness,
        cv2.LINE_AA
    )


# =============================================================================
# VFX ENGINE
# =============================================================================

class VFXEngine:

    def __init__(self, max_particles=700):

        self.particles = []

        self.max_particles = max_particles

        self.time = 0.0

        self.impact_flash = 0.0

        # Moon projectile storage.
        self.moons = []

        # Sound rings.
        self.rings = []

    # -------------------------------------------------------------------------

    def reset(self):

        self.particles.clear()
        self.moons.clear()
        self.rings.clear()

    # -------------------------------------------------------------------------

    def spawn_particle(
        self,
        position,
        velocity,
        color,
        size,
        life,
        gravity=0,
        drag=0
    ):

        if len(self.particles) >= self.max_particles:

            self.particles.pop(
                0
            )

        self.particles.append(
            Particle(
                position=np.array(
                    position,
                    dtype=np.float32
                ),
                velocity=np.array(
                    velocity,
                    dtype=np.float32
                ),
                color=color,
                size=size,
                life=life,
                max_life=life,
                gravity=gravity,
                drag=drag
            )
        )

    # -------------------------------------------------------------------------

    def update_particles(self, dt):

        for particle in self.particles:

            particle.update(dt)

        self.particles = [
            p for p in self.particles
            if p.alive
        ]

    # -------------------------------------------------------------------------

    def render_particles(self, frame):

        layer = np.zeros_like(frame)

        for p in self.particles:

            if p.position[0] < 0:
                continue

            if p.position[1] < 0:
                continue

            if (
                p.position[0] >= frame.shape[1]
                or
                p.position[1] >= frame.shape[0]
            ):
                continue

            radius = max(
                1,
                int(p.size * p.alpha)
            )

            cv2.circle(
                layer,
                tuple(p.position.astype(int)),
                radius,
                p.color,
                -1,
                cv2.LINE_AA
            )

        glow = cv2.GaussianBlur(
            layer,
            (0, 0),
            5
        )

        alpha_overlay(
            frame,
            glow,
            0.45
        )

        alpha_overlay(
            frame,
            layer,
            0.85
        )

    # -------------------------------------------------------------------------
    # MAIN UPDATE
    # -------------------------------------------------------------------------

    def update(
        self,
        frame,
        style,
        sword,
        dt,
        ultimate=False
    ):

        self.time += dt

        self.update_particles(dt)

        if sword is not None:

            hilt, tip = sword

            if style == "Water":
                self.water(frame, hilt, tip)

            elif style == "Flame":
                self.flame(frame, hilt, tip)

            elif style == "Sun":
                self.sun(frame, hilt, tip)

            elif style == "Wind":
                self.wind(frame, hilt, tip)

            elif style == "Stone":
                self.stone(frame, hilt, tip)

            elif style == "Thunder":
                self.thunder(frame, hilt, tip)

            elif style == "Beast":
                self.beast(frame, hilt, tip)

            elif style == "Serpent":
                self.serpent(frame, hilt, tip)

            elif style == "Mist":
                self.mist(frame, hilt, tip)

            elif style == "Love":
                self.love(frame, hilt, tip)

            elif style == "Sound":
                self.sound(frame, hilt, tip)

            elif style == "Insect":
                self.insect(frame, hilt, tip)

            elif style == "Moon":
                self.moon(frame, hilt, tip)

            if ultimate:

                self.ultimate_sun(
                    frame,
                    hilt,
                    tip
                )

        self.update_rings(frame, dt)
        self.update_moons(frame, dt)

        self.render_particles(frame)

    # =========================================================================
    # WATER
    # =========================================================================

    def water(self, frame, hilt, tip):

        h = np.array(hilt, dtype=np.float32)
        t = np.array(tip, dtype=np.float32)

        axis = normalize(t - h)
        normal = perpendicular(axis)

        points = []

        length = distance(h, t)

        for i in range(80):

            u = i / 79.0

            center = h + axis * length * u

            wave = math.sin(
                u * 8.0
                + self.time * 5
            )

            radius = 15 + 25 * u

            p = center + normal * wave * radius

            points.append(
                tuple(p.astype(int))
            )

        overlay = np.zeros_like(frame)

        cv2.polylines(
            overlay,
            [np.array(points)],
            False,
            (255, 220, 120),
            8,
            cv2.LINE_AA
        )

        overlay = cv2.GaussianBlur(
            overlay,
            (0, 0),
            7
        )

        alpha_overlay(
            frame,
            overlay,
            0.65
        )

        cv2.polylines(
            frame,
            [np.array(points)],
            False,
            (255, 255, 240),
            2,
            cv2.LINE_AA
        )

        # Foam particles near tip.
        for _ in range(2):

            angle = random.uniform(
                0,
                math.pi * 2
            )

            velocity = (
                np.array(
                    [
                        math.cos(angle),
                        math.sin(angle)
                    ]
                )
                * random.uniform(20, 100)
            )

            self.spawn_particle(
                t,
                velocity,
                (255, 255, 255),
                random.uniform(2, 5),
                0.45
            )

    # =========================================================================
    # FLAME
    # =========================================================================

    def flame(self, frame, hilt, tip):

        h = np.array(hilt, dtype=np.float32)
        t = np.array(tip, dtype=np.float32)

        axis = normalize(t - h)
        normal = perpendicular(axis)

        overlay = np.zeros_like(frame)

        for i in range(5):

            offset = (
                math.sin(
                    self.time * 7
                    + i
                )
                * 20
            )

            left = (
                h
                + normal * offset
            )

            right = (
                h
                - normal * offset
            )

            end = (
                h
                + axis
                * (80 + i * 35)
            )

            polygon = np.array(
                [
                    left,
                    right,
                    end
                ],
                dtype=np.int32
            )

            cv2.fillPoly(
                overlay,
                [polygon],
                (
                    20,
                    50 + i * 30,
                    255
                )
            )

        blur = cv2.GaussianBlur(
            overlay,
            (0, 0),
            12
        )

        alpha_overlay(
            frame,
            blur,
            0.5
        )

        alpha_overlay(
            frame,
            overlay,
            0.35
        )

        # Fire particles.
        for _ in range(4):

            pos = h + axis * random.uniform(
                0,
                180
            )

            velocity = (
                normal
                * random.uniform(-80, 80)
                - axis * random.uniform(10, 80)
            )

            self.spawn_particle(
                pos,
                velocity,
                random.choice(
                    [
                        (0, 50, 255),
                        (0, 120, 255),
                        (30, 200, 255)
                    ]
                ),
                random.uniform(2, 6),
                0.7,
                gravity=-30
            )

    # =========================================================================
    # SUN
    # =========================================================================

    def sun(self, frame, hilt, tip):

        h = np.array(hilt, dtype=np.float32)
        t = np.array(tip, dtype=np.float32)

        axis = normalize(t - h)
        normal = perpendicular(axis)

        overlay = np.zeros_like(frame)

        # Golden rays.
        for i in range(7):

            angle = (
                self.time * 0.7
                + i * math.pi * 2 / 7
            )

            direction = rotate_vector(
                axis,
                angle
            )

            end = (
                h
                + direction
                * (80 + 30 * math.sin(self.time * 3 + i))
            )

            cv2.line(
                overlay,
                tuple(h.astype(int)),
                tuple(end.astype(int)),
                (0, 180, 255),
                3,
                cv2.LINE_AA
            )

        # Blade ribbon.
        points = []

        for i in range(100):

            u = i / 99

            center = h + axis * distance(h, t) * u

            wave = math.sin(
                self.time * 8 + u * 15
            )

            points.append(
                tuple(
                    (
                        center
                        + normal * wave * 12
                    ).astype(int)
                )
            )

        cv2.polylines(
            overlay,
            [np.array(points)],
            False,
            (0, 220, 255),
            7,
            cv2.LINE_AA
        )

        glow = cv2.GaussianBlur(
            overlay,
            (0, 0),
            10
        )

        alpha_overlay(
            frame,
            glow,
            0.65
        )

        alpha_overlay(
            frame,
            overlay,
            0.8
        )

    # =========================================================================
    # WIND
    # =========================================================================

    def wind(self, frame, hilt, tip):

        h = np.array(hilt, dtype=np.float32)
        t = np.array(tip, dtype=np.float32)

        axis = normalize(t - h)
        normal = perpendicular(axis)

        overlay = np.zeros_like(frame)

        for j in range(8):

            points = []

            phase = (
                self.time * 5
                + j * 0.8
            )

            for i in range(80):

                u = i / 79

                center = (
                    h
                    + axis
                    * distance(h, t)
                    * u
                )

                wave = math.sin(
                    phase
                    + u * 18
                )

                p = (
                    center
                    + normal
                    * wave
                    * (10 + 20 * u)
                )

                points.append(
                    tuple(p.astype(int))
                )

            cv2.polylines(
                overlay,
                [np.array(points)],
                False,
                (80, 255, 100),
                2,
                cv2.LINE_AA
            )

        glow = cv2.GaussianBlur(
            overlay,
            (0, 0),
            8
        )

        alpha_overlay(
            frame,
            glow,
            0.7
        )

        alpha_overlay(
            frame,
            overlay,
            0.8
        )

    # =========================================================================
    # STONE
    # =========================================================================

    def stone(self, frame, hilt, tip):

        t = np.array(tip, dtype=np.float32)

        # Impact shockwave.
        radius = int(
            20
            + 100
            * (
                0.5
                + 0.5
                * math.sin(self.time * 5)
            )
        )

        cv2.circle(
            frame,
            tuple(t.astype(int)),
            radius,
            (160, 160, 160),
            2,
            cv2.LINE_AA
        )

        # Rock fragments.
        for _ in range(3):

            velocity = np.array(
                [
                    random.uniform(-150, 150),
                    random.uniform(-250, 50)
                ]
            )

            self.spawn_particle(
                t,
                velocity,
                (130, 130, 130),
                random.uniform(3, 8),
                1.2,
                gravity=350,
                drag=0.1
            )

    # =========================================================================
    # THUNDER
    # =========================================================================

    def thunder(self, frame, hilt, tip):

        h = np.array(hilt, dtype=np.float32)
        t = np.array(tip, dtype=np.float32)

        axis = normalize(t - h)
        normal = perpendicular(axis)

        overlay = np.zeros_like(frame)

        for j in range(3):

            points = [h.copy()]

            for i in range(1, 12):

                u = i / 11

                center = (
                    h
                    + axis
                    * distance(h, t)
                    * u
                )

                jitter = normal * random.uniform(
                    -20,
                    20
                )

                points.append(
                    center + jitter
                )

            cv2.polylines(
                overlay,
                [
                    np.array(
                        points,
                        dtype=np.int32
                    )
                ],
                False,
                (220, 255, 255),
                3,
                cv2.LINE_AA
            )

        glow = cv2.GaussianBlur(
            overlay,
            (0, 0),
            8
        )

        alpha_overlay(
            frame,
            glow,
            0.8
        )

        alpha_overlay(
            frame,
            overlay,
            1
        )

        # Sparks.
        for _ in range(5):

            p = (
                h
                + axis
                * random.uniform(
                    0,
                    distance(h, t)
                )
            )

            self.spawn_particle(
                p,
                normal * random.uniform(-100, 100),
                (230, 255, 255),
                random.uniform(1, 3),
                0.25
            )

    # =========================================================================
    # BEAST
    # =========================================================================

    def beast(self, frame, hilt, tip):

        h = np.array(hilt, dtype=np.float32)
        t = np.array(tip, dtype=np.float32)

        axis = normalize(t - h)
        normal = perpendicular(axis)

        for side in [-1, 1]:

            points = []

            for i in range(40):

                u = i / 39

                center = (
                    h
                    + axis
                    * distance(h, t)
                    * u
                )

                p = (
                    center
                    + normal
                    * side
                    * math.sin(
                        u * 20
                        + self.time * 5
                    )
                    * 18
                )

                points.append(
                    tuple(p.astype(int))
                )

            cv2.polylines(
                frame,
                [np.array(points)],
                False,
                (0, 140, 255),
                3,
                cv2.LINE_AA
            )

    # =========================================================================
    # SERPENT
    # =========================================================================

    def serpent(self, frame, hilt, tip):

        h = np.array(hilt, dtype=np.float32)
        t = np.array(tip, dtype=np.float32)

        axis = normalize(t - h)
        normal = perpendicular(axis)

        points = []

        length = distance(h, t)

        for i in range(100):

            u = i / 99

            center = h + axis * length * u

            wave = math.sin(
                u * 22
                - self.time * 6
            )

            p = (
                center
                + normal * wave * 25
            )

            points.append(
                tuple(p.astype(int))
            )

        overlay = np.zeros_like(frame)

        cv2.polylines(
            overlay,
            [np.array(points)],
            False,
            (190, 40, 220),
            8,
            cv2.LINE_AA
        )

        glow = cv2.GaussianBlur(
            overlay,
            (0, 0),
            10
        )

        alpha_overlay(
            frame,
            glow,
            0.7
        )

        cv2.polylines(
            frame,
            [np.array(points)],
            False,
            (210, 50, 240),
            3,
            cv2.LINE_AA
        )

    # =========================================================================
    # MIST
    # =========================================================================

    def mist(self, frame, hilt, tip):

        h = np.array(hilt, dtype=np.float32)
        t = np.array(tip, dtype=np.float32)

        axis = normalize(t - h)
        normal = perpendicular(axis)

        layer = np.zeros_like(frame)

        for i in range(18):

            u = random.random()

            center = (
                h
                + axis
                * distance(h, t)
                * u
            )

            offset = normal * random.uniform(
                -45,
                45
            )

            center += offset

            cv2.circle(
                layer,
                tuple(center.astype(int)),
                random.randint(10, 35),
                (210, 220, 255),
                -1
            )

        layer = cv2.GaussianBlur(
            layer,
            (0, 0),
            20
        )

        alpha_overlay(
            frame,
            layer,
            0.20
        )

    # =========================================================================
    # LOVE
    # =========================================================================

    def love(self, frame, hilt, tip):

        h = np.array(hilt, dtype=np.float32)
        t = np.array(tip, dtype=np.float32)

        axis = normalize(t - h)
        normal = perpendicular(axis)

        points = []

        for i in range(120):

            u = i / 119

            center = h + axis * distance(h, t) * u

            wave = math.sin(
                u * 16
                + self.time * 6
            )

            p = (
                center
                + normal
                * wave
                * 45
            )

            points.append(
                tuple(p.astype(int))
            )

        overlay = np.zeros_like(frame)

        cv2.polylines(
            overlay,
            [np.array(points)],
            False,
            (180, 80, 255),
            6,
            cv2.LINE_AA
        )

        blur = cv2.GaussianBlur(
            overlay,
            (0, 0),
            10
        )

        alpha_overlay(
            frame,
            blur,
            0.7
        )

        cv2.polylines(
            frame,
            [np.array(points)],
            False,
            (200, 120, 255),
            2,
            cv2.LINE_AA
        )

    # =========================================================================
    # SOUND
    # =========================================================================

    def sound(self, frame, hilt, tip):

        t = np.array(tip, dtype=np.float32)

        phase = (
            self.time * 100
        ) % 100

        for i in range(3):

            radius = int(
                25
                + (
                    phase
                    + i * 35
                ) % 130
            )

            alpha = max(
                0.1,
                1
                - radius / 160
            )

            overlay = np.zeros_like(frame)

            cv2.circle(
                overlay,
                tuple(t.astype(int)),
                radius,
                (0, 220, 255),
                3,
                cv2.LINE_AA
            )

            alpha_overlay(
                frame,
                overlay,
                alpha
            )

        for _ in range(3):

            angle = random.uniform(
                0,
                2 * math.pi
            )

            velocity = np.array(
                [
                    math.cos(angle),
                    math.sin(angle)
                ]
            ) * random.uniform(
                100,
                250
            )

            self.spawn_particle(
                t,
                velocity,
                (0, 230, 255),
                2,
                0.5
            )

    # =========================================================================
    # INSECT
    # =========================================================================

    def insect(self, frame, hilt, tip):

        h = np.array(hilt, dtype=np.float32)
        t = np.array(tip, dtype=np.float32)

        axis = normalize(t - h)
        normal = perpendicular(axis)

        for _ in range(4):

            u = random.random()

            p = (
                h
                + axis
                * distance(h, t)
                * u
                + normal
                * random.uniform(-20, 20)
            )

            velocity = (
                normal
                * random.uniform(-100, 100)
                - axis
                * random.uniform(0, 50)
            )

            self.spawn_particle(
                p,
                velocity,
                random.choice(
                    [
                        (220, 130, 255),
                        (255, 180, 255),
                        (180, 100, 255)
                    ]
                ),
                random.uniform(2, 5),
                random.uniform(0.6, 1.2),
                gravity=-20
            )

    # =========================================================================
    # MOON
    # =========================================================================

    def moon(self, frame, hilt, tip):

        h = np.array(hilt, dtype=np.float32)
        t = np.array(tip, dtype=np.float32)

        axis = normalize(t - h)

        # Spawn crescent projectiles occasionally.
        if random.random() < 0.12:

            direction = rotate_vector(
                axis,
                random.uniform(
                    -0.8,
                    0.8
                )
            )

            self.moons.append(
                {
                    "position": t.copy(),
                    "velocity": direction * random.uniform(
                        80,
                        180
                    ),
                    "life": 2.0,
                    "rotation": random.random()
                }
            )

        # Blade glow.
        draw_glow_line(
            frame,
            tuple(h.astype(int)),
            tuple(t.astype(int)),
            (255, 230, 170),
            2,
            8
        )

    # -------------------------------------------------------------------------

    def update_moons(self, frame, dt):

        remaining = []

        for moon in self.moons:

            moon["life"] -= dt

            if moon["life"] <= 0:
                continue

            moon["position"] += (
                moon["velocity"]
                * dt
            )

            moon["rotation"] += dt * 4

            p = moon["position"]

            size = 16

            overlay = np.zeros_like(frame)

            # Crescent shape using two circles.
            cv2.circle(
                overlay,
                tuple(p.astype(int)),
                size,
                (255, 235, 170),
                2,
                cv2.LINE_AA
            )

            offset = np.array(
                [
                    math.cos(moon["rotation"]),
                    math.sin(moon["rotation"])
                ]
            ) * 6

            cv2.circle(
                overlay,
                tuple(
                    (p + offset).astype(int)
                ),
                size - 4,
                (0, 0, 0),
                -1
            )

            glow = cv2.GaussianBlur(
                overlay,
                (0, 0),
                6
            )

            alpha_overlay(
                frame,
                glow,
                0.7
            )

            alpha_overlay(
                frame,
                overlay,
                1
            )

            remaining.append(moon)

        self.moons = remaining

    # =========================================================================
    # ULTIMATE SUN
    # =========================================================================

    def ultimate_sun(self, frame, hilt, tip):

        h = np.array(hilt, dtype=np.float32)

        # Large rotating solar disk.
        radius = int(
            100
            + 20 * math.sin(
                self.time * 5
            )
        )

        overlay = np.zeros_like(frame)

        cv2.circle(
            overlay,
            tuple(h.astype(int)),
            radius,
            (0, 100, 255),
            5,
            cv2.LINE_AA
        )

        for i in range(16):

            angle = (
                self.time * 2
                + i * math.pi * 2 / 16
            )

            direction = np.array(
                [
                    math.cos(angle),
                    math.sin(angle)
                ]
            )

            end = (
                h
                + direction
                * (
                    radius
                    + 50
                    * math.sin(
                        self.time * 3 + i
                    )
                )
            )

            cv2.line(
                overlay,
                tuple(h.astype(int)),
                tuple(end.astype(int)),
                (0, 200, 255),
                3,
                cv2.LINE_AA
            )

        blur = cv2.GaussianBlur(
            overlay,
            (0, 0),
            15
        )

        alpha_overlay(
            frame,
            blur,
            0.8
        )

        alpha_overlay(
            frame,
            overlay,
            0.8
        )

    # =========================================================================
    # SOUND RING STORAGE
    # =========================================================================

    def update_rings(self, frame, dt):

        remaining = []

        for ring in self.rings:

            ring["radius"] += 180 * dt
            ring["life"] -= dt

            if ring["life"] <= 0:
                continue

            alpha = ring["life"] / ring["max_life"]

            overlay = np.zeros_like(frame)

            cv2.circle(
                overlay,
                tuple(
                    ring["center"].astype(int)
                ),
                int(ring["radius"]),
                ring["color"],
                3,
                cv2.LINE_AA
            )

            alpha_overlay(
                frame,
                overlay,
                alpha
            )

            remaining.append(ring)

        self.rings = remaining


# =============================================================================
# HSV CALIBRATION TOOL
# =============================================================================

class HSVCalibrator:

    def __init__(self):

        self.window = "Sword HSV Calibration"

        cv2.namedWindow(
            self.window,
            cv2.WINDOW_NORMAL
        )

        cv2.createTrackbar(
            "H Low",
            self.window,
            40,
            179,
            lambda x: None
        )

        cv2.createTrackbar(
            "H High",
            self.window,
            85,
            179,
            lambda x: None
        )

        cv2.createTrackbar(
            "S Low",
            self.window,
            90,
            255,
            lambda x: None
        )

        cv2.createTrackbar(
            "S High",
            self.window,
            255,
            255,
            lambda x: None
        )

        cv2.createTrackbar(
            "V Low",
            self.window,
            90,
            255,
            lambda x: None
        )

        cv2.createTrackbar(
            "V High",
            self.window,
            255,
            255,
            lambda x: None
        )

    def values(self):

        lower = np.array(
            [
                cv2.getTrackbarPos(
                    "H Low",
                    self.window
                ),
                cv2.getTrackbarPos(
                    "S Low",
                    self.window
                ),
                cv2.getTrackbarPos(
                    "V Low",
                    self.window
                )
            ],
            dtype=np.uint8
        )

        upper = np.array(
            [
                cv2.getTrackbarPos(
                    "H High",
                    self.window
                ),
                cv2.getTrackbarPos(
                    "S High",
                    self.window
                ),
                cv2.getTrackbarPos(
                    "V High",
                    self.window
                )
            ],
            dtype=np.uint8
        )

        return lower, upper

    def close(self):

        cv2.destroyWindow(
            self.window
        )


# =============================================================================
# DRAW HUD
# =============================================================================

def draw_hud(
    frame,
    gesture,
    sword_detected,
    fps,
    combo_text,
    debug
):

    h, w = frame.shape[:2]

    # Semi-transparent HUD.
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (15, 15),
        (430, 160),
        (15, 15, 15),
        -1
    )

    alpha_overlay(
        frame,
        overlay,
        0.55
    )

    if gesture:

        color = STYLE_COLORS.get(
            gesture,
            (255, 255, 255)
        )

        cv2.putText(
            frame,
            f"BREATHING: {gesture.upper()}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )

    else:

        cv2.putText(
            frame,
            "BREATHING: NONE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (180, 180, 180),
            2,
            cv2.LINE_AA
        )

    sword_text = (
        "SWORD: DETECTED"
        if sword_detected
        else
        "SWORD: NOT DETECTED"
    )

    cv2.putText(
        frame,
        sword_text,
        (30, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (120, 255, 120)
        if sword_detected
        else
        (100, 100, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (30, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    if combo_text:

        cv2.putText(
            frame,
            f"COMBO: {combo_text}",
            (30, 142),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 220, 100),
            2,
            cv2.LINE_AA
        )

    cv2.putText(
        frame,
        "Q: Quit | D: Debug | C: HSV | R: Reset",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA
    )


# =============================================================================
# DRAW SWORD DEBUG
# =============================================================================

def draw_sword_debug(frame, sword):

    if sword is None:
        return

    hilt, tip = sword

    cv2.line(
        frame,
        hilt,
        tip,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        hilt,
        8,
        (255, 0, 0),
        -1
    )

    cv2.circle(
        frame,
        tip,
        8,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        frame,
        "HILT",
        (
            hilt[0] + 10,
            hilt[1]
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "TIP",
        (
            tip[0] + 10,
            tip[1]
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Demon Slayer Real-Time Breathing VFX"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=CONFIG["camera_index"]
    )

    parser.add_argument(
        "--debug",
        action="store_true"
    )

    parser.add_argument(
        "--calibrate",
        action="store_true"
    )

    parser.add_argument(
        "--sword-lower",
        nargs=3,
        type=int,
        metavar=("H", "S", "V")
    )

    parser.add_argument(
        "--sword-upper",
        nargs=3,
        type=int,
        metavar=("H", "S", "V")
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # HSV configuration.
    # -------------------------------------------------------------------------

    lower = CONFIG[
        "sword_hsv_lower"
    ].copy()

    upper = CONFIG[
        "sword_hsv_upper"
    ].copy()

    if args.sword_lower:

        lower = np.array(
            args.sword_lower,
            dtype=np.uint8
        )

    if args.sword_upper:

        upper = np.array(
            args.sword_upper,
            dtype=np.uint8
        )

    # -------------------------------------------------------------------------
    # Camera.
    # -------------------------------------------------------------------------

    cap = cv2.VideoCapture(
        args.camera
    )

    if not cap.isOpened():

        raise SystemExit(
            f"\nCould not open camera {args.camera}.\n"
            "Try:\n"
            "    python demon_slayer_vfx.py --camera 1\n"
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CONFIG["width"]
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CONFIG["height"]
    )

    # -------------------------------------------------------------------------
    # Components.
    # -------------------------------------------------------------------------

    recognizer = GestureRecognizer()

    stabilizer = GestureStabilizer(
        CONFIG["gesture_history"]
    )

    sequence = SequenceDetector(
        COMBOS,
        CONFIG["combo_window"]
    )

    sword_tracker = SwordTracker(
        lower,
        upper,
        CONFIG["sword_min_contour_area"],
        CONFIG["sword_smoothing"]
    )

    vfx = VFXEngine(
        CONFIG["max_particles"]
    )

    calibrator = None

    if args.calibrate:

        calibrator = HSVCalibrator()

    # -------------------------------------------------------------------------
    # Runtime variables.
    # -------------------------------------------------------------------------

    debug = args.debug

    previous_time = time.time()

    fps = 0.0

    ultimate_style = None
    ultimate_end = 0.0

    print()
    print("=" * 70)
    print(" DEMON SLAYER BREATHING VFX")
    print("=" * 70)
    print()
    print("Camera:", args.camera)
    print("Sword HSV lower:", lower)
    print("Sword HSV upper:", upper)
    print()
    print("Controls:")
    print("  Q / ESC - Quit")
    print("  D       - Debug")
    print("  C       - HSV calibration")
    print("  R       - Reset VFX")
    print()

    # -------------------------------------------------------------------------
    # Main loop.
    # -------------------------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Failed to read frame.")
            break

        # Mirror webcam for natural interaction.
        frame = cv2.flip(
            frame,
            1
        )

        now = time.time()

        dt = min(
            now - previous_time,
            0.05
        )

        previous_time = now

        # ---------------------------------------------------------------------
        # FPS.
        # ---------------------------------------------------------------------

        instant_fps = 1.0 / max(
            dt,
            1e-6
        )

        fps = (
            0.9 * fps
            + 0.1 * instant_fps
        )

        # ---------------------------------------------------------------------
        # HSV calibration.
        # ---------------------------------------------------------------------

        if calibrator is not None:

            lower, upper = calibrator.values()

            sword_tracker.set_hsv(
                lower,
                upper
            )

        # ---------------------------------------------------------------------
        # Detect hand.
        # ---------------------------------------------------------------------

        raw_gesture, features, frame = recognizer.process(
            frame,
            debug
        )

        stable_gesture = stabilizer.update(
            raw_gesture
        )

        # ---------------------------------------------------------------------
        # Detect sword.
        # ---------------------------------------------------------------------

        sword = sword_tracker.update(
            frame
        )

        # ---------------------------------------------------------------------
        # Combo detector.
        # ---------------------------------------------------------------------

        unlocked = sequence.update(
            stable_gesture
        )

        if unlocked:

            ultimate_style = unlocked

            ultimate_end = (
                time.time()
                + 2.5
            )

            print(
                f"\nULTIMATE UNLOCKED: "
                f"{unlocked.upper()}"
            )

        # ---------------------------------------------------------------------
        # Determine active style.
        # ---------------------------------------------------------------------

        active_style = stable_gesture

        ultimate_active = (
            ultimate_style is not None
            and time.time() < ultimate_end
        )

        if not ultimate_active:

            ultimate_style = None

        # ---------------------------------------------------------------------
        # VFX only when both hand + sword exist.
        # ---------------------------------------------------------------------

        if (
            active_style is not None
            and sword is not None
        ):

            vfx.update(
                frame,
                active_style,
                sword,
                dt,
                ultimate=ultimate_active
            )

        else:

            # Still update existing particles.
            vfx.update_particles(dt)
            vfx.update_rings(
                frame,
                dt
            )
            vfx.update_moons(
                frame,
                dt
            )
            vfx.render_particles(
                frame
            )

        # ---------------------------------------------------------------------
        # Debug sword.
        # ---------------------------------------------------------------------

        if debug:

            draw_sword_debug(
                frame,
                sword
            )

            if sword_tracker.last_mask is not None:

                cv2.imshow(
                    "Sword Mask",
                    sword_tracker.last_mask
                )

        # ---------------------------------------------------------------------
        # Calibration preview.
        # ---------------------------------------------------------------------

        if calibrator is not None:

            hsv = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2HSV
            )

            mask = cv2.inRange(
                hsv,
                lower,
                upper
            )

            cv2.imshow(
                "Sword Mask",
                mask
            )

            cv2.setWindowTitle(
                calibrator.window,
                (
                    f"HSV "
                    f"L={lower.tolist()} "
                    f"U={upper.tolist()}"
                )
            )

        # ---------------------------------------------------------------------
        # HUD.
        # ---------------------------------------------------------------------

        draw_hud(
            frame,
            active_style,
            sword is not None,
            fps,
            sequence.text(),
            debug
        )

        # ---------------------------------------------------------------------
        # Ultimate banner.
        # ---------------------------------------------------------------------

        if ultimate_active:

            cv2.putText(
                frame,
                "ULTIMATE BREATHING!",
                (
                    430,
                    80
                ),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (0, 220, 255),
                3,
                cv2.LINE_AA
            )

        # ---------------------------------------------------------------------
        # Display.
        # ---------------------------------------------------------------------

        cv2.imshow(
            "Demon Slayer Breathing VFX",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # ---------------------------------------------------------------------
        # Controls.
        # ---------------------------------------------------------------------

        if key == ord("q") or key == 27:

            break

        elif key == ord("d"):

            debug = not debug

        elif key == ord("r"):

            vfx.reset()
            stabilizer.reset()

            print(
                "VFX reset."
            )

        elif key == ord("c"):

            if calibrator is None:

                calibrator = HSVCalibrator()

            else:

                lower, upper = calibrator.values()

                print()
                print(
                    "Current HSV values:"
                )
                print(
                    "Lower:",
                    lower.tolist()
                )
                print(
                    "Upper:",
                    upper.tolist()
                )

                calibrator.close()

                calibrator = None

    # -------------------------------------------------------------------------
    # Cleanup.
    # -------------------------------------------------------------------------

    cap.release()

    recognizer.close()

    if calibrator is not None:

        calibrator.close()

    cv2.destroyAllWindows()

    print()
    print("Application closed.")
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()