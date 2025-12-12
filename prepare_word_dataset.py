#!/usr/bin/env python3
"""
Script untuk prepare dataset kata BISINDO
- Split train/val/test
- Extract 2 hands landmarks (consistently: left hand first, right hand second)
- Save to CSV

Usage:
    python3 prepare_word_dataset.py --input data/words_raw --output data/words_processed
"""

import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm

# Initialize MediaPipe Hands untuk up to 2 tangan
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,  # penting: up to 2 hands
    min_detection_confidence=0.5
)

def extract_two_hands_landmarks(image_path):
    """
    Extract landmarks dari up to 2 tangan.
    Returns: list of 126 features (21 landmarks x 3 coords x 2 hands)
             atau None kalau gak detect sama sekali atau hasil tidak valid.

    Behavior:
      - Jika dua tangan terdeteksi: urutkan berdasarkan handedness (LEFT, RIGHT)
      - Jika hanya satu tangan terdeteksi: duplikasi vektor 63 -> 126 untuk konsistensi
      - Jika lebih dari 2 tangan (jarang): ambil dua teratas berdasarkan handedness
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image
    results = hands.process(image_rgb)

    # No hands detected
    if not results.multi_hand_landmarks:
        return None

    # Create list of (handedness_label, landmarks_list)
    detected = []

    # If multi_handedness is available, use it to get LEFT/RIGHT labels per hand
    # MediaPipe pairs multi_hand_landmarks and multi_handedness in same order
    handednesses = []
    if results.multi_handedness:
        for h in results.multi_handedness:
            # label can be 'Left' or 'Right' (capitalization)
            label = h.classification[0].label if h.classification else ""
            handednesses.append(label)
    else:
        # fallback: empty labels
        handednesses = [""] * len(results.multi_hand_landmarks)

    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        label = handednesses[idx] if idx < len(handednesses) else ""
        coords = []
        for landmark in hand_landmarks.landmark:
            coords.extend([landmark.x, landmark.y, landmark.z])
        detected.append((label, coords))

    # Sort detected by label to try to get consistent ordering: Left first, then Right.
    # If labels are empty or both same, preserve original order.
    def sort_key(item):
        # Left -> 0, Right -> 1, others -> 2 (preserve relative order beyond that)
        if item[0].lower() == 'left':
            return 0
        if item[0].lower() == 'right':
            return 1
        return 2

    detected_sorted = sorted(detected, key=sort_key)

    # Take up to two hands (if >2, we keep first two after sorting)
    detected_sorted = detected_sorted[:2]

    # Build all_landmarks sequence
    all_landmarks = []
    for _, coords in detected_sorted:
        all_landmarks.extend(coords)

    # Now handle lengths:
    # - 63 -> only one hand detected
    # - 126 -> two hands detected
    # If 63, duplicate it to make 126 (hand1 duplicated as hand2)
    if len(all_landmarks) == 63:
        all_landmarks = all_landmarks + all_landmarks

    # If more/less than expected, return None (so sample counted as failed)
    if len(all_landmarks) != 126:
        # you can log here if needed for debugging
        return None

    return all_landmarks

def scan_dataset(input_folder):
    """
    Scan dataset folder dan buat list semua image paths dengan label
    Returns: list of (image_path, label)
    """
    input_path = Path(input_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Folder {input_folder} tidak ditemukan!")

    print(f"📂 Scanning dataset di: {input_folder}")

    dataset = []
    class_folders = sorted([f for f in input_path.iterdir() if f.is_dir()])

    print(f"📊 Ditemukan {len(class_folders)} kelas kata")

    for class_folder in class_folders:
        class_name = class_folder.name

        # Get all images (common extensions)
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'):
            image_files.extend(list(class_folder.glob(ext)))

        for image_file in image_files:
            dataset.append((str(image_file), class_name))

        print(f"   {class_name}: {len(image_files)} images")

    print(f"\n✅ Total: {len(dataset)} images\n")

    return dataset

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset menjadi train/val/test
    Returns: train_data, val_data, test_data
    """
    print("✂️  Splitting dataset...")

    # Separate paths and labels
    image_paths, labels = zip(*dataset)

    # First split: train + (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=42
    )

    # Second split: val + test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=42
    )

    train_data = list(zip(train_paths, train_labels))
    val_data = list(zip(val_paths, val_labels))
    test_data = list(zip(test_paths, test_labels))

    print(f"   Train: {len(train_data)} ({train_ratio*100:.0f}%)")
    print(f"   Val:   {len(val_data)} ({val_ratio*100:.0f}%)")
    print(f"   Test:  {len(test_data)} ({test_ratio*100:.0f}%)")
    print()

    return train_data, val_data, test_data

def extract_and_save_csv(data, output_file, split_name):
    """
    Extract landmarks dari images dan save ke CSV
    """
    print(f"🔄 Extracting landmarks untuk {split_name}...")

    rows = []
    failed_count = 0
    total_count = 0

    for image_path, label in tqdm(data, desc=f"Processing {split_name}"):
        total_count += 1
        landmarks = extract_two_hands_landmarks(image_path)

        # sekarang kita expect 126 features (63 per hand x 2)
        if landmarks is not None and len(landmarks) == 126:
            # Create row: [label, hand1_x, hand1_y, hand1_z, ..., hand2_x, hand2_y, hand2_z, ...]
            row = [label] + landmarks
            rows.append(row)
        else:
            failed_count += 1

    # Create DataFrame columns
    columns = ['label']

    # Hand 1 landmarks (21 x 3 = 63)
    for i in range(21):
        columns.extend([f'hand1_{i}_x', f'hand1_{i}_y', f'hand1_{i}_z'])

    # Hand 2 landmarks (21 x 3 = 63)
    for i in range(21):
        columns.extend([f'hand2_{i}_x', f'hand2_{i}_y', f'hand2_{i}_z'])

    # Validate shape before creating DataFrame
    if rows:
        # each row should have length == len(columns)
        mismatch_idx = None
        for idx, r in enumerate(rows):
            if len(r) != len(columns):
                mismatch_idx = idx
                break
        if mismatch_idx is not None:
            # safety: print info and skip problematic rows
            print(f"⚠️  Detected row with unexpected length at index {mismatch_idx}: expected {len(columns)} got {len(rows[mismatch_idx])}")
            # filter rows that match expected length
            rows = [r for r in rows if len(r) == len(columns)]

    df = pd.DataFrame(rows, columns=columns)

    # Save to CSV
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"   ✅ Saved: {output_file}")
    print(f"   ✅ Samples: {len(df)} / {total_count}")
    print(f"   ⚠️  Failed: {failed_count} (tidak detect 2 tangan atau hasil tidak valid)")
    print()

    return len(df), failed_count

def main():
    parser = argparse.ArgumentParser(description='Prepare BISINDO word dataset')
    parser.add_argument('--input', type=str, default='data/words_raw',
                        help='Input folder dengan raw images')
    parser.add_argument('--output', type=str, default='data/words_processed',
                        help='Output folder untuk CSV files')

    args = parser.parse_args()

    print("="*60)
    print("🚀 BISINDO Word Dataset Preparation")
    print("="*60)
    print()

    # Create output folder
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)

    try:
        # 1. Scan dataset
        dataset = scan_dataset(args.input)

        if len(dataset) == 0:
            print("❌ Dataset kosong!")
            return

        # 2. Split dataset
        train_data, val_data, test_data = split_dataset(dataset)

        # 3. Extract landmarks dan save ke CSV
        train_file = output_path / 'train_words.csv'
        val_file = output_path / 'val_words.csv'
        test_file = output_path / 'test_words.csv'

        total_success = 0
        total_failed = 0

        for data, file, name in [
            (train_data, train_file, 'Train'),
            (val_data, val_file, 'Val'),
            (test_data, test_file, 'Test')
        ]:
            success, failed = extract_and_save_csv(data, file, name)
            total_success += success
            total_failed += failed

        # Summary
        print("="*60)
        print("🎉 Dataset Preparation Complete!")
        print("="*60)
        print(f"\n📊 Summary:")
        print(f"   Total images processed: {len(dataset)}")
        print(f"   Successful extractions: {total_success}")
        print(f"   Failed extractions: {total_failed}")
        total_processed = total_success + total_failed
        if total_processed > 0:
            print(f"   Success rate: {total_success/total_processed*100:.1f}%")
        print()
        print(f"📁 Output files:")
        print(f"   {train_file}")
        print(f"   {val_file}")
        print(f"   {test_file}")
        print()
        print("📝 Next step: Train model dengan train_word_model.py")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        hands.close()

if __name__ == "__main__":
    main()
