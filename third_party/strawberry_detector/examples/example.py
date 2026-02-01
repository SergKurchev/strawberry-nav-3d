"""
Example usage of strawberry_detector package.
"""

import json
from pathlib import Path

# Import the detector
from strawberry_detector import detect, StrawberryDetector


def example_quick_detection():
    """Quick detection using convenience function."""
    print("=" * 60)
    print("Example 1: Quick Detection")
    print("=" * 60)
    
    # Replace with your image path or URL
    image_path = "https://media.dom-i-remont.info/2021/03/unnamed-16.jpg"
    
    # Single-line detection
    result = detect(image_path)
    
    print(f"Found {result['detections_count']} strawberries")
    
    for det in result['detections']:
        print(f"  ID {det['id']}: {det['class_name']}, "
              f"confidence={det['confidence']:.2f}, "
              f"distance={det['depth'].get('mean_cm', 'N/A')} cm")
    
    return result


def example_batch_detection():
    """Process multiple images with reused detector."""
    print("=" * 60)
    print("Example 2: Batch Detection")
    print("=" * 60)
    
    # Initialize detector once
    detector = StrawberryDetector()
    
    # Example image URLs (replace with your images)
    images = [
        "https://media.dom-i-remont.info/2021/03/unnamed-16.jpg",
        # Add more images here
    ]
    
    all_results = []
    
    for img_path in images:
        print(f"\nProcessing: {img_path}")
        result = detector.detect(img_path)
        all_results.append(result)
        print(f"  Found {result['detections_count']} strawberries")
    
    return all_results


def example_with_visualization():
    """Detection with visualization output."""
    print("=" * 60)
    print("Example 3: Detection with Visualization")
    print("=" * 60)
    
    detector = StrawberryDetector()
    
    image_path = "https://media.dom-i-remont.info/2021/03/unnamed-16.jpg"
    output_image = Path(__file__).parent.parent / "output_visualization.jpg"
    
    result, vis_image = detector.detect_and_visualize(
        image_path,
        output_image_path=output_image
    )
    
    print(f"Visualization saved to: {output_image}")
    print(f"Found {result['detections_count']} strawberries")
    
    return result


def example_save_json():
    """Save results to JSON file."""
    print("=" * 60)
    print("Example 4: Save Results to JSON")
    print("=" * 60)
    
    result = detect("https://media.dom-i-remont.info/2021/03/unnamed-16.jpg")
    
    output_path = Path(__file__).parent.parent / "detection_results.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")
    
    return result


if __name__ == "__main__":
    # Run examples
    print("\n🍓 Strawberry Detector Examples\n")
    
    # You can uncomment the examples you want to run:
    
    result = example_quick_detection()
    
    # result = example_batch_detection()
    
    # result = example_with_visualization()
    
    # result = example_save_json()
    
    print("\n✅ Examples complete!")
