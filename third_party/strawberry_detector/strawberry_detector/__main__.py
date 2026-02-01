"""
CLI interface for strawberry detection.

Usage:
    python -m strawberry_detector --image photo.jpg --output results/
    python -m strawberry_detector --folder images/ --output results/ --save-full
"""

import argparse
import json
import sys
from pathlib import Path

from .detector import StrawberryDetector


def main():
    parser = argparse.ArgumentParser(
        description="🍓 Strawberry Detection with Depth Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image - basic (JSON only)
  python -m strawberry_detector --image photo.jpg --output results.json
  
  # Single image - full output (JSON + depth.npy + masks.npy + visualization)
  python -m strawberry_detector --image photo.jpg --output results/ --save-full
  
  # Folder of images - full output
  python -m strawberry_detector --folder images/ --output results/ --save-full --visualize
  
  # From URL
  python -m strawberry_detector --image "https://example.com/strawberries.jpg" --output results/ --save-full
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", "-i",
        help="Path or URL to input image"
    )
    input_group.add_argument(
        "--folder", "-f",
        help="Path to folder containing images"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path (JSON file for basic mode, folder for --save-full or batch)"
    )
    
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Save visualization image(s)"
    )
    
    parser.add_argument(
        "--save-full",
        action="store_true",
        help="Save full output: JSON + depth.npy + masks.npy + visualization"
    )
    
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (auto-detected if not specified)"
    )
    
    args = parser.parse_args()

    # Force torch default device to CPU when requested (avoid MPS default tensors)
    if getattr(args, 'device', None) == 'cpu':
        import torch
        try:
            torch.set_default_device('cpu')
        except Exception:
            pass
    
    print("=" * 60)
    print("🍓 Strawberry Detection with Depth Estimation")
    print("=" * 60)
    
    try:
        # Initialize detector
        detector = StrawberryDetector(device=args.device)
        
        if args.folder:
            # Folder mode
            print(f"📁 Input folder: {args.folder}")
            print(f"📁 Output folder: {args.output}")
            print(f"💾 Save full: {args.save_full}")
            print()
            
            results = detector.detect_folder(
                args.folder,
                output_folder=args.output,
                conf_threshold=args.conf,
                visualize=args.visualize,
                save_full=args.save_full
            )
            
            # Save combined results
            output_folder = Path(args.output)
            combined_path = output_folder / "all_results.json"
            with open(combined_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print()
            print("=" * 60)
            print(f"✅ Batch processing complete!")
            print(f"📊 Processed {len(results)} images")
            
            # Aggregate stats
            total_strawberries = sum(r.get("detections_count", 0) for r in results if "detections_count" in r)
            print(f"🍓 Total strawberries found: {total_strawberries}")
            print(f"💾 Results saved to: {args.output}")
            print("=" * 60)
            
        else:
            # Single image mode
            print(f"📷 Image: {args.image}")
            print(f"💾 Output: {args.output}")
            print(f"💾 Save full: {args.save_full}")
            print()
            
            if args.save_full:
                # Full output mode - saves to folder
                output_path = Path(args.output)
                result = detector.detect_and_save_full(
                    args.image,
                    output_dir=output_path,
                    conf_threshold=args.conf,
                    save_visualization=args.visualize or True  # Always save vis in full mode
                )
                
                print()
                print("=" * 60)
                print(f"✅ Detection complete!")
                print(f"📊 Found {result['detections_count']} strawberries")
                print(f"\n📁 Output files:")
                for key, path in result["output_files"].items():
                    if isinstance(path, list):
                        print(f"   - {key}: {len(path)} files")
                    else:
                        print(f"   - {key}: {Path(path).name}")
                
            else:
                # Basic mode - JSON only
                if args.visualize:
                    output_path = Path(args.output)
                    vis_path = output_path.parent / f"{output_path.stem}_vis.jpg"
                    result, vis = detector.detect_and_visualize(
                        args.image,
                        output_image_path=vis_path,
                        conf_threshold=args.conf
                    )
                    print(f"🖼️ Visualization saved: {vis_path}")
                else:
                    result = detector.detect(args.image, conf_threshold=args.conf)
                
                # Save JSON
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print()
                print("=" * 60)
                print(f"✅ Detection complete!")
                print(f"📊 Found {result['detections_count']} strawberries")
            
            # Print summary
            stats = result["statistics"]
            print(f"   - Ripe: {stats.get('total_ripe', 0)}")
            print(f"   - Unripe: {stats.get('total_unripe', 0)}")
            
            if "closest_distance_cm" in stats:
                print(f"   - Closest: {stats['closest_distance_cm']:.1f} cm")
            if "furthest_distance_cm" in stats:
                print(f"   - Furthest: {stats['furthest_distance_cm']:.1f} cm")
            
            print(f"💾 Results saved to: {args.output}")
            print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
