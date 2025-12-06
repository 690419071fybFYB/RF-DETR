
import argparse
from rfdetr import RFDETRBase

def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR with specific ideas")
    parser.add_argument('--dataset', type=str, default="/home/fyb/datasets/RSOD_cocoFormat", help="Path to dataset")
    parser.add_argument('--output_dir', type=str, default="results_ideas", help="Output directory")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=6, help="Batch size")
    
    # Idea flags
    parser.add_argument('--use_dual_prior', action='store_true', help="Enable Dual Prior Calibration")
    parser.add_argument('--use_fourier_mixer', action='store_true', help="Enable Fourier Token Mixer")
    parser.add_argument('--use_dynamic_fusion', action='store_true', help="Enable Dynamic Frequency Fusion")
    
    args = parser.parse_args()
    
    print(f"Initializing RFDETRBase with:")
    print(f"  use_dual_prior={args.use_dual_prior}")
    print(f"  use_fourier_mixer={args.use_fourier_mixer}")
    print(f"  use_dynamic_fusion={args.use_dynamic_fusion}")
    
    model = RFDETRBase(
        use_dynamic_query=True,
        use_dual_prior=args.use_dual_prior,
        use_fourier_mixer=args.use_fourier_mixer,
        use_dynamic_fusion=args.use_dynamic_fusion
    )
    
    print(f"Starting training for {args.epochs} epochs...")
    model.train(
        dataset_dir=args.dataset,
        dataset_file="coco",
        coco_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=args.output_dir,
        early_stopping=True,
        early_stopping_patience=5,
    )

if __name__ == "__main__":
    main()
