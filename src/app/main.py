# src/app/main.py
import argparse

def main():
    parser = argparse.ArgumentParser(description="Balance Ball Control System")
    parser.add_argument(
        "--mode",
        choices=["cv", "dl"],
        default="cv",
        help="Select tracking mode: 'cv' for classical computer vision or 'dl' for deep learning."
    )
    args = parser.parse_args()

    if args.mode == "cv":
        from SphereDetectionNN.src.control import balance_ball_pid_cv as balance
    else:
        from control import balance_ball_pid_dl as balance

    balance.main_loop()

if __name__ == "__main__":
    main()
