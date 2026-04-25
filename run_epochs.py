import os
import subprocess

def run():
    # Number of epochs to test
    epochs_to_test = [10, 50, 100]

    for epoch in epochs_to_test:
        print(f"=============================")
        print(f"Training for {epoch} epochs...")
        print(f"=============================")

        # Modify the train script to use the specific number of epochs and save checkpoint
        with open("src/train.py", "r") as f:
            train_code = f.read()

        # Temporarily modify the bottom of the train file
        modified_code = train_code.replace(
            'train(num_epochs=5, batch_size=2)',
            f'train(num_epochs={epoch}, batch_size=2)'
        )

        with open("src/train_temp.py", "w") as f:
            f.write(modified_code)

        # Run the training script
        subprocess.run(["python3", "src/train_temp.py"], check=True)

        # Modify the inference script to use the right checkpoint and save the right output
        with open("src/inference.py", "r") as f:
            infer_code = f.read()

        modified_infer_code = infer_code.replace(
            'checkpoint_file = "checkpoints/sr_model_epoch_5.pth"',
            f'checkpoint_file = "checkpoints/sr_model_epoch_{epoch}.pth"'
        ).replace(
            'output_result = "data/output_comparison.png"',
            f'output_result = "assets/epoch_{epoch}_comparison.png"'
        ).replace(
            'output_result = "assets/example_comparison.png"',
            f'output_result = "assets/epoch_{epoch}_comparison.png"'
        )

        with open("src/inference_temp.py", "w") as f:
            f.write(modified_infer_code)

        # Run the inference script
        print(f"Running inference for {epoch} epochs...")
        subprocess.run(["python3", "src/inference_temp.py"], check=True)

    # Clean up temporary files
    if os.path.exists("src/train_temp.py"):
        os.remove("src/train_temp.py")
    if os.path.exists("src/inference_temp.py"):
        os.remove("src/inference_temp.py")

    print("All trainings completed!")

if __name__ == "__main__":
    run()
