import subprocess
import sys

SCRIPTS = [
    "src/1_fetch_data.py",
    "src/2_extract_mechanics.py",
    "src/3_cluster_mechanics.py",
    "src/4_generate_jokes.py",
]


def main():
    for script in SCRIPTS:
        print(f"\n{'=' * 50}\nRunning {script}...\n{'=' * 50}")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"Error executing {script}. Stopping pipeline.")
            sys.exit(result.returncode)
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
