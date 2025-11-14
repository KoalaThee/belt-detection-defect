from pathlib import Path

ROOT = Path(__file__).parent / "patch_data"
splits = ["train", "val", "test"]
categories = ["empty", "present"]

print("Image count in each folder:")
print("-" * 40)

for split in splits:
    print(f"\n{split}:")
    for category in categories:
        folder = ROOT / split / category
        if folder.exists():
            count = len(list(folder.glob("*.jpg")))
            print(f"  {category}: {count} images")
        else:
            print(f"  {category}: folder not found")