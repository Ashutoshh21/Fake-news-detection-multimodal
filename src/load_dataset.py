from PIL import Image
import pandas as pd
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_data():
    df = pd.read_csv("../dataset/data.csv")
    return df

def load_image(image_name):
    path = f"../dataset/images/{image_name}"
    img = Image.open(path).convert("RGB")
    img = transform(img)
    return img

def load_metadata(row):
    score = row["score"]
    comments = row["comments"]
    return [score, comments]

def load_sample(df, idx):
    row = df.iloc[idx]

    text = row["text"]
    image = load_image(row["image"])
    metadata = load_metadata(row)
    label = row["label"]

    return text, image, metadata, label

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print("\nDataset shape:", df.shape
)
    # test image loading
    img = load_image(df.iloc[0]["image"])
    print("\nImage loaded:", img)

    # test metadata loading
    metadata = load_metadata(df.iloc[0])
    print("\nMetadata:", metadata)
