import torch, glob
def count(root):
    n=0
    for p in glob.glob(root+"/**/*.pt", recursive=True):
        d=torch.load(p, map_location="cpu")
        if "x" in d: n += d["x"].shape[0]
    return n

if __name__ == "__main__":
    print("orig:", count("../data/processed"))
    print("chunk:", count("../data/processed_xy_chunked"))