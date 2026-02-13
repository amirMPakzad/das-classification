# DAS Classification

Deep learning–based event classification using Distributed Acoustic Sensing (DAS) signals.

## Reference

Dataset and methodology adapted from:
**Comprehensive Dataset for Event Classification Using DAS Systems**
[https://springernature.figshare.com/articles/dataset/Comprehensive_Dataset_for_Event_Classification_Using_Distributed_Acoustic_Sensing_DAS_Systems/27004732](https://springernature.figshare.com/articles/dataset/Comprehensive_Dataset_for_Event_Classification_Using_Distributed_Acoustic_Sensing_DAS_Systems/27004732)


## Dataset

Download the dataset from the link above and set the correct path in `configs/app.yaml`.

## Train

Recommended:

```bash
./scripts/train.sh
```

Manual:

```bash
source venv/bin/activate
python -m das_classification.cli train --config configs/app.yaml
```

## Outputs

Models and logs are saved in `runs/` (best.pt, last.pt, history.jsonl).
