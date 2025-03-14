# I-RAVEN

## Get the Data
Generate the I-RAVEN dataset with the instructions proveded [here](https://github.com/husheng12345/SRAN) and save it in this folder.

```bash
git clone https://github.com/husheng12345/SRAN
pip2 install --user -r SRAN/I-RAVEN/requirements.txt
python2 SRAN/I-RAVEN/main.py --save-dir .
```

## Prepare the Data

Run the rule preprocessing script:
```bash
python src/datasets/generation/iraven_task.py --config center_single --load_dir data/I-RAVEN --save_dir data
```

# I-RAVEN-X

## Generate the dataset
Run the rule preprocessing script:
```bash
python src/datasets/generation/iravenx_task.py --n 10 --maxval 1000 --nconf 10 --save_dir data
```
