# Get Wikipedia
Assuming you're in the main project directory.
```
./experiments/get_data.sh <data_dir>
```

# Convert to matrices
```
python -m noallen.torchtext.corpus2context_matrices
```

# Run training
```
python -m noallen.train --config <config_file> --save_path
```
